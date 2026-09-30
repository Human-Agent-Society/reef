"""Sampled OPD's complete loss on CPU, with only the Megatron batch adapters replaced."""

from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from recipes.sdft.slime import SdftAlgorithm, SdftSettings
from reef.train.slime_backend.distill.objective import distill_loss
from reef.train.slime_backend.score_centering import TOPK_INDICES_KEY, TOPK_LOG_PROBS_KEY


@pytest.fixture
def cpu_batch_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real loss and vocabulary math; replace GPU runtime batch extraction."""

    def _responses(logits, *, args, unconcat_tokens, total_lengths, response_lengths):
        rows = logits.split(response_lengths)
        return [
            (row, tokens[-length:])
            for row, tokens, length in zip(rows, unconcat_tokens, response_lengths, strict=True)
        ]

    def _log_probs(logits, *, with_entropy, **options):
        values = [
            torch.log_softmax(rows, -1).gather(-1, tokens[:, None])[:, 0]
            for rows, tokens in _responses(logits, **options)
        ]
        return None, {"log_probs": values}

    def _reducer(total_lengths, response_lengths, loss_masks, packed, calculate_per_token_loss):
        def _reduce(values):
            return sum(
                (part * mask).sum() / mask.sum().clamp_min(1)
                for part, mask in zip(values.split(response_lengths), loss_masks, strict=True)
            )

        return _reduce

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_group=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    loss_module = ModuleType("slime.backends.megatron_utils.loss")
    loss_module.get_responses = _responses
    loss_module.get_log_probs_and_entropy = _log_probs
    monkeypatch.setitem(sys.modules, loss_module.__name__, loss_module)
    cp_module = ModuleType("slime.backends.megatron_utils.cp_utils")
    cp_module.get_sum_of_sample_mean = _reducer
    monkeypatch.setitem(sys.modules, cp_module.__name__, cp_module)


@pytest.mark.unit
@pytest.mark.parametrize("cap", [0.0, 1.2])
@pytest.mark.parametrize("skip", [0, 1])
@pytest.mark.parametrize("head_size", [3, 7])
@pytest.mark.parametrize("matched", [False, True])
def test_opd_loss_matches_full_vocabulary_centered_gradient(
    cpu_batch_adapters, cap: float, skip: int, head_size: int, matched: bool
) -> None:
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(7, 7, generator=generator).requires_grad_()
    probabilities = logits.detach().softmax(-1)
    head_ids = probabilities.topk(head_size, dim=-1).indices
    if matched:
        sampler = probabilities.clone()
    elif head_size == 7:
        sampler = torch.randn(7, 7, generator=generator).softmax(-1)
    else:
        # The missing tail is proportional to p, so the head approximation
        # must agree with the independent full-vocabulary gradient below.
        in_head = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, head_ids, True)
        tail = probabilities.masked_fill(in_head, 0)
        sampler = 0.2 * tail / tail.sum(-1, keepdim=True)
        head = torch.randn(7, head_size, generator=generator).softmax(-1) * 0.8
        sampler.scatter_(-1, head_ids, head)
    sampler_log_probs = sampler.log()
    teacher_log_probs = torch.randn(7, 7, generator=generator).log_softmax(-1).requires_grad_()
    teacher_head_ids = teacher_log_probs.detach().topk(2, dim=-1).indices
    sampled = torch.tensor([0, 3, 6, 2, 5, 1, 4])  # Includes tokens outside the recorded head.
    lengths = [3, 2, 2]
    masks = [torch.tensor([1.0, 0.0, 1.0]), torch.ones(2), torch.ones(2)]
    sample_weights = [1.5, 0.5, 0.0]
    batch = {
        "total_lengths": [length + 1 for length in lengths],
        "response_lengths": lengths,
        "unconcat_tokens": [torch.cat([torch.tensor([0]), tokens]) for tokens in sampled.split(lengths)],
        "loss_masks": masks,
        "distill_sample_weights": sample_weights,
        "rollout_log_probs": list(sampler_log_probs.gather(-1, sampled[:, None])[:, 0].split(lengths)),
        "distill_teacher_topk_ids": list(teacher_head_ids.split(lengths)),
        "distill_teacher_topk_log_probs": list(teacher_log_probs.gather(-1, teacher_head_ids).split(lengths)),
        "distill_teacher_sampled_log_probs": list(teacher_log_probs.gather(-1, sampled[:, None])[:, 0].split(lengths)),
        TOPK_INDICES_KEY: list(head_ids.split(lengths)),
        TOPK_LOG_PROBS_KEY: list(sampler_log_probs.gather(-1, head_ids).split(lengths)),
    }
    args = Namespace(
        score_centering=True,
        score_centering_top_k=head_size,
        score_centering_min_tail_mass=1e-6,
        calculate_per_token_loss=False,
    )
    SdftAlgorithm().apply_driver_options(
        args,
        SdftSettings(
            divergence="reverse",
            top_k=2,
            importance_sampling_cap=cap,
            importance_sampling_level="token",
            skip_response_tokens=skip,
        ),
    )
    reduce = get_sum_of_sample_mean(batch["total_lengths"], lengths, masks, None, False)
    loss, metrics = distill_loss(args, batch, logits, reduce)
    loss.backward()

    # Analytic score vectors, without the implementation's head or centering kernels.
    scores = torch.eye(7)[None, :, :] - probabilities[:, None, :]
    if cap == 0:
        weights = torch.ones_like(probabilities)
    else:
        weights = (probabilities / sampler).clamp(max=cap)
    rows = torch.arange(7)
    advantages = teacher_log_probs.detach()[rows, sampled] - probabilities.log()[rows, sampled]
    expected = -advantages[:, None] * (
        weights[rows, sampled, None] * scores[rows, sampled] - ((sampler * weights)[:, :, None] * scores).sum(1)
    )
    effective_masks = [mask.clone() for mask in masks]
    for mask in effective_masks:
        mask[:skip] = 0
    reduction_weights = torch.cat(
        [mask * weight / mask.sum().clamp_min(1) for mask, weight in zip(effective_masks, sample_weights, strict=True)]
    )
    torch.testing.assert_close(logits.grad, expected * reduction_weights[:, None], atol=2e-6, rtol=2e-5)
    assert teacher_log_probs.grad is None
    assert torch.equal(batch["loss_masks"][0], torch.tensor([1.0, 0.0, 1.0]))
    assert "advantages" not in batch
    torch.testing.assert_close(metrics["loss"], loss.detach())

    args.score_centering = False
    plain_loss, plain_metrics = distill_loss(args, batch, logits, reduce)
    torch.testing.assert_close(loss.detach(), plain_loss.detach() + metrics["score_centering_term"])
    assert "score_centering_term" not in plain_metrics
    if matched:
        torch.testing.assert_close(metrics["score_centering_term"], torch.zeros(()), atol=2e-6, rtol=0)
    else:
        assert abs(metrics["score_centering_term"].item()) > 1e-5
