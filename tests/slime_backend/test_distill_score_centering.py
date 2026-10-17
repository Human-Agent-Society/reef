"""Sampled OPD's complete loss on CPU, with only the Megatron batch adapters replaced."""

from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from recipes.opd.slime import OpdAlgorithm, OpdSettings
from recipes.sdft.slime import SdftAlgorithm, SdftSettings
from reef.train.slime_backend.distill.algorithm import ADVANTAGES_KEY
from reef.train.slime_backend.distill.objective import (
    distill_actor_pre_train,
    distill_loss,
    normalized_sampled_advantages,
)
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
@pytest.mark.parametrize("advantage_clip", [0.0, 0.5])
@pytest.mark.parametrize("normalized", [False, True])
def test_opd_loss_matches_full_vocabulary_centered_gradient(
    cpu_batch_adapters, cap: float, skip: int, head_size: int, matched: bool, advantage_clip: float, normalized: bool
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
    if normalized:
        batch[ADVANTAGES_KEY] = normalized_sampled_advantages(
            batch["distill_teacher_sampled_log_probs"],
            batch["rollout_log_probs"],
            masks,
            skip_response_tokens=skip,
            device="cpu",
            data_parallel_group=None,
            data_parallel_world=1,
        )
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
            advantage_clip=advantage_clip,
            normalize_advantages=normalized,
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
    if normalized:
        advantages = torch.cat(batch[ADVANTAGES_KEY])
    else:
        advantages = teacher_log_probs.detach()[rows, sampled] - probabilities.log()[rows, sampled]
    if advantage_clip > 0:
        advantages = advantages.clamp(min=-advantage_clip, max=advantage_clip)
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


@pytest.mark.unit
def test_normalizing_loss_requires_the_pre_train_advantages(cpu_batch_adapters) -> None:
    sampled = torch.tensor([1, 2])
    batch = {
        "total_lengths": [3],
        "response_lengths": [2],
        "unconcat_tokens": [torch.cat([torch.tensor([0]), sampled])],
        "loss_masks": [torch.ones(2)],
        "distill_sample_weights": [1.0],
        "distill_teacher_topk_ids": [torch.zeros(2, 1, dtype=torch.long)],
        "distill_teacher_topk_log_probs": [torch.full((2, 1), -1.0)],
        "distill_teacher_sampled_log_probs": [torch.full((2,), -1.0)],
    }
    args = Namespace(score_centering=False, calculate_per_token_loss=False)
    SdftAlgorithm().apply_driver_options(
        args,
        SdftSettings(divergence="reverse", top_k=1, importance_sampling_cap=0.0, normalize_advantages=True),
    )

    with pytest.raises(RuntimeError, match=ADVANTAGES_KEY):
        distill_loss(args, batch, torch.randn(2, 4, requires_grad=True), lambda values: values.sum())


@pytest.mark.unit
def test_opd_pre_train_hook_normalizes_the_engine_scored_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_data_parallel_group=lambda: None)
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    args = Namespace()
    OpdAlgorithm().apply_driver_options(
        args, OpdSettings(teacher_url="http://teacher:30001", teacher_model_path="/models/teacher")
    )
    teacher = [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5, -3.0, -1.0])]
    rollout = [torch.tensor([-0.5, -1.0]), torch.tensor([-1.5, -1.0, -1.0])]
    masks = [torch.tensor([1, 1]), torch.tensor([1, 0, 1])]
    rollout_data = {
        "teacher_tokens": [torch.tensor([7, 1, 2]), torch.tensor([7, 3, 4, 5])],
        "distill_teacher_topk_ids": [torch.zeros(2, 1, dtype=torch.long), torch.zeros(3, 1, dtype=torch.long)],
        "distill_teacher_topk_log_probs": [row[:, None] for row in teacher],
        "distill_teacher_sampled_log_probs": teacher,
        "rollout_log_probs": rollout,
        "loss_masks": masks,
    }

    distill_actor_pre_train(SimpleNamespace(args=args), rollout_data)

    advantages = torch.cat(rollout_data[ADVANTAGES_KEY])
    trained = torch.cat(masks).bool()
    torch.testing.assert_close(advantages[trained].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[trained].std(unbiased=False), torch.tensor(1.0))
    assert not advantages[~trained].any()
