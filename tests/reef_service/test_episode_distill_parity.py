"""Full masked episode suffixes through the existing teacher indexing and loss on CPU."""

from __future__ import annotations

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from recipes.sdft.slime import SdftAlgorithm, SdftSettings
from recipes.sdpo.slime import SdpoAlgorithm, SdpoSettings
from reef.train.slime_backend.distill.objective import distill_loss
from reef.train.slime_backend.distill.teacher import gather_teacher_log_probs

from . import test_distill_score_centering


@pytest.fixture
def cpu_batch_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    test_distill_score_centering.cpu_batch_adapters.__wrapped__(monkeypatch)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sdft", "sdpo"])
@pytest.mark.parametrize("importance_sampling_cap", [0.0, 2.0])
def test_episode_loss_matches_dense_selected_mean_and_has_zero_context_gradients(
    cpu_batch_adapters: None, method: str, importance_sampling_cap: float
) -> None:
    generator = torch.Generator().manual_seed(91)
    response_ids = torch.tensor([1, 2, 3, 7, 8, 4, 5, 9, 6])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    student = torch.randn(9, 11, generator=generator).requires_grad_()
    teacher = torch.randn(9, 11, generator=generator).log_softmax(-1)
    sampler_log_probs = student.detach().log_softmax(-1).gather(-1, response_ids[:, None])[:, 0] - 0.2
    batch = {
        "total_lengths": [12],
        "response_lengths": [9],
        "unconcat_tokens": [torch.cat([torch.tensor([0, 0, 0]), response_ids])],
        "loss_masks": [mask],
        "distill_sample_weights": [1.0],
        "rollout_log_probs": [sampler_log_probs],
    }
    args = Namespace(calculate_per_token_loss=False, log_probs_chunk_size=2, score_centering=False)
    if method == "sdft":
        SdftAlgorithm().apply_driver_options(
            args, SdftSettings(importance_sampling_cap=importance_sampling_cap, skip_response_tokens=0)
        )
        batch["distill_teacher_log_probs"] = [teacher]
    else:
        SdpoAlgorithm().apply_driver_options(
            args, SdpoSettings(top_k=11, importance_sampling_cap=importance_sampling_cap)
        )
        selected_ids = torch.arange(11).expand(9, -1)
        batch["distill_teacher_topk_ids"] = [selected_ids]
        batch["distill_teacher_topk_log_probs"] = [teacher.gather(-1, selected_ids)]
        batch["distill_teacher_sampled_log_probs"] = [teacher.gather(-1, response_ids[:, None])[:, 0]]

    def _reduce(values: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum()

    loss, metrics = distill_loss(args, batch, student, _reduce)
    loss.backward()
    dense_student = student.detach().clone().requires_grad_()
    student_log_probs = dense_student.log_softmax(-1)
    teacher_log_probs = teacher.detach()
    if method == "sdft":
        divergences = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1)
    else:
        mixture = torch.logaddexp(student_log_probs, teacher_log_probs) - torch.log(torch.tensor(2.0))
        divergences = 0.5 * (teacher_log_probs.exp() * (teacher_log_probs - mixture)).sum(-1) + 0.5 * (
            student_log_probs.exp() * (student_log_probs - mixture)
        ).sum(-1)
    if importance_sampling_cap == 0:
        weighted_divergences = divergences
    else:
        current = student.detach().log_softmax(-1).gather(-1, response_ids[:, None])[:, 0]
        token_weights = (current - sampler_log_probs).exp().clamp(max=importance_sampling_cap)
        if method == "sdft":
            weighted_divergences = divergences * (token_weights * mask).sum() / mask.sum()
        else:
            weighted_divergences = divergences * token_weights
    dense_loss = _reduce(weighted_divergences)
    dense_loss.backward()
    torch.testing.assert_close(loss, dense_loss, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(student.grad, dense_student.grad, atol=2e-6, rtol=2e-5)
    assert torch.equal(student.grad[mask == 0], torch.zeros_like(student.grad[mask == 0]))
    assert (student.grad[mask == 1].abs().sum(-1) > 0).all()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert teacher.grad is None
    assert torch.equal(batch["loss_masks"][0], mask)
    torch.testing.assert_close(metrics["loss"], loss.detach())


@pytest.mark.unit
def test_teacher_scores_every_position_of_full_episode_suffix(cpu_batch_adapters: None) -> None:
    generator = torch.Generator().manual_seed(17)
    teacher_prompt_length = 5
    suffix_length = 9
    logits = torch.randn(1, teacher_prompt_length + suffix_length, 11, generator=generator)
    args = Namespace(rollout_temperature=0.7, log_probs_chunk_size=2)
    _, result = gather_teacher_log_probs(
        logits,
        args=args,
        unconcat_tokens=[torch.zeros(teacher_prompt_length + suffix_length, dtype=torch.long)],
        total_lengths=[teacher_prompt_length + suffix_length],
        response_lengths=[suffix_length],
    )
    expected = (logits[0, teacher_prompt_length - 1 : -1] / args.rollout_temperature).log_softmax(-1)
    (rows,) = result["distill_teacher_log_probs"]
    assert rows.shape == (9, 11)
    torch.testing.assert_close(rows, expected.to(torch.float16), atol=2e-3, rtol=2e-3)
