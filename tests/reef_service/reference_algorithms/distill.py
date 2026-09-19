"""Pure-Python reference for the distillation divergences.

Transcribes ``_compute_loss`` of the SDFT reference implementation
(idanshen/Self-Distillation at ``d77573212fa0``, ``distil_trainer.py``) with
plain floats, extended with the generalized Jensen-Shannon divergence: at
every response position, the divergence over the full vocabulary between
the teacher's next-token distribution and the student's; the per-sample
mean over the trained tokens; and the truncated importance-sampling weight.
``tests/reef_service/test_distill_parity.py`` pins the tensor code in
``reef/train/slime_backend/distill/divergence.py`` to it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def log_softmax(logits: Sequence[float]) -> list[float]:
    largest = max(logits)
    log_sum_exp = largest + math.log(sum(math.exp(value - largest) for value in logits))
    return [value - log_sum_exp for value in logits]


def divergence(
    student_logits: Sequence[float], teacher_log_probs: Sequence[float], kind: str, jsd_beta: float = 0.5
) -> float:
    """KL(teacher || student) for ``forward``, KL(student || teacher) for ``reverse``, the generalized JSD for ``jsd``."""
    student_log_probs = log_softmax(student_logits)
    pairs = list(zip(teacher_log_probs, student_log_probs, strict=True))
    if kind == "forward":
        return sum(math.exp(teacher) * (teacher - student) for teacher, student in pairs)
    if kind == "reverse":
        return sum(math.exp(student) * (student - teacher) for teacher, student in pairs)
    if kind == "jsd":
        total = 0.0
        for teacher, student in pairs:
            mixture = math.log(jsd_beta * math.exp(teacher) + (1.0 - jsd_beta) * math.exp(student))
            total += jsd_beta * math.exp(teacher) * (teacher - mixture)
            total += (1.0 - jsd_beta) * math.exp(student) * (student - mixture)
        return total
    raise ValueError(f"unknown divergence {kind!r}")


def restricted_divergence(
    student_log_probs: Sequence[float], teacher_log_probs: Sequence[float], kind: str, jsd_beta: float = 0.5
) -> float:
    """The divergence between the two distributions restricted to the given ids and renormalized."""
    return divergence(log_softmax(student_log_probs), log_softmax(teacher_log_probs), kind, jsd_beta)


def sequence_importance_weight(
    student_log_probs: Sequence[float],
    rollout_log_probs: Sequence[float],
    loss_mask: Sequence[int],
    cap: float,
) -> float:
    """The masked mean of ``min(pi_theta / pi_rollout, cap)`` over the response."""
    ratios = [
        min(math.exp(student - rollout), cap)
        for student, rollout in zip(student_log_probs, rollout_log_probs, strict=True)
    ]
    trained = sum(loss_mask)
    return sum(ratio * mask for ratio, mask in zip(ratios, loss_mask, strict=True)) / max(trained, 1)


def sample_loss(per_token: Sequence[float], loss_mask: Sequence[int], weight: float = 1.0) -> float:
    """The reference's per-sample loss: the masked mean token divergence scaled by the sample's weight."""
    trained = sum(loss_mask)
    return weight * sum(value * mask for value, mask in zip(per_token, loss_mask, strict=True)) / max(trained, 1)
