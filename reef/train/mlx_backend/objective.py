"""OpenClaw-RL's top-K OPD surrogate, in MLX.

A port of ``recipes/openclawrl/slime/objective.py``'s ``_opd_one_sample``.
The tensor-parallel machinery around the original disappears here — one
process holds the whole vocabulary — but the arithmetic is the same, step for
step, and ``tests/reef_service/test_mlx_openclawrl.py`` pins it against the
torch original rather than against expectations written by hand.

The distillation term is what makes this recipe worth running on a laptop:
the evaluative signal is one bit per turn, while this puts a teacher-weighted
advantage on every candidate token the policy considered.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

#: Matches the reference's clamp on the log-ratio before it is exponentiated.
PPO_KL_CLAMP = 20.0

_NEG_INF = -1.0e30


def masked_lse(values: mx.array, mask: mx.array) -> mx.array:
    """Numerically stable log-sum-exp over the true positions of ``mask``.

    A row whose mask is empty returns 0 rather than -inf: it is zeroed
    downstream by ``row_valid``, and letting an infinity through here would
    poison the gradient of every other row in the batch.
    """
    masked = mx.where(mask, values, mx.full(values.shape, _NEG_INF, dtype=values.dtype))
    row_max = mx.max(masked, axis=-1, keepdims=True)
    row_max = mx.where(row_max > _NEG_INF / 2, row_max, mx.zeros_like(row_max))
    shifted = masked - row_max
    exponentiated = mx.where(mask, mx.exp(shifted), mx.zeros_like(shifted))
    total = mx.maximum(mx.sum(exponentiated, axis=-1, keepdims=True), mx.array(1e-30))
    return row_max + mx.log(total)


def masked_softmax(values: mx.array, mask: mx.array) -> mx.array:
    """Softmax over the true positions of ``mask``; off-mask entries are 0."""
    out = mx.exp(values - masked_lse(values, mask))
    return mx.where(mask, out, mx.zeros_like(out))


def policy_loss(
    ppo_kl: mx.array,
    advantages: mx.array,
    eps_lo: float,
    eps_hi: float,
) -> tuple[mx.array, mx.array]:
    """Slime's ``compute_policy_loss``: the pessimistic clipped surrogate.

    ``ppo_kl`` is ``log(pi_old / pi_new)``, so the ratio is its negative
    exponential. Returns the per-entry loss and the clip indicator.
    """
    ratio = mx.exp(-ppo_kl)
    unclipped = -ratio * advantages
    clipped = -mx.clip(ratio, 1 - eps_lo, 1 + eps_hi) * advantages
    return mx.maximum(unclipped, clipped), (clipped > unclipped).astype(mx.float32)


@dataclass(frozen=True)
class OPDResult:
    """Per-token outputs of one sample's OPD term.

    Everything except ``per_token_pg`` is a monitor: Reef reports them so an
    operator can see the teacher gap and the clip rate without reading them
    back out of the loss.
    """

    per_token_pg: mx.array
    per_token_clip: mx.array
    per_token_diff: mx.array
    per_token_capture_drift: mx.array
    row_valid: mx.array


def opd_one_sample(
    log_probs_at_candidates: mx.array,
    *,
    student_indices: mx.array,
    student_captured_log_probs: mx.array,
    teacher_indices: mx.array,
    teacher_log_probs: mx.array,
    eps_lo: float,
    eps_hi: float,
    diff_clip: float | None,
) -> OPDResult:
    """The top-K OPD surrogate for one sample's response tokens.

    ``log_probs_at_candidates`` is the student's *current* global log-prob at
    each of ``student_indices`` — the only argument carrying gradient. The
    caller computes it as ``logits[v] - logsumexp(logits)``, which is where
    both halves of the gradient come from.

    The steps follow the reference exactly:

    1. ``S_t`` is the intersection of the student's and teacher's candidate
       sets, expressed on the student axis.
    2. Teacher log-probs are reordered onto the student's ordering.
    3. ``ell_old`` is this same forward, detached, so the ratio starts at 1
       and the surrogate sits at its unclipped linear point where the whole
       teacher signal survives.
    4. The importance weight is a softmax of ``ell_old`` over ``S_t`` alone,
       so clipping the teacher gap below cannot change it.
    5. The advantage is the clipped teacher-versus-old gap, weighted.
    6. A PPO surrogate on the global log-ratio.
    7. Summed over ``S_t`` — not averaged, because the weight already
       normalises within the set.
    """
    rows = student_indices.shape[0]
    if rows == 0:
        empty = mx.zeros((0,), dtype=mx.float32)
        return OPDResult(empty, empty, empty, empty, mx.zeros((0,), dtype=mx.bool_))

    # 1) The subset mask on the student axis: at most one teacher entry can
    #    match each student candidate, so `any` picks it out.
    equal = student_indices[:, :, None] == teacher_indices[:, None, :]
    subset_mask = mx.any(equal, axis=-1)
    row_valid = mx.any(subset_mask, axis=-1)
    mask = subset_mask.astype(mx.float32)

    # 2) Reorder the teacher's log-probs onto the student's candidate order.
    #    The weighted sum picks the unique match; off-mask entries collapse
    #    to zero and are masked out anyway.
    teacher_aligned = mx.sum(equal.astype(mx.float32) * teacher_log_probs[:, None, :].astype(mx.float32), axis=-1)

    current = log_probs_at_candidates.astype(mx.float32)
    # 3) One optimizer step per rollout means the old actor is this actor.
    old = mx.stop_gradient(current)
    # Monitor only: how far the serving engine's capture sits from this
    # forward. Reading the capture into the surrogate instead is what would
    # let cross-engine drift zero the gradient on the teacher's side.
    capture_drift = mx.abs(student_captured_log_probs.astype(mx.float32) - old)

    # 4) Importance weight over the subset, from the old log-probs alone.
    weight = mx.stop_gradient(masked_softmax(old, subset_mask))

    # 5) Advantage: the teacher-versus-old gap, optionally clamped so a
    #    rare-token candidate the two wildly disagree on cannot dominate.
    diff = mx.stop_gradient(teacher_aligned - old)
    if diff_clip is not None:
        diff = mx.clip(diff, -diff_clip, diff_clip)
    advantages = mx.stop_gradient(diff * weight)

    # 6) The surrogate on the global log-ratio.
    ppo_kl = mx.clip(old - current, -PPO_KL_CLAMP, PPO_KL_CLAMP)
    per_entry_pg, per_entry_clip = policy_loss(ppo_kl, advantages, eps_lo, eps_hi)

    # 7) Sum over the subset; the clip rate is a fraction, not a sum.
    per_token_pg = mx.sum(per_entry_pg * mask, axis=-1)
    counted = mx.maximum(mx.sum(mask, axis=-1), mx.array(1.0))
    per_token_clip = mx.sum(per_entry_clip * mask, axis=-1) / counted
    per_token_diff = mx.sum(mx.abs(diff) * weight, axis=-1)
    per_token_drift = mx.sum(capture_drift * weight, axis=-1)

    valid = row_valid.astype(mx.float32)
    return OPDResult(
        per_token_pg * valid,
        per_token_clip * valid,
        per_token_diff * valid,
        per_token_drift * valid,
        row_valid,
    )


def candidate_log_probs(logits: mx.array, indices: mx.array) -> mx.array:
    """Student global log-probs at ``indices``: ``logits[v] - logsumexp(logits)``.

    Both halves are differentiable, which is what carries the teacher's
    instruction back into the policy.
    """
    gathered = mx.take_along_axis(logits, indices, axis=-1)
    return gathered - mx.logsumexp(logits, axis=-1, keepdims=True)


__all__ = [
    "PPO_KL_CLAMP",
    "OPDResult",
    "candidate_log_probs",
    "masked_lse",
    "masked_softmax",
    "opd_one_sample",
    "policy_loss",
]
