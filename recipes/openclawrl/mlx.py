"""OpenClaw-RL's loss on the MLX backend: the reward term, the top-K distillation term and an optional KL term.

The distillation kernel is a port of ``recipes/openclawrl/slime/objective.py``'s
``_opd_one_sample`` with the same arithmetic step by step; one process holds
the whole vocabulary, so no tensor-parallel reduction is needed. The engine
imports this module when it resolves the ``openclawrl`` loss family.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import mlx.core as mx

from reef.core.batches import TrajectoryItem
from reef.core.config import config_option
from reef.train.mlx_backend.contracts import LossOptions
from reef.train.mlx_backend.losses import MLXLoss, MLXLossSettings, MLXLossStep
from reef.train.mlx_backend.training import MLXTrainer, PackedRows, ResponseTokens, TokenObjective, TrainingRow

#: The bound on a log ratio before it is exponentiated, as in the reference.
LOG_RATIO_CLIP = 20.0

#: Stands for minus infinity in masked reductions; finite, so no NaN reaches a gradient.
NEGATIVE_LARGE = -1.0e30

HINT_SELECTIONS = ("shortest", "sequence_optimal")


def masked_lse(values: mx.array, mask: mx.array) -> mx.array:
    """Log-sum-exp over the true positions of ``mask``; a row with an empty mask returns 0."""
    masked = mx.where(mask, values, mx.full(values.shape, NEGATIVE_LARGE, dtype=values.dtype))
    row_max = mx.max(masked, axis=-1, keepdims=True)
    row_max = mx.where(row_max > NEGATIVE_LARGE / 2, row_max, mx.zeros_like(row_max))
    shifted = masked - row_max
    exponentiated = mx.where(mask, mx.exp(shifted), mx.zeros_like(shifted))
    total = mx.maximum(mx.sum(exponentiated, axis=-1, keepdims=True), mx.array(1e-30))
    return row_max + mx.log(total)


def masked_softmax(values: mx.array, mask: mx.array) -> mx.array:
    """Softmax over the true positions of ``mask``; off-mask entries are 0."""
    out = mx.exp(values - masked_lse(values, mask))
    return mx.where(mask, out, mx.zeros_like(out))


def policy_loss(ppo_kl: mx.array, advantages: mx.array, eps_lo: float, eps_hi: float) -> tuple[mx.array, mx.array]:
    """Slime's ``compute_policy_loss``: the pessimistic clipped surrogate and the clip indicator.

    ``ppo_kl`` is ``log(pi_old / pi_new)``, so the ratio is its negative exponential.
    """
    ratio = mx.exp(-ppo_kl)
    unclipped = -ratio * advantages
    clipped = -mx.clip(ratio, 1 - eps_lo, 1 + eps_hi) * advantages
    return mx.maximum(unclipped, clipped), (clipped > unclipped).astype(mx.float32)


@dataclass(frozen=True)
class OPDResult:
    """Per-token outputs of one sample's distillation term; all but ``per_token_pg`` are monitors."""

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
    """The top-K distillation surrogate for one sample's response tokens.

    ``log_probs_at_candidates`` is the policy's current log probability at
    each of ``student_indices`` and is the only argument with a gradient. The
    steps follow the reference:

    1. ``S_t`` is the intersection of the student's and teacher's candidate
       sets, on the student axis.
    2. The teacher's log probabilities are reordered onto the student's order.
    3. ``ell_old`` is this forward pass, detached: one optimizer step per
       rollout, so the ratio starts at 1.
    4. The weight is a softmax of ``ell_old`` over ``S_t``.
    5. The advantage is the clipped teacher-minus-old gap, weighted.
    6. A PPO surrogate on the log ratio.
    7. A sum over ``S_t``; the weight already normalizes within the set.
    """
    rows = student_indices.shape[0]
    if rows == 0:
        empty = mx.zeros((0,), dtype=mx.float32)
        return OPDResult(empty, empty, empty, empty, mx.zeros((0,), dtype=mx.bool_))

    # 1) At most one teacher entry matches each student candidate.
    equal = student_indices[:, :, None] == teacher_indices[:, None, :]
    subset_mask = mx.any(equal, axis=-1)
    row_valid = mx.any(subset_mask, axis=-1)
    mask = subset_mask.astype(mx.float32)

    # 2) The weighted sum picks the unique match; off-mask entries are 0.
    teacher_aligned = mx.sum(equal.astype(mx.float32) * teacher_log_probs[:, None, :].astype(mx.float32), axis=-1)

    current = log_probs_at_candidates.astype(mx.float32)
    # 3)
    old = mx.stop_gradient(current)
    # A monitor of how far the serving capture sits from this forward pass.
    capture_drift = mx.abs(student_captured_log_probs.astype(mx.float32) - old)

    # 4)
    weight = mx.stop_gradient(masked_softmax(old, subset_mask))

    # 5)
    diff = mx.stop_gradient(teacher_aligned - old)
    if diff_clip is not None:
        diff = mx.clip(diff, -diff_clip, diff_clip)
    advantages = mx.stop_gradient(diff * weight)

    # 6)
    ppo_kl = mx.clip(old - current, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    per_entry_pg, per_entry_clip = policy_loss(ppo_kl, advantages, eps_lo, eps_hi)

    # 7) The clip rate is a fraction, not a sum.
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
    """The policy's log probabilities at ``indices``: ``logits[v] - logsumexp(logits)``, both with a gradient."""
    gathered = mx.take_along_axis(logits, indices, axis=-1)
    return gathered - mx.logsumexp(logits, axis=-1, keepdims=True)


def sampled_log_probs(logits: mx.array, tokens: mx.array) -> mx.array:
    """The log probabilities of the sampled ``tokens`` under ``logits`` of ``[tokens, vocab]``."""
    return mx.take_along_axis(logits, tokens[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)


@dataclass(frozen=True)
class TeacherCandidate:
    """One hindsight hint, rendered as the hint-enhanced prompt followed by the policy's response."""

    hint: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class DistillationRow:
    """One judged turn: its policy row, the candidates the policy considered, and the teacher sequences."""

    policy: TrainingRow
    topk_indices: tuple[tuple[int, ...], ...]
    topk_log_probs: tuple[tuple[float, ...], ...]
    candidates: tuple[TeacherCandidate, ...]

    def __post_init__(self) -> None:
        response_length = self.policy.response_length
        if len(self.topk_indices) != response_length or len(self.topk_log_probs) != response_length:
            raise ValueError("the captured candidates must cover exactly the response tokens")
        if not self.candidates:
            raise ValueError("a distillation row needs at least one teacher candidate")
        if any(len(candidate.tokens) <= response_length for candidate in self.candidates):
            raise ValueError("a teacher sequence must be its prompt followed by the whole response")

    @classmethod
    def from_item(cls, index: int, item: TrajectoryItem, advantage: float) -> DistillationRow:
        """The row of one reserved sample; ``index`` names the sample in errors.

        The processor writes the generation top-K and, once the judge
        proposed a hint, ``extras["teacher_cands"]`` into the sample's
        training data. Neither can be rebuilt afterwards.
        """
        training = item.training
        topk_indices = training.get("topk_indices") or ()
        topk_log_probs = training.get("topk_log_probs") or ()
        if not topk_indices or not topk_log_probs:
            raise ValueError(
                f"sample {index} carries no generation top-K; the openclawrl loss distils onto the candidates "
                "the policy considered, so set training.options.capture_topk"
            )
        extras = training.get("extras") or {}
        teacher_candidates = extras.get("teacher_cands") if isinstance(extras, Mapping) else None
        if not teacher_candidates:
            raise ValueError(
                f"sample {index} carries no teacher candidates; the processor attaches them as "
                "extras['teacher_cands'] once the judge proposed a hindsight hint"
            )
        return cls(
            policy=TrainingRow.from_item(item, advantage),
            topk_indices=tuple(tuple(int(token) for token in row) for row in topk_indices),
            topk_log_probs=tuple(tuple(float(value) for value in row) for row in topk_log_probs),
            candidates=tuple(
                TeacherCandidate(str(entry.get("hint", "")), tuple(int(token) for token in entry["teacher_tokens"]))
                for entry in teacher_candidates
            ),
        )


@dataclass(frozen=True)
class OpenClawRLMLXSettings(MLXLossSettings):
    """The openclawrl loss settings under ``training.options.loss_options``; defaults match the Slime recipe."""

    w_rl: float = config_option(1.0, help="Weight of the reward term.")
    w_opd: float = config_option(1.0, help="Weight of the distillation term.")
    eps_lo: float = config_option(0.2, help="Lower clip range of both surrogates.")
    eps_hi: float = config_option(0.28, help="Upper clip range of both surrogates.")
    diff_clip: float | None = config_option(1.0, help="Bound on the teacher-minus-policy gap; null leaves it open.")
    hint_selection: str = config_option(
        "sequence_optimal", help="Which hint the teacher uses: shortest or sequence_optimal."
    )
    native_k: int = config_option(20, help="Teacher candidates per token that sequence_optimal compares.")

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("w_rl", self.w_rl),
            ("w_opd", self.w_opd),
            ("eps_lo", self.eps_lo),
            ("eps_hi", self.eps_hi),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"openclawrl {name} must be finite and non-negative")
        if self.w_rl == 0.0 and self.w_opd == 0.0:
            # AdamW still decays the adapter on a zero gradient.
            raise ValueError("openclawrl needs a non-zero w_rl or w_opd")
        if self.diff_clip is not None and (not math.isfinite(self.diff_clip) or self.diff_clip <= 0):
            raise ValueError("openclawrl diff_clip must be finite and positive, or null")
        if self.hint_selection not in HINT_SELECTIONS:
            raise ValueError(f"openclawrl hint_selection must be one of: {', '.join(HINT_SELECTIONS)}")
        if self.native_k <= 0:
            raise ValueError("openclawrl native_k must be positive")


def select_hint(native_topk: Sequence[mx.array], student_indices: mx.array, hint_selection: str) -> int:
    """The index of the hint the teacher uses for the whole response.

    ``shortest`` takes candidate 0; the processor orders candidates shortest
    first. ``sequence_optimal`` takes the candidate whose own top-K tokens
    the policy also considered most often over the response.
    """
    if hint_selection == "shortest" or len(native_topk) == 1:
        return 0
    totals = mx.stack(
        [
            mx.sum(mx.any(student_indices[:, :, None] == native[:, None, :], axis=-1).astype(mx.int32))
            for native in native_topk
        ]
    )
    return int(mx.argmax(totals).item())


class OpenClawRLTokenLoss(TokenObjective):
    """``w_rl * reward + w_opd * distillation + kl_coef * k3``, each a mean over the sample's masked response tokens."""

    def __init__(
        self,
        packed: PackedRows,
        row: DistillationRow,
        teacher_log_probs: mx.array,
        reference_log_probs: mx.array | None,
        settings: OpenClawRLMLXSettings,
    ) -> None:
        self._sampled = packed.targets[0]
        self._mask = mx.array([float(value) for value in row.policy.loss_mask])
        self._student_indices = mx.array([list(candidates) for candidates in row.topk_indices])
        self._student_captured = mx.array([list(values) for values in row.topk_log_probs])
        self._teacher_log_probs = teacher_log_probs
        self._reference_log_probs = reference_log_probs
        self._reward = row.policy.advantages[0]
        # Constant over the slices, so each slice's sum divides by it.
        self._token_count = max(sum(row.policy.loss_mask), 1)
        self._settings = settings

    def loss(self, logits: mx.array, begin: int, stop: int) -> mx.array:
        settings = self._settings
        logits = logits[0]
        mask = self._mask[begin:stop]
        sampled = sampled_log_probs(logits, self._sampled[begin:stop])
        total = mx.zeros((), dtype=mx.float32)
        if settings.w_rl:
            # One optimizer step per rollout: the old policy is this policy,
            # so the ratio starts at 1.
            ppo_kl = mx.clip(mx.stop_gradient(sampled) - sampled, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
            per_token, _ = policy_loss(
                ppo_kl, mx.full(sampled.shape, self._reward, dtype=mx.float32), settings.eps_lo, settings.eps_hi
            )
            total = total + settings.w_rl * mx.sum(per_token * mask)
        if settings.w_opd:
            indices = self._student_indices[begin:stop]
            result = opd_one_sample(
                candidate_log_probs(logits, indices),
                student_indices=indices,
                student_captured_log_probs=self._student_captured[begin:stop],
                # The teacher was scored at the student's candidates, so the two sets coincide.
                teacher_indices=indices,
                teacher_log_probs=self._teacher_log_probs[begin:stop],
                eps_lo=settings.eps_lo,
                eps_hi=settings.eps_hi,
                diff_clip=settings.diff_clip,
            )
            total = total + settings.w_opd * mx.sum(result.per_token_pg * mask)
        if self._reference_log_probs is not None:
            # k3 = exp(d) - d - 1 with d = reference - current: non-negative,
            # and zero only where the policy matches the base.
            difference = mx.clip(self._reference_log_probs[begin:stop] - sampled, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
            total = total + settings.kl_coef * mx.sum((mx.exp(difference) - difference - 1.0) * mask)
        return total / self._token_count


class OpenClawRLMLXStep(MLXLossStep):
    """One OpenClaw-RL step: one pass per judged turn, after the frozen base scored its teacher sequences."""

    def __init__(self, rows: Sequence[DistillationRow], settings: OpenClawRLMLXSettings) -> None:
        self.rows = tuple(rows)
        self.settings = settings

    def train(self, trainer: MLXTrainer) -> dict[str, float | int]:
        result = trainer.optimizer_step(self.passes(trainer))
        return {
            **result,
            "rows": len(self.rows),
            "response_tokens": sum(row.policy.response_length for row in self.rows),
            "w_rl": self.settings.w_rl,
            "w_opd": self.settings.w_opd,
            "kl_coef": self.settings.kl_coef,
        }

    def passes(self, trainer: MLXTrainer) -> Iterator[tuple[PackedRows, TokenObjective]]:
        """One pass per row, after the frozen base scored the row's teacher sequences."""
        settings = self.settings
        for row in self.rows:
            packed = trainer.pack([row.policy])
            response_length = row.policy.response_length
            student_indices = mx.array([list(candidates) for candidates in row.topk_indices])
            with trainer.base_model():
                at_student = []
                native = []
                for candidate in row.candidates:
                    logits = trainer.response_logits(trainer.pack([ResponseTokens(candidate.tokens, response_length)]))
                    log_probs = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
                    at_student.append(mx.take_along_axis(log_probs, student_indices, axis=-1))
                    native.append(
                        mx.argpartition(-log_probs, kth=settings.native_k - 1, axis=-1)[:, : settings.native_k]
                    )
                    mx.eval(at_student[-1], native[-1])
                reference = None
                if settings.kl_coef:
                    reference = mx.stop_gradient(
                        sampled_log_probs(trainer.response_logits(packed)[0], packed.targets[0])
                    )
                    mx.eval(reference)
            teacher_log_probs = at_student[select_hint(native, student_indices, settings.hint_selection)]
            yield packed, OpenClawRLTokenLoss(packed, row, teacher_log_probs, reference, settings)


class OpenClawRLMLXLoss(MLXLoss):
    """The ``openclawrl`` loss family on MLX."""

    def settings(self, options: LossOptions) -> OpenClawRLMLXSettings:
        return OpenClawRLMLXSettings.from_options(options)

    def step(
        self, samples: Sequence[TrajectoryItem], advantages: Sequence[float], options: LossOptions
    ) -> OpenClawRLMLXStep:
        if not samples:
            raise ValueError("an openclawrl step needs at least one sample")
        rows = [
            DistillationRow.from_item(index, sample, advantage)
            for index, (sample, advantage) in enumerate(zip(samples, advantages, strict=True))
        ]
        return OpenClawRLMLXStep(rows, self.settings(options))
