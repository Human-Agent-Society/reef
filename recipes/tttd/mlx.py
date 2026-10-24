"""TTT-Discover's loss on the MLX backend: the unclipped importance-sampling surrogate and the centered base KL.

The engine imports this module when it resolves the ``tttd`` loss family.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from reef.core.batches import TrajectoryItem
from reef.train.mlx_backend.contracts import LossOptions
from reef.train.mlx_backend.losses import MLXLoss, MLXLossSettings, MLXLossStep
from reef.train.mlx_backend.training import MLXTrainer, PackedRows, TokenObjective, TrainingRow


def frozen_base_kl_advantages(
    rows: Sequence[TrainingRow], base_log_probs: Sequence[Sequence[float]], kl_coef: float
) -> list[TrainingRow]:
    """Add TTT-Discover's centered frozen-base KL term to each response token's advantage.

    A port of ``incorporate_kl_penalty``: each advantage moves by
    ``kl_coef * mask * (mean - difference)``, where ``difference`` is the
    rollout-minus-base log probability and ``mean`` is its batch mean over
    the masked tokens.
    """
    differences = [
        [
            (rollout - base) * mask
            for rollout, base, mask in zip(row.rollout_log_probs, base_row, row.loss_mask, strict=True)
        ]
        for row, base_row in zip(rows, base_log_probs, strict=True)
    ]
    masked_tokens = sum(sum(row.loss_mask) for row in rows)
    if masked_tokens <= 0:
        raise ValueError("the TTT-Discover base KL term needs at least one masked response token")
    mean = sum(sum(row_differences) for row_differences in differences) / masked_tokens
    return [
        replace(
            row,
            advantages=tuple(
                advantage + kl_coef * mask * (mean - difference)
                for advantage, difference, mask in zip(row.advantages, row_differences, row.loss_mask, strict=True)
            ),
        )
        for row, row_differences in zip(rows, differences, strict=True)
    ]


class ImportanceSamplingSurrogate(TokenObjective):
    """``-exp(log_pi - log_pi_rollout) * advantage``, summed over response tokens.

    The sum is float32 although the model may run in float16: it adds many
    terms of similar size and opposite sign, which float16 would round away.
    """

    def __init__(self, packed: PackedRows, rows: Sequence[TrainingRow]) -> None:
        self._targets = packed.targets
        self._mask = packed.response_mask
        self._rollout_log_probs = packed.on_scored([row.rollout_log_probs for row in rows])
        self._advantages = packed.on_scored([row.advantages for row in rows])

    def loss(self, logits: mx.array, begin: int, stop: int) -> mx.array:
        mask = self._mask[:, begin:stop]
        log_probs = -nn.losses.cross_entropy(logits, self._targets[:, begin:stop], reduction="none") * mask
        ratio = mx.exp(log_probs.astype(mx.float32) - self._rollout_log_probs[:, begin:stop].astype(mx.float32)) * mask
        return (-ratio * self._advantages[:, begin:stop] * mask).astype(mx.float32).sum()


class TttdMLXStep(MLXLossStep):
    """One TTT-Discover step: micro-batches of rows, with the base KL term when ``kl_coef`` is set."""

    def __init__(self, rows: Sequence[TrainingRow], settings: MLXLossSettings) -> None:
        self.rows = tuple(rows)
        self.settings = settings

    def train(self, trainer: MLXTrainer) -> dict[str, float | int]:
        size = trainer.micro_batch_size
        starts = range(0, len(self.rows), size)
        packed_batches = [trainer.pack(self.rows[start : start + size]) for start in starts]

        def response_log_probs() -> list[list[float]]:
            return [
                values
                for packed in packed_batches
                for values in packed.response_values(trainer.response_log_probs(packed))
            ]

        # The policy log probabilities before the update measure how far the
        # batch lags the served weights: on a fresh rollout the ratio is 1.
        pairs = [
            (policy, rollout)
            for row, row_policy in zip(self.rows, response_log_probs(), strict=True)
            for policy, rollout in zip(row_policy, row.rollout_log_probs, strict=True)
        ]
        rows = self.rows
        if self.settings.kl_coef:
            with trainer.base_model():
                base = response_log_probs()
            rows = frozen_base_kl_advantages(rows, base, self.settings.kl_coef)
        result = trainer.optimizer_step(
            (packed, ImportanceSamplingSurrogate(packed, rows[start : start + size]))
            for packed, start in zip(packed_batches, starts, strict=True)
        )
        return {
            **result,
            "importance_ratio": sum(math.exp(policy - rollout) for policy, rollout in pairs) / len(pairs),
            "ppo_kl": sum(rollout - policy for policy, rollout in pairs) / len(pairs),
            "response_tokens": len(pairs),
            "rows": len(rows),
        }


class TttdMLXLoss(MLXLoss):
    """The ``tttd`` loss family on MLX."""

    def step(
        self, samples: Sequence[TrajectoryItem], advantages: Sequence[float], options: LossOptions
    ) -> TttdMLXStep:
        if not samples:
            raise ValueError("a TTT-Discover step needs at least one sample")
        rows = [
            TrainingRow.from_item(sample, advantage) for sample, advantage in zip(samples, advantages, strict=True)
        ]
        return TttdMLXStep(rows, self.settings(options))
