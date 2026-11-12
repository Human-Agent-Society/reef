"""Tinker implementation of TTTD's unmasked policy loss and centered base KL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recipes.tttd.objective import tttd_advantages
from reef.train.tinker_backend.loss_families import register_loss_family
from reef.train.tinker_backend.losses import MatrixInput, TinkerLoss, TokenRow
from reef.train.tinker_backend.objectives import TinkerObjective
from reef.train.tinker_backend.schedule import next_steps
from reef.train.tinker_backend.signals import TrainingSignal
from reef.train.types import JobData


class TttdTinkerObjective(TinkerObjective):
    name = "tttd"
    loss_family = "tttd"

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        advantages, metrics = tttd_advantages(data)
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {**metrics, "steps": steps}, advantages)


@register_loss_family
class TttdTinkerLoss(TinkerLoss):
    loss_family = "tttd"
    needs_base_logprobs = True

    def inputs(
        self, rows: Sequence[TokenRow], base_logprobs: Sequence[Sequence[float]], *, kl_coef: float
    ) -> list[dict[str, list[Any] | MatrixInput]]:
        if not kl_coef:
            return [row.inputs([row.advantage] * len(row.mask)) for row in rows]
        if len(base_logprobs) != len(rows):
            raise ValueError("TTTD base KL requires one frozen-base log-probability row per sample")
        differences = []
        count = 0
        for row, base in zip(rows, base_logprobs, strict=True):
            if len(base) != len(row.mask):
                raise ValueError("TTTD base KL must align with response tokens")
            differences.append(
                [(sampled - frozen) * mask for sampled, frozen, mask in zip(row.logprobs, base, row.mask, strict=True)]
            )
            count += sum(row.mask)
        if count <= 0:
            raise ValueError("TTTD base KL requires at least one selected response token")
        mean = sum(sum(values) for values in differences) / count
        # The mask gates only KL. TTTD's policy term retains the trajectory
        # advantage on forced prefill tokens, matching its Slime objective.
        return [
            row.inputs(
                [
                    row.advantage + kl_coef * mask * (mean - difference)
                    for mask, difference in zip(row.mask, values, strict=True)
                ]
            )
            for row, values in zip(rows, differences, strict=True)
        ]
