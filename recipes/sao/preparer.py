"""SAO step preparer: one rollout, one DP unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.types import PolicyBatch, TrainingBatch


@register_step_preparer
class SaoPreparer(StepPreparer):
    name = "sao"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, PolicyBatch):
            raise TypeError(f"{self.name} requires PolicyBatch, got {type(batch).__name__}")
        steps = next_steps(state)
        return StepSignal(
            "train",
            self.name,
            {"steps": steps},
            {"steps": steps, "rollouts": len(batch.samples)},
        )
