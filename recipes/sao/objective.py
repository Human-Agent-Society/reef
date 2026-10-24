"""SAO training objective: one rollout, one DP unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import TrainingSignal
from reef.train.types import JobData, trajectories


@register_objective
class SaoObjective(TrainingObjective):
    name = "sao"
    loss_family = "sao"
    # The SAO ratio is clipped on both sides, so passes after the first stay bounded.
    supports_multiple_epochs = True

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        samples = trajectories(data)
        steps = next_steps(state)
        return TrainingSignal(
            "train",
            {"steps": steps},
            {"steps": steps, "rollouts": len(samples)},
        )
