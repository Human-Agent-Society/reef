"""SAO training objective: one rollout, one DP unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class SaoObjective(TrainingObjective):
    name = "sao"
    loss_family = "sao"
    # The SAO ratio is clipped on both sides, so passes after the first stay bounded.
    supports_multiple_epochs = True

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal(
            "train",
            {"steps": steps},
            {"steps": steps, "rollouts": len(samples)},
        )


@register_objective
class SaoGrpoControlObjective(SaoObjective):
    """Same step signal as SAO; ``loss_family`` routes the payload to the control's Slime side."""

    name = "sao-grpo-dis"
    loss_family = "sao-grpo-dis"
