"""Backend-neutral objective for stock clipped PPO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class PpoRlhfObjective(TrainingObjective):
    name = "ppo_rlhf_reference_reward"
    loss_family = "ppo_rlhf_reference_reward"
    # Slime's clipped PPO ratio remains valid for the configured extra passes.
    supports_multiple_epochs = True

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        steps = next_steps(state)
        return StepSignal(
            "train",
            {"steps": steps},
            {"steps": steps, "rollouts": len(trajectories(batch))},
        )


__all__ = ["PpoRlhfObjective"]
