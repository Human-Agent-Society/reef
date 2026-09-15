"""The Reasoning Agent's step: group relative advantages over the episodes of one task, on Tinker's importance sampling loss."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import TrainingBatch, trajectory_groups

# Tinker's built in loss: the advantage on every response token, the rollout log probs as the reference.
LOSS_FAMILY = "importance_sampling"


@register_step_preparer
class SpadePreparer(StepPreparer):
    """Each episode's advantage is its reward centered and scaled within its task group; a constant group gets 0."""

    name = "spade"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        advantages: list[float] = []
        constant_groups = 0
        for group in trajectory_groups(batch):
            rewards = [trajectory_reward(sample) for sample in group]
            mean = statistics.fmean(rewards)
            spread = statistics.pstdev(rewards)
            if spread == 0.0:
                constant_groups += 1
            advantages.extend((reward - mean) / spread if spread else 0.0 for reward in rewards)
        steps = next_steps(state)
        normalized = tuple(advantages)
        return StepSignal(
            "train",
            LOSS_FAMILY,
            {"steps": steps},
            {"advantages": normalized, "constant_groups": constant_groups, "steps": steps},
            normalized,
            StepScheduling(unit="sample", batch_size="actual"),
        )
