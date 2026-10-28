"""The generated-task objective: group relative advantages over the rollouts of one task, on the importance sampling loss."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import TrainingSignal
from reef.train.types import JobData, trajectory_groups


@register_objective
class TaskGroupObjective(TrainingObjective):
    """Each rollout's advantage is its reward centered and scaled within its task group; a constant group gets 0."""

    name = "task_group"
    # Tinker's built in loss: the advantage on every response token, the rollout log probs as the reference.
    loss_family = "importance_sampling"

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        advantages: list[float] = []
        constant_groups = 0
        groups = 0
        for group in trajectory_groups(data):
            groups += 1
            rewards = [trajectory_reward(sample) for sample in group]
            mean = statistics.fmean(rewards)
            spread = statistics.pstdev(rewards)
            if spread == 0.0:
                constant_groups += 1
            advantages.extend((reward - mean) / spread if spread else 0.0 for reward in rewards)
        if groups and constant_groups == groups:
            # Zero advantages everywhere train nothing: skip the step and keep the weights and the step count.
            return TrainingSignal(
                "skip", dict(state), {"constant_groups": constant_groups, "skipped": "every group is constant"}
            )
        steps = next_steps(state)
        normalized = tuple(advantages)
        return TrainingSignal(
            "train",
            {"steps": steps},
            {"advantages": normalized, "constant_groups": constant_groups, "steps": steps},
            normalized,
        )
