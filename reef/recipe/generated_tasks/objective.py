"""The generated-task objective: group relative advantages over the rollouts of one task, on importance sampling."""

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
    """Each rollout's advantage is its reward minus its task group's mean reward; a constant group gets 0.

    The advantage is not divided by the group's standard deviation, as in the AutoEnvScaling and DAPO recipes: with
    partial credit, a group whose rewards differ a little (1.0 and 0.9) would otherwise train as hard as one whose
    rewards differ fully (1.0 and 0.0).
    """

    name = "task_group"
    # The advantage on every response token, the rollout log probs as the reference: Tinker's built-in loss, and
    # truncated importance sampling on Slime (reef.train.slime_backend.importance_sampling).
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
            advantages.extend(reward - mean if spread else 0.0 for reward in rewards)
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
