"""The AutoEnvScaling objective on Tinker: each rollout's reward minus the mean reward of its task's rollouts.

A task job of the generated-task recipe names it as its ``objective`` by its dotted reference, as the AutoEnvScaling
examples on Tinker do. It trains on Tinker's built-in importance-sampling loss. The advantage is not divided by the
group's standard deviation, as in the AutoEnvScaling and DAPO recipes: with partial credit, a group whose rewards
differ a little (1.0 and 0.9) would otherwise train as hard as one whose rewards differ fully (1.0 and 0.0). A group
whose rewards are all equal gets 0. A job whose groups are all equal trains nothing, so it is skipped and keeps its
weights and step count.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from reef.core.job_data import JobData, trajectory_groups
from reef.core.trajectories import trajectory_reward
from reef.train.tinker_backend.objectives import TinkerObjective
from reef.train.tinker_backend.schedule import next_steps
from reef.train.tinker_backend.signals import TrainingSignal


class AutoEnvScalingTinkerObjective(TinkerObjective):
    """Train each rollout on its reward minus its task's mean reward, on Tinker's ``importance_sampling`` loss."""

    name = "autoenvscaling"
    # The advantage on every response token, the rollout log probs as the reference.
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
        values = tuple(advantages)
        return TrainingSignal(
            "train",
            {"steps": steps},
            {"advantages": values, "constant_groups": constant_groups, "steps": steps},
            values,
        )


__all__ = ["AutoEnvScalingTinkerObjective"]
