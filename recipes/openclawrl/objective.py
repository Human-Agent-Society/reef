"""OpenClaw-RL training objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class OpenClawRLObjective(TrainingObjective):
    name = "openclawrl"
    loss_family = "openclawrl"
    # The OPD branch takes its old-policy log-probs from the same actor forward
    # that produces the current ones, so a second pass would train against the
    # wrong policy; the Slime spec refuses --num-steps-per-rollout>1 for the same reason.
    supports_multiple_epochs = False

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        # The upstream top-K loss consumes raw rewards without normalization.
        advantages = tuple(trajectory_reward(sample) for sample in samples)
        steps = next_steps(state)
        return StepSignal(
            "train",
            {"steps": steps},
            {"advantages": advantages, "steps": steps},
            advantages,
        )
