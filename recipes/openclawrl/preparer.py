"""OpenClaw-RL step preparer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.types import PolicyBatch, TrainingBatch


@register_step_preparer
class OpenClawRLPreparer(StepPreparer):
    name = "openclawrl"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, PolicyBatch):
            raise TypeError(f"{self.name} requires PolicyBatch, got {type(batch).__name__}")
        advantages = tuple(sample.reward for sample in batch.samples)
        steps = next_steps(state)
        return StepSignal(
            "train",
            # The paper objective: the verbatim upstream top-K select loss
            # (recipes/openclawrl/slime); advantages stay the raw
            # per-sample rewards (upstream disables reward normalization).
            self.name,
            {"steps": steps},
            {"advantages": advantages, "steps": steps},
            advantages,
        )
