"""SFT training objective: every sample in the batch, unweighted."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class SftObjective(TrainingObjective):
    name = "sft"
    loss_family = "sft"
    # A demonstration is a fixed target: a second pass over the batch trains the same tokens again.
    supports_multiple_epochs = True

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})
