"""SDFT training objective: every sample distils its own demonstration-conditioned teacher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class SdftObjective(TrainingObjective):
    name = "sdft"
    loss_family = "sdft"
    # The teacher is the policy at the start of the step; a second pass over
    # the batch would distil a teacher the first pass already moved away from.
    supports_multiple_epochs = False

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})
