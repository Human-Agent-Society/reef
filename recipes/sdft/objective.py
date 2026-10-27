"""SDFT training objective: every sample distils its own demonstration-conditioned teacher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import TrainingSignal
from reef.train.types import JobData, trajectories


@register_objective
class SdftObjective(TrainingObjective):
    name = "sdft"
    loss_family = "sdft"
    # The teacher is the policy at the start of the step; a second pass over
    # the data would distil a teacher the first pass already moved away from.
    supports_multiple_epochs = False

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        samples = trajectories(data)
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})
