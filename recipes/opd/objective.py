"""One distillation update per on-policy batch, with no scalar-reward advantages."""

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class OpdObjective(TrainingObjective):
    name = "opd"
    loss_family = "opd"
    supports_multiple_epochs = False

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        if not samples:
            raise ValueError("OPD requires student rollouts")
        steps = next_steps(state)
        return StepSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})
