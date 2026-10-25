"""One distillation update per job of on-policy data, with no scalar-reward advantages."""

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import TrainingSignal
from reef.train.types import JobData, trajectories


@register_objective
class OpdObjective(TrainingObjective):
    name = "opd"
    loss_family = "opd"
    supports_multiple_epochs = False

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        samples = trajectories(data)
        if not samples:
            raise ValueError("OPD requires student rollouts")
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})
