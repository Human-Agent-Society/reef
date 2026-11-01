"""The dataset objective: Slime computes the training signal of a dataset job, so Reef prepares none."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import TrainingSignal
from reef.train.types import JobData


@register_objective
class SlimeDatasetObjective(TrainingObjective):
    """Leave a dataset job's rollouts, rewards, advantages and loss to Slime (the ``slime`` loss family)."""

    name = "slime_dataset"
    loss_family = "slime"

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        raise RuntimeError("a dataset job trains on Slime's own rollouts; Reef prepares no training signal for it")
