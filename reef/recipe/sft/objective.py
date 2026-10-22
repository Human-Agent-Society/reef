"""The SFT objective: the token cross-entropy of each sample's target response, with the same weight for all samples."""

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
    """Train all samples in the batch with the same weight. A sample has no advantage."""

    name = "sft"
    loss_family = "sft"
    # The target is fixed and the loss reads no old-policy log probs. Thus each pass trains the same loss.
    supports_multiple_epochs = True

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})
