"""OPD's thin Slime family on the shared distillation implementation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import TrainingSignal
from reef.train.slime_backend.algorithm import register_loss_family
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings
from reef.train.slime_backend.objectives import SlimeObjective, register_slime_objective
from reef.train.types import JobData, trajectories


@dataclass(frozen=True)
class OpdSettings(DistillSettings):
    """A frozen separate teacher and sampled-token reverse KL.

    A positive ``top_k`` selects the base's sampled-token reverse-KL
    estimator. Only the sampled-token log-probability enters that loss;
    keeping one top entry minimizes the unused top-K payload. The teacher
    still scores the token against the full vocabulary. ``importance_sampling_cap``
    0 drops the truncated importance-sampling ratio between the sampler and the
    trainer that the cookbook loss applies; with one update per on-policy batch
    the ratio is 1 up to numerics.
    """

    teacher: str = "separate"
    divergence: str = "reverse"
    top_k: int = 1
    teacher_update_rate: float = 0.0
    importance_sampling_cap: float = 0.0
    importance_sampling_level: str = "token"


@register_loss_family
class OpdAlgorithm(DistillAlgorithm):
    """Score student tokens with the named teacher, then update the student once."""

    loss_family = "opd"
    settings_type = OpdSettings


@register_slime_objective
class OpdObjective(SlimeObjective):
    name = "opd"
    loss_family = "opd"
    supports_multiple_epochs = False

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        samples = trajectories(data)
        if not samples:
            raise ValueError("OPD requires student rollouts")
        steps = next_steps(state)
        return TrainingSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})


__all__ = ["OpdAlgorithm", "OpdObjective", "OpdSettings"]
