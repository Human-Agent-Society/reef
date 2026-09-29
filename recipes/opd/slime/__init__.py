"""OPD's thin Slime family on the shared distillation implementation."""

from dataclasses import dataclass

from reef.train.slime_backend.algorithm import register_loss_family
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings


@dataclass(frozen=True)
class OpdSettings(DistillSettings):
    """A frozen separate teacher and sampled-token reverse KL.

    A positive ``top_k`` selects the base's sampled-token reverse-KL
    estimator. Only the sampled-token log-probability enters that loss;
    keeping one top entry minimizes the unused top-K payload. The teacher
    still scores the token against the full vocabulary. No truncation or
    sequence-level importance correction is applied to an on-policy step.
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


__all__ = ["OpdAlgorithm", "OpdSettings"]
