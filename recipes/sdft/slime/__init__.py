"""Slime implementation of SDFT (Self-Distillation Fine-Tuning): a thin family on the backend's distillation base.

The base (``reef.train.slime_backend.distill``) carries the wire row, the
flags, the teacher pass and the divergences; this family names itself,
sets the reference implementation's defaults and forwards the worker hooks
(``objective.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.train.slime_backend.algorithm import register_loss_family
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings


@dataclass(frozen=True)
class SdftSettings(DistillSettings):
    """SDFT's defaults, parsed from the ``--sdft-*`` flags.

    The reference trainer's (idanshen/Self-Distillation, ``distil_config.py``
    and ``main.py``): the model as its own teacher, the forward KL over the
    whole vocabulary, truncated importance sampling capped at 2, every
    response token trained, and the teacher a copy of the weights that moves
    toward the policy by ``ref_model_mixup_alpha`` = 0.01 after every step.
    """

    teacher: str = "self"
    divergence: str = "forward"
    top_k: int = 0
    teacher_update_rate: float = 0.01
    importance_sampling_cap: float = 2.0
    skip_response_tokens: int = 0


@register_loss_family
class SdftAlgorithm(DistillAlgorithm):
    """The self-teacher family: a per-token divergence to the demonstration-conditioned model.

    Before each step the pre-train hook runs a forward pass over every
    sample's teacher sequence (the prompt with the demonstration, then the
    student's response) on the teacher's weights and keeps the teacher's
    next-token distribution at each response position. The loss then puts
    the student's distribution at the same positions, from the training
    forward over the plain request, against it.
    """

    loss_family = "sdft"
    settings_type = SdftSettings


__all__ = ["SdftAlgorithm", "SdftSettings"]
