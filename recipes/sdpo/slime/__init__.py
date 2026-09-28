"""Slime implementation of SDPO (Self-Distillation Policy Optimization): a thin family on the backend's distillation base.

The base (``reef.train.slime_backend.distill``) carries the wire row with
its per-sample weight, the flags, the teacher pass with the student's
top-K selection, and the divergences; this family names itself, sets the
reference's Section 3 defaults, pins the reference's reduction and forwards
the worker hooks (``objective.py``).
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from reef.train.slime_backend.algorithm import register_loss_family
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings


@dataclass(frozen=True)
class SdpoSettings(DistillSettings):
    """SDPO's defaults, parsed from the ``--sdpo-*`` flags.

    Section 3 of the reference (lasgroup/SDPO at ``7c457fc1b1f6``,
    ``sdpo.yaml`` and ``actor.yaml``): the model as its own teacher, a copy
    of the weights that moves toward the policy by 0.05 after every step, the
    generalized Jensen-Shannon divergence over the student's top 100 ids plus
    one bucket for the rest of the vocabulary, and detached per-token
    importance weights capped at 2.
    """

    teacher: str = "self"
    divergence: str = "jsd"
    top_k: int = 100
    top_k_source: str = "student"
    top_k_distribution: str = "tail"
    teacher_update_rate: float = 0.05
    importance_sampling_cap: float = 2.0
    importance_sampling_level: str = "token"


@register_loss_family
class SdpoAlgorithm(DistillAlgorithm):
    """The feedback-conditioned self-teacher family: one optimizer step per complete sampling step.

    Before the step the pre-train hook selects the student's top-K ids, then
    runs the teacher over every sample's teacher sequence (the question with
    a successful sibling's response or the environment's feedback, then the
    student's response) and keeps its log-probs at those ids. The loss puts
    the student's distribution at the same ids against it; rollouts whose
    teacher read nothing privileged carry sample weight 0 in the wire row.
    """

    loss_family = "sdpo"
    settings_type = SdpoSettings

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        super().validate_specific_args(args, source)
        # The reference token-means each one-sample microbatch and averages
        # the samples, inactive ones included; Slime's per-token reduction
        # would weight the samples by their lengths instead.
        if args.calculate_per_token_loss:
            raise RuntimeError(f"{source} requires the sequence-mean reduction: omit --calculate-per-token-loss")


__all__ = ["SdpoAlgorithm", "SdpoSettings"]
