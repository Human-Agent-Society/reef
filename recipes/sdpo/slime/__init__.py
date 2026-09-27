"""Slime implementation of SDPO on the shared distillation backend."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from reef.train.slime_backend.algorithm import register_loss_family
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings


@dataclass(frozen=True)
class SdpoSettings(DistillSettings):
    """The no-rich-feedback paper defaults; rich feedback uses top-K 20 and reverse KL."""

    teacher: str = "self"
    divergence: str = "jsd"
    top_k: int = 100
    top_k_tail: bool = True
    teacher_update_rate: float = 0.05
    importance_sampling_cap: float = 2.0
    importance_sampling_mode: str = "token"
    skip_response_tokens: int = 0


@register_loss_family
class SdpoAlgorithm(DistillAlgorithm):
    loss_family = "sdpo"
    settings_type = SdpoSettings

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        super().validate_specific_args(args, source)
        if args.calculate_per_token_loss:
            raise RuntimeError(
                f"{source} requires sequence-mean reduction; omit --calculate-per-token-loss to match "
                "the reference's one-response microbatches"
            )


__all__ = ["SdpoAlgorithm", "SdpoSettings"]
