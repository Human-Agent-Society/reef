"""SDPO's defaults on the shared Slime distillation implementation."""

from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from reef.train.slime_backend.algorithm import register_loss_family
from reef.train.slime_backend.distill import DistillAlgorithm, DistillSettings
from reef.train.types import TrajectoryItem


@dataclass(frozen=True)
class SdpoSettings(DistillSettings):
    """Section 3 of the reference at 7c457fc1b1f6: EMA, JSD and student top-K plus tail."""

    teacher: str = "self"
    divergence: str = "jsd"
    top_k: int = 100
    top_k_source: str = "student"
    top_k_distribution: str = "tail"
    teacher_update_rate: float = 0.05
    importance_sampling_cap: float = 2.0
    importance_sampling_level: str = "token"
    skip_response_tokens: int = 0


@register_loss_family
class SdpoAlgorithm(DistillAlgorithm):
    loss_family = "sdpo"
    settings_type = SdpoSettings
    rollout_data_keys = (*DistillAlgorithm.rollout_data_keys, "distill_sample_weights")
    external_batch_keys = (*DistillAlgorithm.external_batch_keys, "distill_sample_weights")
    rollout_log_skip_keys = (*DistillAlgorithm.rollout_log_skip_keys, "distill_sample_weights")

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        super().validate_specific_args(args, source)
        if args.calculate_per_token_loss:
            raise RuntimeError(f"{source} requires sequence-mean reduction: omit --calculate-per-token-loss")
        if args.distill_skip_response_tokens != 0:
            raise RuntimeError(f"{source} requires --sdpo-skip-response-tokens=0")
        if args.attention_dropout != 0 or args.hidden_dropout != 0:
            raise RuntimeError(f"{source} requires zero attention/hidden dropout to reuse the student's top-K support")

    def shape_sample_row(self, sample: TrajectoryItem) -> list[Any]:
        return [*super().shape_sample_row(sample), sample.training["distill_sample_mask"]]

    def build_rollout_data(self, payload: Mapping[str, Any], samples: Sequence) -> dict:
        rows = []
        masks = []
        for index, row in enumerate(samples):
            if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 7:
                raise ValueError(f"sdpo sample {index} must be a distillation row followed by distill_sample_mask")
            mask = row[6]
            if not isinstance(mask, Integral) or isinstance(mask, bool) or mask not in (0, 1):
                raise ValueError(f"sdpo sample {index} distill_sample_mask must be 0 or 1")
            rows.append(row[:6])
            masks.append(mask)
        data = super().build_rollout_data(payload, rows)
        # The reference uses token-mean inside a one-sample microbatch, then
        # averages all microbatch losses, including zero-target samples. This
        # is a sequence mean over the original grid, not an active-token mean.
        data["distill_sample_weights"] = [float(mask) for mask in masks]
        return data


__all__ = ["SdpoAlgorithm", "SdpoSettings"]
