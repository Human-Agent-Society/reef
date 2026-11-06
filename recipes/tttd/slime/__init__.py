"""Slime implementation of TTT-Discover."""

from __future__ import annotations

import math
from argparse import Namespace
from collections.abc import Mapping
from numbers import Real
from typing import Any

from recipes.tttd.objective import tttd_signal
from reef.train.algos.signals import TrainingSignal
from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family
from reef.train.slime_backend.objectives import SlimeObjective, register_slime_objective
from reef.train.types import JobData


@register_loss_family
class TttdAlgorithm(SlimeAlgorithm):
    loss_family = "tttd"
    loss_type = "custom_loss"
    advantages = "required"
    allows_slime_advantage_computation = True
    required_objective_hooks = ("custom_loss_function_path", "custom_advantage_function_path")

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if not getattr(args, "compute_advantages_and_returns", True):
            raise RuntimeError(f"{source} requires Slime advantage computation for frozen-base KL")
        kl_coef = getattr(args, "kl_coef", None)
        # A zero coefficient would also drop the reference-model group the
        # frozen-base KL reads from (with_ref keys on kl_coef != 0), so the
        # coefficient is a tunable positive, not a pinned constant.
        if (
            not isinstance(kl_coef, Real)
            or isinstance(kl_coef, bool)
            or not math.isfinite(float(kl_coef))
            or kl_coef <= 0
        ):
            raise RuntimeError(f"{source} requires a positive finite --kl-coef (paper setting: 0.1)")


@register_slime_objective
class TttdObjective(SlimeObjective):
    name = "tttd"
    loss_family = "tttd"

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        return tttd_signal(data, state)


__all__ = ["TttdAlgorithm", "TttdObjective"]
