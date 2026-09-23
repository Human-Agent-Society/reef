"""Reef-side configuration and validation for reference-policy KL telemetry.

The numerical controller lives in :mod:`reef.train.adaptive_kl` and remains
dependency-light.  This adapter owns only Slime argument validation and the
boundary between the driver-side controller and worker metrics.
"""

from __future__ import annotations

import argparse
import math
from numbers import Real
from typing import Any

from reef.train.adaptive_kl import (
    AdaptiveKLConfig,
    CONTROLLER_VERSION,
    DEFAULT_VERL_ERROR_CLIP,
    DEFAULT_VERL_HORIZON,
    DEFAULT_LOSS_FAMILY,
)

DEFAULT_TARGET_KL = 0.1
DEFAULT_MIN_BETA = 1e-6
DEFAULT_MAX_BETA = 10.0
DEFAULT_ADAPTATION_RATE = 0.05
DEFAULT_EMA_DECAY = 0.9
DEFAULT_MAX_UPDATE_RATIO = 2.0
DEFAULT_CONTROLLER = "reef"


def add_adaptive_kl_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add explicit, opt-in adaptive-KL arguments to the Slime parser."""
    parser.add_argument(
        "--adaptive-kl-mode",
        choices=("off", "telemetry", "shadow", "reward"),
        default="off",
        help="Reference-policy reward-KL controller mode; off is the default.",
    )
    parser.add_argument(
        "--adaptive-kl-controller",
        choices=("reef", "verl", "fixed"),
        default=DEFAULT_CONTROLLER,
        help="Controller math: existing Reef EMA, VERL-compatible, or fixed beta.",
    )
    parser.add_argument("--adaptive-kl-target", type=float, default=DEFAULT_TARGET_KL)
    parser.add_argument("--adaptive-kl-initial-beta", type=float, default=None)
    parser.add_argument("--adaptive-kl-min-beta", type=float, default=DEFAULT_MIN_BETA)
    parser.add_argument("--adaptive-kl-max-beta", type=float, default=DEFAULT_MAX_BETA)
    parser.add_argument("--adaptive-kl-adaptation-rate", type=float, default=DEFAULT_ADAPTATION_RATE)
    parser.add_argument("--adaptive-kl-ema-decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--adaptive-kl-max-update-ratio", type=float, default=DEFAULT_MAX_UPDATE_RATIO)
    parser.add_argument("--adaptive-kl-horizon", type=int, default=DEFAULT_VERL_HORIZON)
    parser.add_argument("--adaptive-kl-error-clip", type=float, default=DEFAULT_VERL_ERROR_CLIP)
    parser.add_argument("--adaptive-kl-cooldown-steps", type=int, default=0)
    parser.add_argument("--adaptive-kl-spike-threshold", type=float, default=None)
    parser.add_argument("--adaptive-kl-reference-policy-id", type=str, default=None)
    return parser


def _positive_finite(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise RuntimeError(f"{name} must be a positive finite number")
    value = float(value)
    if value <= 0 or not math.isfinite(value):
        raise RuntimeError(f"{name} must be a positive finite number")
    return value


def adaptive_kl_config_from_args(args) -> AdaptiveKLConfig | None:
    """Build the validated controller config, or ``None`` when disabled."""
    mode = getattr(args, "adaptive_kl_mode", "off")
    if mode == "off":
        return None
    initial_beta = getattr(args, "adaptive_kl_initial_beta", None)
    if initial_beta is None:
        initial_beta = getattr(args, "kl_coef", None)
    reference_policy_id = getattr(args, "adaptive_kl_reference_policy_id", None)
    if not isinstance(reference_policy_id, str) or not reference_policy_id.strip():
        raise RuntimeError("adaptive KL requires --adaptive-kl-reference-policy-id")
    try:
        return AdaptiveKLConfig(
            target_kl=getattr(args, "adaptive_kl_target", DEFAULT_TARGET_KL),
            initial_beta=initial_beta,
            min_beta=getattr(args, "adaptive_kl_min_beta", DEFAULT_MIN_BETA),
            max_beta=getattr(args, "adaptive_kl_max_beta", DEFAULT_MAX_BETA),
            adaptation_rate=getattr(args, "adaptive_kl_adaptation_rate", DEFAULT_ADAPTATION_RATE),
            ema_decay=getattr(args, "adaptive_kl_ema_decay", DEFAULT_EMA_DECAY),
            max_update_ratio=getattr(args, "adaptive_kl_max_update_ratio", DEFAULT_MAX_UPDATE_RATIO),
            reference_policy_id=reference_policy_id.strip(),
            mode=mode,
            cooldown_steps=getattr(args, "adaptive_kl_cooldown_steps", 0),
            spike_threshold=getattr(args, "adaptive_kl_spike_threshold", None),
            loss_family=DEFAULT_LOSS_FAMILY,
            controller_version=CONTROLLER_VERSION,
            controller_type=getattr(args, "adaptive_kl_controller", DEFAULT_CONTROLLER),
            horizon=getattr(args, "adaptive_kl_horizon", DEFAULT_VERL_HORIZON),
            error_clip=getattr(args, "adaptive_kl_error_clip", DEFAULT_VERL_ERROR_CLIP),
        )
    except ValueError as error:
        raise RuntimeError(f"invalid adaptive KL configuration: {error}") from error


def validate_adaptive_kl_args(args, *, loss_family: str | None = None) -> AdaptiveKLConfig | None:
    """Validate adaptive KL only on the dedicated standard PPO/RLHF path.

    The current Reef recipes either own another KL family or install a custom
    loss.  They are deliberately rejected here instead of being silently
    treated as PPO/RLHF reference-reward training.
    """
    config = adaptive_kl_config_from_args(args)
    if config is None:
        return None
    if loss_family != DEFAULT_LOSS_FAMILY:
        raise RuntimeError(
            "adaptive KL requires the dedicated ppo_rlhf_reference_reward loss family; "
            "existing SAO, TTTD, OpenClawRL, and custom loss families are not interchangeable"
        )
    if getattr(args, "loss_type", None) != "policy_loss":
        raise RuntimeError("adaptive KL telemetry requires Slime --loss-type policy_loss")
    if not getattr(args, "compute_advantages_and_returns", False):
        raise RuntimeError("adaptive KL telemetry requires Slime advantage computation")
    if getattr(args, "advantage_estimator", None) != "ppo":
        raise RuntimeError(
            "adaptive KL telemetry requires advantage_estimator=ppo; "
            "ppo_kl, SAO, TTTD, and other KL families are not interchangeable"
        )
    kl_coef = _positive_finite(getattr(args, "kl_coef", None), "Slime kl_coef")
    if getattr(args, "use_kl_loss", False) or getattr(args, "use_opd", False):
        raise RuntimeError("adaptive KL telemetry cannot be combined with loss-side KL or OPD divergence")
    # The worker uses the existing fixed beta to compute the current batch's
    # reference KL.  The controller's initial beta is hypothetical in these
    # pre-reward modes and must not silently rewrite that baseline.
    if getattr(args, "adaptive_kl_initial_beta", None) is None:
        args.adaptive_kl_initial_beta = kl_coef
    args.adaptive_kl_reference_policy_id = config.reference_policy_id
    return config


__all__ = [
    "DEFAULT_ADAPTATION_RATE",
    "DEFAULT_EMA_DECAY",
    "DEFAULT_MAX_BETA",
    "DEFAULT_MAX_UPDATE_RATIO",
    "DEFAULT_MIN_BETA",
    "DEFAULT_TARGET_KL",
    "adaptive_kl_config_from_args",
    "add_adaptive_kl_arguments",
    "validate_adaptive_kl_args",
]
