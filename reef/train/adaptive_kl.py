"""Pure controller for PPO/RLHF-style reference-policy KL shaping.

This module deliberately has no Torch, Ray, Slime, or recipe imports.  It owns
only controller state and numerical policy; rollout transport, reward shaping,
and checkpoint I/O belong to the integration layer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

AdaptiveKLMode = Literal["off", "telemetry", "shadow", "reward"]
AdaptiveKLControllerType = Literal["reef", "verl", "fixed"]

SCHEMA_VERSION = 1
CONTROLLER_VERSION = "adaptive-kl-v1"
DEFAULT_LOSS_FAMILY = "ppo_rlhf_reference_reward"
_MODES = frozenset({"off", "telemetry", "shadow", "reward"})
_CONTROLLER_TYPES = frozenset({"reef", "verl", "fixed"})
DEFAULT_VERL_HORIZON = 10_000
DEFAULT_VERL_ERROR_CLIP = 0.2


def _finite_real(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class AdaptiveKLConfig:
    """Validated, serializable configuration for one controller instance."""

    target_kl: float
    initial_beta: float
    min_beta: float
    max_beta: float
    adaptation_rate: float
    ema_decay: float
    max_update_ratio: float
    reference_policy_id: str
    mode: AdaptiveKLMode = "off"
    cooldown_steps: int = 0
    spike_threshold: float | None = None
    loss_family: str = DEFAULT_LOSS_FAMILY
    controller_version: str = CONTROLLER_VERSION
    controller_type: AdaptiveKLControllerType = "reef"
    horizon: int = DEFAULT_VERL_HORIZON
    error_clip: float = DEFAULT_VERL_ERROR_CLIP

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        if self.controller_type not in _CONTROLLER_TYPES:
            raise ValueError(f"controller_type must be one of {sorted(_CONTROLLER_TYPES)}")
        if not isinstance(self.reference_policy_id, str) or not self.reference_policy_id:
            raise ValueError("reference_policy_id must be a non-empty string")
        if not isinstance(self.loss_family, str) or not self.loss_family:
            raise ValueError("loss_family must be a non-empty string")
        if self.controller_version != CONTROLLER_VERSION:
            raise ValueError(f"unsupported controller_version: {self.controller_version!r}")

        target_kl = _finite_real(self.target_kl, "target_kl")
        initial_beta = _finite_real(self.initial_beta, "initial_beta")
        min_beta = _finite_real(self.min_beta, "min_beta")
        max_beta = _finite_real(self.max_beta, "max_beta")
        adaptation_rate = _finite_real(self.adaptation_rate, "adaptation_rate")
        ema_decay = _finite_real(self.ema_decay, "ema_decay")
        max_update_ratio = _finite_real(self.max_update_ratio, "max_update_ratio")
        error_clip = _finite_real(self.error_clip, "error_clip")

        if target_kl <= 0:
            raise ValueError("target_kl must be positive")
        if min_beta <= 0 or initial_beta <= 0 or max_beta <= 0:
            raise ValueError("min_beta, initial_beta, and max_beta must be positive")
        if min_beta > max_beta:
            raise ValueError("min_beta must be less than or equal to max_beta")
        if not min_beta <= initial_beta <= max_beta:
            raise ValueError("initial_beta must be within [min_beta, max_beta]")
        if adaptation_rate <= 0:
            raise ValueError("adaptation_rate must be positive")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if max_update_ratio < 1:
            raise ValueError("max_update_ratio must be at least 1")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if error_clip <= 0:
            raise ValueError("error_clip must be positive")

        _non_negative_int(self.cooldown_steps, "cooldown_steps")
        if self.spike_threshold is not None:
            spike_threshold = _finite_real(self.spike_threshold, "spike_threshold")
            if spike_threshold <= 0:
                raise ValueError("spike_threshold must be positive when provided")

        # Keep normalized values in the frozen object even when callers pass
        # integer-valued numerics.  This also makes state round-trips stable.
        object.__setattr__(self, "target_kl", target_kl)
        object.__setattr__(self, "initial_beta", initial_beta)
        object.__setattr__(self, "min_beta", min_beta)
        object.__setattr__(self, "max_beta", max_beta)
        object.__setattr__(self, "adaptation_rate", adaptation_rate)
        object.__setattr__(self, "ema_decay", ema_decay)
        object.__setattr__(self, "max_update_ratio", max_update_ratio)
        object.__setattr__(self, "error_clip", error_clip)
        if self.spike_threshold is not None:
            object.__setattr__(self, "spike_threshold", float(self.spike_threshold))


@dataclass(frozen=True, slots=True)
class AdaptiveKLDecision:
    """The observable result of one controller observation."""

    observed_kl: float | None
    beta_before: float
    beta_next: float
    ema_kl: float | None
    beta_update_ratio: float
    update_applied: bool
    update_reason: str
    step: int
    cooldown_remaining: int
    controller_kl: float | None = None

    def metrics(
        self,
        *,
        mode: AdaptiveKLMode,
        reference_policy_id: str,
        controller_type: AdaptiveKLControllerType = "reef",
        horizon: int | None = None,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Return the namespaced telemetry contract for this decision."""
        return {
            "adaptive_kl/mode": mode,
            "adaptive_kl/controller": controller_type,
            "adaptive_kl/observed_kl": self.observed_kl,
            "adaptive_kl/smoothed_kl": self.ema_kl,
            "adaptive_kl/controller_kl": self.controller_kl,
            "adaptive_kl/beta": self.beta_before,
            "adaptive_kl/beta_next": self.beta_next,
            "adaptive_kl/beta_update_ratio": self.beta_update_ratio,
            "adaptive_kl/update_applied": self.update_applied,
            "adaptive_kl/update_reason": self.update_reason,
            "adaptive_kl/reference_policy_id": reference_policy_id,
            "adaptive_kl/horizon": horizon,
            "adaptive_kl/n_steps": n_steps,
        }


class AdaptiveKLController:
    """Bounded EMA controller for the reward-side reference KL coefficient."""

    def __init__(self, config: AdaptiveKLConfig) -> None:
        if not isinstance(config, AdaptiveKLConfig):
            raise TypeError("config must be an AdaptiveKLConfig")
        self.config = config
        self._beta = config.initial_beta
        self._ema_kl: float | None = None
        self._step = 0
        self._cooldown_remaining = 0

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def ema_kl(self) -> float | None:
        return self._ema_kl

    @property
    def step(self) -> int:
        return self._step

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    def state_dict(self) -> dict[str, Any]:
        """Return the versioned, JSON-compatible active controller state."""
        return {
            "schema_version": SCHEMA_VERSION,
            "beta": self._beta,
            "target_kl": self.config.target_kl,
            "ema_kl": self._ema_kl,
            "step": self._step,
            "cooldown_remaining": self._cooldown_remaining,
            "reference_policy_id": self.config.reference_policy_id,
            "loss_family": self.config.loss_family,
            "controller_version": self.config.controller_version,
            "controller_type": self.config.controller_type,
            "horizon": self.config.horizon,
            "error_clip": self.config.error_clip,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], config: AdaptiveKLConfig) -> AdaptiveKLController:
        """Restore state after validating schema, identity, and bounds."""
        if not isinstance(state, Mapping):
            raise TypeError("controller state must be a mapping")

        schema_version = state.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported controller state schema_version: {schema_version!r}")
        if state.get("controller_version") != config.controller_version:
            raise ValueError("controller version mismatch")
        if state.get("reference_policy_id") != config.reference_policy_id:
            raise ValueError("reference policy identity mismatch")
        if state.get("loss_family") != config.loss_family:
            raise ValueError("loss family mismatch")
        # Older checkpoints predate the controller selector.  They are
        # unambiguously the original Reef controller and remain restorable.
        if state.get("controller_type", "reef") != config.controller_type:
            raise ValueError("controller type mismatch")
        if state.get("horizon", DEFAULT_VERL_HORIZON) != config.horizon:
            raise ValueError("horizon does not match controller configuration")
        if state.get("error_clip", DEFAULT_VERL_ERROR_CLIP) != config.error_clip:
            raise ValueError("error_clip does not match controller configuration")

        target_kl = _finite_real(state.get("target_kl"), "state.target_kl")
        if target_kl != config.target_kl:
            raise ValueError("target_kl does not match controller configuration")

        beta = _finite_real(state.get("beta"), "state.beta")
        if not config.min_beta <= beta <= config.max_beta:
            raise ValueError("state.beta is outside configured bounds")

        ema_kl = state.get("ema_kl")
        if ema_kl is not None:
            ema_kl = _finite_real(ema_kl, "state.ema_kl")
            if ema_kl < 0:
                raise ValueError("state.ema_kl must be non-negative")

        step = _non_negative_int(state.get("step"), "state.step")
        cooldown_remaining = _non_negative_int(state.get("cooldown_remaining"), "state.cooldown_remaining")
        if cooldown_remaining > config.cooldown_steps:
            raise ValueError("state.cooldown_remaining exceeds configured cooldown_steps")

        controller = cls(config)
        controller._beta = beta
        controller._ema_kl = ema_kl
        controller._step = step
        controller._cooldown_remaining = cooldown_remaining
        return controller

    def observe(
        self,
        observed_kl: Real,
        *,
        training_step_succeeded: bool,
        loss_is_finite: bool,
        valid_token_count: int | None = None,
        n_steps: int = 1,
    ) -> AdaptiveKLDecision:
        """Observe one completed logical step and, when valid, update beta.

        ``observed_kl`` must already use the upstream reference-KL estimator
        and valid-token convention.  The controller never computes KL itself.
        ``valid_token_count`` is optional for callers that perform mask
        validation upstream; when supplied, zero is rejected explicitly.
        """
        beta_before = self._beta
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")

        def decision(
            observed: float | None,
            reason: str,
            *,
            controller_kl: float | None = None,
        ) -> AdaptiveKLDecision:
            return AdaptiveKLDecision(
                observed_kl=observed,
                beta_before=beta_before,
                beta_next=self._beta,
                ema_kl=self._ema_kl,
                beta_update_ratio=self._beta / beta_before,
                update_applied=self._beta != beta_before,
                update_reason=reason,
                step=self._step,
                cooldown_remaining=self._cooldown_remaining,
                controller_kl=controller_kl,
            )

        if self.config.mode == "off":
            return decision(None, "disabled")
        if not isinstance(training_step_succeeded, bool):
            raise ValueError("training_step_succeeded must be a boolean")
        if not training_step_succeeded:
            return decision(None, "training_step_failed")
        if not isinstance(loss_is_finite, bool):
            raise ValueError("loss_is_finite must be a boolean")
        if not loss_is_finite:
            return decision(None, "non_finite_loss")
        if valid_token_count is not None:
            if not isinstance(valid_token_count, int) or isinstance(valid_token_count, bool) or valid_token_count < 0:
                raise ValueError("valid_token_count must be a non-negative integer or None")
            if valid_token_count == 0:
                return decision(None, "empty_valid_token_mask")

        try:
            observed = _finite_real(observed_kl, "observed_kl")
        except ValueError:
            return decision(None, "invalid_observed_kl")
        # A sampled log-ratio KL estimate can be negative even though the
        # population KL is non-negative. VERL's controller consumes that
        # per-batch estimate directly, so finite negative values must remain
        # part of the proportional-control signal. Keep the stricter
        # non-negative guard for the original Reef controller contract.
        if observed < 0 and self.config.controller_type == "reef":
            return decision(None, "invalid_observed_kl")

        if self.config.controller_type == "fixed":
            self._step += 1
            return decision(observed, "fixed", controller_kl=observed)

        if self.config.controller_type == "verl":
            self._step += 1
            proportional_error = max(
                -self.config.error_clip,
                min(self.config.error_clip, observed / self.config.target_kl - 1),
            )
            multiplier = 1 + proportional_error * n_steps / self.config.horizon
            if not math.isfinite(multiplier) or multiplier <= 0:
                return decision(observed, "invalid_beta_candidate", controller_kl=observed)
            raw_beta = self._beta * multiplier
            beta_next = min(max(raw_beta, self.config.min_beta), self.config.max_beta)
            if not math.isfinite(beta_next):
                return decision(observed, "invalid_beta_candidate", controller_kl=observed)
            self._beta = beta_next
            reason = "updated" if beta_next != beta_before else "unchanged"
            return decision(observed, reason, controller_kl=observed)

        if self._ema_kl is None:
            next_ema = observed
        else:
            next_ema = self.config.ema_decay * self._ema_kl + (1 - self.config.ema_decay) * observed

        if not math.isfinite(next_ema):
            return decision(observed, "invalid_ema_candidate")

        self._ema_kl = next_ema
        self._step += 1

        if self.config.spike_threshold is not None and observed > self.config.spike_threshold:
            self._cooldown_remaining = max(self._cooldown_remaining, self.config.cooldown_steps)
            return decision(observed, "kl_spike", controller_kl=next_ema)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return decision(observed, "cooldown", controller_kl=next_ema)

        exponent = self.config.adaptation_rate * (next_ema / self.config.target_kl - 1)
        try:
            raw_beta = self._beta * math.exp(exponent)
        except OverflowError:
            raw_beta = math.inf if exponent > 0 else 0.0

        ratio_min = self._beta / self.config.max_update_ratio
        ratio_max = self._beta * self.config.max_update_ratio
        beta_next = min(max(raw_beta, ratio_min), ratio_max)
        beta_next = min(max(beta_next, self.config.min_beta), self.config.max_beta)
        if not math.isfinite(beta_next):
            return decision(observed, "invalid_beta_candidate", controller_kl=next_ema)

        self._beta = beta_next
        if beta_next == beta_before:
            if next_ema == self.config.target_kl:
                reason = "target_reached"
            elif beta_next == self.config.min_beta:
                reason = "min_beta_bound"
            elif beta_next == self.config.max_beta:
                reason = "max_beta_bound"
            else:
                reason = "unchanged"
        else:
            reason = "updated"
        return decision(observed, reason, controller_kl=next_ema)


__all__ = [
    "CONTROLLER_VERSION",
    "DEFAULT_LOSS_FAMILY",
    "DEFAULT_VERL_ERROR_CLIP",
    "DEFAULT_VERL_HORIZON",
    "SCHEMA_VERSION",
    "AdaptiveKLConfig",
    "AdaptiveKLController",
    "AdaptiveKLControllerType",
    "AdaptiveKLDecision",
    "AdaptiveKLMode",
]
