"""Slime implementation of SAO (Single-Rollout Asynchronous Optimization)."""

from __future__ import annotations

import argparse
import logging
import time
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from recipes.sao.slime.utils.data_builder import build_sao_rollout_data, sao_sample_row
from recipes.sao.slime.utils.schedule import DEFAULT_CRITIC_STEPS_PER_ACTOR, SaoSchedule
from reef.train.slime_backend.algorithm import SlimeAlgorithm, TrainResult, register_loss_family

_logger = logging.getLogger(__name__)

CRITIC_ATTENTION_PARAM_PATTERN = "self_attention"
DEFAULT_LAMBDA_ALPHA = 1.5
# Paper: value targets use lambda=1 (Monte-Carlo returns) while the policy's
# advantages may use the length-adaptive lambda (arXiv:2607.07508, Section 4).
DEFAULT_CRITIC_LAMBDA = 1.0


@dataclass(frozen=True)
class SaoSettings:
    """SAO driver options parsed from the ``--sao-*`` flag family."""

    length_adaptive_lambda: bool = False
    lambda_alpha: float = DEFAULT_LAMBDA_ALPHA
    critic_lambda: float = DEFAULT_CRITIC_LAMBDA


@register_loss_family
class SaoAlgorithm(SlimeAlgorithm):
    loss_family = "sao"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "forbidden"
    # SAO deliberately runs Slime's pre-train pass: the critic supplies
    # values, then the registered skip-observation GAE hook constructs the
    # actor advantages in the backend.
    allows_slime_advantage_computation = True
    forbidden_advantages_message = (
        "sao advantages are computed by the value model in the training backend; the Reef payload must omit them"
    )

    # Wire surface the adapter layer serves generically. ``action_masks`` is
    # an int column per response token: the rollout manager partitions and
    # tensorizes it, the worker slices it for context parallelism like
    # advantages, ``get_batch`` carries it to the critic microbatch, and the
    # critic's explained-variance metric restricts itself to it.
    # Skip-observation GAE reads it from ``rollout_data`` directly.
    # ``rollout_created_ats`` is consumed driver-side by ``provenance_metrics``
    # and dropped before the workers see it; the skip key keeps the numeric
    # rollout logger safe if that ever changes.
    rollout_data_keys = ("action_masks",)
    rollout_tensor_dtypes: Mapping[str, str] = {"action_masks": "int"}
    response_aligned_keys = ("action_masks",)
    external_batch_keys = ("action_masks",)
    critic_value_mask_key = "action_masks"
    rollout_log_skip_keys = ("rollout_created_ats",)
    # The critic's scalar value head has no counterpart in the HF checkpoint;
    # its bias keeps the worker-local zero initialization.
    critic_value_head_zero_init = True
    # Lane B loss injection: keep Slime's stock policy_loss and swap only the
    # per-token pg primitive (sao_loss, registered on
    # custom_pg_loss_function_path). The adapter layer owns the routing.
    uses_pg_loss_primitive = True
    required_objective_hooks = ("custom_pg_loss_function_path", "custom_advantage_function_path")

    _schedule: SaoSchedule

    # --- stage 1: configure ---

    def configure_backend_args(self, args: Namespace) -> None:
        # SAO deliberately keeps Slime's pre-train advantage pass: the critic
        # supplies values, then the registered skip-observation GAE hook
        # constructs the actor advantages in the backend.
        args.compute_advantages_and_returns = True

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if not getattr(args, "use_critic", False):
            raise RuntimeError(
                f"{source} requires a value model; pass --use-critic on the Slime driver so the critic group is created"
            )

        eps_low = getattr(args, "eps_clip", None)
        if not isinstance(eps_low, Real) or isinstance(eps_low, bool) or not 0 <= float(eps_low) < 1:
            raise RuntimeError(f"{source} requires --eps-clip in [0, 1)")

        eps_high = getattr(args, "eps_clip_high", None)
        if not isinstance(eps_high, Real) or isinstance(eps_high, bool) or eps_high < 0:
            raise RuntimeError(f"{source} requires --eps-clip-high >= 0")
        # Slime silently falls an unset --eps-clip-high back to --eps-clip, which
        # yields a symmetric narrow band — very far from SAO's asymmetric trust
        # region. ``eps_clip_high_explicit`` is recorded by slime_validate_args
        # before that fallback; a namespace built without the parse path counts
        # as explicit because it literally set the attribute.
        if not getattr(args, "eps_clip_high_explicit", True):
            raise RuntimeError(
                f"{source} requires an explicit --eps-clip-high: leaving it unset falls back to "
                "--eps-clip and produces a symmetric clip band, not SAO's asymmetric trust region "
                "(paper settings: eps_clip/eps_clip_high = 0.3/5.0 for reasoning, 0.8/3.0 for coding)"
            )

        # An unset --critic-steps-per-actor means the schedule's paper default.
        steps = getattr(args, "critic_steps_per_actor", None)
        if steps is not None and (not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 1):
            raise RuntimeError(f"{source} requires --critic-steps-per-actor >= 1")
        warmup = getattr(args, "num_critic_only_steps", None)
        if not isinstance(warmup, Integral) or isinstance(warmup, bool) or warmup < 0:
            raise RuntimeError(f"{source} requires --num-critic-only-steps >= 0")

        if getattr(args, "sao_length_adaptive_lambda", False):
            alpha = getattr(args, "sao_lambda_alpha", None)
            if not isinstance(alpha, Real) or isinstance(alpha, bool) or alpha <= 0:
                raise RuntimeError(
                    f"{source} requires --sao-lambda-alpha > 0 when --sao-length-adaptive-lambda is enabled"
                )

        critic_lambda = getattr(args, "sao_critic_lambda", DEFAULT_CRITIC_LAMBDA)
        if (
            not isinstance(critic_lambda, Real)
            or isinstance(critic_lambda, bool)
            or not 0 <= float(critic_lambda) <= 1
        ):
            raise RuntimeError(f"{source} requires --sao-critic-lambda in [0, 1]")

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[SaoSettings, list[str]]:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
        parser.add_argument(
            "--sao-length-adaptive-lambda",
            dest="length_adaptive_lambda",
            action="store_true",
            help=(
                "Use SAO's length-adaptive GAE lambda lambda=1-1/(alpha*l) per sample instead of "
                "the fixed --lambd (arXiv:2607.07508, Section 4). alpha is --sao-lambda-alpha."
            ),
        )
        parser.add_argument(
            "--sao-lambda-alpha",
            dest="lambda_alpha",
            type=float,
            help=f"alpha in SAO's length-adaptive lambda=1-1/(alpha*l). Paper default {DEFAULT_LAMBDA_ALPHA}.",
        )
        parser.add_argument(
            "--sao-critic-lambda",
            dest="critic_lambda",
            type=float,
            help=(
                "GAE lambda for the critic role's value targets. The critic never uses the "
                "length-adaptive lambda; its returns regress toward lambda=1 Monte-Carlo returns "
                f"(arXiv:2607.07508). Default {DEFAULT_CRITIC_LAMBDA}."
            ),
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return SaoSettings(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, SaoSettings) else SaoSettings()
        args.sao_length_adaptive_lambda = settings.length_adaptive_lambda
        args.sao_lambda_alpha = settings.lambda_alpha
        args.sao_critic_lambda = settings.critic_lambda

    # --- stage 2: shape row ---

    def shape_sample_row(self, sample):
        return sao_sample_row(sample)

    # --- stage 3: build batch ---

    def build_rollout_data(self, payload, samples):
        return build_sao_rollout_data(payload, samples, self)

    # --- stage 4: prepare rollout ---

    def configure_critic_args(self, critic_args: Namespace) -> None:
        """Project SAO's critic-role config onto a critic ``Namespace``.

        Two things: (1) freeze the attention stack for an MoE value model
        (SAO Section 4) — an MoE critic trains only its experts, layernorms,
        and value head while attention is held fixed; a dense critic trains
        everything. (2) Project the critic-role GAE lambda: value targets use
        ``lambda_critic`` (default 1, Monte-Carlo returns) while the policy's
        advantages may use the length-adaptive lambda. Both roles run the same
        ``sao_advantages`` hook, so the separation lives entirely
        in the critic role's args — ``objective`` needs no role awareness.
        """
        _apply_attention_freeze(critic_args)

        critic_lambda = getattr(critic_args, "sao_critic_lambda", None)
        if critic_lambda is None:
            return
        critic_args.sao_length_adaptive_lambda = False
        critic_args.lambd = float(critic_lambda)

    def bind(self, config=None, *, critic_steps_per_actor=None, critic_only_steps=0):
        if config is not None and not isinstance(config, SaoSettings):
            raise TypeError("sao bridge algorithm config must be SaoSettings")
        bound = SaoAlgorithm()
        if critic_steps_per_actor is None:
            critic_steps_per_actor = DEFAULT_CRITIC_STEPS_PER_ACTOR
        bound._schedule = SaoSchedule(critic_steps_per_actor, critic_only_steps)
        return bound

    def train(self, rollout_id, refs, *, actor_group, critic_group, resolve):
        if critic_group is None:
            raise RuntimeError(
                "SAO requires a value model, but the bridge was booted without one; start the driver with --use-critic"
            )
        plan = self._schedule.plan(rollout_id)
        critic_values = None
        for _ in range(plan.critic_updates):
            critic_values = resolve(critic_group.async_train(rollout_id, refs))
        actor_results = []
        if plan.train_actor:
            actor_results = list(resolve(actor_group.async_train(rollout_id, refs, external_data=critic_values)) or ())
        return TrainResult(
            actor_results, {"sao/critic_updates": plan.critic_updates, "sao/actor_trained": int(plan.train_actor)}
        )

    def provenance_metrics(self, rollout_data: dict[str, Any], serving_version: str) -> dict[str, Any]:
        head, sep, tail = serving_version.rpartition(":")
        serving_step = int(tail) if sep and tail.isdigit() else None
        metrics: dict[str, Any] = {}
        if serving_step is not None and head:
            steps = [
                step
                for token in rollout_data.get("producing_runtime_load_ids") or []
                if token is not None and (step := _runtime_load_id_sequence(str(token), head)) is not None
            ]
            future_steps = [step for step in steps if step > serving_step]
            if future_steps:
                _logger.warning(
                    "dropping %d producing weight step(s) %s ahead of serving step %d from sao policy-lag stats",
                    len(future_steps),
                    sorted(set(future_steps)),
                    serving_step,
                )
            lags = [serving_step - step for step in steps if step <= serving_step]
            if lags:
                metrics.update({"sao/policy_lag_max": max(lags), "sao/policy_lag_mean": sum(lags) / len(lags)})
        ages = [
            max(0.0, time.time() - float(value))
            for value in rollout_data.get("rollout_created_ats") or []
            if value is not None
        ]
        if ages:
            metrics.update({"sao/queue_age_s_max": max(ages), "sao/queue_age_s_mean": sum(ages) / len(ages)})
        total = sum(rollout_data.get("response_lengths") or [])
        if total > 0:
            trained = sum(sum(mask) for mask in rollout_data.get("loss_masks") or [])
            metrics["sao/effective_token_rate"] = trained / total
        return metrics


def _runtime_load_id_sequence(token: str, incarnation: str) -> int | None:
    head, sep, tail = token.rpartition(":")
    return int(tail) if sep and head == incarnation and tail.isdigit() else None


def _apply_attention_freeze(critic_args: Namespace) -> None:
    """Freeze the critic's attention stack for an MoE value model (SAO Section 4).

    An MoE critic trains only its experts, layernorms, and value head while
    attention is held fixed. A dense critic is the fallback: no patterns, every
    parameter trains. The patterns ride the critic's own
    ``freeze_params_name_list`` and never affect the actor.
    """
    is_moe_critic = bool(getattr(critic_args, "num_experts", 0))
    patterns = [CRITIC_ATTENTION_PARAM_PATTERN] if is_moe_critic else []
    if not patterns:
        return
    # The backend forbids setting both freeze and only-train lists; if the run
    # already pins the critic to an only-train list, respect it rather than
    # crash at validation, and say so.
    if getattr(critic_args, "only_train_params_name_list", None):
        _logger.warning(
            "critic has only_train_params_name_list set; skipping SAO attention freeze to avoid conflicting with it"
        )
        return
    existing = list(getattr(critic_args, "freeze_params_name_list", None) or [])
    for pattern in patterns:
        if pattern not in existing:
            existing.append(pattern)
    critic_args.freeze_params_name_list = existing
    _logger.info("SAO: freezing MoE critic attention via patterns %s", patterns)


__all__ = ["SaoAlgorithm", "SaoSettings"]
