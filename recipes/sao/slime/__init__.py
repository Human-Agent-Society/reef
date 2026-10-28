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

from reef.core.job_data import TrajectorySample
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import PolicyGradientWeight, SlimeAlgorithm, TrainResult, register_loss_family
from reef.train.slime_backend.data_builder import build_policy_rollout_data, validate_policy_columns

_logger = logging.getLogger(__name__)

CRITIC_ATTENTION_PARAM_PATTERN = "self_attention"
DEFAULT_LAMBDA_ALPHA = 1.5
# Paper: value targets use lambda=1 (Monte-Carlo returns) while the policy's
# advantages may use the length-adaptive lambda (arXiv:2607.07508, Section 4).
DEFAULT_CRITIC_LAMBDA = 1.0
# Paper default: K critic updates per actor update.
DEFAULT_CRITIC_STEPS_PER_ACTOR = 2
ROW_SHAPE = (
    "[source_id, tokens, loss_mask, rollout_log_probs, reward, action_mask, "
    "producing_runtime_load_id, rollout_created_at]"
)


@dataclass(frozen=True)
class SaoSettings:
    """SAO driver options parsed from the ``--sao-*`` flag family."""

    length_adaptive_lambda: bool = False
    lambda_alpha: float = DEFAULT_LAMBDA_ALPHA
    critic_lambda: float = DEFAULT_CRITIC_LAMBDA
    freeze_critic_attention: bool = False


@dataclass(frozen=True)
class SaoStepPlan:
    """What one training step does: the critic updates, then the actor trains or waits.

    Values: ``critic_updates`` is an integer >= 1 and ``train_actor`` a bool.
    Purpose: :meth:`SaoSchedule.plan` makes one per rollout.
    :meth:`SaoAlgorithm.train` runs the critic ``critic_updates`` times, then
    the actor once if ``train_actor`` is true.
    """

    critic_updates: int
    train_actor: bool

    def __post_init__(self) -> None:
        if self.critic_updates < 1:
            raise ValueError("critic_updates must be >= 1")


@dataclass(frozen=True)
class SaoSchedule:
    """The critic and actor cadence of one run.

    Values: ``critic_steps_per_actor`` (the paper's K) is an integer >= 1 and
    ``critic_only_steps`` an integer >= 0.
    Purpose: :meth:`SaoAlgorithm.bind` makes one from the driver flags and
    :meth:`SaoAlgorithm.train` asks it for each rollout's plan. For a rollout
    index ``i >= 0``, ``plan(i)`` is ``SaoStepPlan(K, i >= critic_only_steps)``:
    every step updates the critic K times, and the actor trains once the
    critic-only warmup is over.
    """

    critic_steps_per_actor: int = DEFAULT_CRITIC_STEPS_PER_ACTOR
    critic_only_steps: int = 0

    def __post_init__(self) -> None:
        if self.critic_steps_per_actor < 1:
            raise ValueError("critic_steps_per_actor (K) must be >= 1")
        if self.critic_only_steps < 0:
            raise ValueError("critic_only_steps must be >= 0")

    def plan(self, rollout_index: int) -> SaoStepPlan:
        if rollout_index < 0:
            raise ValueError("rollout_index must be >= 0")
        return SaoStepPlan(self.critic_steps_per_actor, rollout_index >= self.critic_only_steps)


def sao_sample_row(sample: TrajectorySample) -> list[Any]:
    """Shape one Reef sample into SAO's 8-element wire row.

    The first five columns are the shared policy 5-tuple; SAO appends the
    action mask (for skip-observation GAE), producing runtime load ID, and
    creation time, which the 5-tuple has no slots for. Outbound mirror of
    :func:`build_sao_rollout_data`.
    """
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        list(sample.training.get("action_mask", [])),
        sample.training.get("runtime_load_id", None),
        sample.training.get("rollout_created_at", None),
    ]


def build_sao_rollout_data(
    payload: Mapping[str, Any],
    samples: Sequence,
    spec: SlimeAlgorithm,
) -> dict:
    """Validate and convert Reef SAO rows into Slime's external rollout payload.

    SAO's wire row keeps the policy 5-tuple as its prefix and appends the
    action mask, producing version, and creation time: ``[source_id, tokens,
    loss_mask, rollout_log_probs, reward, action_mask, producing_runtime_load_id,
    rollout_created_at]``. The shared policy builder assembles the 5-tuple
    columns; this builder validates the appended columns and attaches them.
    Each SAO sample is one independently scheduled rollout, so there is no
    comparison-group barrier and ``advantages`` are never shipped — the value
    model computes them from the critic's own forward pass inside the
    skip-observation GAE (the spec declares them forbidden;
    ``to_slime_rollout_data`` enforces it).
    """
    base_rows: list[list[Any]] = []
    action_masks: list[list[int]] = []
    producing_runtime_load_ids: list[str | None] = []
    rollout_created_ats: list[float | None] = []

    for sample_index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 8:
            raise ValueError(f"SAO sample {sample_index} must be {ROW_SHAPE}")
        (
            source_id,
            row_tokens,
            row_loss_mask,
            row_log_probs,
            reward,
            row_action_mask,
            producing_runtime_load_id,
            rollout_created_at,
        ) = row
        row_tokens, row_loss_mask, row_log_probs, row_action_mask, reward = validate_policy_columns(
            f"SAO sample {sample_index}",
            row_tokens,
            row_loss_mask,
            row_log_probs,
            reward,
            action_mask=row_action_mask,
            require_log_probs=True,
        )
        if row_action_mask is None:
            raise RuntimeError("validated SAO sample has no action mask")

        base_rows.append([source_id, row_tokens, row_loss_mask, row_log_probs, reward])
        action_masks.append(row_action_mask)
        producing_runtime_load_ids.append(
            None if producing_runtime_load_id is None else str(producing_runtime_load_id)
        )
        rollout_created_ats.append(None if rollout_created_at is None else float(rollout_created_at))

    data = build_policy_rollout_data({**dict(payload), "samples": base_rows}, base_rows, spec)
    data["action_masks"] = action_masks
    data["producing_runtime_load_ids"] = producing_runtime_load_ids
    data["rollout_created_ats"] = rollout_created_ats
    return data


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
    # ``rollout_created_ats`` is consumed driver-side by ``rollout_metrics``
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

    def policy_gradient_weight(self, args: Namespace) -> PolicyGradientWeight:
        # compute_sao_loss: -sg(f(r)) * A * log p with r against the rollout
        # log-probs and f(r) = r strictly inside (1 - eps_clip, 1 + eps_clip_high).
        return PolicyGradientWeight("masked", lower=1.0 - args.eps_clip, upper=1.0 + args.eps_clip_high)

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if not getattr(args, "use_critic", False):
            raise RuntimeError(
                f"{source} requires a value model; set training.options.use-critic so Slime creates the critic group"
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
        parser.add_argument(
            "--sao-freeze-critic-attention",
            dest="freeze_critic_attention",
            action="store_true",
            help="Freeze critic self-attention, including dense models. MoE critics already freeze it by default.",
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return SaoSettings(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, SaoSettings) else SaoSettings()
        args.sao_length_adaptive_lambda = settings.length_adaptive_lambda
        args.sao_lambda_alpha = settings.lambda_alpha
        args.sao_critic_lambda = settings.critic_lambda
        args.sao_freeze_critic_attention = settings.freeze_critic_attention

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
        and value head while attention is held fixed. Dense critics freeze
        attention only with --sao-freeze-critic-attention.
        (2) Project the critic-role GAE lambda: value targets use
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
            raise TypeError("sao algorithm config must be SaoSettings")
        bound = SaoAlgorithm()
        if critic_steps_per_actor is None:
            critic_steps_per_actor = DEFAULT_CRITIC_STEPS_PER_ACTOR
        bound._schedule = SaoSchedule(critic_steps_per_actor, critic_only_steps)
        return bound

    def train(self, rollout_id, refs, *, actor_group, critic_group, resolve):
        if critic_group is None:
            raise RuntimeError(
                "SAO requires a value model, but Slime started without one; set training.options.use-critic"
            )
        plan = self._schedule.plan(rollout_id)
        critic_values = None
        for _ in range(plan.critic_updates):
            critic_values = resolve(critic_group.async_train(rollout_id, refs))
        actor_results = []
        if plan.train_actor:
            actor_results = list(resolve(actor_group.async_train(rollout_id, refs, external_data=critic_values)) or ())
        return TrainResult(
            actor_results,
            {"sao/critic_updates": plan.critic_updates, "sao/actor_trained": int(plan.train_actor)},
            actor_trained=plan.train_actor,
        )

    def rollout_metrics(self, rollout_data: dict[str, Any]) -> dict[str, Any]:
        """Telemetry of one rollout's data: its producing-version spread, its queue age and its trained token share.

        A sample's producing-version spread is how many versions its producing
        version (``incarnation:sequence``) is behind the newest producing
        version of the same serving incarnation in the data. It measures how
        far apart the data's samples were produced, not how far they are
        behind the weights that train on them. A sample whose version has no
        sequence does not count.
        """
        metrics: dict[str, Any] = {}
        versions = [
            (head, int(tail))
            for token in rollout_data.get("producing_runtime_load_ids") or []
            if token is not None
            for head, sep, tail in (str(token).rpartition(":"),)
            if sep and head and tail.isdigit()
        ]
        if versions:
            newest: dict[str, int] = {}
            for incarnation, sequence in versions:
                newest[incarnation] = max(sequence, newest.get(incarnation, sequence))
            spreads = [newest[incarnation] - sequence for incarnation, sequence in versions]
            metrics.update(
                {
                    "sao/producing_version_spread_max": max(spreads),
                    "sao/producing_version_spread_mean": sum(spreads) / len(spreads),
                }
            )
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


def _apply_attention_freeze(critic_args: Namespace) -> None:
    """Freeze the critic's attention stack for an MoE value model (SAO Section 4).

    An MoE critic trains only its experts, layernorms, and value head while
    attention is held fixed. Dense critics can opt into the same freeze;
    otherwise every parameter trains. The patterns ride the critic's own
    ``freeze_params_name_list`` and never affect the actor.
    """
    is_moe_critic = bool(getattr(critic_args, "num_experts", 0))
    explicit_freeze = getattr(critic_args, "sao_freeze_critic_attention", False)
    patterns = [CRITIC_ATTENTION_PARAM_PATTERN] if is_moe_critic or explicit_freeze else []
    if not patterns:
        return
    # The backend forbids setting both freeze and only-train lists; if the run
    # already pins the critic to an only-train list, respect it rather than
    # crash at validation, and say so.
    if getattr(critic_args, "only_train_params_name_list", None):
        if explicit_freeze:
            raise ValueError("--sao-freeze-critic-attention conflicts with --only-train-params-name-list")
        _logger.warning(
            "critic has only_train_params_name_list set; skipping SAO attention freeze to avoid conflicting with it"
        )
        return
    existing = list(getattr(critic_args, "freeze_params_name_list", None) or [])
    for pattern in patterns:
        if pattern not in existing:
            existing.append(pattern)
    critic_args.freeze_params_name_list = existing
    _logger.info("SAO: freezing critic attention via patterns %s", patterns)


__all__ = ["SaoAlgorithm", "SaoSchedule", "SaoSettings", "SaoStepPlan", "build_sao_rollout_data", "sao_sample_row"]
