"""Reef-specific arguments layered onto the runtime's parser hook."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from reef.train.slime_backend.loss_families import LOSS_FAMILIES
from reef.train.slime_backend.reef_adapters.megatron.lora import validate_megatron_lora_args

REEF_MEGATRON_INIT_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.initialize_megatron_objective"
REEF_MODEL_PROVIDER_PATH = "reef.train.slime_backend.reef_adapters.megatron.model_provider.provide_actor_model"


def add_reef_slime_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register options implemented by Reef rather than the runtime."""
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="Weight-conversion implementation selected by the Reef Slime adapter.",
    )
    parser.add_argument("--megatron-lora-rank", type=int, default=0)
    parser.add_argument("--megatron-lora-alpha", type=int, default=None)
    parser.add_argument("--megatron-lora-dropout", type=float, default=0.0)
    parser.add_argument("--megatron-lora-target-modules", type=str, nargs="+", default=None)
    parser.add_argument(
        "--max-loaded-loras",
        type=int,
        default=1,
        help="Adapter slots the SGLang engine keeps loaded on the shared base model (>= 1).",
    )
    parser.add_argument("--check-lora-weight-equal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-lora-base-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--critic-steps-per-actor",
        type=int,
        default=None,
        help="Critic optimizer steps per actor step; unset means the loss family's own default.",
    )
    parser.add_argument("--critic-save", type=str, default=None)
    parser.add_argument(
        "--custom-pg-loss-function-path",
        type=str,
        default=None,
        help="Reef per-token policy-gradient loss primitive.",
    )
    parser.add_argument(
        "--use-critic",
        action="store_true",
        default=False,
        help="Enable a critic when the selected Slime advantage estimator does not imply one.",
    )
    return parser


def finalize_reef_slime_args(args: Any, arguments: Sequence[str]) -> None:
    """Restore Reef derivations and install runtime extension hooks."""
    explicitly_enabled_critic = any(
        argument == "--use-critic" or argument.startswith("--use-critic=") for argument in arguments
    )
    if explicitly_enabled_critic:
        args.use_critic = True
        args.offload_train = True

    if args.critic_steps_per_actor is not None and args.critic_steps_per_actor <= 0:
        raise ValueError("--critic-steps-per-actor must be positive")
    if args.megatron_lora_alpha is None and args.megatron_lora_rank:
        args.megatron_lora_alpha = args.megatron_lora_rank
    validate_megatron_lora_args(args)

    previous_init = getattr(args, "custom_megatron_init_path", None)
    if previous_init != REEF_MEGATRON_INIT_PATH:
        args.reef_chained_megatron_init_path = previous_init
        args.custom_megatron_init_path = REEF_MEGATRON_INIT_PATH

    if args.megatron_lora_rank:
        previous_provider = getattr(args, "custom_model_provider_path", None)
        if previous_provider != REEF_MODEL_PROVIDER_PATH:
            args.reef_chained_model_provider_path = previous_provider
            args.custom_model_provider_path = REEF_MODEL_PROVIDER_PATH

    configure_reef_loss_args(args)


def configure_reef_loss_args(args: Any) -> None:
    """Project loss-family settings after the driver stamps ``loss_family``.

    Delegates to the family's spec: its declarative wire attributes land on
    ``args`` for the adapter layer to consume, then ``configure_backend_args``
    runs — so this module never names an individual family.
    """
    family = getattr(args, "loss_family", None)
    if not family:
        return
    spec = LOSS_FAMILIES.resolve(family)
    configured = tuple(getattr(args, "custom_rollout_data_keys", ()) or ())
    args.custom_rollout_data_keys = tuple(dict.fromkeys((*configured, *spec.rollout_data_keys)))
    args.reef_rollout_tensor_dtypes = dict(spec.rollout_tensor_dtypes)
    args.reef_external_batch_keys = tuple(spec.external_batch_keys)
    args.reef_rollout_log_skip_keys = tuple(spec.rollout_log_skip_keys)
    spec.configure_backend_args(args)
    if spec.uses_pg_loss_primitive:
        # Route Slime's numerical CISPO callsite onto the family's registered
        # pg primitive: the worker swaps loss.compute_cispo_loss for the
        # function behind custom_pg_loss_function_path (worker_hooks), so no
        # Slime source patch is needed. The routing value is adapter-owned;
        # families only declare the lane.
        args.advantage_estimator = "cispo"


__all__ = [
    "REEF_MEGATRON_INIT_PATH",
    "REEF_MODEL_PROVIDER_PATH",
    "add_reef_slime_arguments",
    "configure_reef_loss_args",
    "finalize_reef_slime_args",
]
