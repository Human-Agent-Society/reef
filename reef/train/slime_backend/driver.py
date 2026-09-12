"""Slime argument preparation and component definitions for Reef's model driver.

No resources are allocated while creating a plan. Framework-specific settings
and compatibility mode selection stay inside this integration.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.core.config import config_value
from reef.runtime.deployment import ModelDeploymentPlan
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.launch import driver_arguments
from reef.train.slime_backend.loss_families import UnknownLossFamilyError, resolve_loss_family
from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig


def load_args_file(path: str | Path) -> list[str]:
    """Read shell-like Slime arguments without executing a shell.

    Environment variables are expanded first, ``#`` comments are supported,
    and quoted values remain one token. Reef derives missing architecture
    flags from ``--hf-checkpoint`` before the runtime parses them, so an args
    file — when used — only carries deployment choices.
    """
    args_path = Path(path)
    try:
        contents = args_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read SLIME_ARGS_FILE {args_path}: {exc}") from exc
    try:
        return shlex.split(os.path.expandvars(contents), comments=True)
    except ValueError as exc:
        raise RuntimeError(f"cannot parse SLIME_ARGS_FILE {args_path}: {exc}") from exc


def _parse_slime_args(arguments: Sequence[str]):
    """Parse prepared file-first/direct-last arguments through Slime."""
    if not arguments:
        raise RuntimeError("no Slime arguments: set SLIME_ARGS_FILE or pass flags on the driver command")

    from reef.train.slime_backend.reef_adapters.megatron.hf_arguments import add_hf_architecture_arguments

    prepared_arguments = add_hf_architecture_arguments(arguments)
    original_argv = sys.argv
    sys.argv = [original_argv[0], *prepared_arguments]
    try:
        from slime.utils.arguments import parse_args

        from reef.train.slime_backend.reef_adapters.slime_arguments import (
            add_reef_slime_arguments,
            finalize_reef_slime_args,
        )

        args = parse_args(add_custom_arguments=add_reef_slime_arguments)
        finalize_reef_slime_args(args, prepared_arguments)
        return args
    finally:
        sys.argv = original_argv


def _configure_executors(args, config: Mapping[str, Any], arguments: Sequence[str]) -> None:
    from reef.runtime.executor.config import executor_settings, role_executor_settings, select_executor

    for role, attribute, flag in (
        ("training", "reef_executor_backend", "--reef-executor-backend"),
        ("rollout", "reef_rollout_executor_backend", "--reef-rollout-executor-backend"),
    ):
        explicit = any(value == flag or value.startswith(flag + "=") for value in arguments)
        settings = (
            executor_settings(config, getattr(args, attribute)) if explicit else role_executor_settings(config, role)
        )
        selection = select_executor(settings, role=role)
        settings = selection.settings
        logging.getLogger(__name__).info("%s: executor=%s (%s)", role, settings.backend, selection.reason)
        setattr(args, attribute, settings.backend)
        options_attribute = "reef_train_executor_options" if role == "training" else "reef_rollout_executor_options"
        setattr(args, options_attribute, dict(settings.options))


def _stamp_loss_family_reference(args, reference: str | None) -> None:
    """Carry a dotted family reference to the workers.

    ``apply_driver_options`` stamps the canonical name, which a fresh worker
    process cannot resolve for an external family: its registry starts empty.
    The reference is what it needs to import the module itself.
    """
    if not reference:
        return
    dotted = reference if ":" in reference else loss_family_refs().get(reference)
    if dotted is not None:
        args.loss_family_ref = dotted


def _apply_bridge_resume_fallback(args) -> None:
    """Mirror raw-mode first-start fallback for Megatron Bridge loading.

    Compose always points ``--load`` at the persistent save directory. On a
    first start that directory has no checkpoint tracker, and slime's parse
    normalizes such a ``--load`` to ``ref_load`` — ``None`` here — so an
    unset/empty ``load`` IS the first-start case, not a case to skip: left
    alone it fails the storage preflight and the actor's load assertion.
    Fall back to the HF checkpoint (slime's own supported bridge first-start
    source) while preserving ``--load`` once a resumable checkpoint exists.
    """
    if getattr(args, "megatron_to_hf_mode", None) != "bridge":
        return
    load = getattr(args, "load", None)
    if isinstance(load, str) and load.strip() and (Path(load) / "latest_checkpointed_iteration.txt").is_file():
        return
    fallback = getattr(args, "ref_load", None) or getattr(args, "hf_checkpoint", None)
    if not isinstance(fallback, str) or not fallback.strip():
        raise RuntimeError(
            "Megatron Bridge first start requires --ref-load or --hf-checkpoint when --load has no checkpoint"
        )
    args.load = fallback
    args.start_rollout_id = 0


def _retention_options(arguments: Sequence[str]) -> tuple[RetentionConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
    parser.add_argument("--reef-checkpoint-policy", dest="policy")
    parser.add_argument("--reef-checkpoint-max-storage-fraction", dest="max_storage_fraction", type=float)
    parser.add_argument("--reef-checkpoint-min-free-space-fraction", dest="min_free_space_fraction", type=float)
    parser.add_argument("--reef-checkpoint-max-storage", dest="max_storage_bytes", type=_size_bytes)
    parser.add_argument("--reef-checkpoint-min-free-space", dest="min_free_space_bytes", type=_size_bytes)
    options, remaining = parser.parse_known_args(arguments)
    return RetentionConfig(**vars(options)), remaining


def _validate_tracking_args(args: Any) -> None:
    if getattr(args, "wandb_key", None):
        raise RuntimeError("--wandb-key is not supported; use WANDB_API_KEY or the W&B credential store")
    if getattr(args, "use_wandb", False):
        raise RuntimeError(
            "--use-wandb in training.slime_flags is not supported; use Reef's observability.wandb config"
        )


def _size_bytes(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*(B|[KMGT]i?B)", value.strip(), re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError("size must include a unit such as GB or GiB")
    unit = match[2].upper()
    size = int(match[1]) * (1024 if "I" in unit else 1000) ** "BKMGT".index(unit[0])
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return size


def _job_runtime_env(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Ray job ``runtime_env`` carrying the driver's ``PYTHONPATH`` to its actors.

    The deploy layer appends the recipe source root to every *service*
    process's ``PYTHONPATH``, but Ray actors are forked from the raylet,
    whose environment predates the stack: with ``execution: training: ray``
    the shared local cluster starts inside ``reef serve``, before any
    service environment exists. Cookbook loss families
    (``recipes.<method>.slime:...``) resolve inside the Megatron workers, so
    without the driver's ``PYTHONPATH`` the first train actor dies with
    ``No module named 'recipes'``. Setting the job-level ``runtime_env``
    here covers every actor this driver creates — the bridge and the train
    workers — on local and external clusters alike; Slime's per-actor
    ``env_vars`` merge over the job-level mapping rather than replacing it.
    """
    source = os.environ if environ is None else environ
    pythonpath = source.get("PYTHONPATH", "").strip()
    if not pythonpath:
        return None
    return {"env_vars": {"PYTHONPATH": pythonpath}}


def create_model_plan(
    config: Mapping[str, Any], direct_args: Sequence[str] = (), *, loss_family: str
) -> ModelDeploymentPlan:
    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is required")
    args_file = os.environ.get("SLIME_ARGS_FILE", "").strip() or None
    namespace = os.environ.get("REEF_RAY_NAMESPACE", DEFAULT_NAMESPACE)
    actor_name = os.environ.get("REEF_RAY_ACTOR_NAME", DEFAULT_ACTOR_NAME)
    if config.get("reef", {}).get("inference_num_gpus") is not None and (args_file or direct_args):
        raise RuntimeError("managed drivers read resolved configuration; pass options through reef serve")
    recipe = config_value(config, "reef", "recipe", expand=False)
    if not loss_family:
        raise RuntimeError(f"reef.recipe {recipe!r} declares no training loss family")
    try:
        spec = resolve_loss_family(loss_family)
    except UnknownLossFamilyError as exc:
        raise RuntimeError(f"reef.recipe {recipe!r} declares unsupported loss family {loss_family!r}: {exc}") from exc
    combined_args = [
        *driver_arguments(config),
        *(load_args_file(args_file) if args_file else []),
        *direct_args,
    ]
    retention, remaining_args = _retention_options(combined_args)
    loss_family_config, slime_args = spec.parse_driver_options(remaining_args)
    args = _parse_slime_args(slime_args)
    _configure_executors(args, config, slime_args)
    _validate_tracking_args(args)
    _apply_bridge_resume_fallback(args)
    spec.apply_driver_options(args, loss_family_config)
    _stamp_loss_family_reference(args, loss_family)
    from reef.train.slime_backend.reef_adapters.bridge import prepare_bridge
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args
    from reef.train.slime_backend.resources import SlimeDeploymentResources, SlimeInferenceService
    from reef.train.slime_backend.training import SlimeTrainingService

    configure_reef_loss_args(args)
    spec.validate_backend_args(args, recipe=recipe)
    prepared = prepare_bridge(args, retention=retention, loss_family=loss_family)
    separate = (
        not prepared.lora and not getattr(args, "colocate", False) and not getattr(args, "rollout_external", False)
    )
    return ModelDeploymentPlan(
        resources=SlimeDeploymentResources(
            args,
            ray_address=ray_address,
            namespace=namespace,
            runtime_env=_job_runtime_env(),
            allocate_models=separate,
        ),
        inference=SlimeInferenceService(args) if separate else None,
        training=SlimeTrainingService(
            args,
            preparation=prepared,
            loss_family_config=loss_family_config,
            actor_name=actor_name,
            namespace=namespace,
            separate_inference=separate,
        ),
    )
