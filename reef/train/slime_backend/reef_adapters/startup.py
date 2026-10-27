"""Build the Slime arguments that the training component's workers start with.

``prepare_slime_arguments`` runs before any placement group or GPU worker
exists. It separates Reef's checkpoint retention options, parses the rest
through Slime, applies the startup loss family and Reef's adjustments, and
refuses arguments the component cannot run with. ``job_runtime_env`` is the
Ray environment of the workers.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.loss_families import UnknownLossFamilyError, resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class
from reef.train.slime_backend.reef_adapters.preflight import (
    configure_critic_args_hook,
    configure_rollout_runtime,
    validate_slime_args,
)
from reef.train.slime_backend.reef_adapters.training_job.storage import (
    MEGATRON_TRACKER,
    RetentionConfig,
    retention_options,
)

logger = logging.getLogger(__name__)


def prepare_slime_arguments(
    arguments: Sequence[str], *, loss_family: str, recipe: str, execution: Mapping[str, Any]
) -> tuple[SlimeArguments, object | None, RetentionConfig]:
    """Return the workers' Slime arguments, the startup loss family's driver config and the checkpoint retention.

    ``arguments`` are the native Slime arguments with Reef's retention
    options. ``loss_family`` is the recipe's startup loss family, by name or
    dotted reference; ``recipe`` names the recipe in error messages.
    ``execution`` holds the deployment's ``execution`` and ``executors``
    sections. Raise ``RuntimeError`` for an unknown loss family, before Slime
    parses anything, and ``ValueError`` or ``RuntimeError`` for arguments
    the component cannot run with.
    """
    try:
        spec = resolve_loss_family(loss_family)
    except UnknownLossFamilyError as exc:
        raise RuntimeError(f"recipe {recipe!r} declares unsupported loss family {loss_family!r}: {exc}") from exc
    retention, remaining = retention_options(arguments)
    loss_family_config, slime_arguments = spec.parse_driver_options(remaining)
    args = parse_slime_args(slime_arguments)
    configure_executors(args, execution, slime_arguments)
    validate_tracking_args(args)
    use_megatron_bridge_first_start_load(args)
    apply_unsaved_optimizer_resume(args)
    spec.apply_driver_options(args, loss_family_config)
    stamp_loss_family_reference(args, loss_family)
    # Imported here: the argument module reaches the Megatron LoRA stack (test_dependency_boundaries).
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    configure_reef_loss_args(args)
    spec.validate_backend_args(args, recipe=recipe)
    validate_slime_args(args, spec)
    configure_critic_args_hook(args)
    configure_rollout_runtime(args)
    # Resolve the executor before any worker starts.
    slime_executor_class(args.reef_executor_backend)
    return args, loss_family_config, retention


def parse_slime_args(arguments: Sequence[str]) -> SlimeArguments:
    """Parse prepared arguments through Slime, with Reef's options and hooks."""
    if not arguments:
        raise RuntimeError("no Slime arguments: configure training options in the deployment config")

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

        args = SlimeArguments(**vars(parse_args(add_custom_arguments=add_reef_slime_arguments)))
        finalize_reef_slime_args(args, prepared_arguments)
        return args
    finally:
        sys.argv = original_argv


def configure_executors(args: SlimeArguments, execution: Mapping[str, Any], arguments: Sequence[str]) -> None:
    """Select the training executor: an explicit Slime flag, else the deployment's role setting."""
    from reef.runtime.executor.config import (
        ExecutorSettings,
        executor_settings,
        role_executor_settings,
        select_executor,
    )

    def flag_given(flag: str) -> bool:
        return any(value == flag or value.startswith(flag + "=") for value in arguments)

    def selected(settings: ExecutorSettings, role: str) -> ExecutorSettings:
        selection = select_executor(settings, role=role)
        logger.info("%s: executor=%s (%s)", role, selection.settings.backend, selection.reason)
        return selection.settings

    if flag_given("--reef-executor-backend"):
        training = selected(executor_settings(execution, args.reef_executor_backend), "training")
    else:
        training = selected(role_executor_settings(execution, "training"), "training")
    args.reef_executor_backend = training.backend
    args.reef_train_executor_options = dict(training.options)


def validate_tracking_args(args: SlimeArguments) -> None:
    """Refuse Slime's own W&B options; Reef's ``observability.wandb`` section configures tracking."""
    if args.wandb_key:
        raise RuntimeError("--wandb-key is not supported; use WANDB_API_KEY or the W&B credential store")
    if args.use_wandb:
        raise RuntimeError(
            "--use-wandb in training.slime_flags is not supported; use Reef's observability.wandb config"
        )


def use_megatron_bridge_first_start_load(args: SlimeArguments) -> None:
    """Load the Hugging Face checkpoint on a Megatron Bridge first start, and the save directory after it.

    ``--load`` always names the persistent save directory. On a first start it
    has no checkpoint tracker, and Slime's parse normalizes such a ``--load`` to
    ``ref_load``, ``None`` here; the HF checkpoint is then the source.
    """
    if args.megatron_to_hf_mode != "bridge":
        return
    load = args.load
    if isinstance(load, str) and load.strip() and (Path(load) / MEGATRON_TRACKER).is_file():
        return
    first_start_load = args.ref_load or args.hf_checkpoint
    if not isinstance(first_start_load, str) or not first_start_load.strip():
        raise RuntimeError(
            "Megatron Bridge first start requires --ref-load or --hf-checkpoint when --load has no checkpoint"
        )
    args.load = first_start_load
    args.start_rollout_id = 0


def apply_unsaved_optimizer_resume(args: SlimeArguments) -> None:
    """Resume a run that never saved its optimizer (``--no-save-optim``) from its weights alone."""
    if not args.no_save_optim:
        return
    load, save = args.load, args.save
    if not (isinstance(load, str) and load.strip() and isinstance(save, str) and save.strip()):
        return
    if Path(load).resolve() != Path(save).resolve():
        return
    if (Path(load) / MEGATRON_TRACKER).is_file():
        args.no_load_optim = True


def stamp_loss_family_reference(args: SlimeArguments, reference: str) -> None:
    """Carry a dotted family reference to the workers, whose registry starts empty."""
    dotted = reference if ":" in reference else loss_family_refs().get(reference)
    if dotted is not None:
        args.loss_family_ref = dotted


def job_runtime_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Ray job ``runtime_env`` for the training workers: the CUDA/NCCL defaults and this process's ``PYTHONPATH``.

    Ray actors are forked from the raylet, whose environment predates the
    service. The workers get ``CUDA_DEVICE_MAX_CONNECTIONS=1`` and
    ``NCCL_NVLS_ENABLE=0`` unless this process sets them, and the service's
    ``PYTHONPATH``: cookbook loss families resolve inside the workers, so
    without it the first actor fails to import ``recipes``.
    """
    source = os.environ if environ is None else environ
    env_vars = {
        "CUDA_DEVICE_MAX_CONNECTIONS": source.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        "NCCL_NVLS_ENABLE": source.get("NCCL_NVLS_ENABLE", "0"),
    }
    pythonpath = source.get("PYTHONPATH", "").strip()
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath
    return {"env_vars": env_vars}


__all__ = ["job_runtime_env", "prepare_slime_arguments"]
