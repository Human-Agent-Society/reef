"""Run the long-lived Slime-to-Reef training bridge driver.

Environment contract:

===============================  =============================================
Variable                         Meaning
===============================  =============================================
``RAY_ADDRESS``                  Required. Ray cluster address to join.
``SLIME_ARGS_FILE``              Optional. Shell-like file of Slime flags,
                                 parsed without a shell; direct command-line
                                 flags are appended after it.
``REEF_RAY_NAMESPACE``           Ray namespace of the bridge actor
                                 (default ``reef``).
``REEF_RAY_ACTOR_NAME``          Name the bridge actor is published under
                                 (default ``reef-train-bridge``).
``REEF_BRIDGE_READY_FILE``       Path of the readiness marker file
                                 (default ``/tmp/reef-slime-bridge.ready``;
                                 ``reef serve`` sets it under the stack's
                                 ``run_dir``); ``--ready-file`` overrides it.
``REEF_BRIDGE_HEALTH_TIMEOUT_S``  Healthcheck RPC timeout in seconds
                                 (default ``10``).
``REEF_TRAINING_LOSS``           Required for training. A registered family
                                 name or dotted ``package.module:SPEC``.
``REEF_TRAINING_RECIPE``         Optional recipe label for logs and errors;
                                 never used to infer the loss family.
===============================  =============================================
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ray

from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import UnknownLossFamilyError, resolve_loss_family
from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig

READY_MARKER = "reef-slime-bridge-ready"
DEFAULT_READY_FILE = "/tmp/reef-slime-bridge.ready"


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


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


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


def _resolve_loss_family(environ) -> tuple[str | None, str | None, SlimeAlgorithm | None]:
    """Resolve the training loss family and its spec from the driver environment.

    ``REEF_TRAINING_LOSS`` names the family directly. Cookbook methods should
    use a dotted reference so the driver and fresh workers can import their
    implementation without Reef importing any method at boot.
    ``REEF_TRAINING_RECIPE`` is only an optional label; it cannot select a
    loss family.
    Returns ``(loss_family, recipe, spec)``, resolved through the loss-family
    registry exactly once, or ``(None, None, None)`` when neither variable is
    set. The registry is the vocabulary authority: an unknown family fails in
    ``resolve_loss_family``.
    """
    loss = environ.get("REEF_TRAINING_LOSS", "").strip()
    recipe = environ.get("REEF_TRAINING_RECIPE", "").strip() or None
    if loss:
        try:
            return loss, recipe, resolve_loss_family(loss)
        except UnknownLossFamilyError as exc:
            raise RuntimeError(f"unsupported REEF_TRAINING_LOSS: {loss!r} ({exc})") from exc
    if recipe:
        raise RuntimeError("REEF_TRAINING_LOSS is required when REEF_TRAINING_RECIPE is set")
    return None, None, None


def _stamp_loss_family_reference(args, reference: str | None) -> None:
    """Carry a dotted family reference to the workers.

    ``apply_driver_options`` stamps the canonical name, which a fresh worker
    process cannot resolve for an external family: its registry starts empty.
    The reference is what it needs to import the module itself.
    """
    if reference and ":" in reference:
        args.loss_family_ref = reference


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


def _driver_options(arguments: Sequence[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--ready-file",
        default=os.environ.get("REEF_BRIDGE_READY_FILE", DEFAULT_READY_FILE),
    )
    options, remaining = parser.parse_known_args(arguments)
    if not options.ready_file:
        raise RuntimeError("--ready-file must be non-empty")
    return Path(options.ready_file), remaining


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


def _write_ready_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(f"{READY_MARKER}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _healthcheck(ready_file: Path) -> int:
    try:
        ray_address = _required_environment("RAY_ADDRESS")
        namespace = os.environ.get("REEF_RAY_NAMESPACE", DEFAULT_NAMESPACE)
        actor_name = os.environ.get("REEF_RAY_ACTOR_NAME", DEFAULT_ACTOR_NAME)
        timeout_s = float(os.environ.get("REEF_BRIDGE_HEALTH_TIMEOUT_S", "10"))
    except ValueError as exc:
        print(f"bridge healthcheck failed: REEF_BRIDGE_HEALTH_TIMEOUT_S must be numeric ({exc})", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"bridge healthcheck failed: {exc}", file=sys.stderr)
        return 1
    if timeout_s <= 0:
        print("bridge healthcheck failed: REEF_BRIDGE_HEALTH_TIMEOUT_S must be positive", file=sys.stderr)
        return 1

    try:
        if ready_file.read_text(encoding="utf-8").strip() != READY_MARKER:
            raise RuntimeError(f"bridge ready marker is missing or invalid: {ready_file}")
        ray.init(address=ray_address, namespace=namespace, logging_level=logging.ERROR)
        bridge = ray.get_actor(actor_name, namespace=namespace)
        result: Any = ray.get(bridge.health.remote(), timeout=timeout_s)
        return 0 if isinstance(result, dict) and result.get("ok") is True else 1
    except Exception as exc:
        print(f"bridge healthcheck failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()


def _serve(direct_args: Sequence[str], ready_file: Path) -> int:
    # Remove a stale marker before parsing or connecting. This also makes a
    # configuration error fail closed instead of advertising the previous job.
    ready_file.unlink(missing_ok=True)
    ray_address = _required_environment("RAY_ADDRESS")
    args_file = os.environ.get("SLIME_ARGS_FILE", "").strip() or None
    namespace = os.environ.get("REEF_RAY_NAMESPACE", DEFAULT_NAMESPACE)
    actor_name = os.environ.get("REEF_RAY_ACTOR_NAME", DEFAULT_ACTOR_NAME)
    loss_family, recipe, spec = _resolve_loss_family(os.environ)
    combined_args = [*(load_args_file(args_file) if args_file else []), *direct_args]
    retention, remaining_args = _retention_options(combined_args)
    loss_family_config, slime_args = (
        spec.parse_driver_options(remaining_args) if spec is not None else (None, remaining_args)
    )
    args = _parse_slime_args(slime_args)
    _validate_tracking_args(args)
    _apply_bridge_resume_fallback(args)
    if spec is not None:
        # Family-specific flags stripped by ``parse_specific_options`` are
        # projected back onto args so Slime's workers can observe them.
        spec.apply_driver_options(args, loss_family_config)
        _stamp_loss_family_reference(args, loss_family)
        from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

        configure_reef_loss_args(args)
        spec.validate_backend_args(args, recipe=recipe)

    stopping = threading.Event()

    def request_stop(signum, frame) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    bridge = None
    try:
        ray.init(address=ray_address, namespace=namespace)
        # Importing the bridge initializes the Slime serving/training modules;
        # keep that work out of the lightweight healthcheck path.
        from reef.train.slime_backend.reef_adapters.bridge import start_bridge

        bridge = start_bridge(
            args,
            retention=retention,
            loss_family=loss_family,
            loss_family_config=loss_family_config,
            actor_name=actor_name,
            namespace=namespace,
        )
        health = ray.get(bridge.health.remote())
        if not isinstance(health, dict) or health.get("ok") is not True:
            raise RuntimeError(f"bridge actor failed its startup health check: {health!r}")
        _write_ready_file(ready_file)
        print(READY_MARKER, flush=True)
        stopping.wait()
        return 0
    finally:
        ready_file.unlink(missing_ok=True)
        if bridge is not None and ray.is_initialized():
            try:
                ray.kill(bridge, no_restart=True)
            except Exception as exc:
                print(f"failed to stop bridge actor cleanly: {exc}", file=sys.stderr)
        if ray.is_initialized():
            ray.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "healthcheck":
        ready_file, remaining = _driver_options(arguments[1:])
        if remaining:
            raise RuntimeError(f"healthcheck does not recognize arguments: {' '.join(remaining)}")
        return _healthcheck(ready_file)
    if arguments and arguments[0] == "serve":
        arguments = arguments[1:]
    ready_file, direct_args = _driver_options(arguments)
    return _serve(direct_args, ready_file)


if __name__ == "__main__":
    raise SystemExit(main())
