"""Compatibility entrypoint for explicit Slime process stacks.

Managed deployments use reef.service.training_driver. Legacy argument files
and healthcheck commands remain available here; model lifecycle orchestration
is shared with the backend-neutral Reef driver.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.service.deploy.config_utils import load_config
from reef.service.training_driver import _driver_options as model_driver_options
from reef.service.training_driver import _required_environment, _resolve_training_recipe, run_deployment

READY_MARKER = "reef-slime-bridge-ready"
DEFAULT_READY_FILE = "/tmp/reef-slime-bridge.ready"


def _driver_options(arguments: Sequence[str]) -> tuple[Path, list[str]]:
    return model_driver_options(arguments, default_ready_file=DEFAULT_READY_FILE)


def _healthcheck(ready_file: Path) -> int:
    import ray

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
    ready_file.unlink(missing_ok=True)
    from reef.train.slime_backend.driver import create_model_plan

    config = load_config(_required_environment("REEF_CONFIG"))
    loss_family, _ = _resolve_training_recipe(config)
    return run_deployment(
        create_model_plan(config, direct_args, loss_family=loss_family), ready_file, marker=READY_MARKER
    )


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
