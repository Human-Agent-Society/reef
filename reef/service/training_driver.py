"""Reef-owned startup and shutdown of inference and training components.

Backend definitions provide configured components, without allocating them.
This entrypoint validates compatibility, starts shared resources, inference
and training, and releases attempted components in reverse dependency order.
No concrete model framework is imported by this module.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.core.config import config_value
from reef.recipe import RecipeConfigError, WeightTrainingRecipe
from reef.recipe.registry import recipe_class_for
from reef.runtime.deployment import ModelDeploymentPlan
from reef.service.deploy.config_utils import load_config
from reef.service.deploy.training import training_deployment_for

READY_MARKER = "reef-training-ready"
DEFAULT_READY_FILE = "/tmp/reef-training.ready"
_logger = logging.getLogger(__name__)


class ModelDeployment:
    """Own the lifecycle of a validated plan, including partial-start cleanup."""

    def __init__(self, plan: ModelDeploymentPlan) -> None:
        self.plan = plan
        self._started = False
        self._closed = False
        self._resources_started = False
        self._inference_started = False
        self._training_started = False

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("model deployment can only be started once")
        self.plan.validate()
        self._started = True
        try:
            self._resources_started = True
            self.plan.resources.start()
            connection = None
            if self.plan.inference is not None:
                self._inference_started = True
                connection = self.plan.inference.start(self.plan.resources)
                if connection.protocol != self.plan.training.inference_protocol:
                    raise ValueError("inference returned a connection with an incompatible protocol")
                self.plan.inference.check_health()
            else:
                _logger.info("Selected backend uses its combined inference/training compatibility lifecycle")
            self._training_started = True
            self.plan.training.start(self.plan.resources, connection)
            self.plan.training.check_health()
            # Training initialization may modify engine state. Do not advertise
            # deployment readiness based only on the pre-training engine probe.
            if self.plan.inference is not None:
                self.plan.inference.check_health()
        except BaseException:
            try:
                self.close()
            except Exception:
                _logger.exception("Failed to clean up model deployment after startup failure")
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        for started, component in (
            (self._training_started, self.plan.training),
            (self._inference_started, self.plan.inference),
            (self._resources_started, self.plan.resources),
        ):
            if started and component is not None:
                try:
                    component.close()
                except Exception as exc:
                    errors.append(exc)
                    _logger.exception("Failed to close deployment component %s", type(component).__name__)
        if errors:
            raise errors[0]


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _driver_options(
    arguments: Sequence[str], *, default_ready_file: str = DEFAULT_READY_FILE
) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--ready-file", default=os.environ.get("REEF_BRIDGE_READY_FILE", default_ready_file))
    options, remaining = parser.parse_known_args(arguments)
    if not options.ready_file:
        raise RuntimeError("--ready-file must be non-empty")
    return Path(options.ready_file), remaining


def _write_ready_file(path: Path, marker: str = READY_MARKER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(f"{marker}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_deployment(plan: ModelDeploymentPlan, ready_file: Path, *, marker: str = READY_MARKER) -> int:
    ready_file.unlink(missing_ok=True)
    stopping = threading.Event()

    def request_stop(signum, frame) -> None:
        stopping.set()

    previous = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    deployment = ModelDeployment(plan)
    try:
        deployment.start()
        if not stopping.is_set():
            _write_ready_file(ready_file, marker)
            print(marker, flush=True)
            stopping.wait()
        return 0
    except BaseException:
        try:
            deployment.close()
        except Exception:
            _logger.exception("Failed to close deployment while handling a driver error")
        raise
    finally:
        ready_file.unlink(missing_ok=True)
        try:
            deployment.close()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def _resolve_training_recipe(config: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve the recipe and its loss family from ``reef.recipe``.

    The deployment already has one authoritative recipe reference. Importing
    that selected class is the extension boundary; its static
    :meth:`WeightTrainingRecipe.training_spec` supplies the loss family without
    a second environment setting.
    """
    recipe = config_value(config, "reef", "recipe", expand=False)
    if not isinstance(recipe, str) or not recipe:
        raise RuntimeError("REEF_CONFIG must define reef.recipe")
    try:
        recipe_class = recipe_class_for(recipe)
    except RecipeConfigError as exc:
        raise RuntimeError(f"cannot load reef.recipe {recipe!r}: {exc}") from exc
    if recipe_class is None or not issubclass(recipe_class, WeightTrainingRecipe):
        raise RuntimeError(f"model driver requires reef.recipe to name a WeightTrainingRecipe class, got {recipe!r}")
    loss_family = recipe_class.training_spec().loss_family.strip()
    return loss_family, recipe


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    ready_file, remaining = _driver_options(list(sys.argv[1:] if argv is None else argv))
    # Remove stale readiness even if backend selection or preflight fails.
    ready_file.unlink(missing_ok=True)
    if remaining:
        raise ValueError("managed drivers read resolved configuration; pass options through reef serve")
    config = load_config(_required_environment("REEF_CONFIG"))
    backend = training_deployment_for(config.get("reef", {}).get("training_backend"))
    loss_family, _ = _resolve_training_recipe(config)
    plan = backend.create_model_plan(config, loss_family=loss_family)
    return run_deployment(plan, ready_file)


if __name__ == "__main__":
    raise SystemExit(main())
