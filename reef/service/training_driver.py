"""Reef-owned startup, supervision and shutdown of model components.

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
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.core.config import config_value
from reef.inference.deployment import inference_service_for
from reef.recipe import RecipeConfigError
from reef.recipe.registry import recipe_class_for
from reef.runtime.deployment import ComponentHealth
from reef.runtime.deployment import ModelDeployment as ModelDeployment
from reef.runtime.deployment import ModelDeploymentPlan, ModelPlanSource
from reef.service.deploy.config_utils import load_config
from reef.service.deploy.deployment_config import selected_weight_training
from reef.service.deploy.training import training_deployment_for
from reef.train.deployment import TrainingDeploymentPlan

READY_MARKER = "reef-training-ready"
DEFAULT_READY_FILE = "/tmp/reef-training.ready"
_logger = logging.getLogger(__name__)


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


class ConfiguredModelPlanSource(ModelPlanSource):
    """Reparse the resolved configuration and inspect current checkpoints."""

    def __init__(self, config: Mapping[str, Any], loss_family: str) -> None:
        self.config = config
        self.loss_family = loss_family

    def create(self) -> ModelDeploymentPlan:
        backend = training_deployment_for(self.config.get("reef", {}).get("training_backend"))
        training = backend.create_training_plan(self.config, loss_family=self.loss_family)
        return assemble_model_plan(self.config, training)


def assemble_model_plan(config: Mapping[str, Any], training: TrainingDeploymentPlan) -> ModelDeploymentPlan:
    """Select inference independently and validate the pair before allocation."""
    inference = inference_service_for(config.get("reef", {}).get("inference_backend"), training.inference_config)
    plan = ModelDeploymentPlan(
        resources=training.resources,
        training=training.training,
        inference=inference,
        coordinator=training.coordinator,
        health=ComponentHealth(inference, training.training) if training.monitor_components else None,
    )
    plan.validate()
    return plan


def supervise_deployment(
    deployment: ModelDeployment,
    source: ModelPlanSource,
    ready_file: Path,
    stopping: threading.Event,
    *,
    marker: str = READY_MARKER,
) -> ModelDeployment:
    """Cold-rebuild failed components, with at most three restarts per five minutes.

    Cleanup and recovery preflight must succeed before replacing any component.
    In particular, an ambiguous optimizer step is never retried by supervision.
    """
    restarts: deque[float] = deque()
    try:
        while not stopping.wait(1):
            if deployment.plan.health is None:
                continue
            try:
                deployment.poll()
            except Exception as failure:
                ready_file.unlink(missing_ok=True)
                _logger.exception("Model component failed; retiring deployment before recovery")
                deployment.close()
                now = time.monotonic()
                while restarts and now - restarts[0] >= 300:
                    restarts.popleft()
                if len(restarts) >= 3:
                    raise RuntimeError("model deployment exceeded three restarts in five minutes") from failure
                restarts.append(now)
                if stopping.wait(2 ** (len(restarts) - 1)):
                    break
                # Reusing the old plan would reuse its initial checkpoint load
                # arguments and miss training jobs committed since startup.
                deployment = ModelDeployment(source.create())
                deployment.start()
                if not stopping.is_set():
                    _write_ready_file(ready_file, marker)
                    _logger.info("Model deployment recovered and ready")
        return deployment
    except BaseException:
        try:
            deployment.close()
        except Exception:
            _logger.exception("Failed to close replacement deployment")
        raise


def run_deployment(
    plan: ModelDeploymentPlan,
    ready_file: Path,
    *,
    marker: str = READY_MARKER,
    source: ModelPlanSource | None = None,
) -> int:
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
            if source is not None and plan.health is not None:
                deployment = supervise_deployment(deployment, source, ready_file, stopping, marker=marker)
            else:
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
    weight_type = None if recipe_class is None else selected_weight_training(recipe_class, config)
    if weight_type is None:
        raise RuntimeError(
            f"model driver requires reef.recipe to name a WeightTrainingRecipe class or a recipe with a "
            f"weight training component, got {recipe!r}"
        )
    loss_family = weight_type.training_spec().loss_family.strip()
    return loss_family, recipe


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    ready_file, remaining = _driver_options(list(sys.argv[1:] if argv is None else argv))
    # Remove stale readiness even if backend selection or preflight fails.
    ready_file.unlink(missing_ok=True)
    if remaining:
        raise ValueError("managed drivers read resolved configuration; pass options through reef serve")
    config = load_config(_required_environment("REEF_CONFIG"))
    loss_family, _ = _resolve_training_recipe(config)
    source = ConfiguredModelPlanSource(config, loss_family)
    return run_deployment(source.create(), ready_file, source=source)


if __name__ == "__main__":
    raise SystemExit(main())
