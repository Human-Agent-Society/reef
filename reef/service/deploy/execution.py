"""Build service executors without embedding a scheduler in the orchestrator."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import (
    ExecutorSelection,
    ExecutorSettings,
    executor_settings,
    role_executor_settings,
    select_executor,
)
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.service.deploy.process import ProcessWorker, RayProcessWorker


def _service_resources(settings: ExecutorSettings, service: Mapping[str, Any]) -> dict[str, Any]:
    resources = service.get("resources", {})
    if not isinstance(resources, Mapping):
        raise ValueError("service.resources must be an object of worker launch options")
    result = dict(resources)
    for name, value in (
        ("num_cpus", settings.resources.cpus_per_worker),
        ("num_gpus", settings.resources.gpus_per_worker),
    ):
        if value is None:
            continue
        if (name in result and result[name] != value) or (
            name in settings.options and settings.options[name] != value
        ):
            raise ValueError(f"service {name} conflicts with executor resources")
        result[name] = value
    return result


def service_executor_selection(config: Mapping[str, Any], service: Mapping[str, Any]) -> ExecutorSelection:
    settings = (
        executor_settings(config, service["executor"])
        if "executor" in service
        else role_executor_settings(config, "services")
    )
    resources = _service_resources(settings, service)
    local_cuda = service.get("cuda") is not None or "CUDA_VISIBLE_DEVICES" in (service.get("env") or {})
    in_placement_group = False
    # Importing Reef must not import/init Ray. An existing placement context,
    # unlike an installed package or RAY_ADDRESS alone, implies scheduling intent.
    ray = sys.modules.get("ray")
    is_initialized = getattr(ray, "is_initialized", None)
    if settings.backend == "auto" and not local_cuda and callable(is_initialized) and is_initialized():
        from ray.util import get_current_placement_group

        in_placement_group = get_current_placement_group() is not None
    return select_executor(
        settings,
        role="services",
        requires_resources=bool(resources),
        local_cuda=local_cuda,
        in_ray_placement_group=in_placement_group,
    )


def service_executor_config(
    config: Mapping[str, Any],
    service: Mapping[str, Any],
    run_dir: Path,
    timeout: int,
    config_path: Path,
    *,
    selection: ExecutorSelection | None = None,
) -> ExecutorConfig:
    selected = selection or service_executor_selection(config, service)
    settings = selected.settings
    backend = Executor.get_class(settings.backend)
    resources = _service_resources(settings, service)
    if issubclass(backend, UniProcExecutor) and (settings.options or resources):
        raise ValueError("local service execution uses cuda for visibility; resource reservations require Ray/custom")
    if issubclass(backend, RayExecutor):
        if service.get("cuda") is not None or "CUDA_VISIBLE_DEVICES" in (service.get("env") or {}):
            raise ValueError("Ray services must use resources.num_gpus, not cuda/CUDA_VISIBLE_DEVICES")
        options = {**settings.options, **resources}
        env_vars = options.get("runtime_env", {}).get("env_vars", {})
        if "CUDA_VISIBLE_DEVICES" in env_vars:
            raise ValueError("Ray service runtime_env must not override CUDA_VISIBLE_DEVICES")
        if options.get("max_restarts", 0) != 0 or options.get("max_task_retries", 0) != 0:
            raise ValueError("service executors must not replay process launches")
    worker_class = RayProcessWorker if issubclass(backend, RayExecutor) else ProcessWorker
    return ExecutorConfig(
        backend=backend,
        options=settings.options,
        launch_timeout_s=float(service.get("ready_timeout", timeout)),
        workers=(
            WorkerSpec(
                worker_class,
                args=(dict(config), [dict(service)], run_dir, timeout, config_path),
                options=resources,
            ),
        ),
    )
