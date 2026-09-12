"""Validate process definitions and dependencies, then select their executors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import (
    ExecutorSelection,
    ExecutorSettings,
    executor_settings,
    in_ray_placement_group,
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
    return select_executor(
        settings,
        role="services",
        requires_resources=bool(resources),
        local_cuda=local_cuda,
        in_ray_placement_group=settings.backend == "auto" and not local_cuda and in_ray_placement_group(),
    )


def service_executor_config(
    config: Mapping[str, Any],
    service: Mapping[str, Any],
    run_dir: Path,
    timeout: int,
    config_path: Path,
    *,
    selection: ExecutorSelection | None = None,
    source_root: Path | None = None,
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
                kwargs={"source_root": source_root},
                options=resources,
            ),
        ),
    )


def _command_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(argument, str) for argument in value)
        and bool(value[0].strip())
    )


def validate_services(config: Mapping[str, Any], config_path: str | Path) -> list[dict[str, Any]]:
    """Services with valid commands and unique names, checked before any process starts."""
    services = config.get("services")
    if not isinstance(services, list) or not services:
        raise DeployConfigError(f"config {config_path} must declare a non-empty 'services' list")
    names: list[str] = []
    for index, service in enumerate(services):
        if not isinstance(service, dict):
            raise DeployConfigError(
                f"config {config_path}: services[{index}] must be an object, not {type(service).__name__}"
            )
        name = service.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DeployConfigError(f"config {config_path}: services[{index}] must have a non-empty 'name'")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise DeployConfigError(f"invalid service name {name!r}; use letters, digits, '_' or '-'")
        command = service.get("command")
        valid_string = isinstance(command, str) and bool(command.strip())
        valid_list = _command_list(command)
        if not valid_string and not valid_list:
            raise DeployConfigError(
                f"config {config_path}: services[{index}] must have a non-empty 'command' string or list of strings"
            )
        names.append(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DeployConfigError(
            f"config {config_path}: service names must be unique; duplicated: {', '.join(duplicates)}"
        )
    endpoints = config.get("endpoints", {})
    if not isinstance(endpoints, Mapping):
        raise DeployConfigError("endpoints must be an object")
    for service in services:
        dependencies = service.get("depends_on", [])
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) for dep in dependencies):
            raise DeployConfigError(f"service {service['name']!r}: depends_on must be a list of names")
        if any(dep not in names for dep in dependencies):
            raise DeployConfigError(f"service {service['name']!r}: unknown dependency")
        if "ready" in service and not isinstance(service["ready"], str) and not _command_list(service["ready"]):
            raise DeployConfigError(
                f"service {service['name']!r}: ready must be a string or non-empty list of strings"
            )
        for field in ("cwd", "endpoint", "advertise_host"):
            if field in service and not isinstance(service[field], str):
                raise DeployConfigError(f"service {service['name']!r}: {field} must be a string")
        if not isinstance(service.get("env", {}), Mapping):
            raise DeployConfigError(f"service {service['name']!r}: env must be an object")
        try:
            if float(service.get("ready_timeout", config.get("ready_timeout", 3600))) <= 0:
                raise ValueError("ready_timeout must be positive")
            service_executor_config(config, service, Path("."), 3600, Path(config_path))
        except (ValueError, TypeError, ImportError, AttributeError) as exc:
            raise DeployConfigError(f"service {service['name']!r}: {exc}") from exc

    # Stable dependency order, rather than relying on hand-ordered YAML lists.
    by_name = {service["name"]: service for service in services}
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise DeployConfigError(f"service dependency cycle at {name!r}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in by_name[name].get("depends_on", []):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for name in names:
        visit(name)
    return ordered
