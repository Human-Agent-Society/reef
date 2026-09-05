"""Shared YAML executor selectors for services, training and rollout roles."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.runtime.executor.requirements import ExecutionRequirements


@dataclass(frozen=True)
class WorkerResources:
    """Optional per-worker requests; None preserves component defaults."""

    cpus_per_worker: float | None = None
    gpus_per_worker: float | None = None

    def __post_init__(self) -> None:
        ExecutionRequirements(
            cpus_per_worker=1 if self.cpus_per_worker is None else self.cpus_per_worker,
            gpus_per_worker=0 if self.gpus_per_worker is None else self.gpus_per_worker,
        )


@dataclass(frozen=True)
class ExecutorSettings:
    backend: str = "auto"
    options: Mapping[str, Any] = field(default_factory=dict)
    workers: int | None = None
    resources: WorkerResources = field(default_factory=WorkerResources)

    def __post_init__(self) -> None:
        if self.workers is not None:
            ExecutionRequirements(workers=self.workers)
        if not isinstance(self.resources, WorkerResources):
            raise ValueError("executor resources must be WorkerResources")


def worker_requirements(settings: ExecutorSettings, declared: ExecutionRequirements) -> ExecutionRequirements:
    """Merge generic placement configuration with a component's defaults."""
    resources = settings.resources
    cpus = resources.cpus_per_worker
    gpus = resources.gpus_per_worker
    result = replace(
        declared,
        workers=declared.workers if settings.workers is None else settings.workers,
        cpus_per_worker=declared.cpus_per_worker if cpus is None else cpus,
        gpus_per_worker=declared.gpus_per_worker if gpus is None else gpus,
    )
    if result.gpus_per_worker < declared.gpus_per_worker:
        raise ValueError("worker GPU allocation is below the component's declared requirement")
    return result


@dataclass(frozen=True)
class ExecutorSelection:
    """A concrete backend decision, before the Executor factory imports it."""

    settings: ExecutorSettings
    reason: str


def select_executor(
    settings: ExecutorSettings,
    *,
    role: str,
    requires_resources: bool = False,
    local_cuda: bool = False,
    in_ray_placement_group: bool = False,
    requirements: ExecutionRequirements | None = None,
) -> ExecutorSelection:
    """Explicit selection wins; auto uses the capabilities of the current role.

    This never starts a Ray runtime. Only declared local GPU worker needs
    probe CUDA capacity; CPU paths do not import Torch.
    Training/rollout are Slime roles, whose built-in distributed backend is
    currently Ray. A single GPU does not make UniProcExecutor a Slime launcher.
    """
    if role not in ("services", "training", "rollout", "evolution"):
        raise ValueError(f"unknown executor role: {role!r}")
    if role == "evolution":
        return select_worker_executor(settings, requirements or ExecutionRequirements())
    if role in ("training", "rollout") and (settings.workers is not None or settings.resources != WorkerResources()):
        raise ValueError(
            f"execution.{role}.workers/resources are not supported by Slime; use its model topology configuration"
        )
    if role == "services" and settings.workers not in (None, 1):
        raise ValueError("execution.services.workers must be 1; service replication is not supported")
    requires_resources = requires_resources or settings.resources != WorkerResources()
    if settings.backend != "auto":
        return ExecutorSelection(settings, "explicit backend/profile selection")
    if role in ("training", "rollout"):
        backend, reason = "ray", "Slime model workers require the built-in Ray launcher"
    elif local_cuda and (requires_resources or settings.options):
        raise ValueError("auto cannot combine local CUDA visibility with cluster resource options")
    elif local_cuda:
        backend, reason = "uni", "one service controller with local CUDA visibility"
    elif requires_resources or settings.options:
        backend, reason = "ray", "service requests cluster resource/worker options"
    elif in_ray_placement_group:
        backend, reason = "ray", "service is running inside a Ray placement group"
    else:
        backend, reason = "uni", "one service controller with no cluster placement requirements"
    return ExecutorSelection(replace(settings, backend=backend), reason)


def select_worker_executor(settings: ExecutorSettings, requirements: ExecutionRequirements) -> ExecutorSelection:
    """Prefer uni/mp for local topology; Ray placement is explicit/contextual.

    CPU-only components never probe CUDA. GPU capacity is checked only for a
    declared local GPU workload, rather than silently switching it to Ray.
    """
    requirements = worker_requirements(settings, requirements)
    backend = settings.backend
    reason = "explicit backend/profile selection"
    if backend == "auto":
        if requirements.cluster or settings.options:
            backend, reason = "ray", "component requests cluster resources or worker options"
        elif requirements.workers > 1 and in_ray_placement_group():
            backend, reason = "ray", "multiple workers inside a Ray placement group"
        elif requirements.workers == 1:
            backend, reason = "uni", "one local worker"
        else:
            backend, reason = "mp", "multiple workers on one host"
    if backend in ("uni", "mp", "ray") and backend not in requirements.supported_backends:
        raise ValueError(f"component does not support executor {backend!r}")
    if backend in ("uni", "mp") and (requirements.cluster or settings.options):
        raise ValueError(f"executor {backend!r} cannot reserve cluster resources or accept worker options")
    if backend == "uni" and requirements.workers != 1:
        raise ValueError("uni requires exactly one worker; reduce execution workers or select mp")
    if backend in ("uni", "mp") and requirements.gpus_per_worker:
        local_gpu_assignments(requirements)
    return ExecutorSelection(replace(settings, backend=backend), reason)


def in_ray_placement_group() -> bool:
    """Inspect an existing runtime without importing or initializing Ray."""
    ray = sys.modules.get("ray")
    initialized = getattr(ray, "is_initialized", None)
    if not callable(initialized) or not initialized():
        return False
    from ray.util import get_current_placement_group

    return get_current_placement_group() is not None


def visible_cuda_devices() -> tuple[str, ...]:
    """CUDA identities after the caller's visibility mask; CPU paths never call this."""
    try:
        import torch
    except ImportError:
        raise ValueError("local GPU workers require PyTorch; install it or explicitly select ray") from None
    count = torch.cuda.device_count()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility is None:
        return tuple(str(index) for index in range(count))
    return tuple(device.strip() for device in visibility.split(",") if device.strip())[:count]


def local_gpu_assignments(requirements: ExecutionRequirements) -> tuple[tuple[str, ...], ...]:
    """Disjoint CUDA masks for spawned workers; not cluster-wide reservations."""
    count = requirements.gpus_per_worker
    if count != int(count):
        raise ValueError("local GPU workers require whole GPUs; explicitly select ray for fractional reservations")
    count = int(count)
    devices = visible_cuda_devices()
    required = requirements.workers * count
    if required > len(devices):
        raise ValueError(
            f"workers require {required} GPUs, but only {len(devices)} are visible on this host; "
            "reduce workers/GPU requirements or explicitly select ray for a cluster"
        )
    return tuple(devices[rank * count : (rank + 1) * count] for rank in range(requirements.workers))


def executor_settings(config: Mapping[str, Any], selection: Any) -> ExecutorSettings:
    """Resolve a built-in/import path, inline object, or named executors profile."""
    profiles = config.get("executors", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("executors must be an object")
    if selection is None:
        selection = "auto"
    if isinstance(selection, str):
        selection = profiles.get(selection, {"backend": selection})
    if not isinstance(selection, Mapping):
        raise ValueError("executor must be a backend name, profile name, or object")
    unknown = set(selection) - {"backend", "options", "workers", "resources"}
    if unknown:
        raise ValueError(f"unknown executor fields: {sorted(unknown)}")
    backend = selection.get("backend", "auto")
    options = selection.get("options", {})
    if not isinstance(backend, str) or not backend:
        raise ValueError("executor.backend must be a non-empty string")
    if not isinstance(options, Mapping):
        raise ValueError("executor.options must be an object")
    workers = selection.get("workers")
    if isinstance(workers, str) and workers.strip().isdigit():
        workers = int(workers)
    if "workers" in selection and workers is None:
        raise ValueError("execution workers must be a positive integer")
    resources = selection.get("resources", {})
    if not isinstance(resources, Mapping) or set(resources) - {"cpus_per_worker", "gpus_per_worker"}:
        raise ValueError("executor.resources accepts only cpus_per_worker and gpus_per_worker")
    parsed = {}
    for name, value in resources.items():
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"{name} must be a finite nonnegative number") from None
        if value is None:
            raise ValueError(f"{name} must be a finite nonnegative number")
        parsed[name] = value
    return ExecutorSettings(
        backend=backend, options=dict(options), workers=workers, resources=WorkerResources(**parsed)
    )


def role_executor_settings(config: Mapping[str, Any], role: str, default: str = "auto") -> ExecutorSettings:
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be an object")
    unknown = set(execution) - {"services", "training", "rollout", "evolution"}
    if unknown:
        raise ValueError(f"unknown execution roles: {sorted(unknown)}")
    return executor_settings(config, execution.get(role, default))
