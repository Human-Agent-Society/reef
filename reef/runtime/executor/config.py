"""Shared YAML executor selectors for services, training and rollout roles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.runtime.executor.requirements import ExecutionRequirements


@dataclass(frozen=True)
class ExecutorSettings:
    backend: str = "auto"
    options: Mapping[str, Any] = field(default_factory=dict)


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

    This is a pure policy: it neither probes GPUs nor starts a Ray runtime.
    Training/rollout are Slime roles, whose built-in distributed backend is
    currently Ray. A single GPU does not make LocalExecutor a Slime launcher.
    """
    if role not in ("services", "training", "rollout", "evolution"):
        raise ValueError(f"unknown executor role: {role!r}")
    if role == "evolution":
        return select_worker_executor(settings, requirements or ExecutionRequirements())
    if settings.backend != "auto":
        return ExecutorSelection(settings, "explicit backend/profile selection")
    if role in ("training", "rollout"):
        backend, reason = "ray", "Slime model workers require the built-in Ray launcher"
    elif local_cuda and (requires_resources or settings.options):
        raise ValueError("auto cannot combine local CUDA visibility with cluster resource options")
    elif local_cuda:
        backend, reason = "local", "service pins local CUDA visibility"
    elif requires_resources or settings.options:
        backend, reason = "ray", "service requests cluster resource/worker options"
    elif in_ray_placement_group:
        backend, reason = "ray", "service is running inside a Ray placement group"
    else:
        backend, reason = "local", "service has no cluster placement requirements"
    return ExecutorSelection(ExecutorSettings(backend, settings.options), reason)


def select_worker_executor(settings: ExecutorSettings, requirements: ExecutionRequirements) -> ExecutorSelection:
    """Select only launchers the component supports, never by installed hardware."""
    backend = settings.backend
    reason = "explicit backend/profile selection"
    if backend == "auto":
        if requirements.gpus_per_worker or requirements.cluster or settings.options:
            backend, reason = "ray", "component requests GPU/cluster resources or worker options"
        elif requirements.workers == 1:
            backend, reason = "uni", "one CPU worker; model inference is external to this worker"
        else:
            backend, reason = "mp", "multiple CPU workers on one host"
    if backend in ("uni", "mp", "local", "ray") and backend not in requirements.supported_backends:
        raise ValueError(f"component does not support executor {backend!r}")
    if backend in ("uni", "mp", "local") and (
        requirements.gpus_per_worker or requirements.cluster or settings.options
    ):
        raise ValueError(f"executor {backend!r} cannot reserve GPU/cluster resources or accept worker options")
    if backend == "uni" and requirements.workers != 1:
        raise ValueError("uni requires exactly one worker; reduce episode_workers or select mp")
    return ExecutorSelection(ExecutorSettings(backend, settings.options), reason)


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
    unknown = set(selection) - {"backend", "options"}
    if unknown:
        raise ValueError(f"unknown executor fields: {sorted(unknown)}")
    backend = selection.get("backend")
    options = selection.get("options", {})
    if not isinstance(backend, str) or not backend:
        raise ValueError("executor.backend must be a non-empty string")
    if not isinstance(options, Mapping):
        raise ValueError("executor.options must be an object")
    return ExecutorSettings(backend=backend, options=dict(options))


def role_executor_settings(config: Mapping[str, Any], role: str, default: str = "auto") -> ExecutorSettings:
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be an object")
    unknown = set(execution) - {"services", "training", "rollout", "evolution"}
    if unknown:
        raise ValueError(f"unknown execution roles: {sorted(unknown)}")
    return executor_settings(config, execution.get(role, default))
