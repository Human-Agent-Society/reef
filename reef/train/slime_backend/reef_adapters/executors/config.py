"""Resolve Slime launcher aliases before allocating model resources."""

from __future__ import annotations

from typing import Literal

from reef.runtime.executor import Executor
from reef.runtime.executor.config import ExecutorSettings, select_executor

DEFAULT_EXECUTOR_BACKEND = "reef.train.slime_backend.reef_adapters.executors.ray:SlimeRayExecutor"
DEFAULT_ROLLOUT_EXECUTOR = "reef.train.slime_backend.reef_adapters.executors.rollout:SlimeRayRolloutExecutor"


def slime_executor_class(
    backend: str | type[Executor] | None, *, role: Literal["training", "rollout"]
) -> type[Executor]:
    if isinstance(backend, type):
        return Executor.get_class(backend)
    selected = select_executor(ExecutorSettings(backend or "auto"), role=role).settings.backend
    if selected in ("mp", "uni"):
        raise ValueError(
            f"Slime {role} requires ray or a Slime-compatible executor; {selected!r} is not a GPU launcher"
        )
    if selected == "ray":
        selected = DEFAULT_EXECUTOR_BACKEND if role == "training" else DEFAULT_ROLLOUT_EXECUTOR
    return Executor.get_class(selected)
