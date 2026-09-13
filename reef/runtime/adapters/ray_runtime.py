"""Ray connection adapter for the executor-independent training runtime.

The historical Ray runtime names remain aliases for compatibility. Ray owns
actor discovery here; training semantics live in ExecutorTrainingRuntime and
worker control RPC lives in RayExecutor.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_option
from reef.runtime.adapters.executor_runtime import connect_executor_runtimes
from reef.runtime.base import InferenceRuntime, TrainingRuntime
from reef.runtime.executor.failure import ExecutorFailedError
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.inference import InferenceBackendFactory, build_http_inference_backend
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.runtime.registry import RuntimeConfigError, RuntimeFactory, register_runtime_kind
from reef.runtime.settings import TrainingRuntimeSettings
from reef.runtime.training_group import ExecutorTrainGroupHandle, TrainingGroupHandle, TrainingRuntimeError

RayRuntimeError = TrainingRuntimeError
RayTrainGroupHandle = TrainingGroupHandle


def _require_ray():
    """Import Ray lazily so other executors need not install it."""
    try:
        import ray
    except ImportError as exc:
        raise RayRuntimeError(
            "remote Ray runtimes require the 'ray' package; install it to connect to a training backend"
        ) from exc
    return ray


class RemoteRayTrainGroupHandle(ExecutorTrainGroupHandle):
    """Attach a non-owning executor to an existing Ray training coordinator."""

    def __init__(self, train_group_actor: Any, *, timeout_s: float = 300.0) -> None:
        super().__init__(
            RayExecutor.from_workers((train_group_actor,), owned=False),
            timeout_s=timeout_s,
        )


class NamedRayTrainGroupHandle(RemoteRayTrainGroupHandle):
    """Discover the current coordinator before each RPC, without replaying writes.

    A missing name is retried within the operation's timeout. Once an RPC is
    submitted, a write's error reaches the caller even if a replacement appears.
    Only liveness/version reads retry actor loss. Durable training reconciliation
    decides whether a training operation is replayable.
    """

    reconnects = True

    def __init__(
        self, actor_name: str, namespace: str, *, timeout_s: float = 300, health_timeout_s: float = 300
    ) -> None:
        if (
            isinstance(health_timeout_s, bool)
            or not isinstance(health_timeout_s, (int, float))
            or not math.isfinite(health_timeout_s)
            or health_timeout_s <= 0
        ):
            raise TrainingRuntimeError("health timeout must be a positive finite number")
        super().__init__(_require_ray().get_actor(actor_name, namespace=namespace), timeout_s=timeout_s)
        self._actor_name = actor_name
        self._namespace = namespace
        self._closed = False
        self._health_timeout_s = min(health_timeout_s, timeout_s)

    def _rpc(self, method: str, *args: Any) -> Any:
        ray = _require_ray()
        timeout = self._health_timeout_s if method in {"health", "serving_runtime_load_id"} else self._timeout_s
        deadline = time.monotonic() + timeout
        while not self._closed:
            try:
                actor = ray.get_actor(self._actor_name, namespace=self._namespace)
            except ValueError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TrainingRuntimeError("training coordinator is unavailable during recovery") from exc
                time.sleep(min(0.1, remaining))
                continue
            executor = RayExecutor.from_workers((actor,), owned=False)
            try:
                return executor.rpc(0, method, args=args, timeout=max(0.001, deadline - time.monotonic()))
            except (ray.exceptions.RayActorError, ExecutorFailedError):
                if method not in {"health", "serving_runtime_load_id"} or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            finally:
                executor.shutdown()
        raise TrainingRuntimeError("training coordinator connection is closed")

    def shutdown(self) -> None:
        self._closed = True
        super().shutdown()


def connect_ray_runtime(
    *,
    inference_url: str | None = None,
    actor_name: str = DEFAULT_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    ray_address: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    train_timeout_s: float | None = None,
    max_staleness: int = 0,
    inference_backend_factory: InferenceBackendFactory = build_http_inference_backend,
    inference_backend_config: Mapping[str, Any] | None = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Connect to a named training actor and return separate training and inference runtimes.

    Reef and the backend run as separate services in one Ray cluster.
    ``namespace`` must match the namespace used when the backend actor was
    created. ``inference_url`` defaults to the address the actor reports.
    """
    ray = _require_ray()
    if not ray.is_initialized():
        ray.init(address=ray_address or "auto", namespace=namespace)
    # A training step legitimately outlasts an inference request.
    return connect_executor_runtimes(
        train_group_handle=NamedRayTrainGroupHandle(
            actor_name,
            namespace,
            timeout_s=train_timeout_s if train_timeout_s is not None else inference_timeout_s,
            health_timeout_s=inference_timeout_s,
        ),
        inference_url=inference_url,
        model_path=model_path,
        inference_timeout_s=inference_timeout_s,
        max_staleness=max_staleness,
        inference_backend_factory=inference_backend_factory,
        inference_backend_config=inference_backend_config,
    )


@dataclass(frozen=True)
class RayRuntimeSettings(TrainingRuntimeSettings):
    actor_name: str = config_option(DEFAULT_ACTOR_NAME, help="Training coordinator actor name.")
    namespace: str = config_option(DEFAULT_NAMESPACE, help="Ray namespace containing the coordinator.")
    ray_address: str | None = config_option(None, help="Ray cluster address.")


@register_runtime_kind
class RayTrainingRuntimeFactory(RuntimeFactory):
    """Build separate training and inference runtimes from runtime configuration.

    The config mirrors :func:`connect_ray_runtime`'s keyword arguments. A
    ``connect`` entry may inject an alternative connector callable (tests use
    this to stub the Ray cluster); it defaults to :func:`connect_ray_runtime`.
    """

    kind = "ray_training"

    def config_type(self) -> type:
        return RayRuntimeSettings

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        # Existing Python assembly can inject these objects. They are not YAML
        # fields and must never be serialized through the argument parser.
        injected = {key: config[key] for key in ("connect", "inference_backend_factory") if key in config}
        values = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**{key: value for key, value in values.items() if key in config}, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        connect = config.get("connect", connect_ray_runtime)
        if not callable(connect):
            raise RuntimeConfigError("runtime.connect must be callable")
        kwargs: dict[str, Any] = {"model_path": model_path}
        for key in (
            "inference_url",
            "actor_name",
            "namespace",
            "ray_address",
            "inference_timeout_s",
            "train_timeout_s",
            "max_staleness",
            "inference_backend_factory",
            "inference_backend_config",
        ):
            if key in config:
                kwargs[key] = config[key]
        runtime = connect(**kwargs)
        if not (
            isinstance(runtime, tuple)
            and len(runtime) == 2
            and isinstance(runtime[0], TrainingRuntime)
            and isinstance(runtime[1], InferenceRuntime)
        ):
            for component in runtime if isinstance(runtime, tuple) else (runtime,):
                with suppress(Exception):
                    component.shutdown()
            raise RuntimeConfigError(
                f"runtime connector returned {type(runtime).__name__}, not a (TrainingRuntime, InferenceRuntime) pair"
            )
        return runtime
