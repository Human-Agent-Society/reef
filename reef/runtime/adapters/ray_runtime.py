"""Ray connection adapter for the executor-independent training runtime.

The historical Ray runtime names remain aliases for compatibility. Ray owns
actor discovery here; training semantics live in ExecutorTrainingRuntime and
worker control RPC lives in RayExecutor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.base import TrainingRuntime
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.inference import InferenceBackendFactory, build_http_inference_backend
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.runtime.registry import RuntimeConfigError, RuntimeFactory, register_runtime_kind
from reef.runtime.training_group import ExecutorTrainGroupHandle, TrainingGroupHandle, TrainingRuntimeError

RayRuntime = ExecutorTrainingRuntime
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
) -> RayRuntime:
    """Connect to a named training actor and return a ready :class:`RayRuntime`.

    Reef and the backend run as separate services in one Ray cluster.
    ``namespace`` must match the namespace used when the backend actor was
    created. ``inference_url`` defaults to the address the actor reports.
    """
    ray = _require_ray()
    if not ray.is_initialized():
        ray.init(address=ray_address or "auto", namespace=namespace)
    train_group_actor = ray.get_actor(actor_name, namespace=namespace)
    # A training step legitimately outlasts an inference request.
    return RayRuntime(
        train_group_handle=RemoteRayTrainGroupHandle(
            train_group_actor, timeout_s=train_timeout_s if train_timeout_s is not None else inference_timeout_s
        ),
        inference_url=inference_url,
        model_path=model_path,
        inference_timeout_s=inference_timeout_s,
        max_staleness=max_staleness,
        inference_backend_factory=inference_backend_factory,
        inference_backend_config=inference_backend_config,
    )


@register_runtime_kind
class RayTrainingRuntimeFactory(RuntimeFactory):
    """Build (connect) a :class:`RayRuntime` from a runtime config section.

    The config mirrors :func:`connect_ray_runtime`'s keyword arguments. A
    ``connect`` entry may inject an alternative connector callable (tests use
    this to stub the Ray cluster); it defaults to :func:`connect_ray_runtime`.
    """

    kind = "ray_training"

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> TrainingRuntime:
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
        if not isinstance(runtime, TrainingRuntime):
            raise RuntimeConfigError(f"runtime connector returned {type(runtime).__name__}, not a TrainingRuntime")
        return runtime
