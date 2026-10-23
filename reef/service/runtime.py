"""Compose separately selected training and inference runtime connections.

Generic executor and Ray config kinds retain their HTTP request default.
Native model runtimes are supplied by their integration factories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from reef.core.config import config_metadata, config_option

from reef.inference.http import HttpInferenceHandler
from reef.inference.runtime import ExecutorInferenceRuntime
from reef.runtime.deployment import (
    ExecutorRuntimeConfig,
    RayRuntimeConfig,
    RuntimeConfigError,
    RuntimeFactory,
    RuntimeRegistry,
    runtime_pair,
)
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.connection import (
    DEFAULT_ACTOR_NAME,
    DEFAULT_NAMESPACE,
    CoordinatorClient,
    ExecutorCoordinatorClient,
    connect_ray_coordinator,
)
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, TrainingRuntime
from reef.train.runtime import ExecutorTrainingRuntime

#: Connection keys the config path forwards verbatim to :func:`connect_executor_runtimes`.
CONNECTION_CONFIG_KEYS = (
    "inference",
    "inference_url",
    "inference_timeout_s",
    "inference_handler_factory",
    "inference_handler_config",
)
#: Inference runtime options the coordinator runtime sets itself; ``inference_runtime_options`` cannot name them.
RESERVED_INFERENCE_OPTIONS = frozenset({*CONNECTION_CONFIG_KEYS, "type", "control"})
#: Keys the config path forwards verbatim to :func:`connect_ray_runtime`.
RAY_CONFIG_KEYS = (
    "actor_name",
    "namespace",
    "ray_address",
    "train_timeout_s",
    "batch_size",
    *CONNECTION_CONFIG_KEYS[1:],
)


def connect_executor_runtimes(
    *,
    train_group_handle: CoordinatorClient,
    batch_size: int,
    inference: InferenceRuntime | None = None,
    inference_url: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    inference_handler_factory: type[InferenceHandler] = HttpInferenceHandler,
    inference_handler_config: Mapping[str, Any] | None = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Assemble independent components over the existing deployment connection.

    ``batch_size`` is the samples in one training step, from the training options.
    """
    if inference is not None and inference_url is not None:
        raise ValueError("pass inference or inference_url, not both")
    training = ExecutorTrainingRuntime(train_group_handle, batch_size=batch_size)
    if inference is None:
        inference = ExecutorInferenceRuntime(
            control=train_group_handle,
            inference_url=inference_url,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            inference_handler_factory=inference_handler_factory,
            inference_handler_config=inference_handler_config,
        )
    return training, inference


def connect_ray_runtime(
    *,
    batch_size: int = 1,
    inference_url: str | None = None,
    actor_name: str = DEFAULT_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    ray_address: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    train_timeout_s: float | None = None,
    inference_handler_factory: type[InferenceHandler] = HttpInferenceHandler,
    inference_handler_config: Mapping[str, Any] | None = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Connect to a named training actor and return separate training and inference runtimes.

    Reef and the backend run as separate services in one Ray cluster.
    ``namespace`` must match the namespace used when the backend actor was
    created. ``inference_url`` defaults to the address the actor reports.
    ``batch_size`` is the samples in one training step.
    """
    return connect_executor_runtimes(
        train_group_handle=connect_ray_coordinator(
            actor_name=actor_name,
            namespace=namespace,
            ray_address=ray_address,
            train_timeout_s=train_timeout_s,
            inference_timeout_s=inference_timeout_s,
        ),
        batch_size=batch_size,
        inference_url=inference_url,
        model_path=model_path,
        inference_timeout_s=inference_timeout_s,
        inference_handler_factory=inference_handler_factory,
        inference_handler_config=inference_handler_config,
    )


def _executor_config(value: Mapping[str, Any]) -> ExecutorConfig:
    workers = value.get("workers", ())
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)):
        raise RuntimeConfigError("runtime.executor.workers must be a sequence of worker specifications")
    specs = []
    for worker in workers:
        if isinstance(worker, WorkerSpec):
            specs.append(worker)
        elif isinstance(worker, Mapping):
            try:
                specs.append(WorkerSpec(**dict(worker)))
            except (TypeError, ValueError) as exc:
                raise RuntimeConfigError(f"invalid runtime.executor worker: {exc}") from exc
        else:
            raise RuntimeConfigError("runtime.executor.workers entries must be WorkerSpec objects or mappings")
    try:
        return ExecutorConfig(
            backend=value.get("backend", "auto"),
            workers=tuple(specs),
            options=value.get("options", {}),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid runtime.executor configuration: {exc}") from exc


def _executor_from_config(value: Any) -> tuple[Executor, bool]:
    """Resolve the ``executor`` entry; the flag says whether this call created it."""
    if isinstance(value, Mapping):
        value = _executor_config(value)
    if isinstance(value, ExecutorConfig):
        return Executor.create(value), True
    if isinstance(value, Executor):
        return value, False
    raise RuntimeConfigError("runtime.executor must be an Executor, ExecutorConfig, or configuration mapping")


class ExecutorTrainingRuntimeFactory(RuntimeFactory):
    """Create a training coordinator using a configured executor.

    The executor entry accepts an existing Executor, an ExecutorConfig, or a
    mapping with backend, workers, and options. The worker at coordinator_rank
    (default 0) implements CoordinatorClient's methods; it may manage its
    own model-parallel worker group.
    """

    kind = "executor_training"

    def config_type(self) -> type:
        return ExecutorRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        # Python assembly may inject these objects; they are not YAML fields.
        injected = {
            key: config[key] for key in ("executor", "inference", "inference_handler_factory") if key in config
        }
        values = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**values, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        executor, created = _executor_from_config(config.get("executor"))
        train_timeout_s = config.get("train_timeout_s")
        if train_timeout_s is None:
            # A training step legitimately outlasts an inference request.
            train_timeout_s = config.get("inference_timeout_s", 300.0)
        try:
            handle = ExecutorCoordinatorClient(
                executor, rank=config.get("coordinator_rank", 0), timeout_s=train_timeout_s
            )
            return connect_executor_runtimes(
                train_group_handle=handle,
                batch_size=config.get("batch_size", 1),
                model_path=model_path,
                **{key: config[key] for key in CONNECTION_CONFIG_KEYS if key in config},
            )
        except BaseException:
            if created:
                with suppress(Exception):
                    executor.shutdown()
            raise


@dataclass(frozen=True)
class CoordinatorRuntimeConfig(RayRuntimeConfig):
    """Connect to a Reef coordinator in Ray and pair it with a separately selected inference runtime.

    Values: the ``RayRuntimeConfig`` values, where also ``inference_runtime`` is
    a non-empty runtime kind or factory reference, and
    ``inference_runtime_options`` maps option names of that runtime to their
    values. No option name is one that the coordinator sets itself
    (``RESERVED_INFERENCE_OPTIONS``). ``CoordinatorRuntimeFactory`` passes the
    options, the connection settings and the coordinator client to the
    selected runtime's factory, which validates the options.
    """

    inference_runtime: str = config_option(
        "", help="Independently selected inference runtime kind or factory reference."
    )
    inference_runtime_options: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Options of the selected inference runtime.")
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.inference_runtime:
            raise ValueError("runtime.inference_runtime must name the selected inference implementation")
        reserved = sorted(set(self.inference_runtime_options) & RESERVED_INFERENCE_OPTIONS)
        if reserved:
            raise ValueError(f"runtime.inference_runtime_options cannot set {', '.join(reserved)}")


class CoordinatorRuntimeFactory(RuntimeFactory):
    """Build the training runtime over a named coordinator and bind the selected inference runtime.

    Any integration whose model driver runs Reef's coordinator, Slime or a
    hosted trainer alike, connects the HTTP service this way; the inference
    factory receives the coordinator client as its control connection.
    """

    kind = "coordinator_training"

    def config_type(self) -> type:
        return CoordinatorRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        injected = {key: config[key] for key in ("connect", "inference_handler_factory") if key in config}
        parsed = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**{key: value for key, value in parsed.items() if key in config}, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        registry = RuntimeRegistry()
        if "connect" in config:
            injected = {key: value for key, value in config.items() if key != "inference_runtime"}
            built = registry.build({**injected, "type": "ray_training"}, model_path=model_path, environ=environ)
            pair = runtime_pair(built)
            if pair is None:
                raise RuntimeConfigError("the configured training connector must return a runtime pair")
            return pair
        training = ExecutorTrainingRuntime(
            connect_ray_coordinator(
                actor_name=config.get("actor_name", DEFAULT_ACTOR_NAME),
                namespace=config.get("namespace", DEFAULT_NAMESPACE),
                ray_address=config.get("ray_address"),
                inference_timeout_s=config.get("inference_timeout_s", 300.0),
                train_timeout_s=config.get("train_timeout_s"),
            ),
            batch_size=config.get("batch_size", 1),
        )
        try:
            inference = registry.build(
                {
                    **dict(config.get("inference_runtime_options") or {}),
                    **{key: config[key] for key in CONNECTION_CONFIG_KEYS if key in config},
                    "type": config["inference_runtime"],
                    "control": training.train_group_handle,
                },
                model_path=model_path,
                recipe_config=recipe_config,
                environ=environ,
            )
            if not isinstance(inference, InferenceRuntime):
                for component in inference if isinstance(inference, tuple) else ():
                    with suppress(Exception):
                        component.shutdown()
                raise RuntimeConfigError("the selected inference factory must return an InferenceRuntime")
            return training, inference
        except BaseException:
            with suppress(Exception):
                training.shutdown()
            raise


class RayTrainingRuntimeFactory(RuntimeFactory):
    """Build separate training and inference runtimes from runtime configuration.

    The config mirrors :func:`connect_ray_runtime`'s keyword arguments. A
    ``connect`` entry may inject an alternative connector callable (tests use
    this to stub the Ray cluster); it defaults to :func:`connect_ray_runtime`.
    """

    kind = "ray_training"

    def config_type(self) -> type:
        return RayRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        # Existing Python assembly can inject these objects. They are not YAML
        # fields and must never be serialized through the argument parser.
        injected = {key: config[key] for key in ("connect", "inference_handler_factory") if key in config}
        values = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        # Only keys the caller set are forwarded, so the connector's own
        # defaults apply to everything else.
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
        built = connect(model_path=model_path, **{key: config[key] for key in RAY_CONFIG_KEYS if key in config})
        pair = runtime_pair(built)
        if pair is None:
            raise RuntimeConfigError(
                f"runtime connector returned {type(built).__name__}, not a (TrainingRuntime, InferenceRuntime) pair"
            )
        return pair
