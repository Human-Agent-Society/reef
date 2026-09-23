"""Deployment plans, resource ownership, connection configuration, and runtime selection.

- Component contracts: what a deployment's resources, inference service and
  training service each own, and how their health is observed.
- :class:`ModelDeployment`: starts those components in dependency order,
  attaches the weight-transfer session, and runs Reef's coordinator.
- Runtime factories: the ``type`` table that turns a config section into
  runtimes, with bundled kinds resolved lazily so importing contracts loads
  no integration.
- Connection settings shared by the executor- and Ray-backed factories.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from reef.core.config import config_arguments, config_metadata, config_option, parse_config_values
from reef.core.errors import ReefError
from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, WorkerSpec
from reef.runtime.executor.connection import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.runtime.executor.failure import ExecutorFailedError
from reef.runtime.interfaces import InferenceBackend, InferenceRuntime, TrainingBackend, TrainingRuntime
from reef.runtime.scheduler import TrainingCoordinator

_logger = logging.getLogger(__name__)

# -- Component contracts ------------------------------------------------------


class DeploymentResources(ABC):
    """A coordinated allocation and its runtime connection, owned by Reef."""

    @abstractmethod
    def start(self) -> None:
        """Acquire resources; close must also handle a partially failed start."""

    @abstractmethod
    def close(self) -> None:
        """Release owned reservations and connections, idempotently."""


class InferenceResources(DeploymentResources):
    """Allocation that supplies a backend-native inference reservation."""

    @property
    @abstractmethod
    def inference_placement(self) -> Any:
        """Return the borrowed placement handle understood by the selected backend."""


class DeploymentHealth(ABC):
    """Nonblocking observation of an already started deployment."""

    @abstractmethod
    def poll(self) -> None:
        """Raise on failure; an outstanding probe is not a failed component."""


class ComponentHealth(DeploymentHealth):
    """Observe selected components without importing their implementations."""

    def __init__(self, *components: DeploymentHealth) -> None:
        self.components = components

    def poll(self) -> None:
        for component in self.components:
            component.poll()


class ModelPlanSource(ABC):
    """Rebuild components and rerun durable recovery preflight for each attempt."""

    @abstractmethod
    def create(self) -> ModelDeploymentPlan:
        """Return a fresh, unallocated plan from the original configuration."""


@dataclass(frozen=True)
class InferenceConnection:
    """Borrowed control transport with an explicitly versioned adapter protocol.

    The protocol identifies the RPC vocabulary, including direct weight-update
    attachment. An HTTP endpoint alone does not satisfy this connection.
    Reusing it for replacement training workers requires the owning deployment
    to retire the prior trainer and perform the protocol's attachment handshake.
    """

    protocol: str
    control: Executor


@dataclass(frozen=True)
class WeightTransferSession:
    """One deployment's borrowed native sender/receiver attachment.

    This carries the existing adapter control protocol, not a universal tensor
    format. Reef creates a new session identity whenever it rebuilds a receiver;
    training workers must not reuse a previous deployment's attachment.
    """

    protocol: str
    receiver: Executor
    session_id: str

    def __post_init__(self) -> None:
        if not self.protocol or not self.session_id:
            raise ValueError("weight transfer sessions require a protocol and identity")


#: The transfer a trainer declares when it delivers adapters as PEFT directories Reef loads itself.
ADAPTER_FILES_PROTOCOL = "reef-adapter-files-v1"


class InferenceService(DeploymentHealth):
    """An inference component that owns its engines but borrows reservations."""

    @property
    @abstractmethod
    def connection_protocol(self) -> str:
        """Control protocol provided by the selected engine integration."""

    @property
    def supported_transfer_protocols(self) -> tuple[str, ...]:
        """Weight transfers this receiver accepts: its native one, plus Reef-owned file delivery if it loads files."""
        return (self.connection_protocol,)

    @abstractmethod
    def start(self, resources: DeploymentResources) -> InferenceConnection:
        """Start engines in supplied resources and return a borrowed connection."""

    @abstractmethod
    def prepare_weight_transfer(self, connection: InferenceConnection) -> None:
        """Fence the receiver and release shared resources before trainer startup."""

    @abstractmethod
    def backend(self, connection: InferenceConnection) -> InferenceBackend:
        """Return this receiver's backend for Reef-owned coordination."""

    @abstractmethod
    def check_health(self) -> None:
        """Raise when the component is not ready."""

    @abstractmethod
    def poll(self) -> None:
        """Observe component failures without blocking behind active work."""

    @abstractmethod
    def close(self) -> None:
        """Release owned engines, including partial starts, idempotently."""


class TrainingService(DeploymentHealth):
    """Training workers allocated independently of the inference component."""

    @property
    @abstractmethod
    def weight_transfer_protocol(self) -> str | None:
        """Native weight transport consumed by a separately attached sender."""

    @abstractmethod
    def start(self, resources: DeploymentResources) -> None:
        """Start training workers using only their supplied reservations."""

    @abstractmethod
    def attach_weight_transport(self, session: WeightTransferSession) -> None:
        """Configure a sender after allocation without taking receiver ownership."""

    @abstractmethod
    def backend(self) -> TrainingBackend:
        """Return the training backend for Reef-owned coordination."""

    @abstractmethod
    def check_health(self) -> None:
        """Raise when the component is not ready."""

    @abstractmethod
    def poll(self) -> None:
        """Observe component failures without blocking behind active work."""

    @abstractmethod
    def close(self) -> None:
        """Release owned training objects, including partial starts, idempotently."""


# -- Deployment lifecycle -----------------------------------------------------


@dataclass(frozen=True)
class CoordinatorConfig:
    """Executor selection for Reef's coordinator, independent of backend code."""

    backend: str | type[Executor] = "ray"
    options: Mapping[str, Any] = field(default_factory=dict)
    launch_timeout_s: float | None = None


@dataclass(frozen=True)
class ModelDeploymentPlan:
    """Configured components; constructing a plan must not allocate resources.

    A missing inference component explicitly selects a combined compatibility
    lifecycle. It is never a fallback after a separate component fails to start.
    """

    resources: DeploymentResources
    inference: InferenceService | None
    training: TrainingService
    health: DeploymentHealth | None = None
    coordinator: CoordinatorConfig | None = None

    def validate(self) -> None:
        if self.coordinator is not None and self.inference is None:
            raise ValueError("a Reef coordinator requires separate training and inference components")
        required = self.training.weight_transfer_protocol
        if self.inference is None:
            if required is not None:
                raise ValueError(f"training requires {required!r} but no inference component is configured")
            return
        provided = self.inference.supported_transfer_protocols
        if not self.inference.connection_protocol or required not in provided:
            raise ValueError(
                f"incompatible inference control protocol: training requires {required!r}, got {provided!r}"
            )


class ModelDeployment:
    """Own allocation, backend attachment and the generic coordinator process.

    Start order is resources, inference, training, coordinator; close order
    is the reverse, and close runs for every component that was started even
    when a later one failed.
    """

    def __init__(self, plan: ModelDeploymentPlan) -> None:
        self.plan = plan
        self._started = False
        self._closed = False
        self._resources_started = False
        self._inference_started = False
        self._training_started = False
        self.weight_transfer_session: WeightTransferSession | None = None
        self._coordinator: Executor | None = None
        self._coordinator_probe: ExecutorFuture | None = None

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("model deployment can only be started once")
        self.plan.validate()
        self._started = True
        try:
            self._resources_started = True
            self.plan.resources.start()
            connection = self._start_inference()
            self._start_training()
            if connection is not None:
                self._start_coordinator(connection)
                self.plan.inference.check_health()  # type: ignore[union-attr]
        except BaseException:
            try:
                self.close()
            except Exception:
                _logger.exception("Failed to clean up model deployment after startup failure")
            raise

    def _start_inference(self) -> InferenceConnection | None:
        """Start the receiver and fence it for the trainer; ``None`` without a separate component."""
        inference = self.plan.inference
        if inference is None:
            return None
        self._inference_started = True
        connection = inference.start(self.plan.resources)
        required = self.plan.training.weight_transfer_protocol
        if (
            connection.protocol != inference.connection_protocol
            or required not in inference.supported_transfer_protocols
        ):
            raise ValueError("inference returned a connection with an incompatible protocol")
        inference.check_health()
        # Colocated trainers cannot initialize until inference has
        # acknowledged releasing its initial device allocations.
        inference.prepare_weight_transfer(connection)
        self.weight_transfer_session = WeightTransferSession(required, connection.control, uuid4().hex)
        return connection

    def _start_training(self) -> None:
        self._training_started = True
        self.plan.training.start(self.plan.resources)
        if self.weight_transfer_session is not None:
            self.plan.training.attach_weight_transport(self.weight_transfer_session)
        self.plan.training.check_health()

    def _start_coordinator(self, connection: InferenceConnection) -> None:
        config = self.plan.coordinator
        inference = self.plan.inference
        if config is None or inference is None:
            return
        self._coordinator = Executor.create(
            ExecutorConfig(
                backend=config.backend,
                workers=(
                    WorkerSpec(
                        TrainingCoordinator,
                        args=(self.plan.training.backend(), inference.backend(connection)),
                        kwargs={"owns_training": True},
                    ),
                ),
                options=config.options,
                launch_timeout_s=config.launch_timeout_s,
            )
        )
        # Constructor recovery can take as long as checkpoint loading. The
        # launcher's ready timeout bounds startup; a short RPC timeout must
        # not interrupt a healthy recovery and leave its result ambiguous.
        self._validate_health(self._coordinator.rpc(0, "health"))

    def poll(self) -> None:
        """Observe components without queuing repeated health work behind training."""
        if self.plan.health is not None:
            self.plan.health.poll()
        coordinator = self._coordinator
        if coordinator is None:
            return
        if coordinator.failure is not None:
            raise ExecutorFailedError(coordinator.failure)
        if self._coordinator_probe is None:
            self._coordinator_probe = coordinator.rpc(0, "health", non_block=True)
        try:
            result = self._coordinator_probe.result(timeout=0)
        except TimeoutError:
            return
        self._coordinator_probe = None
        self._validate_health(result, allow_recovery=True)

    @staticmethod
    def _validate_health(result: Any, *, allow_recovery: bool = False) -> None:
        if isinstance(result, Mapping) and (
            result.get("ok") is True or (allow_recovery and result.get("recoverable") is True)
        ):
            return
        raise RuntimeError(f"training coordinator failed its health check: {result!r}")

    def _close_coordinator(self) -> None:
        coordinator = self._coordinator
        if coordinator is not None:
            try:
                coordinator.rpc(0, "shutdown", timeout=90)
            finally:
                coordinator.shutdown()
                self._coordinator = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            self._close_coordinator()
        except Exception as exc:
            errors.append(exc)
            _logger.exception("Failed to close runtime coordinator")
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


# -- Runtime factories --------------------------------------------------------

RuntimeBuild = InferenceRuntime | tuple[TrainingRuntime, InferenceRuntime]


class RuntimeConfigError(ReefError):
    """Raised when runtime configuration is invalid or required keys are missing."""


class RuntimeFactory(ABC):
    """Base class for runtime factories.

    Required (subclass must set):
        ``kind`` — canonical kind name used in config ``type``.

    Implement:
        ``__call__`` — receive ``(config, model_path, recipe_config, environ)``
        and return an :class:`InferenceRuntime` or a ``(TrainingRuntime, InferenceRuntime)`` pair.
    """

    kind: str

    def config_type(self) -> type | None:
        """Return this adapter's settings dataclass, or None for legacy factories.

        Declarations use ``reef.core.config.config_option``. Discovery reads
        metadata only; it must not connect services or allocate resources.
        """
        return None

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        """Parse only this selected adapter's configuration."""
        settings_type = self.config_type()
        if settings_type is None:
            return dict(config)
        try:
            values = parse_config_values(
                config_arguments(settings_type, prefix=("runtime",)),
                {key: value for key, value in config.items() if key != "type"},
                environ=environ,
            )
            # Component constructors retain range and cross-field validation.
            settings_type(**values)
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc
        return {"type": config.get("type", self.kind), **values}

    @abstractmethod
    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> RuntimeBuild:
        """Build inference alone or separate training and inference components."""


class _CallableRuntimeFactory(RuntimeFactory):
    """Adapter wrapping a plain callable as a :class:`RuntimeFactory` instance.

    Used when a dotted ``"module:callable"`` reference resolves to a function
    rather than a ``RuntimeFactory`` subclass instance, so
    :func:`runtime_factory_for` always returns a uniform type.
    """

    kind = ""

    def __init__(self, fn: Callable[..., RuntimeBuild]) -> None:
        self._fn = fn

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> RuntimeBuild:
        return self._fn(config, model_path, recipe_config, environ)


#: Bundled kinds, resolved on first use so importing contracts loads no integration.
_BUILTIN_FACTORIES = {
    "coordinator_training": "reef.service.runtime:CoordinatorRuntimeFactory",
    "executor_training": "reef.service.runtime:ExecutorTrainingRuntimeFactory",
    "inference_proxy": "reef.inference.http:InferenceProxyRuntimeFactory",
    "ray_training": "reef.service.runtime:RayTrainingRuntimeFactory",
    "sglang": "reef.inference.sglang.runtime:SGLangRuntimeFactory",
    "slime_training": "reef.train.slime_backend.runtime:SlimeRuntimeFactory",
    "tinker": "reef.inference.tinker:TinkerInferenceRuntimeFactory",
    "vllm": "reef.inference.vllm.runtime:VLLMRuntimeFactory",
}

#: Explicitly registered extensions, plus cached instances of resolved builtins.
_runtime_kinds: dict[str, RuntimeFactory] = {}


def _register(factory: RuntimeFactory) -> None:
    if factory.kind in _runtime_kinds or factory.kind in _BUILTIN_FACTORIES:
        raise ValueError(f"runtime kind {factory.kind!r} is already registered")
    _runtime_kinds[factory.kind] = factory


def register_runtime_kind(cls: type[RuntimeFactory]) -> type[RuntimeFactory]:
    """Register an extension factory when its owning module is explicitly loaded."""
    _register(cls())
    return cls


def register_runtime_factory(factory: RuntimeFactory) -> RuntimeFactory:
    """Register an external runtime factory (module-level convenience)."""
    if not isinstance(factory, RuntimeFactory):
        raise TypeError(f"a runtime factory registers a RuntimeFactory, got {type(factory).__name__}")
    _register(factory)
    return factory


def runtime_kinds() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*_BUILTIN_FACTORIES, *_runtime_kinds)))


def runtime_factory_for(kind: str) -> RuntimeFactory | None:
    """Return the runtime factory for ``kind``.

    A kind is a registered name, a bundled name, or a dotted reference
    ``package.module:factory_name`` to a factory reef does not bundle —
    the config path's way of serving deployment-specific runtimes without a
    hand-rolled service assembly.
    """
    registered = _runtime_kinds.get(kind)
    if registered is not None:
        return registered
    reference = _BUILTIN_FACTORIES.get(kind, kind)
    if ":" not in reference:
        return None
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise RuntimeConfigError(f"dotted runtime kind {kind!r} must be 'package.module:factory_name'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeConfigError(f"cannot import runtime kind {kind!r}: {exc}") from exc
    if isinstance(candidate, RuntimeFactory):
        return candidate
    if isinstance(candidate, type) and issubclass(candidate, RuntimeFactory):
        factory = candidate()
        if kind in _BUILTIN_FACTORIES:
            _runtime_kinds[kind] = factory
        return factory
    if callable(candidate):
        return _CallableRuntimeFactory(candidate)
    raise RuntimeConfigError(f"runtime kind {kind!r} is not callable")


class RuntimeRegistry:
    """Build runtimes from config sections; injected factories shadow registered kinds."""

    def __init__(self, factories: Mapping[str, RuntimeFactory] | None = None) -> None:
        self._factories: dict[str, RuntimeFactory] = dict(factories or {})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._factories) | set(runtime_kinds())))

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_path: str,
        recipe_config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeBuild:
        runtime_type = config.get("type")
        if not isinstance(runtime_type, str) or not runtime_type:
            raise RuntimeConfigError("runtime.type must be a non-empty string")
        factory = self._factories.get(runtime_type) or runtime_factory_for(runtime_type)
        if factory is None:
            raise RuntimeConfigError(
                f"unknown runtime type {runtime_type!r}; available runtimes: {', '.join(self.names)}"
            )
        values = os.environ if environ is None else environ
        if isinstance(factory, RuntimeFactory):
            # Injected entries may be plain callables that parse nothing.
            config = factory.parse_config(config, values)
        return factory(config, model_path, recipe_config or {}, values)


def runtime_pair(value: Any) -> tuple[TrainingRuntime, InferenceRuntime] | None:
    """Return ``value`` as a ``(training, inference)`` pair, or ``None`` after shutting down what it held.

    Factories and connectors are user-supplied; anything other than the pair
    is a configuration error, and any runtime they did build must not leak.
    """
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], TrainingRuntime)
        and isinstance(value[1], InferenceRuntime)
    ):
        return value
    for component in value if isinstance(value, tuple) else (value,):
        with suppress(Exception):
            component.shutdown()
    return None


# -- Config section helpers ---------------------------------------------------


def config_string(config: Mapping[str, Any], name: str) -> str:
    """A required non-empty string key of a runtime config section."""
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeConfigError(f"runtime.{name} must be a non-empty string")
    return value


def config_secret(
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    name: str,
    env_name: str,
) -> str | None:
    """An optional secret: literal ``name`` wins, else the variable named by ``env_name``."""
    if config.get(name) is not None:
        return config_string(config, name)
    if config.get(env_name) is None:
        return None
    return environ.get(config_string(config, env_name))


@dataclass(frozen=True)
class RuntimeConnectionConfig:
    """Connection and request options; model-worker topology stays with the backend."""

    inference_url: str | None = config_option(None, help="Inference worker URL; omitted uses the coordinator.")
    inference_timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")
    train_timeout_s: float | None = config_option(None, help="Training request timeout; omitted follows inference.")
    max_staleness: int = config_option(0, help="Maximum admitted training-version lag.")
    inference_handler_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Inference adapter-owned options.")
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("inference_timeout_s", self.inference_timeout_s),
            ("train_timeout_s", self.train_timeout_s),
        ):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"runtime.{name} must be a positive finite number")
        if isinstance(self.max_staleness, bool) or self.max_staleness < 0:
            raise ValueError("runtime.max_staleness must be a non-negative integer")


@dataclass(frozen=True)
class ExecutorRuntimeConfig(RuntimeConnectionConfig):
    coordinator_rank: int = config_option(0, help="Rank of the training coordinator worker.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coordinator_rank < 0:
            raise ValueError("runtime.coordinator_rank must be non-negative")


@dataclass(frozen=True)
class RayRuntimeConfig(RuntimeConnectionConfig):
    actor_name: str = config_option(DEFAULT_ACTOR_NAME, help="Training coordinator actor name.")
    namespace: str = config_option(DEFAULT_NAMESPACE, help="Ray namespace containing the coordinator.")
    ray_address: str | None = config_option(None, help="Ray cluster address.")
