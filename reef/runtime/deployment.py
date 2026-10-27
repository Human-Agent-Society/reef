"""Inference components, resource ownership, and runtime selection.

- Component contracts: what an inference service and the resources it borrows
  each own, and how their health is checked.
- Runtime factories: the ``type`` table that turns a config section into a
  serving runtime, with bundled kinds resolved lazily so importing contracts
  loads no integration.
- Training component factories: the dotted reference in a training
  deployment's component config that builds its training component.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_arguments, parse_config_values
from reef.core.errors import ReefError
from reef.runtime.executor import Executor
from reef.runtime.interfaces import InferenceBackend, InferenceRuntime
from reef.runtime.jobs import TrainingComponent

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


@dataclass(frozen=True)
class InferenceConnection:
    """What a started ``InferenceService`` returns: its control protocol and, when it has one, its control executor.

    ``protocol`` is the service's non-empty ``connection_protocol``; it names
    the control vocabulary. ``control`` is the borrowed executor that the
    service's ``InferenceBackend`` calls, or None for a receiver that the
    backend reaches by other means, such as HTTP.
    """

    protocol: str
    control: Executor | None


class InferenceService(ABC):
    """An inference component that owns its engines but borrows reservations."""

    @property
    @abstractmethod
    def connection_protocol(self) -> str:
        """Control protocol provided by the selected engine integration."""

    @abstractmethod
    def start(self, resources: DeploymentResources) -> InferenceConnection:
        """Start engines in supplied resources and return a borrowed connection."""

    @abstractmethod
    def backend(self, connection: InferenceConnection) -> InferenceBackend:
        """Return this receiver's backend for Reef-owned coordination."""

    @abstractmethod
    def check_health(self) -> None:
        """Raise when the component is not ready."""

    @abstractmethod
    def poll(self) -> None:
        """Raise when the engines failed, without blocking behind active work: a busy engine has not failed.

        The inference component's health route calls it.
        """

    @abstractmethod
    def close(self) -> None:
        """Release owned engines, including partial starts, idempotently."""


# -- Runtime factories --------------------------------------------------------


class RuntimeConfigError(ReefError):
    """Raised when runtime configuration is invalid or required keys are missing."""


class ConfiguredFactory(ABC):
    """A factory that builds one component from a config section of its own settings type.

    ``kind`` is the non-empty name that the section's ``type`` key holds.
    ``config_type`` is the settings dataclass whose fields the section sets,
    or None for a factory that reads the section as it is. ``parse_config``
    reads the section into those fields. ``RuntimeFactory`` and
    ``TrainingComponentFactory`` add what they build.
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


class RuntimeFactory(ConfiguredFactory):
    """Build the serving runtime that a ``runtime`` config section describes.

    ``__call__`` receives ``(config, model_path, recipe_config, environ)``
    and returns an :class:`InferenceRuntime`. ``RuntimeRegistry`` selects the
    factory by the section's ``type``.
    """

    @abstractmethod
    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        """Build the serving runtime that ``config`` describes."""


class TrainingComponentFactory(ConfiguredFactory):
    """Build the training component that a training deployment's component config describes.

    ``kind`` is the dotted reference ``package.module:factory`` that the
    config's ``type`` holds (``TrainingDeployment.component_type``).
    ``__call__`` receives ``(config, model_path, environ)`` and returns a
    :class:`TrainingComponent`; ``build_training_component`` calls it.
    """

    @abstractmethod
    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        """Build the training component that ``config`` describes."""


class _CallableRuntimeFactory(RuntimeFactory):
    """Adapter wrapping a plain callable as a :class:`RuntimeFactory` instance.

    Used when a dotted ``"module:callable"`` reference resolves to a function
    rather than a ``RuntimeFactory`` subclass instance, so
    :func:`runtime_factory_for` always returns a uniform type.
    """

    kind = ""

    def __init__(self, fn: Callable[..., InferenceRuntime]) -> None:
        self._fn = fn

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        return self._fn(config, model_path, recipe_config, environ)


#: Bundled kinds, resolved on first use so importing contracts loads no integration.
_BUILTIN_FACTORIES = {
    "inference_proxy": "reef.inference.http:InferenceProxyRuntimeFactory",
    "none": "reef.inference.null:NullRuntimeFactory",
    "remote_serving": "reef.inference.remote.runtime:RemoteServingRuntimeFactory",
    "tinker": "reef.inference.tinker:TinkerInferenceRuntimeFactory",
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
    """Build serving runtimes from config sections; injected factories shadow registered kinds."""

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
    ) -> InferenceRuntime:
        """The serving runtime of ``config``; ``RuntimeConfigError`` for an unknown type, and ``TypeError`` when
        the factory builds anything but an ``InferenceRuntime``. A training component it built is closed first."""
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
        runtime = factory(config, model_path, recipe_config or {}, values)
        if isinstance(runtime, TrainingComponent):
            runtime.close()
            raise TypeError(
                f"runtime type {runtime_type!r} built a training component: a runtime serves, and training.backend "
                "selects training"
            )
        if not isinstance(runtime, InferenceRuntime):
            raise TypeError(
                f"runtime type {runtime_type!r} must build an InferenceRuntime, not {type(runtime).__name__}"
            )
        return runtime


def training_component_factory_for(reference: str) -> TrainingComponentFactory:
    """The ``TrainingComponentFactory`` that the dotted ``reference`` (``package.module:factory``) names.

    The reference names a factory instance or a subclass, which is made with no
    arguments. ``RuntimeConfigError`` for another reference.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeConfigError(f"training component type {reference!r} must be 'package.module:factory'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeConfigError(f"cannot import training component type {reference!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, TrainingComponentFactory):
        candidate = candidate()
    if not isinstance(candidate, TrainingComponentFactory):
        raise RuntimeConfigError(f"training component type {reference!r} is not a TrainingComponentFactory")
    return candidate


def build_training_component(
    config: Mapping[str, Any], *, model_path: str, environ: Mapping[str, str] | None = None
) -> TrainingComponent:
    """The training component of ``config``, whose ``type`` names its factory (``training_component_factory_for``).

    ``TypeError`` when the factory builds anything but a ``TrainingComponent``; a serving runtime it built is
    shut down first.
    """
    reference = config.get("type")
    if not isinstance(reference, str) or not reference:
        raise RuntimeConfigError("a training component config needs its factory's dotted reference as type")
    factory = training_component_factory_for(reference)
    values = os.environ if environ is None else environ
    component = factory(factory.parse_config(config, values), model_path, values)
    if isinstance(component, InferenceRuntime):
        component.shutdown()
    if not isinstance(component, TrainingComponent):
        raise TypeError(f"training component type {reference!r} must build a TrainingComponent")
    return component


def close_runtimes(
    training_component: TrainingComponent | None, runtime: InferenceRuntime | None
) -> list[BaseException]:
    """Close a recipe's training component and inference runtime, and return their close errors in that order.

    Admission closes first, so that no new request starts. Then the training
    component closes, and then the runtime shuts down. A failed close does not
    stop the next one. ``None`` stands for an absent component or runtime.
    """
    if runtime is not None:
        runtime.close_admission()
    errors: list[BaseException] = []
    if training_component is not None:
        try:
            training_component.close()
        except BaseException as exc:
            errors.append(exc)
    if runtime is not None:
        try:
            runtime.shutdown()
        except BaseException as exc:
            errors.append(exc)
    return errors


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


def http_readiness_command(python: str, endpoint: str) -> list[str]:
    """Check HTTP readiness without a shell or an optional HTTP client."""
    return [
        python,
        "-c",
        "import sys, urllib.request; "
        "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=5).close()",
        endpoint,
    ]
