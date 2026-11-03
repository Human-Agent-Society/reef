"""Serving runtime selection: the ``type`` table that turns a ``runtime`` config section into a serving runtime.

Bundled kinds resolve on first use, so importing this module loads no
integration. The helpers at the end read the keys of a ``runtime`` section.
"""

from __future__ import annotations

import importlib
import os
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from reef.core.config import ConfiguredFactory
from reef.core.errors import RuntimeConfigError
from reef.inference.interfaces import InferenceRuntime


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
        the factory builds anything but an ``InferenceRuntime``."""
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
        if not isinstance(runtime, InferenceRuntime):
            raise TypeError(
                f"runtime type {runtime_type!r} must build an InferenceRuntime, not {type(runtime).__name__}: a "
                "runtime serves, and training.backend selects training"
            )
        return runtime


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
