"""Runtime kind table and config-backed construction of runtimes.

The same two axes as :mod:`reef.recipe.registry`, deliberately separate:

- The module-level *kind table* maps a config ``type`` to a runtime factory
  (``RuntimeFactory`` ABC, ``@register_runtime_kind`` class decorator,
  ``runtime_factory_for`` resolver). Registration is process-wide and happens
  at import time; the bundled adapters register themselves in
  :mod:`reef.runtime.adapters`.
- :class:`RuntimeRegistry` is an *instance* resolver that builds runtimes
  from config mappings, resolving injected factories first, then registered
  kinds, then dotted references.

Subclass :class:`RuntimeFactory`, set the class attribute ``kind``, implement
:meth:`RuntimeFactory.__call__`, and decorate with ``@register_runtime_kind``::

    @register_runtime_kind
    class MyFactory(RuntimeFactory):
        kind = "my-runtime"

        def __call__(self, config, model_path, recipe_config, environ):
            ...

External factories are either registered at runtime via
:func:`register_runtime_factory` or named as a dotted
``"package.module:factory"`` reference resolved by
:func:`runtime_factory_for`; plain callables from a dotted reference are
wrapped by :class:`_CallableRuntimeFactory`.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from reef.core.errors import ReefError
from reef.runtime.base import InferenceRuntime


class RuntimeConfigError(ReefError):
    """Raised when runtime configuration is invalid or required keys are missing."""


class RuntimeFactory(ABC):
    """Base class for runtime factories.

    Required (subclass must set):
        ``kind`` — canonical kind name used in config ``type``.

    Implement:
        ``__call__`` — receive ``(config, model_path, recipe_config, environ)``
        and return a ready :class:`InferenceRuntime`.
    """

    kind: str

    @abstractmethod
    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        """Build a ready runtime instance from a config section."""


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


_runtime_kinds: dict[str, RuntimeFactory] = {}
"""Bundled runtime factories registered at import time by
:func:`register_runtime_kind`.

Read by :func:`runtime_factory_for`; external factories are registered at
runtime through :func:`register_runtime_factory`.
"""


def register_runtime_kind(cls: type[RuntimeFactory]) -> type[RuntimeFactory]:
    """Class decorator: instantiate and register a bundled runtime factory.

    Each adapter module applies this to its ``RuntimeFactory`` subclass so
    the kind table is populated at import time — no explicit dict entry.
    """
    instance = cls()
    kind = instance.kind
    if kind in _runtime_kinds:
        raise ValueError(f"runtime kind {kind!r} is already registered")
    _runtime_kinds[kind] = instance
    return cls


def register_runtime_factory(factory: RuntimeFactory) -> RuntimeFactory:
    """Register an external runtime factory (module-level convenience)."""
    if not isinstance(factory, RuntimeFactory):
        raise TypeError(f"a runtime factory registers a RuntimeFactory, got {type(factory).__name__}")
    kind = factory.kind
    if kind in _runtime_kinds:
        raise ValueError(f"runtime kind {kind!r} is already registered")
    _runtime_kinds[kind] = factory
    return factory


def runtime_kinds() -> tuple[str, ...]:
    return tuple(_runtime_kinds)


def runtime_factory_for(kind: str) -> RuntimeFactory | None:
    """Return the runtime factory for ``kind``.

    A kind is either a registered name or a dotted reference
    ``package.module:factory_name`` to a factory reef does not bundle —
    the config path's way of serving deployment-specific runtimes without a
    hand-rolled service assembly.
    """
    registered = _runtime_kinds.get(kind)
    if registered is not None or ":" not in kind:
        return registered
    module_name, _, attribute = kind.partition(":")
    if not module_name or not attribute:
        raise RuntimeConfigError(f"dotted runtime kind {kind!r} must be 'package.module:factory_name'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeConfigError(f"cannot import runtime kind {kind!r}: {exc}") from exc
    if isinstance(candidate, RuntimeFactory):
        return candidate
    if callable(candidate):
        return _CallableRuntimeFactory(candidate)
    raise RuntimeConfigError(f"runtime kind {kind!r} is not callable")


class RuntimeRegistry:
    def __init__(self, factories: Mapping[str, RuntimeFactory] | None = None) -> None:
        self._factories: dict[str, RuntimeFactory] = dict(factories or {})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._factories) | set(_runtime_kinds)))

    def build(
        self,
        config: Mapping[str, Any],
        *,
        model_path: str,
        recipe_config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> InferenceRuntime:
        runtime_type = config.get("type")
        if not isinstance(runtime_type, str) or not runtime_type:
            raise RuntimeConfigError("runtime.type must be a non-empty string")
        factory = self._factories.get(runtime_type)
        if factory is None:
            factory = runtime_factory_for(runtime_type)
        if factory is None:
            available = ", ".join(self.names)
            raise RuntimeConfigError(f"unknown runtime type {runtime_type!r}; available runtimes: {available}")
        recipe_context = {} if recipe_config is None else recipe_config
        return factory(config, model_path, recipe_context, os.environ if environ is None else environ)


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
    value = config.get(name)
    if value is not None:
        if not isinstance(value, str) or not value:
            raise RuntimeConfigError(f"runtime.{name} must be a non-empty string")
        return value
    configured_env = config.get(env_name)
    if configured_env is None:
        return None
    if not isinstance(configured_env, str) or not configured_env:
        raise RuntimeConfigError(f"runtime.{env_name} must be a non-empty string")
    return environ.get(configured_env)


__all__ = [
    "RuntimeConfigError",
    "RuntimeFactory",
    "RuntimeRegistry",
    "config_secret",
    "config_string",
    "register_runtime_factory",
    "register_runtime_kind",
    "runtime_factory_for",
    "runtime_kinds",
]
