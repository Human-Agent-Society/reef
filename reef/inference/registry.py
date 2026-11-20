"""Serving runtime selection: the ``type`` table that turns a ``runtime`` config section into a serving runtime.

Bundled kinds are imported on first use, so importing this module loads no
integration. The helpers at the end read the keys of a ``runtime`` section.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.config import ConfiguredFactory
from reef.core.errors import RuntimeConfigError
from reef.core.plugins import PluginConfigError, resolve_plugin_class
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


#: Bundled kinds, as references so that importing this module loads no integration.
BUILTIN_FACTORIES = {
    "inference_proxy": "reef.inference.http:InferenceProxyRuntimeFactory",
    "none": "reef.inference.null:NullRuntimeFactory",
    "remote_serving": "reef.inference.remote.runtime:RemoteServingRuntimeFactory",
    "tinker": "reef.inference.tinker:TinkerInferenceRuntimeFactory",
}


def runtime_factory_for(kind: str) -> RuntimeFactory:
    """The runtime factory that the ``runtime.type`` value ``kind`` selects.

    A kind is a bundled name or a ``package.module:ClassName`` reference to a
    ``RuntimeFactory`` subclass that Reef does not bundle. The class is made
    with no arguments. Raise ``PluginConfigError`` for any other kind.
    """
    factory_class = resolve_plugin_class(
        kind,
        builtins=BUILTIN_FACTORIES,
        base=RuntimeFactory,  # type: ignore[type-abstract]  # the base is a value; the result is concrete
        setting="runtime type",
    )
    return factory_class()


class RuntimeRegistry:
    """Build serving runtimes from config sections; injected factories shadow the kinds of ``runtime_factory_for``."""

    def __init__(self, factories: Mapping[str, RuntimeFactory] | None = None) -> None:
        self._factories: dict[str, RuntimeFactory] = dict(factories or {})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._factories) | set(BUILTIN_FACTORIES)))

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
        factory = self._factories.get(runtime_type)
        if factory is None:
            try:
                factory = runtime_factory_for(runtime_type)
            except PluginConfigError as exc:
                raise RuntimeConfigError(str(exc)) from exc
        values = os.environ if environ is None else environ
        runtime = factory(factory.parse_config(config, values), model_path, recipe_config or {}, values)
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
