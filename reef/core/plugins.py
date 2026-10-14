"""The one form of a plugin that a deployment file names: a class reference and its configuration.

A plugin setting is either ``"package.module:ClassName"`` or a mapping with
``module`` (that reference) and an optional ``config`` mapping. The class must
inherit the plugin's abstract base class. Loading a setting imports and checks
the class only; the recipe builds the plugin later with the class method
``from_config``, so loading configuration allocates nothing.

Each plugin base declares its own ``from_config`` with the arguments that
plugin needs; :class:`Plugin` is their common base. :class:`ConfiguredPlugin`
is the base for plugins that need only their configuration.
:class:`PluginSettings` is the base of the dataclass a plugin's config parses into.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from reef.core.config import Settings
from reef.core.errors import ReefError


class PluginConfigError(ReefError, ValueError):
    """A plugin setting does not name a usable plugin class, or its configuration is invalid."""


@dataclass(frozen=True)
class PluginReference:
    """A checked plugin class and the configuration its ``from_config`` receives.

    A plugin base reads the class back as its own subclass with ``referenced_class``.
    """

    plugin_class: type[Plugin]
    config: Mapping[str, Any] = field(default_factory=dict)


def plugin_reference(value: object, base: type[Plugin], *, setting: str) -> PluginReference:
    """Read one plugin setting: import the class it names and check that it inherits ``base``."""
    if isinstance(value, str):
        reference: object = value
        config: object = {}
    elif isinstance(value, Mapping):
        unknown = sorted(str(key) for key in value if key not in ("module", "config"))
        if unknown:
            raise PluginConfigError(f"{setting} has unknown key(s) {', '.join(unknown)}; expected module and config")
        reference = value.get("module")
        config = value.get("config", {})
    else:
        raise PluginConfigError(f"{setting} must be 'package.module:ClassName' or a mapping with module and config")
    if not isinstance(reference, str):
        raise PluginConfigError(f"{setting}.module must be 'package.module:ClassName'")
    if not isinstance(config, Mapping):
        raise PluginConfigError(f"{setting}.config must be a mapping")
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise PluginConfigError(f"{setting} {reference!r} must be 'package.module:ClassName'")
    try:
        resolved = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise PluginConfigError(f"cannot import {setting} {reference!r}: {exc}") from exc
    if not isinstance(resolved, type) or not issubclass(resolved, base):
        raise PluginConfigError(f"{setting} {reference!r} must name a class that inherits {base.__name__}")
    if inspect.isabstract(resolved):
        raise PluginConfigError(f"{setting} {reference!r} names an abstract class")
    return PluginReference(resolved, dict(config))


class Plugin:
    """Mixin of every plugin base class a deployment file names.

    Each plugin base is an ABC that declares its own ``from_config`` with the
    arguments that plugin needs.
    """

    @classmethod
    def reference(cls, value: object, *, setting: str) -> PluginReference:
        """Read one setting that names a subclass of this plugin base."""
        return plugin_reference(value, cls, setting=setting)

    @classmethod
    def referenced_class(cls, reference: PluginReference) -> type[Self]:
        """The class that ``reference`` names, as a subclass of this plugin base."""
        if not issubclass(reference.plugin_class, cls):
            raise PluginConfigError(f"{reference.plugin_class.__name__} does not inherit {cls.__name__}")
        return reference.plugin_class


class ConfiguredPlugin(Plugin):
    """A plugin built from its configuration alone.

    The default ``from_config`` builds the class with no arguments and accepts
    no configuration. A plugin with settings overrides it.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        if config:
            raise PluginConfigError(f"{cls.__name__} takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls()

    @classmethod
    def from_setting(cls, value: object, *, setting: str) -> Self:
        """Build the plugin a setting names; Python callers may pass a built instance instead."""
        if isinstance(value, cls):
            return value
        reference = cls.reference(value, setting=setting)
        return cls.referenced_class(reference).from_config(reference.config)


class PluginSettings(Settings):
    """Base of the settings dataclass that a plugin's config parses into, in the plugin's ``from_config``."""

    @classmethod
    def parse(cls, config: Mapping[str, Any], *, plugin_class: type[Plugin]) -> Self:
        """Parse the config of ``plugin_class``, which names the config in errors; a bad config is a ``PluginConfigError``."""
        if cls is NoSettings and config:
            names = ", ".join(sorted(map(str, config)))
            raise PluginConfigError(f"{plugin_class.__name__} takes no config, got {names}")
        try:
            return cls.from_config(config, setting=plugin_class.__name__)
        except ValueError as exc:
            raise PluginConfigError(str(exc)) from exc


@dataclass(frozen=True)
class NoSettings(PluginSettings):
    """The settings of a plugin that takes no config. An empty config parses into it; any key is refused."""


__all__ = [
    "ConfiguredPlugin",
    "NoSettings",
    "Plugin",
    "PluginConfigError",
    "PluginReference",
    "PluginSettings",
    "plugin_reference",
]
