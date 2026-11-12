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

A setting that also accepts short names resolves its value with
``resolve_plugin_class``: a built-in name, then an installed entry point, then
the reference. ``import_plugin_class`` reads the reference.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Self, TypeVar

from reef.core.config import Settings
from reef.core.errors import ConfigError


BaseT = TypeVar("BaseT")


class PluginConfigError(ConfigError):
    """A plugin setting does not name a usable plugin class, or its configuration is invalid."""


@dataclass(frozen=True)
class PluginReference:
    """A checked plugin class and the configuration its ``from_config`` receives.

    A plugin base reads the class back as its own subclass with ``referenced_class``.
    """

    plugin_class: type[Plugin]
    config: Mapping[str, Any] = field(default_factory=dict)


def plugin_reference(
    value: object, base: type[Plugin], *, setting: str, builtins: Mapping[str, type[Plugin]] | None = None
) -> PluginReference:
    """Read one plugin setting: import the class it names and check that it inherits ``base``.

    With ``builtins``, the setting may also name one of their keys instead of a reference.
    """
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
    if builtins is None:
        plugin_class = import_plugin_class(reference, base, setting=setting)
    else:
        plugin_class = resolve_plugin_class(reference, builtins=builtins, base=base, setting=setting)
    return PluginReference(plugin_class, dict(config))


def import_plugin_class(reference: str, base: type[BaseT], *, setting: str) -> type[BaseT]:
    """Import the class that the reference ``package.module:ClassName`` names.

    The result is a concrete class that inherits ``base``. Raise ``PluginConfigError``, which names ``setting``,
    for another syntax, a failed import, and a value that is not such a class.
    """
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
    return resolved


def plugin_class_reference(
    name: str,
    *,
    builtins: Mapping[str, type[BaseT] | str],
    entry_point_group: str | None = None,
    setting: str,
) -> type[BaseT] | str:
    """The class, or the ``package.module:ClassName`` reference of the class, that a setting's value ``name`` selects.

    The order is fixed:

    1. A key of ``builtins`` gives its value. A built-in may be written as a reference, so that reading the table
       imports no integration.
    2. With an ``entry_point_group``, the installed entry point of that group and name gives its value.
    3. Any other ``name`` that contains ``:`` is itself the reference.

    This imports nothing, so a host that cannot import the class can still check the name. Raise
    ``PluginConfigError``, which names ``setting`` and the built-in names, for a name that is none of the three and
    for two entry points of one name.
    """
    if not isinstance(name, str) or not name:
        raise PluginConfigError(f"{setting} must be a non-empty string")
    builtin = builtins.get(name)
    if builtin is not None:
        return builtin
    accepted = sorted(builtins)
    if entry_point_group is not None:
        accepted.append(f"an installed {entry_point_group} entry point")
        matches = tuple(entry_points(group=entry_point_group, name=name))
        if len(matches) > 1:
            raise PluginConfigError(f"ambiguous {setting} {name!r}: {len(matches)} {entry_point_group} entry points")
        if matches:
            return matches[0].value
    if ":" not in name:
        choices = ", ".join((*accepted, "'package.module:ClassName'"))
        raise PluginConfigError(f"unknown {setting} {name!r}; use one of: {choices}")
    return name


def resolve_plugin_class(
    name: str,
    *,
    builtins: Mapping[str, type[BaseT] | str],
    entry_point_group: str | None = None,
    base: type[BaseT],
    setting: str,
) -> type[BaseT]:
    """The class that a setting's value ``name`` selects: a built-in name, then an entry point, then a reference.

    ``plugin_class_reference`` defines the order. The result is a concrete class that inherits ``base``. Raise
    ``PluginConfigError``, which names ``setting``, for a name that selects nothing and for a reference that
    ``import_plugin_class`` refuses.
    """
    selected = plugin_class_reference(name, builtins=builtins, entry_point_group=entry_point_group, setting=setting)
    if isinstance(selected, type):
        return selected
    return import_plugin_class(selected, base, setting=f"{setting} {name!r}" if selected != name else setting)


class Plugin:
    """Mixin of every plugin base class a deployment file names.

    Each plugin base is an ABC that declares its own ``from_config`` with the
    arguments that plugin needs.
    """

    @classmethod
    def reference(
        cls, value: object, *, setting: str, builtins: Mapping[str, type[Self]] | None = None
    ) -> PluginReference:
        """Read one setting that names a subclass of this plugin base, by reference or by a key of ``builtins``."""
        return plugin_reference(value, cls, setting=setting, builtins=builtins)

    @classmethod
    def resolve_class(
        cls,
        name: str,
        *,
        setting: str,
        builtins: Mapping[str, type[Self] | str] | None = None,
        entry_point_group: str | None = None,
    ) -> type[Self]:
        """The subclass of this plugin base that a setting's value ``name`` selects.

        Without ``builtins`` and ``entry_point_group``, ``name`` is a ``package.module:ClassName`` reference
        (``import_plugin_class``). With either, ``resolve_plugin_class`` resolves it.
        """
        if builtins is None and entry_point_group is None:
            return import_plugin_class(name, cls, setting=setting)
        return resolve_plugin_class(
            name, builtins=builtins or {}, entry_point_group=entry_point_group, base=cls, setting=setting
        )

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
    def from_setting(cls, value: object, *, setting: str, default: PluginReference | None = None) -> Self:
        """Build the plugin a setting names, or ``default`` when the setting is absent.

        Python callers may pass a built instance instead.
        """
        if isinstance(value, cls):
            return value
        reference = default if value is None and default is not None else cls.reference(value, setting=setting)
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
    "import_plugin_class",
    "plugin_class_reference",
    "plugin_reference",
    "resolve_plugin_class",
]
