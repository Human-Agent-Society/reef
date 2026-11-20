"""Recipe-owned declarations backed by the shared component argument parser.

``config_field`` declares one value of the recipe's ``data`` section. Config
wins over a declared environment fallback and the dataclass default.
``section_field`` declares a field that holds one config section, parsed into
the field's ``Settings`` dataclass. ``reader_field`` declares a field whose
value a ``ConfigReader`` reads from the config, for a value that is neither:
one that several sections give, a file that a section names, or a value that
needs the environment. The loader (``reef.recipe.loading``) reads the three
declarations. Range and cross-field validation stay in the recipe's
``__post_init__``.
"""

from __future__ import annotations

import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import MISSING, field, fields
from typing import Any, get_type_hints

from reef.core.config import Settings, config_arguments, config_option, parse_config_values
from reef.core.errors import InternalError, ReefError
from reef.recipe.errors import RecipeConfigError


def config_field(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
    env: str | None = None,
    help: str = "",
    allow_nonfinite: bool = True,
) -> Any:
    """Declare a recipe field with the same types and parser as public settings.

    Legacy recipe floats permit infinity (for example an unbounded score
    threshold). Set ``allow_nonfinite=False`` for a finite-only field.
    """
    return config_option(default, default_factory=default_factory, env=env, help=help, allow_nonfinite=allow_nonfinite)


#: The metadata key of a ``section_field``: the dotted path of its config section.
SECTION_PATH = "recipe_config_section"
#: The metadata key of a ``section_field``: the error message for a config without the section.
SECTION_REQUIRED = "recipe_config_section_required"


def section_field(path: str, *, required: str) -> Any:
    """Declare a required recipe field that holds the config section at ``path``, parsed.

    ``path`` is dotted from the top of the flat recipe config: ``evolution.evaluation`` is the ``evaluation``
    object inside the ``evolution`` section. The field's annotation is the ``Settings`` dataclass that parses the
    section. ``required`` is the message of the ``RecipeConfigError`` for a config whose top-level section is
    missing; a missing object inside a section is read as empty.
    """
    return field(metadata={SECTION_PATH: path, SECTION_REQUIRED: required})


def resolve_section_field_values(recipe_class: type, config: Mapping[str, Any]) -> dict[str, Settings]:
    """Parse each ``section_field`` of ``recipe_class`` from ``config``; the result maps field names to settings.

    Each section is parsed with ``Settings.from_config`` of the field's annotation, named by its path in errors.
    A key that another declared path names inside a section is left out of the outer section, so each key has
    one reader. Raise ``RecipeConfigError`` for a missing top-level section and for a value that a settings
    dataclass refuses.
    """
    declared = [item for item in fields(recipe_class) if SECTION_PATH in item.metadata]
    paths = [item.metadata[SECTION_PATH] for item in declared]
    values: dict[str, Settings] = {}
    for item in declared:
        path = item.metadata[SECTION_PATH]
        # Resolve only this declaration, as ``config_arguments`` does.
        owner = next(base for base in recipe_class.__mro__ if item.name in vars(base).get("__annotations__", {}))
        declaration = types.SimpleNamespace(__annotations__={item.name: item.type})
        settings_type = get_type_hints(declaration, globalns=vars(sys.modules[owner.__module__]))[item.name]
        if not (isinstance(settings_type, type) and issubclass(settings_type, Settings)):
            raise TypeError(
                f"{recipe_class.__name__}.{item.name} is a section_field: its type must be a Settings class"
            )
        keys = path.split(".")
        section: Any = config.get(keys[0])
        if not isinstance(section, Mapping):
            raise RecipeConfigError(item.metadata[SECTION_REQUIRED])
        for key in keys[1:]:
            section = section.get(key, {}) if isinstance(section, Mapping) else section
        if isinstance(section, Mapping):
            inner = {other.rsplit(".", 1)[1] for other in paths if "." in other and other.rsplit(".", 1)[0] == path}
            section = {key: value for key, value in section.items() if key not in inner}
        try:
            values[item.name] = settings_type.from_config(section, setting=path)
        except (TypeError, ValueError) as exc:
            raise RecipeConfigError(str(exc)) from exc
    return values


#: The metadata key of a ``reader_field``: its ``ConfigReader``.
FIELD_READER = "recipe_config_reader"


class ConfigReader(ABC):
    """Reads the value of one recipe field from the flat recipe config.

    Values: a reader holds no state. ``read(config, environ, sections)`` returns the value of the field: ``config``
    is the flat recipe config after the class's ``section_defaults``, ``environ`` the environment, and ``sections``
    the parsed ``section_field`` values of the class by field name. The result is configuration: a value that the
    config, the environment and the files that the config names decide, and never an object that runs. ``read``
    raises ``ValueError`` or a ``ReefError`` for a config that gives no such value.
    Purpose: ``reader_field`` declares one on a recipe field, and the loader calls ``read`` once for each load
    (``resolve_reader_field_values``).
    """

    @abstractmethod
    def read(self, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]) -> object:
        """The value of the field for ``config``."""


def reader_field(reader: ConfigReader, *, default: Any = MISSING, default_factory: Any = MISSING) -> Any:
    """Declare a recipe field whose value ``reader`` reads from the flat recipe config.

    ``default`` or ``default_factory`` is the value of a recipe that code builds without the loader.
    """
    if not isinstance(reader, ConfigReader):
        raise TypeError("reader_field takes a ConfigReader")
    return field(default=default, default_factory=default_factory, metadata={FIELD_READER: reader})


def resolve_reader_field_values(
    recipe_class: type, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]
) -> dict[str, object]:
    """Read each ``reader_field`` of ``recipe_class`` from ``config``; the result maps field names to values.

    Raise ``RecipeConfigError`` with the reader's message for a ``TypeError``, a ``ValueError`` or a ``ReefError``
    that is not an ``InternalError``.
    """
    values: dict[str, object] = {}
    for item in fields(recipe_class):
        reader = item.metadata.get(FIELD_READER)
        if reader is None:
            continue
        try:
            values[item.name] = reader.read(config, environ, sections)
        except (InternalError, RecipeConfigError):
            raise
        except (TypeError, ValueError, ReefError) as exc:
            raise RecipeConfigError(str(exc)) from exc
    return values


def resolve_config_field_values(
    recipe_class: type,
    config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve a recipe's data section using the shared typed parser.

    A key is read in either spelling: a deployment file writes ``groups-per-step``
    where the field is ``groups_per_step``; a field named in both is refused.
    """
    if not isinstance(config, Mapping):
        raise RecipeConfigError("recipe data must be an object")
    folded: dict[str, Any] = {}
    for key, value in config.items():
        name = str(key).replace("-", "_")
        if name in folded:
            raise RecipeConfigError(f"{recipe_class.__name__} config names the field {name!r} twice")
        folded[name] = value
    config = folded
    arguments = config_arguments(recipe_class)
    known = {argument.name for argument in arguments}
    unknown = sorted(set(config) - known, key=str)
    if unknown:
        raise RecipeConfigError(
            f"{recipe_class.__name__} does not consume config key(s) {', '.join(map(repr, unknown))}; "
            f"known {recipe_class.__name__} config fields: {', '.join(sorted(known)) or 'none'}"
        )
    try:
        return parse_config_values(arguments, config, environ=environ)
    except ValueError as exc:
        raise RecipeConfigError(str(exc)) from exc
