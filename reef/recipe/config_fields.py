"""Recipe config fields: one dataclass field defines one setting.

A recipe setting is declared through :func:`config_field` instead of a bare
default::

    batch_size: int = config_field(1, env="REEF_SAO_BATCH_SIZE")

That declaration defines:

- the YAML key, using the field name in either flat ``reef.<name>`` settings
  or a recipe config's ``data.<name>`` section;
- an optional environment-variable fallback;
- type-aware parsing from the field annotation (``int``, ``float``, ``bool``,
  or ``str``).

Precedence is config over environment over the field default. Range and
cross-field validation stay in the recipe's ``__post_init__`` so the same
validation also covers direct construction.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from reef.recipe.errors import RecipeConfigError

_CONFIG_FIELD_METADATA_KEY = "reef_config_field"
_MISSING = object()

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def config_field(default: Any, *, env: str | None = None) -> Any:
    """Declare a typed recipe setting.

    ``default`` is the dataclass default. ``env`` optionally names the
    environment variable used when config does not supply a value. The field
    annotation must be ``int``, ``float``, ``bool``, or ``str`` and determines
    how config and environment values are parsed.
    """
    return dataclasses.field(
        default=default,
        metadata={_CONFIG_FIELD_METADATA_KEY: _ConfigFieldMarker(env=env)},
    )


@dataclasses.dataclass(frozen=True)
class _ConfigFieldMarker:
    """Metadata stored by :func:`config_field`."""

    env: str | None = None


@dataclasses.dataclass(frozen=True)
class ConfigField:
    """One resolved recipe config field and its value parser."""

    name: str
    env: str | None
    default: Any
    parse: Callable[[Any, str], Any]

    def resolve(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> Any:
        """Resolve config first, then the environment, then the default."""
        value = config.get(self.name, _MISSING)
        if value is not _MISSING:
            return self.parse(value, self.name)
        if self.env is not None:
            raw = environ.get(self.env)
            if raw is not None:
                return self.parse(raw, self.env)
        return self.default


def parse_int(value: Any, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise RecipeConfigError(f"{label} must be an integer, got {value!r}")


def _parse_float(value: Any, label: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise RecipeConfigError(f"{label} must be a number, got {value!r}")


def _parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise RecipeConfigError(f"{label} must be a boolean (true/false), got {value!r}")


def _parse_str(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    raise RecipeConfigError(f"{label} must be a string, got {value!r}")


_PARSERS: dict[type, Callable[[Any, str], Any]] = {
    int: parse_int,
    float: _parse_float,
    bool: _parse_bool,
    str: _parse_str,
}

_PARSER_NAMES: dict[str, type] = {parser_type.__name__: parser_type for parser_type in _PARSERS}


def recipe_config_fields(recipe_class: type) -> dict[str, ConfigField]:
    """Return the config fields declared by a recipe class, keyed by name."""
    if not dataclasses.is_dataclass(recipe_class):
        raise TypeError(f"{recipe_class.__name__} is not a dataclass; config fields are dataclass fields")
    config_fields: dict[str, ConfigField] = {}
    for field in dataclasses.fields(recipe_class):
        marker = field.metadata.get(_CONFIG_FIELD_METADATA_KEY)
        if marker is None:
            continue
        # Resolve this annotation alone. typing.get_type_hints over the whole
        # class fails on the dataclass KW_ONLY sentinel under Python 3.10.
        annotation: Any = field.type
        if isinstance(annotation, str):
            annotation = _PARSER_NAMES.get(annotation, annotation)
        parser = _PARSERS.get(annotation)
        if parser is None:
            supported = ", ".join(parser_type.__name__ for parser_type in _PARSERS)
            raise TypeError(
                f"{recipe_class.__name__}.{field.name} is a config field annotated {annotation!r}; "
                f"config fields must be annotated with one of: {supported}"
            )
        config_fields[field.name] = ConfigField(
            name=field.name,
            env=marker.env,
            default=field.default,
            parse=parser,
        )
    return config_fields


def resolve_config_field_values(
    recipe_class: type,
    config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve every declared config field from config and the environment."""
    config_fields = recipe_config_fields(recipe_class)
    unknown = sorted(key for key in config if key not in config_fields)
    if unknown:
        known = ", ".join(sorted(config_fields)) or "none"
        raise RecipeConfigError(
            f"{recipe_class.__name__} does not consume config key(s) {', '.join(map(repr, unknown))}; "
            f"known {recipe_class.__name__} config fields: {known}"
        )
    return {name: value.resolve(config, environ) for name, value in config_fields.items()}
