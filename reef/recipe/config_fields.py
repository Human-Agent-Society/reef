"""Recipe-owned declarations backed by the shared component argument parser.

Config wins over a declared environment fallback and the dataclass default.
Range and cross-field validation stay in the recipe's ``__post_init__``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, replace
from typing import Any

from reef.core.config import ConfigArgument, config_arguments, config_option, parse_config_values
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


@dataclass(frozen=True)
class ConfigField:
    """Compatibility view of a recipe's shared argument definition."""

    argument: ConfigArgument

    @property
    def name(self) -> str:
        return self.argument.name

    @property
    def env(self) -> str | None:
        return self.argument.env

    @property
    def default(self) -> Any:
        return self.argument.default

    def parse(self, value: Any, label: str) -> Any:
        argument = replace(self.argument, path=tuple(label.split(".")))
        try:
            return parse_config_values((argument,), {self.name: value})[self.name]
        except ValueError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def resolve(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> Any:
        values = {self.name: config[self.name]} if self.name in config else {}
        try:
            return parse_config_values((self.argument,), values, environ=environ)[self.name]
        except ValueError as exc:
            raise RecipeConfigError(str(exc)) from exc


def parse_int(value: Any, label: str) -> int:
    """Compatibility entrypoint for artifact cadence; use the shared parser."""
    argument = ConfigArgument(label, tuple(label.split(".")), "int", False, None, "")
    try:
        return parse_config_values((argument,), {label: value})[label]
    except ValueError as exc:
        raise RecipeConfigError(str(exc)) from exc


def recipe_config_fields(recipe_class: type) -> dict[str, ConfigField]:
    """Read only declarations belonging to the selected recipe class."""
    return {argument.name: ConfigField(argument) for argument in config_arguments(recipe_class)}


def resolve_config_field_values(
    recipe_class: type,
    config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve a recipe's data section using the shared typed parser."""
    if not isinstance(config, Mapping):
        raise RecipeConfigError("recipe data must be an object")
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
