"""The one loader of recipes: a flat recipe config in, a recipe value out.

A recipe class declares its configuration and does not load itself.
:func:`load_recipe` reads the declarations of a class and makes its value:

- ``config_field`` fields from the ``data`` section, with their environment
  fallbacks (``reef.recipe.config_fields.resolve_config_field_values``);
- ``section_field`` fields from their sections
  (``resolve_section_field_values``), over the class's ``section_defaults``;
- ``reader_field`` fields from their readers (``resolve_reader_field_values``);
- the ``pipeline`` section over the class's ``pipeline_spec`` and the
  ``activation`` section, which every recipe has;
- ``refused_sections``, which fail the load when the config sets them;
- the components of a composite, each loaded the same way.

:func:`build_recipe` loads the class that a reference names, and
:func:`named_recipe_config` reads the config of the preset that a deployment
names. :func:`weight_training_selection` answers from the config alone whether a
deployment trains weights. No recipe class decides that for the launcher.

The loader builds no runtime and no training component, and a recipe holds
none (RFC #1390): the service assembly builds them
(``reef.service.assembly.load_deployment``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.plugins import PluginConfigError
from reef.core.values import deep_merge
from reef.evaluation.activation import activation_reference
from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.composite import (
    RESERVED_COMPONENT_NAMES,
    CompositeRecipe,
    component_recipe_config,
    configured_training_mode,
)
from reef.recipe.config import load_recipe_config, recipe_config_from_mapping
from reef.recipe.config_fields import (
    resolve_config_field_values,
    resolve_reader_field_values,
    resolve_section_field_values,
)
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import recipe_class_for
from reef.train.pipeline import PipelineSettings

#: The names that a deployment may give a preset: the stem of a YAML file.
RECIPE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_recipe(
    recipe_class: type[Recipe],
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    field_values: Mapping[str, Any] | None = None,
) -> Recipe:
    """The value of ``recipe_class`` for the flat recipe ``config`` and the environment ``environ``.

    ``config`` is the internal layout that ``reef.recipe.config`` produces: the class's fields under ``data``, and
    its sections at the top. ``field_values`` are the ``config_field`` values when the caller resolved them before,
    for example to connect the training component first; else the loader resolves them from ``data`` and
    ``environ``.

    The class's own ``__post_init__`` checks the relations between the values. Raise ``RecipeConfigError`` for an
    invalid config.
    """
    if field_values is None:
        field_values = resolve_config_field_values(recipe_class, config.get("data", {}), environ)
    if issubclass(recipe_class, CompositeRecipe):
        return load_composite_recipe(recipe_class, config, environ, field_values)
    try:
        if recipe_class.section_defaults:
            config = deep_merge(recipe_class.section_defaults, config)
        for section, message in recipe_class.refused_sections.items():
            if config.get(section):
                raise RecipeConfigError(message)
        sections = resolve_section_field_values(recipe_class, config)
        arguments: dict[str, Any] = {
            **sections,
            **resolve_reader_field_values(recipe_class, config, environ, sections),
        }
        try:
            if config.get("pipeline") is not None:
                arguments["pipeline_stages"] = PipelineSettings.from_setting(
                    config["pipeline"], spec=recipe_class.pipeline_spec(), environ=environ
                )
            if config.get("activation") is not None:
                arguments["activation"] = activation_reference(config["activation"])
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc
        return recipe_class(**arguments, **field_values)
    except RecipeConfigError:
        raise
    except ValueError as exc:
        raise RecipeConfigError(f"invalid {recipe_class.__name__} configuration: {exc}") from exc


def load_composite_recipe(
    recipe_class: type[CompositeRecipe],
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    field_values: Mapping[str, Any],
) -> CompositeRecipe:
    """The composite of ``recipe_class`` whose ``components`` section names each component's recipe.

    Each component is loaded as a file's ``recipe`` section is, with the deployment's model. A component without a ``training_mode`` takes the composite's. ``field_values`` are the composite's
    own ``config_field`` values. Raise ``RecipeConfigError`` for a config without components, for a reserved
    component name, and for components that run different training modes.
    """
    raw = config.get("components")
    if not isinstance(raw, Mapping) or not raw:
        raise RecipeConfigError("a composite recipe config requires a non-empty 'components' object")
    for component in raw:
        if component in RESERVED_COMPONENT_NAMES:
            raise RecipeConfigError(
                f"components.{component} is a metric prefix experiment tracking reserves "
                f"({', '.join(RESERVED_COMPONENT_NAMES)}); rename the component"
            )
    configured_mode = configured_training_mode(config)
    components: dict[str, Recipe] = {}
    for component, section in raw.items():
        settings = component_recipe_config(str(component), section)
        settings["model"] = dict(config.get("model", {}))
        if configured_mode is not None and configured_training_mode(settings) is None:
            # The composite's mode is every component's mode: a component sets its own only to disagree.
            settings["data"]["training_mode"] = configured_mode
        settings = recipe_config_from_mapping(settings)
        components[str(component)] = build_recipe(settings["implementation"], environ, config=settings)
    modes = {recipe.training_mode for recipe in components.values()}
    if len(modes) != 1:
        raise RecipeConfigError(f"components run different training modes {sorted(modes)}; they share one")
    values = dict(field_values)
    if configured_mode is None:
        values["training_mode"] = next(iter(modes))
    if config.get("activation") is not None:
        # One rule activates the artifacts of every component.
        try:
            values["activation"] = activation_reference(config["activation"])
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc
    try:
        return recipe_class(components=components, **values)
    except RecipeConfigError:
        raise
    except ValueError as exc:
        raise RecipeConfigError(f"invalid {recipe_class.__name__} configuration: {exc}") from exc


def weight_training_selection(
    recipe_class: type[Recipe], config: Mapping[str, Any]
) -> tuple[type[WeightTrainingRecipe], Mapping[str, Any]] | None:
    """The weight training recipe class that a deployment of ``recipe_class`` trains with, and that class's config.

    Read from the flat recipe ``config`` alone, before anything is constructed: the service decides from it
    whether to connect a training component and to launch a training backend. A weight training class selects
    itself with ``config``. A composite selects its one component that trains weights, with that component's
    flat config. ``None`` for a deployment that trains no weights.

    A snapshot loads one component into a runtime, so a composite with two weight training components raises
    ``RecipeConfigError`` here, before any runtime is connected.
    """
    if issubclass(recipe_class, WeightTrainingRecipe):
        return recipe_class, config
    if not issubclass(recipe_class, CompositeRecipe):
        return None
    components = config.get("components")
    if not isinstance(components, Mapping):
        return None
    selected: list[tuple[str, tuple[type[WeightTrainingRecipe], Mapping[str, Any]]]] = []
    for name, section in components.items():
        settings = component_recipe_config(str(name), section)
        component_class = recipe_class_for(settings["implementation"])
        selection = None if component_class is None else weight_training_selection(component_class, settings)
        if selection is not None:
            selected.append((str(name), selection))
    if len(selected) > 1:
        raise RecipeConfigError(
            f"components {[name for name, _ in selected]} all train weights; a snapshot loads one component"
        )
    return selected[0][1] if selected else None


def build_recipe(
    implementation: str,
    environ: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Recipe:
    """The recipe of the class that ``implementation`` names, loaded by :func:`load_recipe`.

    ``environ`` defaults to the process environment and ``config`` to the empty config. Raise ``ValueError`` for a
    name that is no class reference.
    """
    recipe_class = recipe_class_for(implementation)
    if recipe_class is None:
        raise ValueError(f"unknown recipe reference {implementation!r}")
    return load_recipe(recipe_class, config or {}, os.environ if environ is None else environ)


def named_recipe_config(
    name: str, environ: Mapping[str, str], *, config_directory: str | Path | None = None
) -> dict[str, Any] | None:
    """The flat recipe config of the recipe that a deployment names by its public name.

    ``name`` is a YAML preset ``<name>.yaml`` under ``config_directory``
    (defaulting to ``REEF_RECIPE_CONFIG_DIR`` in ``environ``), or the reserved
    core name ``recipe``. Presets are deployment data (see
    ``docs/reference/configuration.rst``); Reef bundles none. The result is
    the preset's config (``load_recipe_config``), or ``None`` for the core
    name ``recipe`` without a preset file. Dotted references are not names:
    they are operator configuration for :func:`build_recipe`, so a name never
    triggers an import.

    Raise ``RecipeConfigError`` for an invalid name and for a name with no preset.
    """
    if not RECIPE_NAME.fullmatch(name):
        raise RecipeConfigError(f"invalid recipe name {name!r}")
    configured = config_directory or environ.get("REEF_RECIPE_CONFIG_DIR") or None
    directory = None if configured is None else Path(configured)
    path = None if directory is None else directory / f"{name}.yaml"
    if path is not None and path.exists():
        return load_recipe_config(path)
    if name == "recipe":
        return None
    available = {"recipe"}
    if directory is not None:
        available.update(preset.stem for preset in directory.glob("*.yaml"))
    raise RecipeConfigError(f"unknown deployment recipe {name!r}; available recipes: {', '.join(sorted(available))}")
