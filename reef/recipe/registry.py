"""Recipe references and config-backed resolution by public name.

Two axes live here, deliberately separate:

- :func:`recipe_class_for` and :func:`build_recipe` resolve the core
  record-only recipe or an explicit ``package.module:ClassName`` reference.
- :func:`build_named_recipe` builds the recipe a deployment names: a YAML
  preset from the recipe config directory, or the core record-only recipe.
- :class:`RecipeRegistry` is the closed *instance* table mapping the public
  names scenarios bind to onto built ``Recipe`` objects.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.recipe.base import Recipe
from reef.recipe.config import load_recipe_config
from reef.recipe.errors import RecipeConfigError, UnknownScenarioRecipe
from reef.runtime.base import InferenceRuntime
from reef.runtime.registry import RuntimeRegistry

RECIPE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RecipeType = type[Recipe]


def recipe_class_for(reference: str) -> RecipeType | None:
    """Return the Recipe implementation for ``reference``.

    ``recipe`` is the core record-only recipe. Every learning method is an
    explicit ``package.module:ClassName`` reference, so importing Reef never
    imports method code and operators can see exactly what a deployment loads.
    A different bare name is a public preset name, not a class reference, and
    returns ``None`` here.
    """
    if reference == "recipe":
        return Recipe
    if ":" not in reference:
        return None
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise RecipeConfigError(f"dotted recipe reference {reference!r} must be 'package.module:ClassName'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise RecipeConfigError(f"cannot import recipe reference {reference!r}: {exc}") from exc
    if not (isinstance(candidate, type) and issubclass(candidate, Recipe)):
        raise RecipeConfigError(f"recipe reference {reference!r} is not a Recipe class")
    return candidate


def build_recipe(
    implementation: str,
    environ: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    runtime: InferenceRuntime | None = None,
) -> Recipe:
    recipe_class = recipe_class_for(implementation)
    if recipe_class is None:
        raise ValueError(f"unknown recipe reference {implementation!r}")
    return recipe_class.from_environment(environ, config=config, runtime=runtime)


def build_named_recipe(
    name: str,
    environ: Mapping[str, str] | None = None,
    *,
    config_directory: str | Path | None = None,
    default_runtime: InferenceRuntime | None = None,
    runtime_registry: RuntimeRegistry | None = None,
) -> Recipe:
    """Build the recipe a deployment names by its public name.

    ``name`` is a YAML preset ``<name>.yaml`` under ``config_directory``
    (defaulting to ``REEF_RECIPE_CONFIG_DIR``), or the reserved core name
    ``recipe``. Reef bundles no presets — they are deployment data (see
    ``docs/reference/configuration.rst``). A preset's ``runtime`` section builds the recipe's
    runtime; without one the recipe gets ``default_runtime``. Dotted references are
    not names: they are operator configuration for :func:`build_recipe`, so a
    name never triggers an import.
    """
    if not RECIPE_NAME.fullmatch(name):
        raise RecipeConfigError(f"invalid recipe name {name!r}")
    values = os.environ if environ is None else environ
    configured = config_directory or values.get("REEF_RECIPE_CONFIG_DIR") or None
    directory = None if configured is None else Path(configured)
    path = None if directory is None else directory / f"{name}.yaml"
    if path is None or not path.exists():
        if name == "recipe":
            return build_recipe(name, values, runtime=default_runtime)
        available = {"recipe"}
        if directory is not None:
            available.update(preset.stem for preset in directory.glob("*.yaml"))
        raise UnknownScenarioRecipe(
            f"unknown scenario recipe {name!r}; available recipes: {', '.join(sorted(available))}"
        )

    settings = load_recipe_config(path)
    model_path = settings["model"].get("path")
    if not isinstance(model_path, str) or not model_path:
        raise RecipeConfigError(f"recipe {name!r} must configure a non-empty model.path")
    runtime_config = settings["runtime"]
    # Training runtimes are Ray-based and cannot be built from YAML, so a
    # preset without a runtime section gets the default and the recipe itself
    # reports whether it needs a training runtime injected instead.
    runtime = (
        (runtime_registry or RuntimeRegistry()).build(runtime_config, model_path=model_path, recipe_config=settings)
        if runtime_config
        else default_runtime
    )
    return build_recipe(settings["implementation"], values, config=settings, runtime=runtime)


class RecipeRegistry:
    """The recipes a process serves, by the public name scenarios bind to.

    A closed set: request-time names resolve only to what the operator
    injected, never materializing another implementation or preset. A reef deployment
    serves exactly one recipe (see :attr:`served_recipe`); multi-recipe
    registries exist for tests that exercise scenario bindings across
    deployments.
    """

    def __init__(self, recipes: Mapping[str, Recipe]) -> None:
        self._recipes = dict(recipes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._recipes))

    @property
    def served_recipe(self) -> str | None:
        """The one recipe this registry serves, or ``None`` for a multi-recipe
        registry.

        A request that names no recipe binds its scenario to the served one;
        a multi-recipe registry has no served recipe and requests must name one.
        """
        if len(self._recipes) != 1:
            return None
        return next(iter(self._recipes))

    def resolve(self, recipe: str) -> Recipe:
        selected = self._recipes.get(recipe)
        if selected is None:
            raise UnknownScenarioRecipe(
                f"unknown scenario recipe {recipe!r}; available recipes: {', '.join(self.names)}"
            )
        return selected
