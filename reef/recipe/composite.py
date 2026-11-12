"""One scenario served and evolved by several recipes, one per snapshot component.

A composite recipe is a mapping from component name to that component's
recipe, and it computes nothing from the recipes. ``component_recipes`` gives
the recipes of any deployment, and each owner of a result reads them: the
scenario's components hold each recipe's component under its name
(``reef.scenario.live_objects.scenario_components``), the scenario's default
start keeps each component's seed under that component's directory
(``reef.scenario.live_objects.default_start``), and its trainers are
the components' trainers, run as independent workers that meet at the
scenario's commit boundary (``reef.scenario.factory.read_trainer_specs``; see
``docs/advanced_topics/state-model.rst``).

Configured by a dotted ``implementation`` whose config carries a
``components`` object. Each component has the shape of a file's ``recipe``
section, without ``runtime``: ``implementation`` names its recipe class and
``config`` holds its fields, sections and pipeline stages. Each component
uses the deployment's model:

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         weights:
           implementation: recipes.sao.recipe:SAORecipe
           config:
             pipeline: {sample2job: {config: {job_samples: 8, buffer_size: 32}}}
         harness:
           implementation: reef.recipe.harness:HarnessEvolutionRecipe
           config:
             evolution: {adapter: pi, ...}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from reef.core.components import validate_component_name
from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.config import recipe_config_from_section
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe

#: Metric prefixes experiment tracking keeps for itself; a component's metrics are prefixed with its name.
RESERVED_COMPONENT_NAMES = ("train", "step", "reef", "operations")
#: The sections a component keeps under its own ``config``; a file keeps them at its top level.
COMPONENT_FRAMEWORK_SECTIONS = ("pipeline", "evaluation")


def configured_training_mode(config: Mapping[str, Any]) -> str | None:
    """The ``data.training_mode`` a config names, in either spelling the field parser accepts; ``None`` unset."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        return None
    for key in ("training_mode", "training-mode"):
        if data.get(key) is not None:
            return str(data[key])
    return None


def component_recipe_config(component: str, section: Any) -> dict[str, Any]:
    """The flat recipe config of ``components.<component>``, read the way a file's ``recipe`` section is read.

    A component has no runtime of its own: every component shares the deployment's runtime.
    """
    path = f"components.{component}"
    settings = recipe_config_from_section(section, path, COMPONENT_FRAMEWORK_SECTIONS)
    if "runtime" in settings:
        raise RecipeConfigError(
            f"{path}.runtime is not accepted: every component shares the deployment's runtime, set by recipe.runtime"
        )
    return settings


@dataclass(frozen=True, kw_only=True)
class CompositeRecipe(Recipe):
    """A mapping from component name to the recipe of that component, and nothing more.

    Values: ``components`` maps each snapshot component's name to its recipe value. The mapping meets all of
    these conditions, which ``__post_init__`` checks; a mapping that fails one raises ``RecipeConfigError``:

    - It has at least two entries.
    - Each name is a valid component name (``reef.core.components.validate_component_name``) and is none of
      ``RESERVED_COMPONENT_NAMES``.
    - Each value is a ``Recipe`` and is no ``CompositeRecipe``.
    - Each value's ``training_mode`` equals the composite's.
    - At most one value is a ``WeightTrainingRecipe``: a snapshot loads one component into a runtime.
    - At most one value is a ``HarnessEvolutionRecipe``: a process has one harness training component
      (``reef.scenario.live_objects.LiveObjects.harness_component``).

    The other fields are ``Recipe``'s: ``name``, ``training_mode`` and ``activation`` are the deployment's own
    values. The composite computes nothing from its children. ``report_type``, ``build_components``,
    ``default_start`` and ``stages`` are the inherited defaults of ``Recipe`` and say nothing about the children;
    a reader iterates the children (``component_recipes``) and asks each one.

    Purpose: the loader makes it from the ``components`` section (``reef.recipe.loading.load_composite_recipe``).
    Each owner of a result reads the children with ``component_recipes``: the scenario's components, default start
    and root check (``reef.scenario.live_objects``), the trainers and the report contract
    (``reef.scenario.factory``), and the harness and state directory questions (``reef.recipe.deployment``).
    """

    components: Mapping[str, Recipe] = field(default_factory=dict)
    name: str = "composite"
    #: A deployment file keeps ``components`` under ``recipe.config``.
    config_sections: ClassVar[tuple[str, ...]] = ("components",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.components, Mapping) or len(self.components) < 2:
            raise RecipeConfigError("a composite recipe binds at least two components")
        for component, recipe in self.components.items():
            validate_component_name(component)
            if component in RESERVED_COMPONENT_NAMES:
                # Experiment tracking prefixes a component's metrics with its name; these prefixes are its own.
                raise RecipeConfigError(f"component {component!r} is a metric prefix experiment tracking reserves")
            if not isinstance(recipe, Recipe):
                raise RecipeConfigError(f"component {component!r} must be a Recipe")
            if isinstance(recipe, CompositeRecipe):
                raise RecipeConfigError(f"component {component!r} must not itself be a composite recipe")
            if recipe.training_mode != self.training_mode:
                raise RecipeConfigError(
                    f"component {component!r} runs training_mode {recipe.training_mode!r}, "
                    f"the composite recipe {self.training_mode!r}: every component shares one mode"
                )
        weight_components = [
            name for name, recipe in self.components.items() if isinstance(recipe, WeightTrainingRecipe)
        ]
        if len(weight_components) > 1:
            raise RecipeConfigError(
                f"components {weight_components} all train weights; a snapshot loads one component"
            )
        harness_components = [
            name for name, recipe in self.components.items() if isinstance(recipe, HarnessEvolutionRecipe)
        ]
        if len(harness_components) > 1:
            raise RecipeConfigError(
                f"components {harness_components} each evolve a harness; a composite evolves one, because a process "
                "has one harness training component"
            )
        object.__setattr__(self, "components", dict(self.components))


def component_recipes(recipe: Recipe) -> tuple[tuple[str | None, Recipe], ...]:
    """The recipes of the deployment whose recipe is ``recipe``, each with the name of its snapshot component.

    A ``CompositeRecipe`` gives one pair ``(name, child)`` for each entry of ``components``, in the mapping's
    order. Any other recipe gives the one pair ``(None, recipe)``: the recipe binds no component name of a
    composite. No pair holds a ``CompositeRecipe``.
    """
    if isinstance(recipe, CompositeRecipe):
        return tuple(recipe.components.items())
    return ((None, recipe),)


__all__ = ["CompositeRecipe", "component_recipes"]
