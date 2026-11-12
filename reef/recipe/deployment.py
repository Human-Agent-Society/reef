"""What the recipes of a deployment answer from their configuration alone.

A deployment has one recipe, or one recipe for each component of a ``CompositeRecipe``
(``reef.recipe.composite.component_recipes``). A caller that holds the deployment's recipe asks these functions,
so no recipe class answers for another recipe. The questions that also need a live object are in
``reef.scenario.live_objects``.
"""

from __future__ import annotations

from pathlib import Path

from reef.recipe.base import Recipe
from reef.recipe.composite import component_recipes
from reef.recipe.harness import HarnessEvolutionRecipe


def harness_recipe(recipe: Recipe) -> tuple[str | None, HarnessEvolutionRecipe] | None:
    """The harness recipe of the deployment whose recipe is ``recipe``, with the name of its component in a
    composite (``None`` for a harness recipe on its own); ``None`` for a deployment that evolves no harness.

    A deployment has at most one harness recipe (``CompositeRecipe``).
    """
    for component, child in component_recipes(recipe):
        if isinstance(child, HarnessEvolutionRecipe):
            return component, child
    return None


def harness_adapter(recipe: Recipe) -> str | None:
    """The adapter that the deployment's harness artifacts render for, as ``GET /reef/harness`` serves them: the
    ``evolution.adapter`` of the deployment's harness recipe; ``None`` for a deployment that evolves no harness.

    Every scenario of the process shares it, so the scenario list names it before a scenario loads.
    """
    harness = harness_recipe(recipe)
    return None if harness is None else harness[1].evolution.adapter


def harness_state_dir(recipe: Recipe, scenario: str, component: str) -> Path | None:
    """The directory of ``component``'s harness evolution state in ``scenario``: its history, its step records and
    the running step's progress (``EvolutionSettings.component_state_dir``).

    ``None`` when the deployment evolves no harness in ``component``: a deployment with no harness recipe, or a
    composite whose harness recipe has another component name. A harness recipe on its own answers for any
    ``component``.
    """
    harness = harness_recipe(recipe)
    if harness is None or harness[0] not in (None, component):
        return None
    return harness[1].evolution.component_state_dir(scenario, component)


def refuses_images(recipe: Recipe) -> bool:
    """Whether the deployment whose recipe is ``recipe`` refuses image content: whether any of its recipes does
    (``Recipe.refuses_images``). Each recipe of a composite reads every record, so this holds for a harness client too.
    """
    return any(child.refuses_images for _, child in component_recipes(recipe))


def scenario_state_dirs(recipe: Recipe, scenario: str) -> tuple[Path, ...]:
    """The directories that belong to ``scenario`` alone, which the dispatcher archives when the scenario is
    deleted: those of each recipe of the deployment, in the order of ``component_recipes``.

    A harness recipe gives the scenario's directory of history and job records
    (``EvolutionSettings.scenario_state_dir``). Any other recipe gives what its record-to-sample stage's config
    names (``PipelineSettings.scenario_state_dirs``).
    """
    directories: list[Path] = []
    for _, child in component_recipes(recipe):
        if isinstance(child, HarnessEvolutionRecipe):
            directories.append(child.evolution.scenario_state_dir(scenario))
        else:
            directories.extend(child.stages().scenario_state_dirs(scenario))
    return tuple(directories)


__all__ = ["harness_adapter", "harness_recipe", "harness_state_dir", "refuses_images", "scenario_state_dirs"]
