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


def scenario_state_dirs(recipe: Recipe, scenario: str) -> tuple[Path, ...]:
    """The directories of the record-to-sample stages that belong to ``scenario`` alone, which the dispatcher
    archives when the scenario is deleted: what the stage config of each recipe of the deployment names
    (``PipelineSettings.scenario_state_dirs``), in the order of ``component_recipes``.

    A harness recipe keeps the scenario's history and job records in a directory of the training service. That
    service moves it aside itself (``TrainingComponent.archive_scenario``), so it is not listed here.
    """
    directories: list[Path] = []
    for _, child in component_recipes(recipe):
        if not isinstance(child, HarnessEvolutionRecipe):
            directories.extend(child.stages().scenario_state_dirs(scenario))
    return tuple(directories)


__all__ = ["harness_adapter", "harness_recipe", "scenario_state_dirs"]
