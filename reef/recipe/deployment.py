"""What the recipes of a deployment answer from their configuration alone.

A deployment has one recipe, or one recipe for each component of a ``CompositeRecipe``
(``reef.recipe.composite.component_recipes``). A caller that holds the deployment's recipe asks these functions,
so no recipe class answers for another recipe. The questions that also need a live object are in
``reef.scenario.live_objects``.
"""

from __future__ import annotations

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


__all__ = ["harness_adapter", "harness_recipe"]
