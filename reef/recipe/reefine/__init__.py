"""Reefine: built-in harness refinement from a user's training instructions.

``ReefineRecipe`` binds the served-model proposer to the shared Cordis loop.
It accepts the same ``evolution`` settings as ``CordisRecipe``; the shipped
``reefine`` profile supplies the health task and seed. Custom tasks should
also supply their own ``evolution.evaluate`` scorer. The bundled scorer only
recognizes the profile's health task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.recipe.config_fields import config_field
from reef.recipe.cordis import CordisRecipe
from reef.recipe.errors import RecipeConfigError


@dataclass(frozen=True, kw_only=True)
class ReefineRecipe(CordisRecipe):
    """Refine a pi harness once per instruction, with extensions held for review.

    Configuration defaults enable harness requests and update notices.
    Selection is ``floor``: the evaluation runs the candidate alone on the
    profile's health task and publishes it when every task scores at least
    ``evolution.floor_score``. The floor checks that the tree still works
    (the model binding, the tools, the extensions load), not that the
    requested change does; the step's design and review notes and the
    person judge that. Override ``evolution.selection`` to compare scores
    against the current release, and ``training-mode`` to learn from
    reports too.
    """

    name: str = field(default="reefine", kw_only=True)
    training_mode: str = config_field("manual")

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution", {})
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("reefine requires an 'evolution' config mapping")
        defaults = {
            "propose": "reef.recipe.reefine.evolution:propose",
            "evaluate": "reef.recipe.reefine.evolution:evaluate",
            "requests": True,
            "version_check": True,
            "review_kinds": ["code_extension"],
            "selection": "floor",
        }
        return super()._recipe_kwargs({**settings, "evolution": {**defaults, **evolution}}, values)
