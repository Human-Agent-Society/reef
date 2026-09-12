"""Reefine: built-in harness refinement from a user's training instructions.

``ReefineRecipe`` binds the served-model proposer to the shared Cordis loop.
It accepts the same ``evolution`` settings as ``CordisRecipe``; the shipped
``reefine`` profile supplies the example tasks and seed. Custom tasks should
also supply their own ``evolution.evaluate`` scorer. The bundled scorer only
recognizes the profile's three arithmetic tasks.
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

    Configuration defaults enable harness requests and update notices. As in
    the tutorial, selection is ``always``: evaluation scores are recorded but
    need not improve for publication. Override ``evolution.selection`` to
    enforce score comparison, and ``training-mode`` to learn from reports too.
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
            "selection": "always",
        }
        return super()._recipe_kwargs({**settings, "evolution": {**defaults, **evolution}}, values)
