"""Recipe class references by public name.

:func:`recipe_class_for` resolves the core record-only recipe or an explicit
``package.module:ClassName`` reference. The loader (``reef.recipe.loading``)
makes the recipe value of a class.
"""

from __future__ import annotations

import importlib

from reef.recipe.base import Recipe
from reef.recipe.errors import RecipeConfigError

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
