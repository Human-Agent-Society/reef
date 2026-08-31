"""Errors shared by recipe registries, stores, and request handling."""

from reef.core.errors import ReefError


class RecipeConfigError(ReefError):
    """Raised when a recipe's environment or YAML config is invalid."""


class ScenarioRecipeError(ReefError):
    """Base class for invalid scenario/recipe operations."""


class UnknownScenarioRecipe(ScenarioRecipeError):
    pass


class ScenarioRecipeConflict(ScenarioRecipeError):
    pass
