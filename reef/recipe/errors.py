"""Errors raised while selecting and configuring a deployment recipe."""

from reef.core.errors import ReefError


class RecipeConfigError(ReefError):
    """Raised when a recipe's environment or YAML config is invalid."""
