"""Errors raised while selecting and configuring a deployment recipe."""

from reef.core.errors import ConfigError


class RecipeConfigError(ConfigError):
    """Raised when a recipe's environment or YAML config is invalid."""
