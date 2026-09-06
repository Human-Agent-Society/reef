"""Parsing and validation for recipe YAML configuration and environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from reef.recipe.errors import RecipeConfigError


def load_recipe_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeConfigError(f"cannot load recipe config {config_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise RecipeConfigError("recipe config must be a YAML object")
    config = dict(loaded)
    if not isinstance(config.get("implementation"), str) or not config["implementation"]:
        raise RecipeConfigError("recipe config must contain a non-empty 'implementation'")
    for section in ("data", "artifact", "runtime", "model", "rollout", "optimization"):
        value = config.get(section, {})
        if not isinstance(value, Mapping):
            raise RecipeConfigError(f"recipe config '{section}' must be an object")
        config[section] = dict(value)
    return config


def config_positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RecipeConfigError(f"{name} must be a positive integer")
    return value
