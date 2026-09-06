"""Parsing and validation for recipe YAML configuration and environment."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from reef.recipe.errors import RecipeConfigError


def load_recipe_config(path: str | Path, *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Load a recipe, resolving ``${VAR}`` in ``model.path`` from the environment.

    Expand the model after YAML parsing; prompts, skills and other recipe text
    retain their literal contents. Missing model variables fail at startup.
    """
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
    model_path = config["model"].get("path")
    if isinstance(model_path, str):
        values = os.environ if environ is None else environ

        def replace_variable(match: re.Match[str]) -> str:
            name = match.group(1)
            value = values.get(name)
            if value is None or not value.strip():
                raise RecipeConfigError(
                    f"model.path requires environment variable {name}; set it before starting Reef"
                )
            return value

        config["model"]["path"] = re.sub(r"\$\{(\w+)\}", replace_variable, model_path)
    return config


def config_positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RecipeConfigError(f"{name} must be a positive integer")
    return value
