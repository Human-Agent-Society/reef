"""The training backend and the recipe config of a weight-training deployment, as app assembly reads them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.config import config_arguments
from reef.core.errors import DeployConfigError
from reef.core.plugins import PluginConfigError, resolve_plugin_class
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.train.deployment import TrainingDeployment

#: The bundled training backends, as references so that reading the table imports no training dependency.
BUILTIN_DEPLOYMENTS = {
    "slime": "reef.train.slime_backend.launch:SlimeDeployment",
    "tinker": "reef.train.tinker_backend.launch:TinkerDeployment",
    "spindle_full": "reef.train.tinker_backend.launch:SpindleFullDeployment",
}


def training_deployment_for(name: str | None) -> TrainingDeployment:
    """The deployment definition of the ``training.backend`` value ``name`` (default ``slime``).

    ``name`` is a bundled backend, an installed ``reef.training_backends`` entry point, or a
    ``package.module:ClassName`` reference; each names a ``TrainingDeployment`` class, made with no arguments.
    Raise ``DeployConfigError`` for another name.
    """
    try:
        deployment_class = resolve_plugin_class(
            name or "slime",
            builtins=BUILTIN_DEPLOYMENTS,
            entry_point_group="reef.training_backends",
            base=TrainingDeployment,  # type: ignore[type-abstract]  # the base is a value; the result is concrete
            setting="training backend",
        )
    except PluginConfigError as exc:
        raise DeployConfigError(str(exc)) from exc
    return deployment_class()


def training_recipe_config(
    recipe_type: type[WeightTrainingRecipe], settings: Mapping[str, Any], *, model_path: str
) -> dict[str, Any]:
    """The flat recipe config of a weight training deployment, from the recipe's keys of its ``reef`` section.

    ``settings`` is the recipe-owned slice of the deployment's flat ``reef`` section: the caller removes the
    service's own keys first (``service_owned_keys``). Only keys the operator set are forwarded under ``data``, so
    every default lives with the config fields of ``recipe_type``, never in the service layer. This step only
    translates the layout; the loader's field resolution converts the types. ``model_path`` goes under
    ``model.path``. Raise ``RecipeConfigError`` for a key that ``recipe_type`` does not declare: the operator set
    a value that nothing would consume.
    """
    field_names = {argument.name for argument in config_arguments(recipe_type)}
    data: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in settings.items():
        if key in field_names:
            # A YAML key left empty (None) is unset, not a value to parse.
            if value is not None:
                data[key] = value
        else:
            unknown.append(key)
    if unknown:
        known = ", ".join(f"reef.{name}" for name in sorted(field_names))
        raise RecipeConfigError(
            f"{recipe_type.__name__} does not consume {', '.join(f'reef.{key}' for key in sorted(unknown))}; "
            f"the recipe would silently run with its defaults instead. "
            f"Settings {recipe_type.__name__} consumes: {known}. "
            f"A new setting is declared with config_field() on the recipe dataclass "
            f"(see reef.recipe.config_fields)."
        )
    return {"data": data, "model": {"path": model_path}}
