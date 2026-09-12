"""Views of shipped examples through the production config translation paths."""

from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.registry import recipe_class_for
from reef.service.deploy.components import component_config_arguments
from reef.service.deploy.config import load_config
from reef.service.deploy.layout import normalize_component_layout, translate_layout, translate_references
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.provider import assemble_provider_services
from reef.service.deploy.training import assemble_training_services


def load_deployment(path):
    return resolve_deployment_config(load_config(path, interpolate_env=False), None, path)[0]


def deployment_layout(config):
    """Inspect placement without expanding environment-dependent recipe values."""
    resolved = translate_layout(config)
    arguments = component_config_arguments(resolved)
    resolved = translate_references(normalize_component_layout(resolved, arguments), arguments)
    if "services" not in resolved:
        recipe_type = recipe_class_for(resolved["reef"].get("recipe", "recipe"))
        if recipe_type is not None and issubclass(recipe_type, WeightTrainingRecipe):
            assemble_training_services(resolved)
        else:
            assemble_provider_services(resolved)
    return resolved


def load_harness_deployment(path):
    from reef.recipe.config import recipe_config_from_mapping

    raw = load_config(path)
    config = load_deployment(path)
    return {**config, **recipe_config_from_mapping(raw)}
