"""Load configuration definitions from selected recipes and runtime adapters."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any

from reef.core.config import ConfigArgument, config_arguments, parse_config_values
from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.registry import recipe_class_for
from reef.runtime.executor.config import ExecutorSettings, WorkerResources, executor_settings
from reef.runtime.registry import runtime_factory_for
from reef.service.deploy.config import config_value, interpolate_config, interpolate_config_values
from reef.service.deploy.settings import service_config_arguments, service_owned_keys

_EXECUTION_ROLES = ("services", "training", "rollout", "evolution")


def _recipe_definition(config: Mapping[str, Any]) -> tuple[type[Recipe] | None, tuple[str, ...]]:
    reference = config_value(config, "reef", "recipe", expand=False)
    if isinstance(reference, str) and ":" in reference:
        recipe_type = recipe_class_for(reference)
        prefix = (
            ("reef",)
            if recipe_type is not None and issubclass(recipe_type, WeightTrainingRecipe)
            else ("reef", "data")
        )
        return recipe_type, prefix
    implementation = config.get("implementation")
    if isinstance(implementation, str):
        return recipe_class_for(implementation), ("data",)
    # A separately stored named preset is resolved by the recipe registry;
    # do not advertise fields for a class that the launcher has not selected.
    return None, ()


def component_config_arguments(config: Mapping[str, Any]) -> tuple[ConfigArgument, ...]:
    """Inspect only the selected class; never construct a recipe or runtime."""
    arguments: list[ConfigArgument] = []
    recipe_type, prefix = _recipe_definition(config)
    if recipe_type is not None:
        arguments.extend(config_arguments(recipe_type, prefix=prefix))
    runtime_prefix = ("runtime",) if prefix == ("data",) else ("reef", "runtime")
    runtime = config.get("runtime", {}) if prefix == ("data",) else config.get("reef", {}).get("runtime", {})
    if isinstance(runtime, Mapping) and isinstance(runtime.get("type"), str):
        factory = runtime_factory_for(runtime["type"])
        if factory is None:
            raise ValueError(f"unknown runtime type {runtime['type']!r}")
        settings_type = factory.config_type()
        if settings_type is not None:
            arguments.extend(config_arguments(settings_type, prefix=runtime_prefix))
    for role in _EXECUTION_ROLES:
        prefix = ("execution", role)
        arguments.extend(config_arguments(ExecutorSettings, prefix=prefix))
        arguments.extend(config_arguments(WorkerResources, prefix=(*prefix, "resources")))
    profiles = config.get("executors", {})
    if isinstance(profiles, Mapping):
        for name in profiles:
            prefix = ("executors", name)
            arguments.extend(config_arguments(ExecutorSettings, prefix=prefix))
            arguments.extend(config_arguments(WorkerResources, prefix=(*prefix, "resources")))
    # Namespaces may reuse field names. Only flat reef fields get short flags;
    # component fields always retain their full path to avoid silent collisions.
    public_flags = {
        flag for argument in service_config_arguments() for flag in (*argument.flags, *argument.negative_flags)
    }
    seen = public_flags | {"--recipe", "--model", "--config", "-c", "--help", "-h"}
    for argument in arguments:
        for flag in (*argument.flags, *argument.negative_flags):
            if flag in seen:
                raise ValueError(f"component config flag conflicts with an existing setting: {flag}")
            seen.add(flag)
    return tuple(arguments)


def normalize_component_config(config: Mapping[str, Any], arguments: tuple[ConfigArgument, ...]) -> dict[str, Any]:
    """Validate supplied component fields, preserving omission and opaque maps."""
    normalized = copy.deepcopy(dict(config))
    for argument in arguments:
        node: Any = normalized
        for key in argument.path[:-1]:
            if not isinstance(node, Mapping):
                break
            node = node.get(key)
        if not isinstance(node, dict) or argument.path[-1] not in node:
            continue
        value = node[argument.path[-1]]
        # Flat recipe YAML has historically treated an empty key as omitted.
        if value is None and argument.path[:1] == ("reef",) and len(argument.path) == 2:
            continue
        value = interpolate_config_values(normalized, value)
        node[argument.path[-1]] = parse_config_values((argument,), {argument.name: value})[argument.name]
    recipe_type, prefix = _recipe_definition(normalized)
    if recipe_type is not None and issubclass(recipe_type, WeightTrainingRecipe) and prefix == ("reef",):
        owned = {key: value for key, value in normalized["reef"].items() if key not in service_owned_keys()}
        recipe_type.service_config(owned, model_path="")
    elif recipe_type is not None:
        data = normalized.get("data", {}) if prefix == ("data",) else normalized.get("reef", {}).get("data", {})
        parse_config_values(config_arguments(recipe_type), data, include_defaults=False)
    runtime = normalized.get("runtime") if prefix == ("data",) else normalized.get("reef", {}).get("runtime")
    if runtime is not None and runtime != {}:
        if not isinstance(runtime, Mapping) or not isinstance(runtime.get("type"), str):
            raise ValueError("reef.runtime must be an object with a runtime type")
        runtime = dict(runtime)
        runtime["type"] = interpolate_config(normalized, runtime["type"])
        factory = runtime_factory_for(runtime["type"])
        if factory is None:
            raise ValueError(f"unknown runtime type {runtime['type']!r}")
        factory.parse_config(runtime, os.environ)
    execution = normalized.get("execution", {})
    if not isinstance(execution, Mapping) or set(execution) - set(_EXECUTION_ROLES):
        raise ValueError("execution must be an object with services, training, rollout or evolution roles")
    for selection in execution.values():
        executor_settings(normalized, selection)
    return normalized
