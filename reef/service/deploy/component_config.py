"""Load selected component schemas, translate public paths, and validate their values.

Field declarations own types and defaults. Layout translation maps versioned YAML
and references onto those declarations without interpreting native backend options.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.config import ConfigArgument, config_arguments, config_metadata, parse_config_values
from reef.core.errors import DeployConfigError
from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.registry import recipe_class_for
from reef.runtime.executor.config import ExecutorSettings, WorkerResources, executor_settings
from reef.runtime.registry import runtime_factory_for
from reef.service.deploy.config import config_value, interpolate_config, interpolate_config_values
from reef.service.deploy.service_config import service_config_arguments, service_owned_keys

_MISSING = object()


@dataclass(frozen=True)
class DeploymentSettings:
    run_dir: str = field(
        default=".reef/run",
        metadata=config_metadata("Service log directory.", path=("run_dir",), public_path=("reef", "run_dir")),
    )
    ready_timeout: int = field(
        default=30,
        metadata=config_metadata(
            "Service readiness deadline in seconds.", path=("ready_timeout",), public_path=("reef", "ready_timeout")
        ),
    )


def deployment_config_arguments() -> tuple[ConfigArgument, ...]:
    return config_arguments(DeploymentSettings)


def _take(values: dict[str, Any], path: tuple[str, ...]) -> Any:
    node = values
    for index, part in enumerate(path):
        spellings = tuple(dict.fromkeys((part, part.replace("_", "-"))))
        present = [name for name in spellings if name in node]
        if len(present) > 1:
            raise DeployConfigError(f"duplicate config field: {'.'.join(path)}")
        if not present:
            return _MISSING
        key = present[0]
        if index == len(path) - 1:
            return node.pop(key)
        node = node[key]
        if not isinstance(node, dict):
            raise DeployConfigError(f"{'.'.join(path[:index + 1])} must be an object")
    return _MISSING


def _put(values: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = values
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def _unknown(values: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths = []
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths.extend(_unknown(value, path))
        else:
            paths.append(path)
    return paths


def translate_layout(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate schema-version 2 while keeping unversioned files unchanged."""
    if "schema-version" not in config:
        return copy.deepcopy(dict(config))
    if type(config["schema-version"]) is not int or config["schema-version"] != 2:
        raise DeployConfigError("unsupported schema-version; expected 2")
    if "service" in config or "services" in config:
        raise DeployConfigError(
            "schema-version 2 does not accept service/services: move HTTP settings to reef; "
            "processes are assembled by Reef and the selected recipe"
        )
    if isinstance(config.get("execution"), Mapping) and "services" in config["execution"]:
        raise DeployConfigError(
            "execution.services belongs to legacy process stacks; configure component resources instead"
        )
    pending = copy.deepcopy(dict(config))
    pending.pop("schema-version")
    known_sections = {
        "reef",
        "inference",
        "recipe",
        "training",
        "storage",
        "execution",
        "executors",
        "evaluation",
        "observability",
    }
    extra = set(pending) - known_sections
    if extra:
        raise DeployConfigError(f"unknown config sections: {', '.join(sorted(extra))}")
    resolved: dict[str, Any] = {"reef": {"recipe": "recipe", "host": "127.0.0.1"}, "run_dir": ".reef/run"}
    for argument in (*service_config_arguments(), *deployment_config_arguments()):
        path = argument.public_path or argument.path
        value = _take(pending, path)
        if value is not _MISSING:
            _put(resolved, argument.path, value)
    for path, target in (
        (("recipe", "config"), ("reef", "data")),
        (("recipe", "runtime"), ("reef", "runtime")),
        (("execution",), ("execution",)),
        (("executors",), ("executors",)),
    ):
        value = _take(pending, path)
        if value is not _MISSING:
            _put(resolved, target, value)
    # Empty known sections are allowed; unknown empty objects are still typos.
    for section, content in pending.items():
        if not isinstance(content, Mapping):
            raise DeployConfigError(f"{section} must be an object")
        if content:
            paths = _unknown({section: content}) or [f"{section}.{key}" for key in content]
            raise DeployConfigError(f"unknown config fields: {', '.join(paths)}")
    return resolved


def translate_recipe_fields(config: dict[str, Any], arguments: tuple[ConfigArgument, ...]) -> dict[str, Any]:
    """Translate selected recipe fields and owned sections from their public namespace."""
    recipe_arguments = [argument for argument in arguments if argument.public_path[:2] == ("recipe", "config")]
    if not recipe_arguments:
        return config
    resolved = copy.deepcopy(config)
    recipe_values = resolved.get("reef", {}).pop("data", {})
    if not isinstance(recipe_values, Mapping):
        raise DeployConfigError("recipe.config must be an object")
    by_name = {argument.name: argument for argument in recipe_arguments}
    seen = set()
    for name, value in recipe_values.items():
        name = name.replace("-", "_")
        if name in seen:
            raise DeployConfigError(f"duplicate recipe.config field: {name}")
        seen.add(name)
        if name not in by_name:
            raise DeployConfigError(f"unknown recipe.config field: {name}")
        _put(resolved, by_name[name].path, value)
    return resolved


def normalize_component_layout(config: dict[str, Any], arguments: tuple[ConfigArgument, ...]) -> dict[str, Any]:
    """Accept YAML field spellings from the selected component declarations."""
    resolved = translate_recipe_fields(config, arguments)
    for argument in arguments:
        node: Any = resolved
        for part in argument.path[:-1]:
            if not isinstance(node, dict):
                break
            node = node.get(part)
        if isinstance(node, dict):
            value = _take(node, (argument.path[-1],))
            if value is not _MISSING:
                node[argument.path[-1]] = value
    return resolved


def translate_references(config: dict[str, Any], arguments: tuple[ConfigArgument, ...]) -> dict[str, Any]:
    """Resolve public config references through the same field path declarations."""
    paths = {
        ".".join(argument.public_path or argument.path): ".".join(argument.path)
        for argument in (*service_config_arguments(), *arguments)
    }
    paths.update({"recipe.runtime": "reef.runtime", "recipe.config": "reef.data"})

    def convert(value: Any) -> Any:
        if isinstance(value, str):

            def replace_reference(match: re.Match[str]) -> str:
                reference = match.group(1).replace("-", "_")
                for source in sorted(paths, key=len, reverse=True):
                    if reference == source or reference.startswith(source + "."):
                        return "${" + paths[source] + match.group(1)[len(source) :] + "}"
                return match.group(0)

            return re.sub(r"\$\{([^}]+)\}", replace_reference, value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(config)


_EXECUTION_ROLES = ("services", "training", "rollout", "evolution")


def _recipe_definition(config: Mapping[str, Any]) -> tuple[type[Recipe] | None, tuple[str, ...]]:
    reference = config_value(config, "reef", "recipe", expand=False)
    if isinstance(reference, str) and (":" in reference or reference == "recipe"):
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
    arguments: list[ConfigArgument] = list(deployment_config_arguments())
    recipe_type, prefix = _recipe_definition(config)
    if recipe_type is not None:
        arguments.extend(
            replace(argument, public_path=("recipe", "config", argument.name))
            for argument in config_arguments(recipe_type, prefix=prefix)
        )
        if issubclass(recipe_type, WeightTrainingRecipe):
            # The legacy service contract projects this control into artifact
            # settings rather than the recipe's data fields; keep its default there.
            arguments.append(
                ConfigArgument(
                    "checkpoint_every_n_versions",
                    (*prefix, "checkpoint_every_n_versions"),
                    "int",
                    True,
                    None,
                    "Artifact checkpoint interval in published versions.",
                    public_path=("recipe", "config", "checkpoint_every_n_versions"),
                )
            )
        section_prefix = prefix[:-1] if prefix[-1:] == ("data",) else prefix
        arguments.extend(
            ConfigArgument(
                section,
                (*section_prefix, section),
                "object",
                False,
                {},
                f"Recipe-owned {section} configuration.",
                public_path=("recipe", "config", section),
            )
            for section in recipe_type.config_sections
        )
    runtime_prefix = ("runtime",) if prefix == ("data",) else ("reef", "runtime")
    arguments.append(
        ConfigArgument(
            "runtime_type",
            (*runtime_prefix, "type"),
            "str",
            True,
            None,
            "Selected recipe runtime factory.",
            public_path=("recipe", "runtime", "type"),
        )
    )
    runtime = config.get("runtime", {}) if prefix == ("data",) else config.get("reef", {}).get("runtime", {})
    if isinstance(runtime, Mapping) and isinstance(runtime.get("type"), str):
        factory = runtime_factory_for(runtime["type"])
        if factory is None:
            raise ValueError(f"unknown runtime type {runtime['type']!r}")
        settings_type = factory.config_type()
        if settings_type is not None:
            arguments.extend(
                replace(argument, public_path=("recipe", "runtime", argument.name))
                for argument in config_arguments(settings_type, prefix=runtime_prefix)
            )
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
