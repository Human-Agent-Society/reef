"""Translate versioned public configuration into existing component input paths.

Only layout changes here. Types, defaults and CLI aliases belong to field
metadata; component parsers retain validation and opaque backend payloads.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.config import ConfigArgument, config_arguments, config_metadata
from reef.service.deploy.config import DeployConfigError
from reef.service.deploy.settings import service_config_arguments

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
