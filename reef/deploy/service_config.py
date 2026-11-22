"""A deployment configuration becomes the typed settings of the HTTP process.

CLI and YAML values use the shared parser in ``reef.core.config``. This module
converts effective deployment configuration into ``reef.api.settings.ServiceSettings`` for app assembly.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Mapping
from typing import Any

from reef.api.cors import console_origins
from reef.api.settings import ServiceSettings, service_config_arguments
from reef.core.config import ConfigArgument, parse_config_values
from reef.deploy.config_utils import interpolate_config, interpolate_config_values
from reef.routing.settings import routing_deployment_settings


def _reef_section(config: Mapping[str, Any]) -> dict[str, Any]:
    """The flat ``reef`` section with per-value interpolation applied."""
    section = config.get("reef")
    if not isinstance(section, Mapping):
        return {}

    return {key: interpolate_config_values(config, value) for key, value in section.items()}


def service_override(key: str, value: str) -> tuple[ConfigArgument, str] | None:
    """Identify a declared option by its flag."""
    flag = f"--{key}"
    for argument in service_config_arguments():
        if flag in argument.flags:
            return argument, value
        if flag in argument.negative_flags:
            return argument, "false"
    return None


def _argument_value(config: Mapping[str, Any], argument: ConfigArgument) -> Any:
    node: Any = config
    for key in argument.path:
        if not isinstance(node, Mapping):
            raise ValueError(f"{'.'.join(argument.path[:-1])} must be an object")
        node = node.get(key)
        if node is None:
            return None
    if isinstance(node, str):
        return os.path.expanduser(interpolate_config(config, node.strip())) if node.strip() else None
    return interpolate_config_values(config, node)


def parse_service_arguments(config: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the merged public values with the shared component parser.

    The caller has already applied overrides and environment references, so
    precedence is already present in the mapping.
    """
    arguments = service_config_arguments()
    supplied = {}
    for argument in arguments:
        value = _argument_value(config, argument)
        if value is not None:
            supplied[argument.name] = value
    values = parse_config_values(
        tuple(dataclasses.replace(argument, required=False) for argument in arguments), supplied, environ=os.environ
    )
    if "inference_retry_timeout_s" not in supplied:
        values["inference_retry_timeout_s"] = values["inference_timeout_s"]
    return values


def normalize_service_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate explicit public settings and pass the same values to children.

    Do not add defaults to the deployment mapping: recipes must still know
    which of their fields the operator supplied, and custom stacks may have
    no Reef HTTP child at all.
    """
    values = parse_service_arguments(config)
    # Reject removed or invalid routing settings while loading, before any process starts.
    routing_deployment_settings(values["routing_deployment"])
    normalized = copy.deepcopy(dict(config))
    for argument in service_config_arguments():
        if _argument_value(config, argument) is None:
            continue
        node = normalized
        for key in argument.path[:-1]:
            node = node[key]
        node[argument.path[-1]] = values[argument.name]
    return normalized


def service_config_from_mapping(config: Mapping[str, Any]) -> ServiceSettings:
    """Read HTTP service settings from a resolved deployment's ``reef`` section."""
    values = parse_service_arguments(config)
    if not isinstance(values["recipe"], str) or not values["recipe"]:
        raise ValueError("config must declare a non-empty reef.recipe")
    # ``reef.token`` (one) and ``reef.tokens`` (a list) are equivalent credentials;
    # several let a caller rotate without downtime. An empty entry, such as an
    # unset ``${REEF_TOKEN}``, is not a credential.
    tokens = (token.strip() for token in (values.pop("token") or "", *values["tokens"]))
    values["tokens"] = tuple(dict.fromkeys(token for token in tokens if token))
    values["console_origins"] = console_origins(values["console_origins"])
    values["routing_deployment"] = routing_deployment_settings(values["routing_deployment"])
    # Preserve shared execution settings for directly selected recipes.
    preset = dict(config) if ":" in values["recipe"] else None
    execution = {key: config[key] for key in ("execution", "executors") if key in config}
    return ServiceSettings(
        **values, recipe_settings=_reef_section(config), preset_config=preset, execution_config=execution
    )
