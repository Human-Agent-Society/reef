"""Startup diagnostics: every resolved setting and where it came from.

Printed once by ``reef serve`` before models download or processes start, so
an operator can confirm which file, flag, environment variable or default
produced each effective choice. Credentials are masked.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from reef.core.config import ConfigArgument
from reef.service.deploy.cli import _apply_overrides
from reef.service.deploy.deployment_config import component_config_arguments, public_name, translate_layout
from reef.service.deploy.service_config import service_config_arguments

_MISSING = object()
_SECRET_KEY = re.compile(r"token|api[_-]?key|secret|password|database[_-]?url", re.IGNORECASE)
#: Settings reported even when they keep their defaults: the recipe and the HTTP bind.
_ALWAYS_REPORTED = {("reef", "recipe"), ("reef", "host"), ("reef", "port")}


def _lookup(config: Any, path: tuple[str, ...]) -> Any:
    node = config
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _leaf_paths(value: Any, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(value, Mapping) and value:
        return set().union(*(_leaf_paths(item, (*path, str(key))) for key, item in value.items()))
    return {path}


def mask_secrets(value: Any, name: str = "") -> Any:
    """Replace credential values, by key name, at any depth; absent or empty ones stay visible."""
    if _SECRET_KEY.search(name):
        if value is None or value == "":
            return value
        return ["****"] * len(value) if isinstance(value, (list, tuple)) else "****"
    if isinstance(value, Mapping):
        return {key: mask_secrets(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_secrets(item, name) for item in value]
    return value


def _render(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _source(
    argument: ConfigArgument,
    *,
    supplied: Any,
    supplied_source: str,
    seed: Any,
    overridden: bool,
    environ: Mapping[str, str],
) -> str | None:
    """Where the effective value came from, or None when it is an unreported default."""
    explicit = supplied is not _MISSING and supplied != seed
    if overridden:
        return f"{supplied_source}, command line" if explicit and argument.kind == "object" else "command line"
    if explicit:
        return supplied_source
    if argument.env and environ.get(argument.env, "").strip():
        return f"environment {argument.env}"
    return None


def startup_report(
    base: Mapping[str, Any],
    resolved: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    environ: Mapping[str, str],
    from_file: bool,
    include_defaults: bool = False,
) -> list[str]:
    """Lines describing each explicitly supplied, automatically chosen or key default setting.

    ``base`` is the loaded file or the command-line seed config and ``resolved``
    the normalized deployment config after component assembly. With
    ``include_defaults`` every declared setting is listed, defaults included.
    """
    arguments = (*service_config_arguments(), *component_config_arguments(resolved))
    supplied = translate_layout(base) if from_file else base
    supplied_source = "file" if from_file else "environment"
    # Version 2 and command-line startup seed the record-only recipe and a loopback bind.
    seeds = translate_layout({"schema-version": 2}) if not from_file or "schema-version" in base else {}
    overridden = _leaf_paths(_apply_overrides({}, dict(overrides), arguments=arguments))
    lines = ["resolved settings:"]
    for argument in arguments:
        seed = _lookup(seeds, argument.path)
        value = _lookup(resolved, argument.path)
        supplied_value = _lookup(supplied, argument.path)
        source = _source(
            argument,
            supplied=supplied_value,
            supplied_source=supplied_source,
            seed=seed,
            overridden=any(path[: len(argument.path)] == argument.path for path in overridden),
            environ=environ,
        )
        if source is None:
            if value is _MISSING or value == argument.default or value == seed:
                if not include_defaults and argument.path not in _ALWAYS_REPORTED:
                    continue
                source = "default"
                value = argument.default if value is _MISSING else value
            else:
                source = "automatic"
        elif value is _MISSING:
            value = environ[argument.env] if source.startswith("environment ") and argument.env else supplied_value
        lines.append(f"  {public_name(argument)} = {_render(mask_secrets(value, argument.name))}  ({source})")
    return lines
