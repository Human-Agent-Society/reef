"""Native engine argument transport shared by integration launch definitions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from reef.core.errors import DeployConfigError


def normalize_native_options(options: Any) -> dict[str, Any]:
    """Normalize flag spellings, preserving payload values and rejecting aliases."""
    if not isinstance(options, Mapping):
        raise DeployConfigError("backend options must be an object")
    normalized = {}
    for key, value in options.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", key):
            raise DeployConfigError("backend option names must be native flag names without leading --")
        name = key.replace("_", "-")
        if name in normalized:
            raise DeployConfigError(f"duplicate backend option: {name}")
        normalized[name] = value
    return normalized


def native_arguments(options: Any, *, reserved: set[str] | None = None) -> list[str]:
    """Encode native argv: true is a switch, false/null omit, lists supply values.

    Objects are JSON values. Backend parsers own accepted names, types and
    defaults; managed launch fields cannot also appear in the options object.
    """
    arguments = []
    for name, value in normalize_native_options(options).items():
        if any(protected.startswith(name) for protected in (reserved or set())):
            raise DeployConfigError(f"backend option {name} is managed by Reef or unsupported in managed serving")
        if value is None or value is False:
            continue
        flag = f"--{name}"
        if value is True:
            arguments.append(flag)
        elif isinstance(value, list):
            if any(isinstance(item, str) and item.startswith("--") for item in value):
                raise DeployConfigError(f"backend option {name} list values cannot contain flags")
            arguments.append(flag)
            arguments.extend(json.dumps(item) if isinstance(item, (dict, list, bool)) else str(item) for item in value)
        else:
            encoded = json.dumps(value) if isinstance(value, Mapping) else str(value)
            arguments.append(f"{flag}={encoded}")
    return arguments
