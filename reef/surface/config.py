"""Serve a configuration component: request defaults read from one JSON file.

The ``config`` component of a release holds ``config.json``, a JSON object.
Its ``request_defaults`` object supplies provider request fields the caller
left unset (``temperature``, ``max_tokens``, ...), so a change of defaults is
a release like a change of weights or harness, frozen per request and
recorded against the release that served it. The component is applied by the
service and exposes no client-pulled tree, so it composes beside a harness.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.core.errors import ReefError
from reef.surface.base import ComponentSurface, InferenceHooks, Surface

#: The component name a configuration surface binds.
CONFIG_COMPONENT = "config"
#: The one file a configuration component holds.
CONFIG_FILE = "config.json"
REQUEST_DEFAULTS_KEY = "request_defaults"


def read_config(artifact: Artifact) -> dict[str, Any]:
    """The configuration object of a component; an empty object when the file is absent."""
    local_path = artifact.materialize().local_path
    if local_path is None:
        return {}
    path = local_path / CONFIG_FILE
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReefError(f"{CONFIG_FILE} is not a JSON object: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReefError(f"{CONFIG_FILE} must hold a JSON object")
    return dict(value)


def request_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """The provider request fields the configuration supplies when a caller leaves them unset."""
    defaults = config.get(REQUEST_DEFAULTS_KEY, {})
    if not isinstance(defaults, Mapping):
        raise ReefError(f"{CONFIG_FILE} {REQUEST_DEFAULTS_KEY} must be an object")
    return dict(defaults)


@dataclass(frozen=True)
class ConfigValidator(ArtifactValidator):
    """Admit a configuration component: one well-formed ``config.json``."""

    def validate(self, artifact: Artifact) -> None:
        local_path = artifact.materialize().local_path
        if local_path is None or not (local_path / CONFIG_FILE).is_file():
            raise ReefError(f"configuration component requires {CONFIG_FILE}")
        request_defaults(read_config(artifact))


@dataclass(frozen=True)
class ConfigInferenceHooks(InferenceHooks):
    """Fill provider request fields the caller left unset from the frozen configuration."""

    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]:
        defaults = request_defaults(read_config(artifact))
        return {**defaults, **request} if defaults else request

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None:
        return None


def create_config_surface() -> Surface:
    """Build the ``config`` component: validation and request defaults."""
    return Surface(
        components={CONFIG_COMPONENT: ComponentSurface(validator=ConfigValidator(), inference=ConfigInferenceHooks())}
    )


__all__ = [
    "CONFIG_COMPONENT",
    "CONFIG_FILE",
    "ConfigInferenceHooks",
    "ConfigValidator",
    "create_config_surface",
    "read_config",
    "request_defaults",
]
