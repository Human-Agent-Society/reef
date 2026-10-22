"""Settings and HTTP routes shared by the remote inference component and its clients."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlsplit

#: The control vocabulary of a remote inference component.
CONTROL_PROTOCOL = "reef-remote-control-v1"
#: Every control route is under this path. Any other path is an inference request for the engines.
CONTROL_PATH = "/reef/control/v1"
DEFAULT_CONTROL_TOKEN_ENV = "REEF_REMOTE_CONTROL_TOKEN"
DEFAULT_INFERENCE_TOKEN_ENV = "REEF_REMOTE_INFERENCE_TOKEN"


@dataclass(frozen=True)
class RemoteInferenceSettings:
    """How the control plane reaches one remote inference component.

    - ``url`` is the component's HTTP(S) base URL, without credentials, query or fragment.
    - ``control_token_env`` and ``inference_token_env`` name two different
      environment variables. They hold the bearer tokens for control routes
      and for inference requests.
    - ``startup_timeout_s`` is positive and bounds one engine start, including
      the model download on the component.

    The values come from the ``inference.remote`` section, whose presence runs
    the engines that ``inference.backend`` names on the component. Service
    deployment validates them; the remote inference service reads them.
    """

    url: str
    control_token_env: str = DEFAULT_CONTROL_TOKEN_ENV
    inference_token_env: str = DEFAULT_INFERENCE_TOKEN_ENV
    startup_timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise ValueError("inference.remote.url must be a string")
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("inference.remote.url must be an HTTP(S) URL without credentials, query or fragment")
        object.__setattr__(self, "url", self.url.rstrip("/"))
        for name, value in (
            ("control-token-env", self.control_token_env),
            ("inference-token-env", self.inference_token_env),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"inference.remote.{name} must be a non-empty string")
        if self.control_token_env == self.inference_token_env:
            raise ValueError("inference.remote control and inference tokens must use different variables")
        if (
            isinstance(self.startup_timeout_s, bool)
            or not isinstance(self.startup_timeout_s, (int, float))
            or not math.isfinite(self.startup_timeout_s)
            or self.startup_timeout_s <= 0
        ):
            raise ValueError("inference.remote.startup-timeout-s must be a positive number")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RemoteInferenceSettings:
        """Read the public section; keys may use dashes or underscores."""
        if not isinstance(value, Mapping):
            raise ValueError("inference.remote must be an object")
        values = {str(key).replace("-", "_"): item for key, item in value.items()}
        unknown = sorted(set(values) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"unknown inference.remote fields: {', '.join(unknown)}")
        if "url" not in values:
            raise ValueError("inference.remote requires a url")
        return cls(**values)

    def token(self, variable: str, environ: Mapping[str, str]) -> str:
        """The bearer token in ``variable``; a missing or empty value is an error."""
        token = environ.get(variable, "").strip()
        if not token:
            raise ValueError(f"the remote inference component requires the environment variable {variable}")
        return token
