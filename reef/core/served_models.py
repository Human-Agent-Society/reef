"""The values that tell an inference service what to serve: served artifacts and the keys that name them.

Control decides what each scenario serves and writes it to the ``served_models`` table
(``reef.storage.served_models``). An inference service reads that table on each request and sends the request to
the engine that the value names. Reef never changes the weights under a served artifact: to serve other weights,
control serves a new ``ServedArtifact`` and stops serving the old one later. So the ``artifact_id`` of the value
tells which weights answered a request, and no other identity crosses services.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

#: The name of a scenario's active artifact in ``ScenarioModel``.
ACTIVE = "active"


def require_text(value: object, name: str) -> str:
    """``value`` when it is a non-empty string without surrounding spaces; else ``ValueError``."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding spaces")
    return value


@dataclass(frozen=True)
class ServedArtifact:
    """One artifact that an engine serves, and the address and name that reach it.

    Values: ``artifact_id`` names an artifact of a scenario. ``url`` is an HTTP(S) base URL without credentials,
    query or fragment: the engine outside the Reef process. ``model_name`` is the name that the engine serves the
    artifact under, a non-empty string. From ``serve`` to ``stop_serving``, ``(url, model_name)`` answers with the
    weights of ``artifact_id`` only.

    Purpose: an inference service makes it when it starts serving an artifact (``InferenceServing.serve``);
    control writes it to ``served_models``, and inference replicas send requests with it.
    """

    artifact_id: str
    url: str
    model_name: str

    def __post_init__(self) -> None:
        require_text(self.artifact_id, "artifact_id")
        require_text(self.model_name, "model_name")
        parsed = urlparse(require_text(self.url, "url"))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"url must be an HTTP(S) URL, not {self.url!r}")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("url must not carry credentials, a query or a fragment")
        object.__setattr__(self, "url", self.url.rstrip("/"))

    def to_json(self) -> dict[str, str]:
        """The value as a JSON object with the three fields."""
        return {"artifact_id": self.artifact_id, "url": self.url, "model_name": self.model_name}

    @classmethod
    def from_json(cls, value: object) -> ServedArtifact:
        """The value that ``to_json`` wrote; ``ValueError`` for any other object."""
        if not isinstance(value, dict) or set(value) != {"artifact_id", "url", "model_name"}:
            raise ValueError("a served artifact is an object with artifact_id, url and model_name")
        return cls(value["artifact_id"], value["url"], value["model_name"])


@dataclass(frozen=True)
class ScenarioModel:
    """The key of one row of ``served_models``: a model that a scenario serves.

    Values: ``scenario`` is a non-empty scenario name. ``name`` is ``ACTIVE`` for the scenario's active artifact, an
    artifact ID for an artifact served beside the active one, or the name of an auxiliary model that the scenario
    configures, such as a judge. ``name`` is a non-empty string.

    Purpose: control writes ``served_models`` under it; an inference replica reads the row of the request's key.
    """

    scenario: str
    name: str = ACTIVE

    def __post_init__(self) -> None:
        require_text(self.scenario, "scenario")
        require_text(self.name, "name")


@dataclass(frozen=True)
class ScheduledStop:
    """A served artifact that control stops serving at a time.

    Values: ``served`` is a ``ServedArtifact``; ``stop_after`` is a finite time in seconds since the epoch.

    Purpose: control adds one when an activation replaces a served artifact, so that requests that already use the
    old one end before it stops (``stop_delay_seconds``); control reads the due ones and stops serving them.
    """

    served: ServedArtifact
    stop_after: float

    def __post_init__(self) -> None:
        if not isinstance(self.served, ServedArtifact):
            raise TypeError("a scheduled stop's served value is a ServedArtifact")
        if isinstance(self.stop_after, bool) or not isinstance(self.stop_after, (int, float)):
            raise TypeError("stop_after is a number of seconds")
        if not math.isfinite(self.stop_after):
            raise ValueError("stop_after is a finite time")

    def to_json(self) -> dict[str, Any]:
        return {"served": self.served.to_json(), "stop_after": self.stop_after}


__all__ = ["ACTIVE", "ScenarioModel", "ScheduledStop", "ServedArtifact", "require_text"]
