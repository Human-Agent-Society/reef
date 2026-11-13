"""The values that tell an inference service what to serve: served artifacts and the keys that name them.

Control decides what each scenario serves and writes it to the ``served_models`` table
(``reef.storage.served_models``). An inference service reads that table on each request and sends the request to
the engine that the value names. The ``artifact_id`` of the value is the only identity that crosses services.

Reef's engines serve each load under its own model name: an adapter by its scenario and load, and full weights by
their load. Every artifact therefore has its own value, and the inference component refuses a name that it no longer
serves before any output. It also changes no weights while a request that it admitted still generates, so a request
answered under a value's name was answered by its artifact. A runtime that serves an immutable provider or sampler
revision gives the same guarantee without the component.

A scenario whose snapshot loads no weights, such as a record-only or harness scenario, is answered by the deployment's
configured model. The value of each artifact that it serves names that artifact at that model
(``InferenceRuntime.endpoint_model_name``) and identifies no weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlparse

from reef.core.values import require_text

#: The name of a scenario's active artifact in ``ScenarioModel``.
ACTIVE = "active"
#: The seconds an inference replica caches ``served_models`` at most. Control keeps a replaced value serving at
#: least this long, so that a replica that read it before the change still reaches it.
CACHE_SECONDS = 5.0


@dataclass(frozen=True)
class ServedArtifact:
    """The route of one artifact of a scenario: the address and name that answer requests for it.

    Values: ``artifact_id`` is a non-empty ID of an artifact of a scenario. ``url`` is an HTTP(S) base URL without
    credentials, query or fragment: the engine outside the Reef process. ``model_name`` is a non-empty name that the
    engine answers under. For an artifact with weights, the engine served that artifact's weights under ``model_name``
    when the runtime made the value. The value does not keep that binding for a request; an immutable revision or a
    receiver that pins requests does (``InferenceRuntime.served_artifact``). For an artifact of a scenario without a
    weight component, such as a record-only or harness scenario, ``model_name`` is the deployment's configured model:
    the value associates that model with the artifact and identifies no weights.

    Purpose: a runtime makes it for the weights that it serves (``InferenceRuntime.served_artifact`` and
    ``serve_artifact``), and the inference worker makes it for each artifact that a scenario without a weight
    component serves (``InferenceRuntime.endpoint_model_name``); inference writes it to ``served_models``, and
    inference replicas send requests with it.
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

    Purpose: inference writes ``served_models`` under it; an inference replica reads the row of the request's key.
    """

    scenario: str
    name: str = ACTIVE

    def __post_init__(self) -> None:
        require_text(self.scenario, "scenario")
        require_text(self.name, "name")


@dataclass(frozen=True)
class ScheduledStop:
    """A served artifact that inference stops serving at a time.

    Values: ``served`` is a ``ServedArtifact``; ``stop_after`` is a finite time in seconds since the epoch.

    Purpose: a route update adds one when it replaces or removes a ``served_models`` value that no row names any more, so
    that requests that already read the old value end before it stops (``stop_delay_seconds``). Inference reads the
    due ones, asks the runtime at ``served.url`` to stop serving the value when no row names it, and deletes them.
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


__all__ = ["ACTIVE", "CACHE_SECONDS", "ScenarioModel", "ScheduledStop", "ServedArtifact"]
