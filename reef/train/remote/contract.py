"""The HTTP contract between Reef and a training backend service (RFC #1146), as JSON values.

Reef calls a backend service at these routes, with ``Authorization: Bearer <token>``:

- ``GET /v1/backend``: the ``BackendInfo``.
- ``POST /v1/jobs/check`` with a ``RemoteJob``: 200, or 422 with the reason.
- ``POST /v1/jobs`` with a ``RemoteJob``: 202. The same ``job_id`` again does nothing.
- ``GET /v1/jobs/{job_id}``: the ``RemoteJobStatus``; 404 for a job that the service does not know.
- ``POST /v1/jobs/{job_id}/cancel``: 202.
- ``POST /v1/releases/check`` with ``{"paths": [...], "adapter_config": {...} | null}``: 200, or 422 with the reason.

A job starts from an artifact and reads a job data directory, each at a URI that the backend can read. An artifact
directory holds ``release/``, what serves, and ``state/``, what only training reads. A directory without ``release/``
is a release only, and a job from it starts a new training. A refusal is ``{"error": {"message": str}}``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from reef.core.weight_layout import ReleaseFormat

#: The version of the contract, the first part of each route.
CONTRACT_VERSION = "v1"
#: The part of an artifact directory that serves.
RELEASE_DIR = "release"
#: The part of an artifact directory that only training reads.
STATE_DIR = "state"
RemoteJobState = Literal["running", "finished", "failed"]


def json_object(value: Any, name: str) -> Mapping[str, Any]:
    """``value`` when it is a JSON object; ``ValueError`` names ``name`` otherwise."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def json_text(value: Any, name: str) -> str:
    """``value`` when it is a non-empty string; ``ValueError`` names ``name`` otherwise."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def release_format_to_dict(release_format: ReleaseFormat | None) -> dict[str, Any] | None:
    if release_format is None:
        return None
    return {
        "layout": release_format.layout,
        "adapter_rank": release_format.adapter_rank,
        "adapter_target_modules": list(release_format.adapter_target_modules),
    }


def release_format_from_dict(value: Any) -> ReleaseFormat | None:
    if value is None:
        return None
    value = json_object(value, "release_format")
    return ReleaseFormat(value["layout"], value.get("adapter_rank"), tuple(value.get("adapter_target_modules") or ()))


@dataclass(frozen=True)
class BackendInfo:
    """What a backend service trains: its ``name``, the ``release_format`` of the releases that its checkpoints hold
    (None when no engine loads their files, such as a Tinker manifest), and whether jobs of several scenarios can run
    on it (``shares_scenarios``)."""

    name: str
    release_format: ReleaseFormat | None
    shares_scenarios: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "release_format": release_format_to_dict(self.release_format),
            "shares_scenarios": self.shares_scenarios,
        }

    @classmethod
    def from_dict(cls, value: Any) -> BackendInfo:
        value = json_object(value, "the backend")
        return cls(
            json_text(value.get("name"), "name"),
            release_format_from_dict(value.get("release_format")),
            bool(value.get("shares_scenarios", False)),
        )


@dataclass(frozen=True)
class RemoteJob:
    """One training job as a backend service gets it.

    Values: ``job_id`` and ``scenario`` are non-empty. ``artifact_uri`` is the artifact that the job starts from.
    ``data`` is ``{uri, data_id, sample_count, kind}`` of the job data directory (``reef.storage.job_data``).
    ``parameters`` are the job's parameters, a JSON object.
    """

    job_id: str
    scenario: str
    artifact_uri: str
    data: Mapping[str, Any]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scenario": self.scenario,
            "artifact": {"uri": self.artifact_uri},
            "data": dict(self.data),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RemoteJob:
        value = json_object(value, "the job")
        data = json_object(value.get("data"), "data")
        json_text(data.get("uri"), "data.uri")
        return cls(
            json_text(value.get("job_id"), "job_id"),
            json_text(value.get("scenario"), "scenario"),
            json_text(json_object(value.get("artifact"), "artifact").get("uri"), "artifact.uri"),
            dict(data),
            dict(json_object(value.get("parameters", {}), "parameters")),
        )


@dataclass(frozen=True)
class RemoteCheckpoint:
    """One checkpoint of a job: its ``step``, the ``artifact_uri`` of its artifact directory, and the backend's
    ``metrics``, a JSON object that Reef records without reading."""

    step: int
    artifact_uri: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "artifact_uri": self.artifact_uri, "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, value: Any) -> RemoteCheckpoint:
        value = json_object(value, "a checkpoint")
        step = value.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("a checkpoint's step must be a non-negative int")
        metrics = dict(json_object(value.get("metrics", {}), "a checkpoint's metrics"))
        return cls(step, json_text(value.get("artifact_uri"), "artifact_uri"), metrics)


@dataclass(frozen=True)
class RemoteJobStatus:
    """What a backend service reports about one job: its ``state``, its ``checkpoints`` in step order, its
    ``metrics``, its ``error`` (a non-empty string exactly when the job failed), and its ``progress``, a JSON object
    that the backend defines, or None."""

    job_id: str
    state: RemoteJobState
    checkpoints: tuple[RemoteCheckpoint, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    progress: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "metrics": dict(self.metrics),
            "error": self.error,
            "progress": None if self.progress is None else dict(self.progress),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RemoteJobStatus:
        value = json_object(value, "a job status")
        state = value.get("state")
        if state not in get_args(RemoteJobState):
            raise ValueError(f"a job's state must be running, finished or failed, got {state!r}")
        progress = value.get("progress")
        return cls(
            json_text(value.get("job_id"), "job_id"),
            state,
            tuple(RemoteCheckpoint.from_dict(item) for item in value.get("checkpoints", ())),
            dict(json_object(value.get("metrics", {}), "metrics")),
            value.get("error"),
            None if progress is None else dict(json_object(progress, "progress")),
        )


def refusal(message: str) -> bytes:
    """The body of a refusal."""
    return json.dumps({"error": {"message": message}}).encode()


__all__ = [
    "CONTRACT_VERSION",
    "RELEASE_DIR",
    "STATE_DIR",
    "BackendInfo",
    "RemoteCheckpoint",
    "RemoteJob",
    "RemoteJobState",
    "RemoteJobStatus",
    "refusal",
    "release_format_from_dict",
    "release_format_to_dict",
]
