"""The serving capabilities attached to one scenario.

A surface describes how one frozen release reaches inference or a
client pulling files. The capabilities are explicit: record-only surfaces
have none, while model, adapter, and harness surfaces compose only the pieces
they use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from reef.artifact.artifact import Artifact, ArtifactRef


class ServingRuntime(Protocol):
    """The runtime shape visible to surface loaders."""

    @property
    def base_url(self) -> str: ...


@runtime_checkable
class WeightRuntime(ServingRuntime, Protocol):
    """A runtime that can inspect and restore served model weights."""

    def serving_runtime_load_id(self) -> str | None: ...

    def restore_checkpoint(self, artifact: Artifact) -> str: ...


class ArtifactLoader(Protocol):
    """Runtime-backed artifact loading and startup recovery."""

    def recover(
        self,
        current: ArtifactRef | None,
        checkpoint: ArtifactRef,
        runtime: ServingRuntime | None,
    ) -> ArtifactRef: ...

    def load(self, artifact: Artifact, runtime: ServingRuntime | None) -> str: ...


@runtime_checkable
class ArtifactActivator(Protocol):
    """Optional loader capability: make a published release servable.

    Called once a release is final — after startup recovery has a
    materializable head and after a publication or rollback commit has
    minted its release — and before the scenario routes traffic to it.
    ``source`` names the artifact whose bytes a rollback republished.
    """

    def activate(
        self, artifact: Artifact, runtime: ServingRuntime | None, *, source: Artifact | None = None
    ) -> str: ...


class InferenceHooks(Protocol):
    """Request and response hooks around one provider inference."""

    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]: ...

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None: ...


class InferenceLease(Protocol):
    """Serving state held for one inference attempt; released exactly once."""

    def release(self) -> None: ...


@runtime_checkable
class LeasingInferenceHooks(Protocol):
    """Optional inference capability: hold serving state for one attempt.

    ``begin_request`` runs after ``prepare_request`` froze the artifact and
    returns the lease the service releases when the attempt ends, whether it
    completed, aborted, or failed.
    """

    def begin_request(self, artifact: Artifact, path: str) -> InferenceLease: ...


class FileTree(Protocol):
    """A client-readable file tree derived from an artifact."""

    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None: ...


@dataclass(frozen=True)
class HarnessInfo:
    """What the harness routes need beyond the file tree: the seed behind the base release and the served model."""

    seed_entries: tuple[Mapping[str, Any], ...] = ()
    served_model: str | None = None


@dataclass(frozen=True)
class Surface:
    """The explicit serving capabilities bound to one scenario.

    ``None`` means the capability is absent. Callers inspect the corresponding
    field; every recipe binds an instance of this same type.
    """

    loader: ArtifactLoader | None = None
    inference: InferenceHooks | None = None
    files: FileTree | None = None
    harness: HarnessInfo | None = None


__all__ = [
    "ArtifactActivator",
    "ArtifactLoader",
    "FileTree",
    "HarnessInfo",
    "InferenceHooks",
    "InferenceLease",
    "LeasingInferenceHooks",
    "ServingRuntime",
    "Surface",
    "WeightRuntime",
]
