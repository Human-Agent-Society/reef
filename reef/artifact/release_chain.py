"""Artifact-only operations for one ordered chain of immutable releases."""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from reef.artifact.artifact import Artifact, ArtifactRef, LiveWeightArtifactRef
from reef.artifact.repository import Repository
from reef.core.errors import ReefError


class ReleaseNotRestorable(ReefError):
    """A release has identity but no durable bytes to restore."""


class ArtifactReleaseChain:
    """Own artifact heads, parent links, staging, resolution, and publication.

    This layer knows nothing about scenarios, trainers, commit logs, or
    checkpoint cadence. Its caller decides transaction ordering and records
    the outcome after each artifact operation.
    """

    def __init__(self, repository: Repository, *, process_id: str | None = None) -> None:
        self._repository = repository
        self._process_id = process_id or uuid.uuid4().hex

    @property
    def repository(self) -> Repository:
        return self._repository

    @property
    def base(self) -> ArtifactRef:
        return self._repository.base_artifact

    @property
    def current(self) -> ArtifactRef:
        return self._repository.require_current_artifact()

    @property
    def checkpoint(self) -> ArtifactRef:
        return self._repository.require_checkpoint_artifact()

    def resolve(self, ref: ArtifactRef) -> Artifact:
        return self._repository.resolve(ref)

    def prepare_live(self, *, step: int, runtime_load_id: str) -> tuple[ArtifactRef, LiveWeightArtifactRef]:
        """Build a live child of the durable checkpoint without moving a head."""
        expected = self.current
        ref = LiveWeightArtifactRef(
            content_id=f"weights:{uuid.uuid4().hex}",
            release_id=f"live:{self._process_id}:{runtime_load_id}:{step}",
            parent_release_id=self.checkpoint.release_id,
            runtime_load_id=runtime_load_id,
        )
        return expected, ref

    def advance(self, ref: ArtifactRef, *, expected: ArtifactRef) -> None:
        self._repository.advance_current(ref, expected=expected)

    def stage(self, step: int, artifact: Artifact, *, parent: ArtifactRef | None = None) -> Artifact:
        return self._repository.stage(step, artifact, parent=self.checkpoint if parent is None else parent)

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef | None = None,
        metadata: Mapping[str, object] | None = None,
        advance_heads: bool = True,
    ) -> ArtifactRef:
        return self._repository.publish(
            artifact,
            expected_parent=self.checkpoint if expected_parent is None else expected_parent,
            metadata=metadata,
            advance_heads=advance_heads,
        )

    def discard(self, artifact: Artifact) -> None:
        self._repository.discard(artifact)


__all__ = ["ArtifactReleaseChain", "ReleaseNotRestorable"]
