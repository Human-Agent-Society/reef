"""Artifact-only operations for one ordered chain of immutable releases."""

from __future__ import annotations

from collections.abc import Mapping

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.repository import Repository
from reef.core.errors import ReefError


class ReleaseNotRestorable(ReefError):
    """A release has identity but no durable bytes to restore."""


class ArtifactReleaseChain:
    """Own artifact heads, parent links, staging, resolution, and publication.

    ``current`` is the active artifact, the one that serves. ``checkpoint``
    is the tip of the published chain: the parent of the next publication.
    Publication of an activated artifact moves both; an activation moves
    only ``current``.

    This layer knows nothing about scenarios, trainers, or commit histories. Its
    caller decides transaction ordering and records the outcome after each
    artifact operation.
    """

    def __init__(self, repository: Repository) -> None:
        self._repository = repository

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

    def activate(self, ref: ArtifactRef, *, expected: ArtifactRef) -> None:
        """Make ``ref`` the active artifact (``current``); the checkpoint, the chain tip, stays."""
        self._repository.activate(ref, expected=expected)

    def install_checkpoint(
        self, ref: ArtifactRef, *, expected: ArtifactRef, expected_checkpoint: ArtifactRef, activate: bool
    ) -> None:
        self._repository.install_checkpoint(
            ref, expected=expected, expected_checkpoint=expected_checkpoint, activate=activate
        )

    def stage(self, step: int, artifact: Artifact, *, parent: ArtifactRef | None = None) -> Artifact:
        return self._repository.stage(step, artifact, parent=self.checkpoint if parent is None else parent)

    def stage_composed(
        self, step: int, components: Mapping[str, Artifact], *, parent: ArtifactRef | None = None
    ) -> Artifact:
        return self._repository.stage_composed(step, components, parent=self.checkpoint if parent is None else parent)

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef | None = None,
        metadata: Mapping[str, object] | None = None,
        advance_heads: bool = True,
        activate: bool = True,
    ) -> ArtifactRef:
        return self._repository.publish(
            artifact,
            expected_parent=self.checkpoint if expected_parent is None else expected_parent,
            metadata=metadata,
            advance_heads=advance_heads,
            activate=activate,
        )

    def discard(self, artifact: Artifact) -> None:
        self._repository.discard(artifact)


__all__ = ["ArtifactReleaseChain", "ReleaseNotRestorable"]
