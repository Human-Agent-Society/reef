from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

# Re-exported for backward compatibility: the pure identity types moved to
# reef.core so reef.core never imports from reef.artifact.
from reef.core.artifact_ref import ArtifactRef, LiveWeightArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError

__all__ = [
    "LOCAL_VERSION_PREFIX",
    "Artifact",
    "ArtifactConflict",
    "ArtifactError",
    "ArtifactMaterializationError",
    "ArtifactNotFound",
    "ArtifactPublicationError",
    "ArtifactRef",
    "ArtifactRepository",
    "ArtifactSourceError",
    "LiveWeightArtifactRef",
    "decode_artifact_ref",
    "encode_artifact_ref",
    "is_local_version",
]

#: Version prefix for process-local (staged, never published) artifacts.
#: ``Artifact.local`` and repository staging both mint versions under it;
#: ``is_local_version`` is the shared predicate for recognizing them.
LOCAL_VERSION_PREFIX = "local:"


def is_local_version(version: str) -> bool:
    """True when ``version`` names a process-local artifact without durable bytes."""
    return version.startswith(LOCAL_VERSION_PREFIX)


class ArtifactError(ReefError):
    """Base class for artifact storage and version-chain errors."""


class ArtifactSourceError(ArtifactError):
    pass


class ArtifactNotFound(ArtifactError):
    pass


class ArtifactConflict(ArtifactError):
    pass


class ArtifactMaterializationError(ArtifactError):
    pass


class ArtifactPublicationError(ArtifactError):
    pass


class ArtifactRepository(Protocol):
    """Minimal repository capability needed by an artifact."""

    def materialize(self, ref: ArtifactRef) -> Artifact: ...


class Artifact:
    """A concrete artifact bound to one scenario-scoped repository."""

    def __init__(
        self,
        ref: ArtifactRef,
        repository: ArtifactRepository | None,
        *,
        local_path: Path | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._ref = ref
        self._repository = repository
        self._local_path = None if local_path is None else Path(local_path)
        self._metadata = dict(metadata or {})

    @property
    def ref(self) -> ArtifactRef:
        return self._ref

    @property
    def repository(self) -> ArtifactRepository | None:
        return self._repository

    @property
    def local_path(self) -> Path | None:
        return self._local_path

    @property
    def is_live(self) -> bool:
        """True when this artifact references a live serving weight without a materialized local path.

        Live artifacts use ``LiveWeightArtifactRef``; their bytes live in the
        serving engine, not on disk.
        """
        return self._local_path is None and isinstance(self._ref, LiveWeightArtifactRef)

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self._metadata)

    @classmethod
    def local(
        cls,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
        repository: ArtifactRepository | None = None,
    ) -> Artifact:
        token = uuid.uuid4().hex
        return cls(
            ArtifactRef(
                artifact_id=f"{LOCAL_VERSION_PREFIX}{token}",
                version=f"{LOCAL_VERSION_PREFIX}{token}",
                parent_version=None,
            ),
            repository,
            local_path=path,
            metadata=metadata,
        )

    def with_repository(self, repository: ArtifactRepository) -> Artifact:
        return Artifact(
            self.ref,
            repository,
            local_path=self.local_path,
            metadata=self.metadata,
        )

    def materialize(self) -> Artifact:
        if self.local_path is not None or isinstance(self.ref, LiveWeightArtifactRef):
            return self
        if self.repository is None:
            raise RuntimeError("artifact has no repository")
        return self.repository.materialize(self.ref)

    def discard(self) -> None:
        if self._local_path is not None:
            shutil.rmtree(self._local_path, ignore_errors=True)
