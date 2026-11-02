from __future__ import annotations

import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from reef.core.components import ReleaseComponents, release_components
from reef.core.errors import ReefError

# Re-export the pure identity types while keeping ``reef.core`` independent of
# storage implementations.
from reef.core.snapshot import SnapshotRef, decode_snapshot_ref, encode_snapshot_ref

__all__ = [
    "LOCAL_SNAPSHOT_PREFIX",
    "Artifact",
    "ArtifactConflict",
    "ArtifactError",
    "ArtifactMaterializationError",
    "ArtifactNotFound",
    "ArtifactPublicationError",
    "ArtifactRepository",
    "ArtifactSourceError",
    "ArtifactValidator",
    "SnapshotRef",
    "decode_snapshot_ref",
    "encode_snapshot_ref",
    "is_local_snapshot",
]

#: Release prefix for process-local (staged, never published) artifacts.
#: ``Artifact.local`` and repository staging both mint releases under it;
#: ``is_local_snapshot`` is the shared predicate for recognizing them.
LOCAL_SNAPSHOT_PREFIX = "local:"
#: Files a repository adds to a materialized release; a release with nothing else holds no weights.
REPOSITORY_FILES = frozenset({".git", ".gitattributes", "reef-artifact.json"})


def is_local_snapshot(revision: str) -> bool:
    """True when ``revision`` names a process-local artifact without durable bytes."""
    return revision.startswith(LOCAL_SNAPSHOT_PREFIX)


class ArtifactError(ReefError):
    """Base class for artifact storage and release-chain errors."""


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


class ArtifactRepository(ABC):
    """Minimal repository capability needed by an artifact."""

    @abstractmethod
    def materialize(self, ref: SnapshotRef) -> Artifact: ...


class ArtifactValidator(ABC):
    """Artifact admission checks selected for one scenario."""

    @abstractmethod
    def validate(self, artifact: Artifact) -> None: ...


class Artifact:
    """A concrete artifact bound to one scenario-scoped repository."""

    def __init__(
        self,
        ref: SnapshotRef,
        repository: ArtifactRepository | None,
        *,
        local_path: Path | None = None,
        metadata: Mapping[str, object] | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self._ref = ref
        self._repository = repository
        self._local_path = None if local_path is None else Path(local_path)
        self._metadata = dict(metadata or {})
        stored_identity = self._metadata.pop("artifact_id", None)
        if stored_identity is not None and (not isinstance(stored_identity, str) or not stored_identity):
            raise ValueError("stored artifact_id must be a non-empty string")
        if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id):
            raise ValueError("artifact_id must be a non-empty string")
        if artifact_id is not None and stored_identity is not None and artifact_id != stored_identity:
            raise ValueError("artifact_id differs from its stored identity")
        self._artifact_id = artifact_id if artifact_id is not None else stored_identity

    @property
    def ref(self) -> SnapshotRef:
        return self._ref

    @property
    def repository(self) -> ArtifactRepository | None:
        return self._repository

    @property
    def local_path(self) -> Path | None:
        return self._local_path

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self._metadata)

    @property
    def storage_metadata(self) -> dict[str, object]:
        """Serialize the artifact's metadata and explicit identity for snapshot storage."""
        metadata = dict(self._metadata)
        manifest = self.components
        if manifest is not None:
            metadata["components"] = manifest.carried_from(self.ref.revision).to_dict()
        if self._artifact_id is not None:
            metadata["artifact_id"] = self._artifact_id
        return metadata

    @property
    def artifact_id(self) -> str:
        """The artifact that this value holds: the artifact of a component view, or the only component of a release.

        New artifacts record an independent ID before publication. A component
        view keeps that ID when another snapshot carries it. Legacy snapshots
        without an explicit ID use their original revision-based identity.
        A snapshot containing several components has no single artifact ID;
        callers must select a component.
        """
        if self._artifact_id is not None:
            return self._artifact_id
        materialized = self if self.repository is None or self.local_path is not None else self.materialize()
        if materialized._artifact_id is not None:
            return materialized._artifact_id
        # Provider calls can carry a ref without local files or a repository.
        manifest = materialized.components
        if manifest is None:
            return self.ref.revision
        if not manifest.single:
            raise ArtifactNotFound(
                f"release {self.ref.revision!r} holds components {list(manifest.names)}; name one of them"
            )
        return next(iter(manifest.artifact_ids(self.ref.revision).values()))

    @property
    def components(self) -> ReleaseComponents | None:
        """The release's component manifest; ``None`` for a release published without one."""
        return release_components(self._metadata)

    def component(self, name: str) -> Artifact:
        """The named component of this release, materialized.

        A release without a manifest, or with one component, is that component
        itself. A multi-component release keeps each component in a directory
        named after it; the view carries the component's own artifact ID and metadata.
        """
        materialized = self.materialize()
        manifest = materialized.components
        if manifest is None:
            return materialized
        if name not in manifest.entries:
            raise ArtifactNotFound(f"release {self.ref.revision!r} binds no component {name!r}")
        if manifest.single:
            return materialized
        if materialized.local_path is None:
            raise ArtifactMaterializationError(f"release {self.ref.revision!r} has no local component directories")
        entry = manifest.entries[name]
        directory = materialized.local_path / manifest.relative_path(name)
        # A component the base seeded nothing for has no directory in the
        # release (git keeps none for an empty tree); it is empty, not missing.
        # Only the leaf is created, and only inside a release tree that is
        # there: a release directory that is gone must not come back empty.
        if materialized.local_path.is_dir() and not directory.exists():
            # Two readers of one release race here; the second finds the leaf made.
            directory.mkdir(exist_ok=True)
        return Artifact(
            SnapshotRef(self.ref.revision, self.ref.parent_revision),
            None,
            local_path=directory,
            metadata=entry.metadata,
            artifact_id=manifest.artifact_ids(self.ref.revision)[name],
        )

    @classmethod
    def local(
        cls,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
        repository: ArtifactRepository | None = None,
    ) -> Artifact:
        """Create an unpublished artifact with a fresh ID that staging and publication preserve."""
        token = uuid.uuid4().hex
        stored_metadata = dict(metadata or {})
        stored_metadata.pop("artifact_id", None)
        manifest = release_components(stored_metadata)
        if manifest is None:
            stored_metadata["artifact_id"] = f"artifact:{token}"
        else:
            manifest = ReleaseComponents(
                {
                    name: replace(entry, artifact_id=entry.artifact_id or f"artifact:{uuid.uuid4().hex}")
                    for name, entry in manifest.entries.items()
                }
            )
            stored_metadata["components"] = manifest.to_dict()
        return cls(
            SnapshotRef(revision=f"{LOCAL_SNAPSHOT_PREFIX}{token}", parent_revision=None),
            repository,
            local_path=path,
            metadata=stored_metadata,
        )

    def with_repository(self, repository: ArtifactRepository) -> Artifact:
        return Artifact(
            self.ref,
            repository,
            local_path=self.local_path,
            metadata=self.metadata,
            artifact_id=self._artifact_id,
        )

    def materialize(self) -> Artifact:
        if self.local_path is not None:
            return self
        if self.repository is None:
            raise RuntimeError("artifact has no repository")
        return self.repository.materialize(self.ref)

    def discard(self) -> None:
        if self._local_path is not None:
            shutil.rmtree(self._local_path, ignore_errors=True)
