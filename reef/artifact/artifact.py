from __future__ import annotations

import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from reef.core.components import COMPONENTS_METADATA_KEY, SnapshotComponents, snapshot_components
from reef.core.errors import ReefError

# Re-export the pure identity types while keeping ``reef.core`` independent of
# storage implementations.
from reef.core.snapshot import LOCAL_SNAPSHOT_PREFIX, SnapshotRef, decode_snapshot_ref, encode_snapshot_ref

__all__ = [
    "ARTIFACT_ID_METADATA_KEY",
    "LOCAL_SNAPSHOT_PREFIX",
    "SNAPSHOT_METADATA_KEYS",
    "TRAINING_STARTS_METADATA_KEY",
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
    "artifact_components",
    "content_metadata",
    "decode_snapshot_ref",
    "encode_snapshot_ref",
    "training_starts_of",
    "with_training_starts",
]

#: Files a repository adds to a materialized snapshot; a snapshot with nothing else holds no weights.
REPOSITORY_FILES = frozenset({".git", ".gitattributes", "reef-artifact.json"})
#: The stored metadata key of a flat artifact's ID; ``Artifact`` reads it into ``artifact_id`` and writes it back.
ARTIFACT_ID_METADATA_KEY = "artifact_id"
#: Metadata keys that describe a snapshot, not the content of the artifact it holds.
SNAPSHOT_METADATA_KEYS = frozenset({COMPONENTS_METADATA_KEY, "scenario_commit_record"})
#: The key of an artifact's own metadata that names its training starts (RFC #1263).
TRAINING_STARTS_METADATA_KEY = "training_starts"


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
        stored_identity = self._metadata.pop(ARTIFACT_ID_METADATA_KEY, None)
        if stored_identity is not None and (not isinstance(stored_identity, str) or not stored_identity):
            raise ValueError("stored artifact_id must be a non-empty string")
        if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id):
            raise ValueError("artifact_id must be a non-empty string")
        if artifact_id is not None and stored_identity is not None and artifact_id != stored_identity:
            raise ValueError("artifact_id differs from its stored identity")
        self._artifact_id = artifact_id or stored_identity

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
            metadata[COMPONENTS_METADATA_KEY] = manifest.carried_from(self.ref.revision).to_dict()
        if self._artifact_id is not None:
            metadata[ARTIFACT_ID_METADATA_KEY] = self._artifact_id
        return metadata

    @property
    def artifact_id(self) -> str:
        """The artifact that this value holds: the artifact of a component view, or the only component of a snapshot.

        New artifacts record an independent ID before publication. A component
        view keeps that ID when another snapshot carries it, and a use through an artifact item keeps it in another
        scenario (RFC #1218).
        A snapshot containing several components has no single artifact ID;
        callers must select a component.
        """
        if self._artifact_id is not None:
            return self._artifact_id
        # Provider calls can carry a ref without local files or a repository.
        materialized = self if self.repository is None else self.materialize()
        if materialized._artifact_id is not None:
            return materialized._artifact_id
        manifest = materialized.components
        if manifest is None:
            raise ArtifactNotFound(f"snapshot {self.ref.revision!r} names no artifact ID")
        if not manifest.single:
            raise ArtifactNotFound(
                f"snapshot {self.ref.revision!r} holds components {list(manifest.names)}; name one of them"
            )
        return next(iter(manifest.artifact_ids(self.ref.revision).values()))

    @property
    def components(self) -> SnapshotComponents | None:
        """The snapshot's component manifest; ``None`` for a snapshot published without one."""
        return snapshot_components(self._metadata)

    def component(self, name: str) -> Artifact:
        """The named component of this snapshot, materialized.

        A snapshot without a manifest, or with one component, is that component
        itself. A multi-component release keeps each component in a directory
        named after it; the view carries the component's own artifact ID and metadata.
        """
        materialized = self.materialize()
        manifest = materialized.components
        if manifest is None:
            return materialized
        if name not in manifest.entries:
            raise ArtifactNotFound(f"snapshot {self.ref.revision!r} binds no component {name!r}")
        if manifest.single:
            return materialized
        if materialized.local_path is None:
            raise ArtifactMaterializationError(f"snapshot {self.ref.revision!r} has no local component directories")
        entry = manifest.entries[name]
        directory = materialized.local_path / manifest.relative_path(name)
        # A component the base seeded nothing for has no directory in the
        # release (git keeps none for an empty tree); it is empty, not missing.
        # Only the leaf is created, and only inside a snapshot tree that is
        # there: a snapshot directory that is gone must not come back empty.
        if materialized.local_path.is_dir() and not directory.exists():
            # Two readers of one snapshot race here; the second finds the leaf made.
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
        artifact_id: str | None = None,
    ) -> Artifact:
        """Create an unpublished artifact with the ID ``artifact_id``, or a fresh one, that staging and publication
        preserve. ``artifact_id`` names an artifact that already exists elsewhere, with the same content."""
        token = uuid.uuid4().hex
        stored_metadata = dict(metadata or {})
        stored_metadata.pop(ARTIFACT_ID_METADATA_KEY, None)
        manifest = snapshot_components(stored_metadata)
        if manifest is None:
            stored_metadata[ARTIFACT_ID_METADATA_KEY] = artifact_id or f"artifact:{token}"
        else:
            manifest = SnapshotComponents(
                {
                    name: replace(entry, artifact_id=entry.artifact_id or f"artifact:{uuid.uuid4().hex}")
                    for name, entry in manifest.entries.items()
                }
            )
            stored_metadata[COMPONENTS_METADATA_KEY] = manifest.to_dict()
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


def artifact_components(snapshot: Artifact) -> dict[str, str | None]:
    """Map each artifact ID in ``snapshot`` to the name of its component.

    A snapshot without a manifest holds one artifact, which maps to None.
    """
    manifest = snapshot.components
    if manifest is None:
        return {snapshot.artifact_id: None}
    return {identity: name for name, identity in manifest.artifact_ids(snapshot.ref.revision).items()}


def content_metadata(snapshot: Artifact, component: str | None) -> dict[str, object]:
    """The metadata of one artifact in ``snapshot``: of ``component``, or of the whole snapshot when None.

    The result excludes ``SNAPSHOT_METADATA_KEYS``, so two snapshots that hold
    one artifact give equal results for it.
    """
    manifest = snapshot.components
    if component is None or manifest is None:
        return {key: value for key, value in snapshot.metadata.items() if key not in SNAPSHOT_METADATA_KEYS}
    return dict(manifest.entries[component].metadata)


def training_starts_of(artifact: Artifact) -> tuple[str, ...]:
    """The artifact IDs of the training starts of ``artifact`` (RFC #1263), nearest first: the training start of
    the trainer that made it, then that start's own training starts. Empty for an artifact that no training made.

    ``artifact`` is one artifact: a flat snapshot or a component view. Raise ``ValueError`` when its metadata holds
    another value than a list of non-empty strings.
    """
    starts = artifact.metadata.get(TRAINING_STARTS_METADATA_KEY, [])
    if not isinstance(starts, list) or not all(isinstance(start, str) and start for start in starts):
        raise ValueError(f"an artifact's training starts are a list of artifact IDs, not {starts!r}")
    return tuple(starts)


def with_training_starts(artifact: Artifact, starts: tuple[str, ...]) -> Artifact:
    """``artifact``, unpublished, with ``starts`` as the training starts in its own metadata."""
    if not starts or not all(isinstance(start, str) and start for start in starts):
        raise ValueError("training starts are a non-empty tuple of artifact IDs")
    return Artifact(
        artifact.ref,
        artifact.repository,
        local_path=artifact.local_path,
        metadata={**artifact.storage_metadata, TRAINING_STARTS_METADATA_KEY: list(starts)},
    )
