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
    "ArtifactConflictError",
    "ArtifactError",
    "ArtifactMaterializationError",
    "ArtifactNotFoundError",
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
    """Base class for artifact storage and snapshot-chain errors."""


class ArtifactSourceError(ArtifactError):
    pass


class ArtifactNotFoundError(ArtifactError):
    pass


class ArtifactConflictError(ArtifactError):
    pass


class ArtifactMaterializationError(ArtifactError):
    pass


class ArtifactPublicationError(ArtifactError):
    pass


class ArtifactRepository(ABC):
    """Minimal repository capability needed by an artifact."""

    @abstractmethod
    def materialize(self, ref: SnapshotRef) -> Artifact: ...

    def materialize_component(self, ref: SnapshotRef, name: str) -> Artifact:
        """The component ``name`` of the snapshot ``ref`` with its files in a local directory, as
        ``Artifact.component`` gives it.

        This default materializes the whole snapshot. A repository whose snapshots hold large components makes
        only the files of ``name`` local, so a reader of one component does not fetch the others.
        """
        return self.materialize(ref).component(name)

    def materialize_component_files(
        self, ref: SnapshotRef, name: str | None, paths: tuple[str, ...] | None
    ) -> Artifact:
        """The component ``name`` of the snapshot ``ref`` (the snapshot itself for None) in a local directory that
        holds at least each file of ``paths``, relative to the component, that the component has: all its files
        when ``paths`` is None. Other files may be missing, so a reader reads only ``paths``.

        This default makes the whole component local (``materialize_component``). A repository whose snapshots
        hold large files makes only ``paths`` local, so a reader of a few small files fetches no weights.
        """
        return self.materialize(ref) if name is None else self.materialize_component(ref, name)


class ArtifactValidator(ABC):
    """An admission check of the artifacts of one component of a scenario.

    ``validate(artifact)`` returns when the check admits ``artifact``, and raises ``ValueError`` or a ``ReefError``
    that names what does not hold. ``artifact`` holds the files of one component alone, in a local directory
    (``Artifact.local_path``); the check reads only those files and the artifact's metadata.

    Purpose: a recipe binds it on ``ComponentSpec.validator`` or ``ScenarioComponents.validator``. The process that
    holds an artifact's files for its own work runs the check once, before the artifact is stored: the training
    service on each checkpoint of a job (``reef.train.job_queue.TrainingJobWorker``), and control on an import, a
    sync and the artifact item that starts a scenario, whose files it copies. A stored artifact does not change, so
    an activation, a restore, a serve and a train-from run no check, and no process fetches files to check them.
    """

    @abstractmethod
    def validate(self, artifact: Artifact) -> None: ...


class Artifact:
    """A concrete artifact bound to one scenario-scoped repository.

    ``content`` is None, or the non-empty name of the stored content that holds the files of an unpublished
    artifact (``Artifact.stored``); such an artifact has no ``local_path``.
    """

    def __init__(
        self,
        ref: SnapshotRef,
        repository: ArtifactRepository | None,
        *,
        local_path: Path | None = None,
        metadata: Mapping[str, object] | None = None,
        artifact_id: str | None = None,
        content: str | None = None,
    ) -> None:
        if content is not None and (not isinstance(content, str) or not content or local_path is not None):
            raise ValueError("content is the non-empty name of stored content, on an artifact with no local_path")
        self._ref = ref
        self._repository = repository
        self._local_path = None if local_path is None else Path(local_path)
        self._content = content
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
    def content(self) -> str | None:
        """The name of the stored content that holds this unpublished artifact's files, or None."""
        return self._content

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
        # Provider calls can carry a ref without local files or a repository. An artifact that carries its
        # metadata answers from it, and no file is fetched.
        materialized = self if self.repository is None or self._metadata else self.materialize()
        if materialized._artifact_id is not None:
            return materialized._artifact_id
        manifest = materialized.components
        if manifest is None:
            raise ArtifactNotFoundError(f"snapshot {self.ref.revision!r} names no artifact ID")
        if not manifest.single:
            raise ArtifactNotFoundError(
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
        itself. A multi-component snapshot keeps each component in a directory
        named after it; the view carries the component's own artifact ID and metadata. A snapshot of several
        components that is not local yet, and that carries its manifest, makes only this component local
        (``ArtifactRepository.materialize_component``).
        """
        known = self.components
        if self.local_path is None and self.repository is not None and known is not None and not known.single:
            if name not in known.entries:
                raise ArtifactNotFoundError(f"snapshot {self.ref.revision!r} binds no component {name!r}")
            return self.repository.materialize_component(self.ref, name)
        materialized = self.materialize()
        manifest = materialized.components
        if manifest is None:
            return materialized
        if name not in manifest.entries:
            raise ArtifactNotFoundError(f"snapshot {self.ref.revision!r} binds no component {name!r}")
        if manifest.single:
            return materialized
        if materialized.local_path is None:
            raise ArtifactMaterializationError(f"snapshot {self.ref.revision!r} has no local component directories")
        entry = manifest.entries[name]
        directory = materialized.local_path / manifest.relative_path(name)
        # A component the base seeded nothing for has no directory in the
        # snapshot (git keeps none for an empty tree); it is empty, not missing.
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

    @classmethod
    def stored(cls, content: str) -> Artifact:
        """An unpublished artifact with a fresh ID whose files are the stored content ``content`` of its scenario's
        repository backend (``RepositoryBackend.store_content``). It has no local path: the process that makes it
        need not hold the files."""
        token = uuid.uuid4().hex
        return cls(
            SnapshotRef(revision=f"{LOCAL_SNAPSHOT_PREFIX}{token}", parent_revision=None),
            None,
            metadata={ARTIFACT_ID_METADATA_KEY: f"artifact:{token}"},
            content=content,
        )

    def with_repository(self, repository: ArtifactRepository) -> Artifact:
        return Artifact(
            self.ref,
            repository,
            local_path=self.local_path,
            metadata=self.metadata,
            artifact_id=self._artifact_id,
            content=self.content,
        )

    def component_files(self, name: str | None, paths: tuple[str, ...] | None) -> Artifact:
        """The component ``name`` of this snapshot (the snapshot itself for None) in a local directory that holds
        at least each file of ``paths`` that the component has, or all its files for None
        (``ArtifactRepository.materialize_component_files``). A local snapshot gives its local component."""
        if self.local_path is not None or self.repository is None:
            return self if name is None else self.component(name)
        return self.repository.materialize_component_files(self.ref, name, paths)

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
        content=artifact.content,
    )
