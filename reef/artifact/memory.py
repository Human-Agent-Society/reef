from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import Lock

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactMaterializationError,
    ArtifactNotFound,
    ArtifactPublicationError,
    SnapshotRef,
)
from reef.artifact.files import directory_checksum
from reef.artifact.repository import CachedRepositoryBackendFactory, StagedReleaseRepositoryBackend


class _InMemoryStorage:
    def __init__(self, directory: Path | None) -> None:
        self.root = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.refs: dict[str, SnapshotRef] = {}
        self.paths: dict[str, Path] = {}
        self.metadata: dict[str, dict[str, object]] = {}
        self.artifacts: dict[str, Artifact] = {}

    def copy_artifact(self, source: Path, *, parent_revision: str | None) -> SnapshotRef:
        revision = uuid.uuid4().hex
        destination = self.root / revision
        try:
            shutil.copytree(source, destination)
        except OSError as exc:
            raise ArtifactPublicationError(f"failed to copy artifact into repository from {source}: {exc}") from exc
        ref = SnapshotRef(revision=revision, parent_revision=parent_revision)
        self.refs[revision] = ref
        self.paths[revision] = destination
        return ref


class InMemoryRepositoryBackend(StagedReleaseRepositoryBackend):
    """One scenario's releases in process memory, with their files under ``directory`` (a new temporary
    directory when ``None``)."""

    def __init__(self, scenario: str, *, directory: Path | None = None) -> None:
        if not scenario:
            raise ValueError("scenario must be non-empty")
        self._storage = _InMemoryStorage(directory)
        self.root = self._storage.root
        self._current: SnapshotRef | None = None
        self._metadata: dict[str, object] | None = None

    @classmethod
    def factory(cls, *, directory: Path | None = None) -> CachedRepositoryBackendFactory:
        return _InMemoryRepositoryBackendFactory(cls, directory)

    def resolve_snapshot(self, revision: str) -> SnapshotRef:
        ref = self._storage.refs.get(revision)
        if ref is not None:
            return ref
        raise ArtifactNotFound(f"release does not exist: {revision}")

    def index_snapshot(self, ref: SnapshotRef) -> None:
        """Add immutable ID bindings for a previously stored snapshot."""
        with self._storage.lock:
            self.bind_artifact_ids(ref)

    def resolve_artifact(self, artifact_id: str) -> Artifact:
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("artifact_id must be a non-empty string")
        with self._storage.lock:
            artifact = self._storage.artifacts.get(artifact_id)
            if artifact is None:
                raise ArtifactNotFound(f"repository has no artifact {artifact_id!r}")
            return artifact

    def bind_artifact_ids(self, ref: SnapshotRef) -> None:
        """Index a stored snapshot while the caller holds the storage lock; reject changed ID bindings."""
        snapshot = self.materialize(ref)
        manifest = snapshot.components
        artifacts = [snapshot] if manifest is None else [snapshot.component(name) for name in manifest.names]
        identities = {artifact.artifact_id for artifact in artifacts}
        if len(identities) != len(artifacts):
            raise ArtifactPublicationError("one artifact ID cannot name several components in a snapshot")
        for artifact in artifacts:
            existing = self._storage.artifacts.get(artifact.artifact_id)
            if existing is None:
                continue
            if existing.local_path is None or artifact.local_path is None:
                raise ArtifactPublicationError("stored artifacts require local files")
            existing_metadata, incoming_metadata = (
                {
                    key: value
                    for key, value in item.metadata.items()
                    if key not in ("artifact_id", "components", "scenario_commit_record")
                }
                for item in (existing, artifact)
            )
            if (
                directory_checksum(existing.local_path) != directory_checksum(artifact.local_path)
                or existing_metadata != incoming_metadata
            ):
                raise ArtifactConflict(f"artifact {artifact.artifact_id!r} already names different files or metadata")
        for artifact in artifacts:
            self._storage.artifacts.setdefault(artifact.artifact_id, artifact)

    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> SnapshotRef:
        if root.local_path is None or not root.local_path.is_dir():
            raise ArtifactPublicationError("a root release needs an existing local artifact directory")
        with self._storage.lock:
            if self._current is not None:
                return self._current
            ref = self._storage.copy_artifact(root.local_path, parent_revision=None)
            stored_metadata = {**root.storage_metadata, **dict(metadata)}
            self._storage.metadata[ref.revision] = stored_metadata
            self.bind_artifact_ids(ref)
            self._current = ref
            self._metadata = dict(stored_metadata)
            return ref

    def metadata(self) -> Mapping[str, object] | None:
        if self._current is None:
            return None
        return {key: value for key, value in (self._metadata or {}).items() if key != "artifact_id"}

    def current(self) -> SnapshotRef:
        if self._current is None:
            raise ArtifactNotFound("repository has no current artifact")
        return self._current

    def materialize(self, ref: SnapshotRef) -> Artifact:
        path = self._storage.paths.get(ref.revision)
        if path is None or not path.is_dir():
            raise ArtifactMaterializationError(f"release does not exist: {ref.revision}")
        return Artifact(ref, None, local_path=path, metadata=self._storage.metadata.get(ref.revision))

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: SnapshotRef,
        advance_head: bool = True,
    ) -> SnapshotRef:
        if artifact.local_path is None or not artifact.local_path.is_dir():
            raise ArtifactPublicationError("artifact ref must contain an existing local artifact directory")
        with self._storage.lock:
            current = self.current()
            if current.revision != expected_parent.revision:
                raise ArtifactConflict(
                    f"repository is at {current.revision}, not expected parent {expected_parent.revision}"
                )
            ref = self._storage.copy_artifact(
                artifact.local_path,
                parent_revision=current.revision,
            )
            self._storage.metadata[ref.revision] = artifact.storage_metadata
            self.bind_artifact_ids(ref)
            # advance_head=False mints a pending release the head does not move to.
            if advance_head:
                self._current = ref
                self._metadata = artifact.storage_metadata
            return ref

    def commit_release(self, ref: SnapshotRef, *, expected_parent: SnapshotRef) -> None:
        with self._storage.lock:
            current = self.current()
            if self._storage.refs.get(ref.revision) != ref or ref.parent_revision != expected_parent.revision:
                raise ArtifactPublicationError("committed release differs from its staged identity or parent")
            if current == ref:
                return
            if current != expected_parent:
                raise ArtifactConflict("repository head changed before the staged release was committed")
            self._current = ref
            self._metadata = dict(self._storage.metadata[ref.revision])


class _InMemoryRepositoryBackendFactory(CachedRepositoryBackendFactory):
    def __init__(self, backend_type: type[InMemoryRepositoryBackend], directory: Path | None) -> None:
        super().__init__()
        self._backend_type = backend_type
        self._directory = None if directory is None else Path(directory)

    def _build_backend(self, scenario: str) -> InMemoryRepositoryBackend:
        return self._backend_type(scenario, directory=self._directory)
