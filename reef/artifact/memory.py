from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import RLock

from reef.artifact.artifact import (
    ARTIFACT_ID_METADATA_KEY,
    Artifact,
    ArtifactConflict,
    ArtifactMaterializationError,
    ArtifactNotFound,
    ArtifactPublicationError,
    SnapshotRef,
    artifact_components,
    content_metadata,
)
from reef.artifact.files import directory_checksum
from reef.artifact.repository import CachedRepositoryBackendFactory, StagedSnapshotRepositoryBackend


class _InMemoryStorage:
    def __init__(self, directory: Path | None) -> None:
        self.root = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.refs: dict[str, SnapshotRef] = {}
        self.paths: dict[str, Path] = {}
        self.metadata: dict[str, dict[str, object]] = {}
        #: The snapshot that each artifact ID is bound to.
        self.artifacts: dict[str, SnapshotRef] = {}

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


class InMemoryRepositoryBackend(StagedSnapshotRepositoryBackend):
    """One scenario's snapshots in process memory, with their files under ``directory`` (a new temporary
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

    def artifact_snapshot(self, artifact_id: str) -> SnapshotRef:
        with self._storage.lock:
            ref = self._storage.artifacts.get(artifact_id)
        if ref is None:
            raise ArtifactNotFound(f"repository has no artifact {artifact_id!r}")
        return ref

    def artifact_content(self, ref: SnapshotRef, artifact_id: str) -> tuple[object, dict[str, object]]:
        snapshot = self.materialize(ref)
        component = artifact_components(snapshot)[artifact_id]
        files = snapshot if component is None else snapshot.component(component)
        if files.local_path is None:
            raise ArtifactPublicationError("stored artifacts require local files")
        return directory_checksum(files.local_path), content_metadata(snapshot, component)

    def bind_artifact_id(self, artifact_id: str, ref: SnapshotRef) -> SnapshotRef | None:
        with self._storage.lock:
            existing = self._storage.artifacts.get(artifact_id)
            if existing is None:
                self._storage.artifacts[artifact_id] = ref
            return existing

    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> SnapshotRef:
        if root.local_path is None or not root.local_path.is_dir():
            raise ArtifactPublicationError("a root snapshot needs an existing local artifact directory")
        with self._storage.lock:
            if self._current is not None:
                return self._current
            ref = self._storage.copy_artifact(root.local_path, parent_revision=None)
            stored_metadata = {**root.storage_metadata, **dict(metadata)}
            self._storage.metadata[ref.revision] = stored_metadata
            self.bind_artifact_ids(self.materialize(ref))
            self._current = ref
            self._metadata = dict(stored_metadata)
            return ref

    def metadata(self) -> Mapping[str, object] | None:
        if self._current is None:
            return None
        return {key: value for key, value in (self._metadata or {}).items() if key != ARTIFACT_ID_METADATA_KEY}

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
            self.bind_artifact_ids(self.materialize(ref))
            # advance_head=False mints a pending snapshot the head does not move to.
            if advance_head:
                self._current = ref
                self._metadata = artifact.storage_metadata
            return ref

    def commit_snapshot(self, ref: SnapshotRef, *, expected_parent: SnapshotRef) -> None:
        with self._storage.lock:
            current = self.current()
            if self._storage.refs.get(ref.revision) != ref or ref.parent_revision != expected_parent.revision:
                raise ArtifactPublicationError("committed snapshot differs from its staged identity or parent")
            if current == ref:
                return
            if current != expected_parent:
                raise ArtifactConflict("repository head changed before the staged snapshot was committed")
            self._current = ref
            self._metadata = dict(self._storage.metadata[ref.revision])


class _InMemoryRepositoryBackendFactory(CachedRepositoryBackendFactory):
    def __init__(self, backend_type: type[InMemoryRepositoryBackend], directory: Path | None) -> None:
        super().__init__()
        self._backend_type = backend_type
        self._directory = None if directory is None else Path(directory)

    def _build_backend(self, scenario: str) -> InMemoryRepositoryBackend:
        return self._backend_type(scenario, directory=self._directory)
