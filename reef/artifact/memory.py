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
    ArtifactRef,
)
from reef.artifact.repository import CachedRepositoryBackendFactory, StagedReleaseRepositoryBackend


class _InMemoryStorage:
    def __init__(self, directory: Path | None) -> None:
        self.root = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.refs: dict[str, ArtifactRef] = {}
        self.paths: dict[str, Path] = {}
        self.metadata: dict[str, dict[str, object]] = {}

    def copy_artifact(self, source: Path, *, parent_release_id: str | None, content_id: str) -> ArtifactRef:
        release_id = uuid.uuid4().hex
        destination = self.root / release_id
        try:
            shutil.copytree(source, destination)
        except OSError as exc:
            raise ArtifactPublicationError(f"failed to copy artifact into repository from {source}: {exc}") from exc
        ref = ArtifactRef(content_id=content_id, release_id=release_id, parent_release_id=parent_release_id)
        self.refs[release_id] = ref
        self.paths[release_id] = destination
        return ref


class InMemoryRepositoryBackend(StagedReleaseRepositoryBackend):
    """One scenario's releases in process memory, with their files under ``directory`` (a new temporary
    directory when ``None``)."""

    def __init__(self, scenario: str, *, directory: Path | None = None) -> None:
        if not scenario:
            raise ValueError("scenario must be non-empty")
        self._storage = _InMemoryStorage(directory)
        self.root = self._storage.root
        self._current: ArtifactRef | None = None
        self._metadata: dict[str, object] | None = None

    @classmethod
    def factory(cls, *, directory: Path | None = None) -> CachedRepositoryBackendFactory:
        return _InMemoryRepositoryBackendFactory(cls, directory)

    def resolve_release(self, release_id: str) -> ArtifactRef:
        ref = self._storage.refs.get(release_id)
        if ref is not None:
            return ref
        raise ArtifactNotFound(f"release does not exist: {release_id}")

    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> ArtifactRef:
        if root.local_path is None or not root.local_path.is_dir():
            raise ArtifactPublicationError("a root release needs an existing local artifact directory")
        with self._storage.lock:
            if self._current is not None:
                return self._current
            ref = self._storage.copy_artifact(root.local_path, parent_release_id=None, content_id=root.ref.content_id)
            self._current = ref
            self._metadata = dict(metadata)
            self._storage.metadata[ref.release_id] = dict(metadata)
            return ref

    def metadata(self) -> Mapping[str, object] | None:
        if self._current is None:
            return None
        return dict(self._metadata or {})

    def current(self) -> ArtifactRef:
        if self._current is None:
            raise ArtifactNotFound("repository has no current artifact")
        return self._current

    def materialize(self, ref: ArtifactRef) -> Artifact:
        path = self._storage.paths.get(ref.release_id)
        if path is None or not path.is_dir():
            raise ArtifactMaterializationError(f"release does not exist: {ref.release_id}")
        return Artifact(ref, None, local_path=path, metadata=self._storage.metadata.get(ref.release_id))

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef,
        advance_head: bool = True,
    ) -> ArtifactRef:
        if artifact.local_path is None or not artifact.local_path.is_dir():
            raise ArtifactPublicationError("artifact ref must contain an existing local artifact directory")
        with self._storage.lock:
            current = self.current()
            if current.release_id != expected_parent.release_id:
                raise ArtifactConflict(
                    f"repository is at {current.release_id}, not expected parent {expected_parent.release_id}"
                )
            ref = self._storage.copy_artifact(
                artifact.local_path,
                parent_release_id=current.release_id,
                content_id=artifact.ref.content_id,
            )
            self._storage.metadata[ref.release_id] = dict(artifact.metadata)
            # advance_head=False mints a pending release the head does not move to.
            if advance_head:
                self._current = ref
                self._metadata = dict(artifact.metadata)
            return ref

    def commit_release(self, ref: ArtifactRef, *, expected_parent: ArtifactRef) -> None:
        with self._storage.lock:
            current = self.current()
            if self._storage.refs.get(ref.release_id) != ref or ref.parent_release_id != expected_parent.release_id:
                raise ArtifactPublicationError("committed release differs from its staged identity or parent")
            if current == ref:
                return
            if current != expected_parent:
                raise ArtifactConflict("repository head changed before the staged release was committed")
            self._current = ref
            self._metadata = dict(self._storage.metadata[ref.release_id])


class _InMemoryRepositoryBackendFactory(CachedRepositoryBackendFactory):
    def __init__(self, backend_type: type[InMemoryRepositoryBackend], directory: Path | None) -> None:
        super().__init__()
        self._backend_type = backend_type
        self._directory = None if directory is None else Path(directory)

    def _build_backend(self, scenario: str) -> InMemoryRepositoryBackend:
        return self._backend_type(scenario, directory=self._directory)
