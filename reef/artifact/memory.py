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
from reef.artifact.repository import CachedRepositoryBackendFactory, RepositoryBackend


class _InMemoryStorage:
    def __init__(self, bootstrap_path: Path, root: Path | None) -> None:
        source = Path(bootstrap_path)
        if not source.is_dir():
            raise ArtifactNotFound(f"bootstrap artifact directory does not exist: {source}")
        self.root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.refs: dict[str, ArtifactRef] = {}
        self.paths: dict[str, Path] = {}
        self.latest = self.copy_artifact(source, parent_version=None, artifact_id="initial")

    def copy_artifact(
        self,
        source: Path,
        *,
        parent_version: str | None,
        artifact_id: str | None = None,
    ) -> ArtifactRef:
        version = uuid.uuid4().hex
        destination = self.root / version
        try:
            shutil.copytree(source, destination)
        except OSError as exc:
            raise ArtifactPublicationError(f"failed to copy artifact into repository from {source}: {exc}") from exc
        ref = ArtifactRef(
            artifact_id=artifact_id or f"artifact:{version}",
            version=version,
            parent_version=parent_version,
        )
        self.refs[version] = ref
        self.paths[version] = destination
        return ref


class InMemoryRepositoryBackend(RepositoryBackend):
    def __init__(
        self,
        scenario: str,
        bootstrap_path: Path,
        *,
        root: Path | None = None,
    ) -> None:
        if not scenario:
            raise ValueError("scenario must be non-empty")
        self._storage = _InMemoryStorage(bootstrap_path, root)
        self.root = self._storage.root
        self._current: ArtifactRef | None = None
        self._metadata: dict[str, object] | None = None

    @classmethod
    def factory(
        cls,
        bootstrap_path: Path,
        *,
        root: Path | None = None,
    ) -> CachedRepositoryBackendFactory:
        return _InMemoryRepositoryBackendFactory(
            cls,
            bootstrap_path,
            root,
        )

    def resolve_version(self, version: str | None = None) -> ArtifactRef:
        if version is None or version == "latest":
            return self._storage.latest
        ref = self._storage.refs.get(version)
        if ref is not None:
            return ref
        for candidate in self._storage.refs.values():
            if candidate.artifact_id == version:
                return candidate
        raise ArtifactNotFound(f"artifact version does not exist: {version}")

    def fork(
        self,
        artifact_version: str | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        with self._storage.lock:
            if self._current is not None:
                return self._current
            selected = self.resolve_version(artifact_version)
            ref = self._storage.copy_artifact(
                self._storage.paths[selected.version],
                parent_version=selected.version,
            )
            self._current = ref
            self._metadata = dict(metadata or {})
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
        path = self._storage.paths.get(ref.version)
        if path is None or not path.is_dir():
            raise ArtifactMaterializationError(f"artifact version does not exist: {ref.version}")
        return Artifact(ref, None, local_path=path)

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef,
    ) -> ArtifactRef:
        if artifact.local_path is None or not artifact.local_path.is_dir():
            raise ArtifactPublicationError("artifact ref must contain an existing local artifact directory")
        with self._storage.lock:
            current = self.current()
            if current.version != expected_parent.version:
                raise ArtifactConflict(
                    f"repository is at {current.version}, not expected parent {expected_parent.version}"
                )
            ref = self._storage.copy_artifact(
                artifact.local_path,
                parent_version=current.version,
            )
            self._current = ref
            self._metadata = dict(artifact.metadata)
            self._storage.latest = ref
            return ref


class _InMemoryRepositoryBackendFactory(CachedRepositoryBackendFactory):
    def __init__(
        self,
        backend_type: type[InMemoryRepositoryBackend],
        bootstrap_path: Path,
        root: Path | None,
    ) -> None:
        super().__init__()
        self._backend_type = backend_type
        self._bootstrap_path = Path(bootstrap_path)
        self._root = None if root is None else Path(root)

    def _build_backend(self, scenario: str) -> InMemoryRepositoryBackend:
        return self._backend_type(
            scenario,
            self._bootstrap_path,
            root=self._root,
        )
