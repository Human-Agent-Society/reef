from __future__ import annotations

import contextlib
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Protocol, runtime_checkable

from reef.artifact.artifact import (
    LOCAL_VERSION_PREFIX,
    Artifact,
    ArtifactConflict,
    ArtifactPublicationError,
    ArtifactRef,
)


class RepositoryBackend(ABC):
    """Durable storage bound to one scenario repository."""

    @abstractmethod
    def resolve_version(self, version: str | None = None) -> ArtifactRef: ...

    @abstractmethod
    def fork(
        self,
        artifact_version: str | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef: ...

    @abstractmethod
    def metadata(self) -> Mapping[str, object] | None: ...

    @abstractmethod
    def current(self) -> ArtifactRef: ...

    @abstractmethod
    def materialize(self, ref: ArtifactRef) -> Artifact: ...

    @abstractmethod
    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef,
    ) -> ArtifactRef: ...


class CachedRepositoryBackendFactory(ABC):
    """Own per-scenario backend caching instead of hiding it in a closure."""

    _REGISTRATION_MISS_TTL_SECONDS = 5.0
    _REGISTRATION_MISS_CACHE_LIMIT = 1024

    def __init__(self) -> None:
        self._backends: dict[str, RepositoryBackend] = {}
        self._registration_misses: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()

    def __call__(self, scenario: str) -> RepositoryBackend:
        with self._lock:
            backend = self._backends.get(scenario)
            if backend is None:
                backend = self._build_backend(scenario)
                self._backends[scenario] = backend
            self._registration_misses.pop(scenario, None)
            return backend

    def has_registration(self, scenario: str) -> bool:
        """Check registration without constructing a new scenario backend."""
        now = monotonic()
        with self._lock:
            backend = self._backends.get(scenario)
            miss_expires_at = self._registration_misses.get(scenario)
            if miss_expires_at is not None:
                if miss_expires_at > now:
                    self._registration_misses.move_to_end(scenario)
                    return False
                self._registration_misses.pop(scenario, None)

        registered = (
            backend.metadata() is not None if backend is not None else self._has_persisted_registration(scenario)
        )
        with self._lock:
            if registered:
                self._registration_misses.pop(scenario, None)
            else:
                self._registration_misses[scenario] = monotonic() + self._REGISTRATION_MISS_TTL_SECONDS
                self._registration_misses.move_to_end(scenario)
                while len(self._registration_misses) > self._REGISTRATION_MISS_CACHE_LIMIT:
                    self._registration_misses.popitem(last=False)
        return registered

    def list_registrations(self) -> tuple[str, ...]:
        """All scenario names registered with this factory's storage."""
        with self._lock:
            loaded = {name for name, backend in self._backends.items() if backend.metadata() is not None}
        return tuple(sorted(loaded | set(self._list_persisted_registrations())))

    @abstractmethod
    def _build_backend(self, scenario: str) -> RepositoryBackend: ...

    def _has_persisted_registration(self, scenario: str) -> bool:
        return False

    def _list_persisted_registrations(self) -> tuple[str, ...]:
        return ()


class RepositoryBackendFactory(Protocol):
    def __call__(self, scenario: str) -> RepositoryBackend: ...


@runtime_checkable
class RegistrationAwareRepositoryBackendFactory(RepositoryBackendFactory, Protocol):
    def has_registration(self, scenario: str) -> bool: ...


@runtime_checkable
class EnumerableRepositoryBackendFactory(RepositoryBackendFactory, Protocol):
    def list_registrations(self) -> tuple[str, ...]: ...


class Repository:
    """Scenario-scoped artifact version-chain and persistence facade."""

    def __init__(
        self,
        backend: RepositoryBackend,
        base_artifact: ArtifactRef,
        *,
        current_artifact: ArtifactRef | None = None,
        checkpoint_artifact: ArtifactRef | None = None,
        local_dir: Path | None = None,
    ) -> None:
        self._backend = backend
        self._base_artifact = base_artifact
        self._current_artifact = current_artifact
        self._checkpoint_artifact = checkpoint_artifact
        # Guards every mutation of the (current, checkpoint) head pair so the
        # two refs can only move as one atomic version record. Reads stay
        # lock-free: a ref read is atomic under the GIL and readers must not
        # be serialized behind a publish.
        self._head_lock = Lock()
        # Non-checkpointed pull-surface versions remain process-local. Keep
        # every staged artifact addressable for snapshot and rollback reads; a
        # single "latest local" slot would make an older snapshot dangle as
        # soon as the next version staged.
        self._local_artifacts: dict[str, Artifact] = {}
        self._process_id = uuid.uuid4().hex
        self._temporary_directory = None
        if local_dir is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="reef-artifact-versions-")
            self._local_root = Path(self._temporary_directory.name)
        else:
            self._local_root = Path(local_dir)
        self.local_root.mkdir(parents=True, exist_ok=True)

    @property
    def local_root(self) -> Path:
        return self._local_root

    @property
    def backend(self) -> RepositoryBackend:
        return self._backend

    @property
    def base_artifact(self) -> ArtifactRef:
        return self._base_artifact

    @property
    def current_artifact(self) -> ArtifactRef | None:
        return self._current_artifact

    @property
    def checkpoint_artifact(self) -> ArtifactRef | None:
        return self._checkpoint_artifact

    def require_current_artifact(self) -> ArtifactRef:
        if self._current_artifact is None:
            raise ArtifactPublicationError("repository has no current artifact")
        return self._current_artifact

    def require_checkpoint_artifact(self) -> ArtifactRef:
        if self._checkpoint_artifact is None:
            raise ArtifactPublicationError("repository has no checkpoint artifact")
        return self._checkpoint_artifact

    def advance_current(self, ref: ArtifactRef, *, expected: ArtifactRef) -> None:
        """Move serving state without changing the last durable checkpoint.

        The move is a compare-and-swap: ``expected`` is the head the caller
        observed when it prepared the new version. A mismatch means a
        concurrent writer advanced the head in between, and fails loudly
        instead of blind-overwriting it (lost update).
        """
        with self._head_lock:
            if self._current_artifact != expected:
                raise ArtifactConflict(
                    f"serving head advanced from {expected.version} to "
                    f"{None if self._current_artifact is None else self._current_artifact.version}; "
                    f"refusing to advance to {ref.version}"
                )
            self._current_artifact = ref

    def fork(self, *, metadata: Mapping[str, object] | None = None) -> ArtifactRef:
        ref = self.backend.fork(self.base_artifact.version, metadata=metadata)
        with self._head_lock:
            self._current_artifact = ref
            self._checkpoint_artifact = ref
        return ref

    def materialize(self, ref: ArtifactRef) -> Artifact:
        local = self._local_artifacts.get(ref.version)
        if local is not None and local.ref == ref:
            return local
        return self.backend.materialize(ref).with_repository(self)

    def resolve(self, ref: ArtifactRef) -> Artifact:
        """Resolve an artifact reference to a usable Artifact.

        Returns a live artifact for a ``LiveWeightArtifactRef`` — the bytes are in
        the serving engine, not on disk. Otherwise materializes the artifact
        to a local path. Use ``Artifact.is_live`` to distinguish the two.
        """
        artifact = Artifact(ref, self)
        if artifact.is_live:
            return artifact
        return self.materialize(ref)

    def stage(
        self,
        step: int,
        artifact: Artifact,
        *,
        parent: ArtifactRef,
    ) -> Artifact:
        if artifact.local_path is None:
            raise ArtifactPublicationError("staged artifact requires local_path")
        destination = self.local_root / self._process_id / uuid.uuid4().hex
        try:
            shutil.copytree(artifact.local_path, destination)
        except OSError as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise ArtifactPublicationError(f"failed to stage artifact from {artifact.local_path}: {exc}") from exc
        staged = Artifact(
            ArtifactRef(
                artifact_id=f"{LOCAL_VERSION_PREFIX}{uuid.uuid4().hex}",
                version=f"{LOCAL_VERSION_PREFIX}{self._process_id}:{uuid.uuid4().hex}:{step}",
                parent_version=parent.version,
            ),
            self,
            local_path=destination,
            metadata=artifact.metadata,
        )
        self._local_artifacts[staged.ref.version] = staged
        return staged

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        ref = self._backend.publish(
            (
                artifact.with_repository(self)
                if metadata is None
                else Artifact(
                    artifact.ref,
                    self,
                    local_path=artifact.local_path,
                    metadata=metadata,
                )
            ),
            expected_parent=expected_parent,
        )
        with self._head_lock:
            self._current_artifact = ref
            self._checkpoint_artifact = ref
        self.discard(artifact)
        return ref

    def discard(self, artifact: Artifact) -> None:
        if self._local_artifacts.get(artifact.ref.version) is artifact:
            self._local_artifacts.pop(artifact.ref.version, None)
        path = artifact.local_path
        artifact.discard()
        if path is not None:
            with contextlib.suppress(OSError):
                path.parent.rmdir()
