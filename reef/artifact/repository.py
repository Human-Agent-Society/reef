from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock
from time import monotonic
from typing import Any, Self

from reef.artifact.artifact import (
    LOCAL_RELEASE_PREFIX,
    Artifact,
    ArtifactConflict,
    ArtifactPublicationError,
    ArtifactRef,
    ArtifactRepository,
)
from reef.artifact.composite import compose_release, link_or_copy
from reef.core.components import validate_component_name
from reef.core.plugins import Plugin


@dataclass(frozen=True)
class FileDownload:
    """One file of a release that a client downloads straight from a repository backend's storage.

    Values: ``path`` is the file's path in the release, as the materialized release names it: a relative POSIX
    path, not empty, with no ``.`` or ``..`` part. ``byte_count`` is the file's size, at least 0. ``url`` is an
    absolute ``https`` or ``http`` URL that opens the file with a GET and no Reef credential until ``expires_at``
    (Unix seconds, positive).

    Purpose: ``RepositoryBackend.file_downloads`` makes them, and the data service's artifact file list gives them
    to clients, so that the bytes of a large artifact do not pass through a Reef process.
    """

    path: str
    byte_count: int
    url: str
    expires_at: int

    def __post_init__(self) -> None:
        parts = PurePosixPath(self.path).parts if isinstance(self.path, str) else ()
        if not parts or PurePosixPath(self.path).is_absolute() or any(part in (".", "..") for part in parts):
            raise ValueError(f"a file download's path is a relative path in the release, got {self.path!r}")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int) or self.byte_count < 0:
            raise ValueError("a file download's byte_count is an int of at least 0")
        if not isinstance(self.url, str) or not self.url.startswith(("https://", "http://")):
            raise ValueError("a file download's url is an absolute http or https URL")
        if isinstance(self.expires_at, bool) or not isinstance(self.expires_at, int) or self.expires_at <= 0:
            raise ValueError("a file download's expires_at is a positive int of Unix seconds")


class RepositoryBackend(ArtifactRepository):
    """Durable storage bound to one scenario repository."""

    def file_downloads(self, ref: ArtifactRef, expires_in_seconds: int) -> tuple[FileDownload, ...] | None:
        """Every file of the release ``ref``, in path order, each with a URL that opens it for ``expires_in_seconds``
        seconds; the backend lists and signs them without materializing the release.

        None, the default, when this backend keeps no copy of its releases that a client can open; Reef then lists
        the files of the materialized release, and serves them itself. Raise ``ArtifactNotFound`` when the backend
        holds no release ``ref``.
        """
        return None

    def release_uri(self, ref: ArtifactRef) -> str | None:
        """The URI of the directory of the release ``ref``, as ``ArtifactSources`` reads it, ending with ``/``.

        Another scenario imports the release from this URI, also in another deployment that can read the backend's
        storage. None, the default, when the backend keeps no copy that a URI names. Raise ``ArtifactNotFound``
        when the backend holds no release ``ref``.
        """
        return None

    @abstractmethod
    def resolve_release(self, release_id: str) -> ArtifactRef:
        """The release that the release ID ``release_id`` names in this repository.

        Raise ``ArtifactNotFound`` for a release the repository does not hold.
        """

    @abstractmethod
    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> ArtifactRef:
        """Register the scenario with ``root`` as its root release, and return the scenario's head.

        ``root`` is an unpublished artifact whose ``local_path`` is a directory. When the scenario has no head, the
        backend stores the files of ``root`` as a release with no parent, with ``root.ref.content_id`` and
        ``metadata``, and makes that release the head in one atomic step. When the scenario has a head, also one
        that another caller registered first, the backend stores nothing and returns that head. Raise
        ``ArtifactPublicationError`` when the files cannot be stored.
        """

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
        advance_head: bool = True,
    ) -> ArtifactRef: ...


class StagedReleaseRepositoryBackend(RepositoryBackend):
    """Optional storage contract for publication followed by head promotion.

    ``publish(advance_head=False)`` must store resolvable release bytes and
    metadata without moving the head. ``commit_release`` then promotes that
    release without publishing its bytes again.
    """

    @abstractmethod
    def commit_release(self, ref: ArtifactRef, *, expected_parent: ArtifactRef) -> None:
        """Advance a staged durable release after the scenario commits it.

        Implementations must accept an already-current release so recovery can
        repeat an interrupted post-commit mirror update without publishing new
        bytes. They must reject an unrelated current head.
        """
        ...


class RepositoryBackendFactory(ABC):
    @abstractmethod
    def __call__(self, scenario: str) -> RepositoryBackend: ...


class RegistrationAwareRepositoryBackendFactory(RepositoryBackendFactory):
    @abstractmethod
    def has_registration(self, scenario: str) -> bool: ...


class EnumerableRepositoryBackendFactory(RepositoryBackendFactory):
    @abstractmethod
    def list_registrations(self) -> tuple[str, ...]: ...


class BaseModelFiles(ABC):
    """The model files of a default start, which a process makes only when a scenario's root release holds them.

    Purpose: a training component gives a value in ``DefaultStart.model_files``. ``ScenarioFactory`` calls
    ``directory`` only when it creates a scenario from the default start. Thus a process that only opens scenarios
    makes no model files.
    """

    @abstractmethod
    def directory(self) -> Path:
        """A directory of the model files. The first call makes them; a later call gives the same directory."""


@dataclass(frozen=True)
class ModelDirectory(BaseModelFiles):
    """Model files that exist already.

    Values: ``path`` is an existing directory; the constructor raises ``ValueError`` for another path.
    ``directory`` returns ``path``.
    """

    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_dir():
            raise ValueError(f"model directory {self.path} does not exist")

    def directory(self) -> Path:
        return self.path


@dataclass(frozen=True)
class DefaultStart:
    """The content of a new scenario's root release when the scenario starts from the deployment's configuration.

    Values:

    - ``files`` maps relative POSIX paths inside the release to the text of each file.
    - ``model_files`` gives the directory of model files, or is ``None`` for no model files.
    - ``model_component`` is the component directory that the model files go into. ``None`` is the release root.
      It is ``None`` when ``model_files`` is ``None``.
    - ``inputs`` maps non-empty names to non-empty strings. They identify the configuration, for example the
      configured model and the checksum of the configured seed.

    Purpose: a recipe makes the value (``Recipe.default_start``). ``ScenarioFactory`` writes it as the root release
    of a scenario that it creates without an artifact item; only that write reads the model files. The root records
    ``inputs`` in its ``RootSource``, so a training component that cannot import a release (Slime) can compare
    them with its own configuration.
    """

    files: Mapping[str, str] = field(default_factory=dict)
    model_files: BaseModelFiles | None = None
    model_component: str | None = None
    inputs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for relative in self.files:
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"default start file {relative!r} must be a relative path inside the release")
        if self.model_files is None and self.model_component is not None:
            raise ValueError("model_component names where model files go, so it requires model_files")
        if self.model_component is not None:
            validate_component_name(self.model_component)
        for name, value in self.inputs.items():
            if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
                raise ValueError(f"default start input {name!r} must map a non-empty name to a non-empty string")

    @property
    def content_id(self) -> str:
        """The content ID of a root release made from this start. Equal ``inputs`` give equal content IDs."""
        digest = hashlib.sha256(json.dumps(dict(self.inputs), sort_keys=True).encode()).hexdigest()
        return f"content:{digest}"

    def write(self, directory: Path) -> None:
        """Write the start's files into the empty directory ``directory``: the model files under ``model_component``
        as hard links where the file system allows, and each of ``files`` at its path."""
        if self.model_files is not None:
            target = directory if self.model_component is None else directory / self.model_component
            shutil.copytree(self.model_files.directory(), target, dirs_exist_ok=True, copy_function=link_or_copy)
        for relative, text in self.files.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class RepositoryBackendContext:
    """The deployment values that a configured factory gets in addition to its own config.

    ``work_dir`` and ``cache_dir`` are ``storage.artifact_work_dir`` and
    ``storage.artifact_cache_dir``.
    """

    work_dir: Path
    cache_dir: Path


class ConfigurableRepositoryBackendFactory(RepositoryBackendFactory, Plugin):
    """A repository backend factory that a deployment names in ``storage.artifact_backend``.

    Optional capabilities are separate base classes. To find or list stored
    scenarios, also inherit ``RegistrationAwareRepositoryBackendFactory`` or
    ``EnumerableRepositoryBackendFactory``. ``CachedRepositoryBackendFactory``
    gives both. To work with a scenario commit history, the backends must
    subclass ``StagedReleaseRepositoryBackend``.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, context: RepositoryBackendContext) -> Self:
        """Build the factory a deployment names.

        ``config`` is the plugin's own configuration. Raise ``ValueError`` for
        an invalid configuration.
        """


class CachedRepositoryBackendFactory(RegistrationAwareRepositoryBackendFactory, EnumerableRepositoryBackendFactory):
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

    def archive_registration(self, scenario: str) -> tuple[str, ...]:
        """Forget the scenario's backend and move its durable registration aside; what was archived, by name.

        After this the factory answers ``has_registration`` false and
        ``list_registrations`` without the name, so a later create under the
        same name registers a new root release. Content-addressed storage
        the scenario shared with others stays.
        """
        with self._lock:
            self._backends.pop(scenario, None)
            self._registration_misses.pop(scenario, None)
        return self._archive_persisted_registration(scenario)

    @abstractmethod
    def _build_backend(self, scenario: str) -> RepositoryBackend: ...

    def _has_persisted_registration(self, scenario: str) -> bool:
        return False

    def _list_persisted_registrations(self) -> tuple[str, ...]:
        return ()

    def _archive_persisted_registration(self, scenario: str) -> tuple[str, ...]:
        """Move the durable registration aside; nothing to do for a backend that registers in memory only."""
        return ()


class Repository(ArtifactRepository):
    """Scenario-scoped release chain and persistence facade.

    ``start_artifact`` is the scenario's root release, the release that the scenario was created with.
    """

    def __init__(
        self,
        backend: RepositoryBackend,
        start_artifact: ArtifactRef,
        *,
        current_artifact: ArtifactRef | None = None,
        checkpoint_artifact: ArtifactRef | None = None,
        local_dir: Path | None = None,
    ) -> None:
        self._backend = backend
        self._start_artifact = start_artifact
        self._current_artifact = current_artifact
        self._checkpoint_artifact = checkpoint_artifact
        # Guards every mutation of the (current, checkpoint) head pair so the
        # two refs can only move as one atomic release record. Reads stay
        # lock-free: a ref read is atomic under the GIL and readers must not
        # be serialized behind a publish.
        self._head_lock = Lock()
        # Non-checkpointed pull-surface releases remain process-local. Keep
        # every staged artifact addressable for snapshot and activation reads; a
        # single "latest local" slot would make an older snapshot dangle as
        # soon as the next release is staged.
        self._local_artifacts: dict[str, Artifact] = {}
        self._process_id = uuid.uuid4().hex
        self._temporary_directory = None
        if local_dir is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="reef-artifact-releases-")
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

    def file_downloads(self, ref: ArtifactRef, expires_in_seconds: int) -> tuple[FileDownload, ...] | None:
        """The backend's ``file_downloads`` of the durable release ``ref``; None for a release staged on this host,
        which only the local copy holds."""
        if ref.release_id in self._local_artifacts:
            return None
        return self.backend.file_downloads(ref, expires_in_seconds)

    def release_uri(self, ref: ArtifactRef) -> str | None:
        """The backend's ``release_uri`` of the durable release ``ref``; None for a release staged on this host.

        Raise ``ValueError`` when the backend gives a URI that does not end with ``/``.
        """
        if ref.release_id in self._local_artifacts:
            return None
        uri = self.backend.release_uri(ref)
        if uri is not None and not uri.endswith("/"):
            raise ValueError(f"the repository backend gave release URI {uri!r}, which does not end with '/'")
        return uri

    @property
    def start_artifact(self) -> ArtifactRef:
        return self._start_artifact

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

    def activate(self, ref: ArtifactRef, *, expected: ArtifactRef) -> None:
        """Make ``ref`` the active artifact; the checkpoint and the backend head stay where they are.

        No storage I/O runs: the commit history records the activation, and
        recovery reads the active artifact back from it. The move is a
        compare-and-swap: ``expected`` is the active artifact the caller
        observed. A mismatch raises ``ArtifactConflict`` and changes nothing.
        """
        with self._head_lock:
            if self._current_artifact != expected:
                raise ArtifactConflict(
                    f"active artifact moved from {expected.release_id} to "
                    f"{None if self._current_artifact is None else self._current_artifact.release_id}; "
                    f"refusing to activate {ref.release_id}"
                )
            self._current_artifact = ref

    def install_checkpoint(
        self, ref: ArtifactRef, *, expected: ArtifactRef, expected_checkpoint: ArtifactRef, activate: bool
    ) -> None:
        """Install an already-committed release as the checkpoint, the chain tip, without storage I/O. When
        ``activate`` is true, also make it the active artifact (``current``)."""
        with self._head_lock:
            if self._current_artifact != expected or self._checkpoint_artifact != expected_checkpoint:
                raise ArtifactConflict("repository heads changed before the committed release was installed")
            if activate:
                self._current_artifact = ref
            self._checkpoint_artifact = ref

    def synchronize_checkpoint(self) -> None:
        """Repair a stale backend head from the committed checkpoint, never vice versa."""
        checkpoint = self.require_checkpoint_artifact()
        if self.backend.current() == checkpoint:
            return
        if checkpoint.parent_release_id is None:
            raise ArtifactConflict("a committed checkpoint without a parent cannot repair a different head")
        parent = self.backend.resolve_release(checkpoint.parent_release_id)
        self.require_staged_commit_support().commit_release(checkpoint, expected_parent=parent)

    def require_staged_commit_support(self) -> StagedReleaseRepositoryBackend:
        if not isinstance(self.backend, StagedReleaseRepositoryBackend):
            raise ArtifactPublicationError("repository backend must implement StagedReleaseRepositoryBackend")
        return self.backend

    def materialize(self, ref: ArtifactRef) -> Artifact:
        local = self._local_artifacts.get(ref.release_id)
        if local is not None and local.ref == ref:
            return local
        return self.backend.materialize(ref).with_repository(self)

    def resolve(self, ref: ArtifactRef) -> Artifact:
        """Resolve an artifact reference to an Artifact materialized at a local path."""
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
        return self.register_staged(step, artifact, destination, parent=parent)

    def stage_composed(
        self,
        step: int,
        components: Mapping[str, Artifact],
        *,
        parent: ArtifactRef,
    ) -> Artifact:
        """Compose several components straight into a staged local release; one copy, not two."""
        destination = self.local_root / self._process_id / uuid.uuid4().hex
        composed = compose_release(components, directory=destination)
        return self.register_staged(step, composed, destination, parent=parent)

    def register_staged(self, step: int, source: Artifact, destination: Path, *, parent: ArtifactRef) -> Artifact:
        """Record the tree staged at ``destination`` as a local release carrying ``source``'s content id and metadata."""
        staged = Artifact(
            ArtifactRef(
                content_id=source.ref.content_id,
                release_id=f"{LOCAL_RELEASE_PREFIX}{self._process_id}:{uuid.uuid4().hex}:{step}",
                parent_release_id=parent.release_id,
            ),
            self,
            local_path=destination,
            metadata=source.metadata,
        )
        self._local_artifacts[staged.ref.release_id] = staged
        return staged

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: ArtifactRef,
        metadata: Mapping[str, object] | None = None,
        advance_heads: bool = True,
        activate: bool = True,
    ) -> ArtifactRef:
        """Publish ``artifact`` as the next release.

        ``advance_heads`` moves the chain tip, the checkpoint ref, and the one
        storage head the backend owns (``RepositoryBackend.publish``'s
        ``advance_head``); ``activate`` then also makes the release the active
        artifact (``current``). ``advance_heads=False`` mints a release beside
        the chain that no head moves to.
        """
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
            advance_head=advance_heads,
        )
        if advance_heads:
            with self._head_lock:
                if activate:
                    self._current_artifact = ref
                self._checkpoint_artifact = ref
        self.discard(artifact)
        return ref

    def discard(self, artifact: Artifact) -> None:
        if self._local_artifacts.get(artifact.ref.release_id) is artifact:
            self._local_artifacts.pop(artifact.ref.release_id, None)
        path = artifact.local_path
        artifact.discard()
        if path is not None:
            with contextlib.suppress(OSError):
                path.parent.rmdir()
