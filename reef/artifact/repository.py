from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock
from time import monotonic
from typing import Any, Self

from reef.artifact.artifact import (
    LOCAL_SNAPSHOT_PREFIX,
    Artifact,
    ArtifactConflictError,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactPublicationError,
    ArtifactRepository,
    SnapshotRef,
    artifact_components,
    content_metadata,
)
from reef.artifact.composite import compose_snapshot
from reef.core.artifact_selection import ArtifactSelection
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, SnapshotComponents
from reef.core.plugins import Plugin
from reef.core.snapshot import SELECTION_VIEW_PREFIX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileDownload:
    """One file of a snapshot that a client downloads straight from a repository backend's storage.

    Values: ``path`` is the file's path in the snapshot, as the materialized snapshot names it: a relative POSIX
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
            raise ValueError(f"a file download's path is a relative path in the snapshot, got {self.path!r}")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int) or self.byte_count < 0:
            raise ValueError("a file download's byte_count is an int of at least 0")
        if not isinstance(self.url, str) or not self.url.startswith(("https://", "http://")):
            raise ValueError("a file download's url is an absolute http or https URL")
        if isinstance(self.expires_at, bool) or not isinstance(self.expires_at, int) or self.expires_at <= 0:
            raise ValueError("a file download's expires_at is a positive int of Unix seconds")


@dataclass(frozen=True)
class StoredFiles:
    """The files of one artifact, named where a repository backend holds them already.

    Values: exactly one of ``content`` and ``artifact_id`` is a non-empty string, and the other is None.
    ``content`` names stored content of the repository (``RepositoryBackend.store_content``). ``artifact_id`` names
    an artifact that a stored snapshot of the repository binds: the files are that snapshot's files of the
    artifact, without the files that the backend adds to a snapshot.

    Purpose: the committer makes them for the snapshot of a training commit, and
    ``RepositoryBackend.publish_stored`` reads them, so that the process that commits a snapshot need not hold
    its files.
    """

    content: str | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        named = [value for value in (self.content, self.artifact_id) if value is not None]
        if len(named) != 1 or not isinstance(named[0], str) or not named[0]:
            raise ValueError("stored files name exactly one of content and artifact_id, a non-empty string")


#: The files of a snapshot to publish: of a flat snapshot, or of each component of a snapshot of several.
StoredTree = StoredFiles | Mapping[str, StoredFiles]


class RepositoryBackend(ArtifactRepository):
    """Durable storage bound to one scenario repository.

    Besides its snapshots, the repository holds stored content: named file trees with no parent, no metadata and
    no artifact ID, which no head moves to. The process that has a checkpoint's files stores them
    (``store_content``), and another process commits them as a snapshot by name (``publish_stored``, whose
    storage part a backend implements as ``publish_parts``). Content lives until ``discard_content``, and at most
    until the scenario is deleted (``discard_all_content``); a snapshot made of it keeps its files after that.
    """

    @abstractmethod
    def store_content(self, name: str, directory: Path) -> None:
        """Store the files of ``directory`` as the content ``name``, a non-empty string.

        Afterwards the content ``name`` holds exactly the files of ``directory``, by relative path and bytes,
        whatever their names, also when ``name`` held other files before; so a repeated call after a failure is
        safe. A symlink is stored as the file that it names. The change is atomic for a reader: a concurrent
        ``publish_stored`` or ``materialize_content`` of ``name`` sees all the files of before or all the files of
        after, and a directory that ``materialize_content`` returned keeps its files. No snapshot, head or
        artifact ID changes. Raise ``ArtifactPublicationError`` when the files cannot be stored; the content
        ``name`` is then as it was before.
        """

    @abstractmethod
    def materialize_content(self, name: str) -> Path:
        """A local directory that holds exactly the files of the content ``name``. The caller does not change
        it. Raise ``ArtifactNotFoundError`` when the repository holds no content ``name``."""

    @abstractmethod
    def discard_content(self, name: str) -> None:
        """Remove the content ``name``; a name that the repository does not hold changes nothing.

        Every snapshot keeps its files. Raise ``ArtifactPublicationError`` when the content cannot be removed.
        """

    @abstractmethod
    def discard_all_content(self) -> None:
        """Remove every content of this scenario's repository, whatever its name. Every snapshot keeps its files.

        Reef calls it when the scenario is deleted: content lives no longer than its scenario. Raise
        ``ArtifactPublicationError`` when the content cannot be removed.
        """

    def publish_stored(
        self,
        files: StoredTree,
        *,
        metadata: Mapping[str, object],
        expected_parent: SnapshotRef,
        advance_head: bool = True,
    ) -> SnapshotRef:
        """Store a snapshot of files that the repository holds already, and return it; the caller reads no file.

        The new snapshot has the parent ``expected_parent`` and the metadata ``metadata``, which is complete: it
        holds the component manifest and the artifact IDs, as ``Artifact.storage_metadata`` gives them for
        ``publish``. For one ``StoredFiles``, the snapshot's files are those files, at its root. For a mapping,
        the snapshot holds the files of each value, byte for byte, in the directory that its key names; a value
        with no files gives no file. The backend binds the artifact IDs of the metadata (``bind_artifact_ids``)
        and moves the head as ``publish`` does for ``advance_head``. Every named content stays stored.

        The head must be ``expected_parent``, also when ``advance_head`` is false: this method checks it first,
        and a backend that moves the head checks it again in the same step as the move.

        Raise ``ArtifactConflictError`` when the head is not ``expected_parent`` or an artifact ID of the
        metadata names other content, ``ArtifactNotFoundError`` when the repository holds no such content or
        binds no such artifact, and ``ArtifactPublicationError`` when the snapshot cannot be stored. The head
        does not move then. A failure after the snapshot was stored can leave it stored with its artifact IDs
        bound, and no head on it: a repeated call with the same artifact IDs and files is safe, and the committer
        finds such a snapshot by its artifact ID.
        """
        current = self.current()
        if current.revision != expected_parent.revision:
            raise ArtifactConflictError(
                f"repository is at {current.revision}, not expected parent {expected_parent.revision}"
            )
        parts: dict[str | None, StoredFiles] = {}
        if isinstance(files, StoredFiles):
            parts[None] = files
        else:
            parts.update(files)
        return self.publish_parts(parts, metadata=metadata, expected_parent=expected_parent, advance_head=advance_head)

    @abstractmethod
    def publish_parts(
        self,
        parts: Mapping[str | None, StoredFiles],
        *,
        metadata: Mapping[str, object],
        expected_parent: SnapshotRef,
        advance_head: bool,
    ) -> SnapshotRef:
        """The backend's part of ``publish_stored``, which has checked the head once: store the snapshot of
        ``parts`` and move the head for ``advance_head``. ``parts`` has the one key None for a flat snapshot,
        else the directory of each component."""

    def snapshot_metadata(self, ref: SnapshotRef) -> Mapping[str, object]:
        """The metadata that the stored snapshot ``ref`` was published or registered with.

        The result is the snapshot's storage metadata (``Artifact.storage_metadata``). This default materializes
        the snapshot. A backend whose snapshots hold large files reads the metadata without them. Raise
        ``ArtifactError`` for a snapshot that the repository does not hold.
        """
        return self.materialize(ref).storage_metadata

    @abstractmethod
    def artifact_snapshot(self, artifact_id: str) -> SnapshotRef:
        """The stored snapshot that ``artifact_id`` is bound to in this repository.

        Raise ``ArtifactNotFoundError`` when this repository has no binding for the ID.
        """

    def artifact_snapshots(self, artifact_ids: Collection[str]) -> dict[str, SnapshotRef]:
        """The stored snapshot of each of ``artifact_ids`` that has a binding; an ID without a binding is absent.

        This default makes one lookup for each ID. A backend whose lookup is slow can look up all of them at once.
        """
        snapshots = {}
        for artifact_id in artifact_ids:
            try:
                snapshots[artifact_id] = self.artifact_snapshot(artifact_id)
            except ArtifactNotFoundError:
                continue
        return snapshots

    def resolve_artifact(self, artifact_id: str, paths: tuple[str, ...] | None = None) -> Artifact:
        """Materialize the component that ``artifact_id`` names, without a caller knowing its snapshot or path,
        and without the files of the other components of that snapshot where the backend can
        (``materialize_component_files``): all its files, or at least those of ``paths``.

        Raise ``ArtifactNotFoundError`` when this repository has no binding for the ID.
        """
        ref = self.artifact_snapshot(artifact_id)
        snapshot = Artifact(ref, None, metadata=self.snapshot_metadata(ref))
        components = artifact_components(snapshot)
        if artifact_id not in components:
            raise ArtifactPublicationError(f"artifact binding does not contain {artifact_id!r}")
        component = components[artifact_id]
        manifest = snapshot.components
        if component is None or manifest is None or manifest.single:
            return self.materialize_component_files(ref, None, paths)
        # Only the files of this artifact are made local, not the other components of its snapshot.
        return self.materialize_component_files(ref, component, paths)

    @abstractmethod
    def bind_artifact_id(self, artifact_id: str, ref: SnapshotRef) -> SnapshotRef | None:
        """Bind ``artifact_id`` to the stored snapshot ``ref`` unless it is bound already.

        Return the snapshot that ``artifact_id`` was bound to before, or None when this call bound it. A binding
        never changes.
        """

    @abstractmethod
    def artifact_content(self, ref: SnapshotRef, artifact_id: str) -> tuple[object, dict[str, object]]:
        """A value of the files, and the metadata (``content_metadata``), of the artifact ``artifact_id`` in ``ref``.

        Two snapshots give equal values for an artifact exactly when they hold the same files and metadata for it.
        """

    def bind_artifact_ids(self, snapshot: Artifact) -> None:
        """Bind each artifact ID in the stored ``snapshot`` to it, before a head moves to it.

        ``snapshot`` needs its metadata, not its files. An ID bound to a snapshot with the same content keeps that
        binding. Raise ``ArtifactConflictError`` for an ID bound to other content. IDs bound before a failure stay
        bound, so a repeated call completes the rest.
        """
        for artifact_id in artifact_components(snapshot):
            self.require_same_content(artifact_id, self.bind_artifact_id(artifact_id, snapshot.ref), snapshot.ref)

    def require_same_content(self, artifact_id: str, bound: SnapshotRef | None, ref: SnapshotRef) -> None:
        """Raise ``ArtifactConflictError`` when ``bound``, the snapshot that ``artifact_id`` was bound to before
        (``bind_artifact_id``), holds other files or metadata for the artifact than the snapshot ``ref``."""
        if (
            bound is not None
            and bound != ref
            and self.artifact_content(bound, artifact_id) != self.artifact_content(ref, artifact_id)
        ):
            raise ArtifactConflictError(f"artifact {artifact_id!r} already names different files or metadata")

    def file_downloads(self, ref: SnapshotRef, expires_in_seconds: int) -> tuple[FileDownload, ...] | None:
        """Every file of the snapshot ``ref``, in path order, each with a URL that opens it for ``expires_in_seconds``
        seconds; the backend lists and signs them without materializing the snapshot.

        None, the default, when this backend keeps no copy of its snapshots that a client can open; Reef then lists
        the files of the materialized snapshot, and serves them itself. Raise ``ArtifactNotFoundError`` when the
        backend holds no snapshot ``ref``.
        """
        return None

    def snapshot_uri(self, ref: SnapshotRef) -> str | None:
        """The URI of the directory of the snapshot ``ref``, as ``ArtifactSources`` reads it, ending with ``/``.

        Another scenario imports the snapshot from this URI, also in another deployment that can read the backend's
        storage. None, the default, when the backend keeps no copy that a URI names. Raise ``ArtifactNotFoundError``
        when the backend holds no snapshot ``ref``.
        """
        return None

    @abstractmethod
    def resolve_snapshot(self, revision: str) -> SnapshotRef:
        """The stored snapshot that the backend revision ``revision`` names in this repository.

        Raise ``ArtifactNotFoundError`` for a snapshot the repository does not hold.
        """

    @abstractmethod
    def register(self, root: Artifact, *, metadata: Mapping[str, object]) -> SnapshotRef:
        """Register the scenario with ``root`` as its root snapshot, and return the scenario's head.

        ``root`` is an unpublished artifact whose ``local_path`` is a directory. When the scenario has no head, the
        backend stores the files of ``root`` as a snapshot with no parent, with its artifact IDs and ``metadata``, and makes that snapshot the head in one atomic step. When the scenario has a head, also one
        that another caller registered first, the backend stores nothing and returns that head. Raise
        ``ArtifactPublicationError`` when the files cannot be stored.
        """

    @abstractmethod
    def metadata(self) -> Mapping[str, object] | None: ...

    @abstractmethod
    def current(self) -> SnapshotRef: ...

    @abstractmethod
    def materialize(self, ref: SnapshotRef) -> Artifact: ...

    @abstractmethod
    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: SnapshotRef,
        advance_head: bool = True,
    ) -> SnapshotRef: ...


class StagedSnapshotRepositoryBackend(RepositoryBackend):
    """Optional storage contract for publication followed by head promotion.

    ``publish(advance_head=False)`` must store resolvable snapshot bytes and
    metadata without moving the head. ``commit_snapshot`` then promotes that
    snapshot without publishing its bytes again.
    """

    @abstractmethod
    def commit_snapshot(self, ref: SnapshotRef, *, expected_parent: SnapshotRef) -> None:
        """Advance a staged durable snapshot after the scenario commits it.

        Implementations must accept an already-current snapshot so recovery can
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


@dataclass(frozen=True)
class DefaultStart:
    """The content of a new scenario's root snapshot when the scenario starts from the deployment's configuration.

    Values:

    - ``files`` maps relative POSIX paths inside the snapshot to the text of each file. No path is empty or
      absolute, and none has a ``..`` part.
    - ``inputs`` maps non-empty names to non-empty strings. They identify the configuration, for example the
      configured model and the checksum of the configured seed.

    The constructor raises ``ValueError`` for another value. The value holds text only: a start of weights names
    its base model in a small file (a base root, ``reef.artifact.base_root``, or a reference,
    ``reef.artifact.sources.REFERENCE_FILE``) and holds no weight file.

    Purpose: a recipe or a training backend's facts make the value from configuration (``Recipe.default_start``,
    ``TrainingBackendFacts.default_start``). ``ScenarioFactory`` writes it as the root snapshot of a scenario that
    it creates without an artifact item. The root records ``inputs`` in its ``RootSource``, so a training component
    that cannot import a snapshot (Slime) can compare them with its own configuration.
    """

    files: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for relative, text in self.files.items():
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"default start file {relative!r} must be a relative path inside the snapshot")
            if not isinstance(text, str):
                raise ValueError(f"default start file {relative!r} must hold text")
        for name, value in self.inputs.items():
            if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
                raise ValueError(f"default start input {name!r} must map a non-empty name to a non-empty string")

    def write(self, directory: Path) -> None:
        """Write each of ``files`` at its path under the directory ``directory``."""
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
    subclass ``StagedSnapshotRepositoryBackend``.
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
        same name registers a new root snapshot. Content-addressed storage
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

    @property
    def archives_registrations(self) -> bool:
        """Whether ``archive_registration`` can move a scenario's registration aside here. The default is true. A
        caller asks before it starts to delete a scenario, so that the delete refuses before it changes anything."""
        return True

    def _archive_persisted_registration(self, scenario: str) -> tuple[str, ...]:
        """Move the durable registration aside; nothing to do for a backend that registers in memory only."""
        return ()


def selection_digest(selection: ArtifactSelection) -> str:
    """The key of the local view of ``selection``. Equal selections have equal keys."""
    return hashlib.sha256(json.dumps(selection.to_dict(), sort_keys=True).encode()).hexdigest()


class Repository(ArtifactRepository):
    """Scenario-scoped snapshot chain and persistence facade.

    ``start_artifact`` is the scenario's root snapshot, the snapshot that the scenario was created with.
    """

    def __init__(
        self,
        backend: RepositoryBackend,
        start_artifact: SnapshotRef,
        *,
        current_artifact: SnapshotRef | None = None,
        checkpoint_artifact: SnapshotRef | None = None,
        local_dir: Path | None = None,
    ) -> None:
        self._backend = backend
        self._start_artifact = start_artifact
        self._current_artifact = current_artifact
        self._checkpoint_artifact = checkpoint_artifact
        # Guards every mutation of the (current, checkpoint) head pair so the
        # two refs can only move as one atomic record. Reads stay
        # lock-free: a ref read is atomic under the GIL and readers must not
        # be serialized behind a publish.
        self._head_lock = Lock()
        # Non-checkpointed snapshots of pulled files remain process-local. Keep
        # every staged artifact addressable for snapshot and activation reads; a
        # single "latest local" slot would make an older snapshot dangle as
        # soon as the next snapshot is staged.
        self._local_artifacts: dict[str, Artifact] = {}
        #: The files of each staged snapshot that the backend holds already, by the staged revision. Such a
        #: snapshot is in ``_local_artifacts`` with no local path, and it is made local only when a reader asks.
        self.stored_trees: dict[str, StoredTree] = {}
        #: The local copy of each staged snapshot of ``stored_trees`` that a reader asked for.
        self.materialized_stored: dict[str, Artifact] = {}
        #: The stored content that exists for one staged snapshot of ``stored_trees`` alone, by its revision.
        self.owned_content: dict[str, str] = {}
        self.selections: dict[str, ArtifactSelection] = {}
        self.selection_lock = Lock()
        self._process_id = uuid.uuid4().hex
        self._temporary_directory = None
        if local_dir is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="reef-artifact-snapshots-")
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

    def file_downloads(self, ref: SnapshotRef, expires_in_seconds: int) -> tuple[FileDownload, ...] | None:
        """The backend's ``file_downloads`` of the durable snapshot ``ref``; None for a snapshot staged on this host,
        which only the local copy holds."""
        if ref.revision in self._local_artifacts:
            return None
        return self.backend.file_downloads(ref, expires_in_seconds)

    def snapshot_uri(self, ref: SnapshotRef) -> str | None:
        """The backend's ``snapshot_uri`` of the durable snapshot ``ref``; None for a snapshot staged on this host.

        Raise ``ValueError`` when the backend gives a URI that does not end with ``/``.
        """
        if ref.revision in self._local_artifacts:
            return None
        uri = self.backend.snapshot_uri(ref)
        if uri is not None and not uri.endswith("/"):
            raise ValueError(f"the repository backend gave snapshot URI {uri!r}, which does not end with '/'")
        return uri

    @property
    def start_artifact(self) -> SnapshotRef:
        return self._start_artifact

    @property
    def current_artifact(self) -> SnapshotRef | None:
        return self._current_artifact

    @property
    def checkpoint_artifact(self) -> SnapshotRef | None:
        return self._checkpoint_artifact

    def require_current_artifact(self) -> SnapshotRef:
        if self._current_artifact is None:
            raise ArtifactPublicationError("repository has no current artifact")
        return self._current_artifact

    def require_checkpoint_artifact(self) -> SnapshotRef:
        if self._checkpoint_artifact is None:
            raise ArtifactPublicationError("repository has no checkpoint artifact")
        return self._checkpoint_artifact

    def activate(self, ref: SnapshotRef, *, expected: SnapshotRef) -> None:
        """Make ``ref`` the active artifact; the checkpoint and the backend head stay where they are.

        No storage I/O runs: the commit history records the activation, and
        recovery reads the active artifact back from it. The move is a
        compare-and-swap: ``expected`` is the active artifact the caller
        observed. A mismatch raises ``ArtifactConflictError`` and changes nothing.
        """
        with self._head_lock:
            if self._current_artifact != expected:
                raise ArtifactConflictError(
                    f"active artifact moved from {expected.revision} to "
                    f"{None if self._current_artifact is None else self._current_artifact.revision}; "
                    f"refusing to activate {ref.revision}"
                )
            self._current_artifact = ref

    def install_checkpoint(
        self, ref: SnapshotRef, *, expected: SnapshotRef, expected_checkpoint: SnapshotRef, activate: bool
    ) -> None:
        """Install an already-committed snapshot as the checkpoint, the chain tip, without storage I/O. When
        ``activate`` is true, also make it the active artifact (``current``)."""
        with self._head_lock:
            if self._current_artifact != expected or self._checkpoint_artifact != expected_checkpoint:
                raise ArtifactConflictError("repository heads changed before the committed snapshot was installed")
            if activate:
                self._current_artifact = ref
            self._checkpoint_artifact = ref

    def synchronize_checkpoint(self) -> None:
        """Repair a stale backend head from the committed checkpoint, never vice versa."""
        checkpoint = self.require_checkpoint_artifact()
        if self.backend.current() == checkpoint:
            return
        if checkpoint.parent_revision is None:
            raise ArtifactConflictError("a committed checkpoint without a parent cannot repair a different head")
        parent = self.backend.resolve_snapshot(checkpoint.parent_revision)
        self.require_staged_commit_support().commit_snapshot(checkpoint, expected_parent=parent)

    def require_staged_commit_support(self) -> StagedSnapshotRepositoryBackend:
        if not isinstance(self.backend, StagedSnapshotRepositoryBackend):
            raise ArtifactPublicationError("repository backend must implement StagedSnapshotRepositoryBackend")
        return self.backend

    def materialize(self, ref: SnapshotRef) -> Artifact:
        local = self._local_artifacts.get(ref.revision)
        if local is not None and local.ref == ref:
            return local if ref.revision not in self.stored_trees else self.materialize_stored(local)
        selection = self.selections.get(ref.revision)
        if selection is not None:
            materialized = self.materialize_selection(selection)
            if materialized.ref == ref:
                return materialized
            return Artifact(ref, self, local_path=materialized.local_path, metadata=materialized.storage_metadata)
        return self.backend.materialize(ref).with_repository(self)

    def resolve(self, ref: SnapshotRef) -> Artifact:
        """Resolve an artifact reference to an Artifact materialized at a local path."""
        return self.materialize(ref)

    def materialize_stored(self, staged: Artifact) -> Artifact:
        """The staged snapshot ``staged`` of ``stored_trees`` with its files in a local directory.

        The files of stored content come from ``RepositoryBackend.materialize_content``, and the files of a bound
        artifact from ``resolve_artifact``. A snapshot of several components is composed once under ``local_root``
        and kept until ``discard``.
        """
        revision = staged.ref.revision
        tree = self.stored_trees[revision]
        if isinstance(tree, StoredFiles):
            return Artifact(staged.ref, self, local_path=self.stored_files(tree), metadata=staged.storage_metadata)
        with self.selection_lock:
            composed = self.materialized_stored.get(revision)
            if composed is None:
                manifest = staged.components
                if manifest is None:
                    raise ArtifactPublicationError("a staged snapshot of several components carries their manifest")
                destination = self.local_root / self._process_id / uuid.uuid4().hex
                compose_snapshot(
                    {
                        name: Artifact(
                            staged.ref,
                            None,
                            local_path=self.stored_files(files),
                            metadata=manifest.entries[name].metadata,
                            artifact_id=manifest.entries[name].artifact_id,
                        )
                        for name, files in tree.items()
                    },
                    directory=destination,
                )
                composed = Artifact(staged.ref, self, local_path=destination, metadata=staged.storage_metadata)
                self.materialized_stored[revision] = composed
            return composed

    def stored_files(self, files: StoredFiles) -> Path:
        """A local directory with the files that ``files`` names; it need not exist when they are no files."""
        if files.content is not None:
            return self.backend.materialize_content(files.content)
        path = self.backend.resolve_artifact(str(files.artifact_id)).local_path
        if path is None:
            raise ArtifactPublicationError(f"artifact {files.artifact_id!r} has no local files")
        return path

    def materialize_component(self, ref: SnapshotRef, name: str) -> Artifact:
        """The component ``name`` of the snapshot ``ref``, local. A snapshot of this host is local as a whole;
        of a stored snapshot, the backend makes only the component local where it can."""
        if ref.revision in self._local_artifacts:
            return self.materialize(ref).component(name)
        selection = self.selections.get(ref.revision)
        if selection is None:
            return self.backend.materialize_component(ref, name)
        if name not in selection.components:
            raise ArtifactNotFoundError(f"snapshot {ref.revision!r} binds no component {name!r}")
        # A view of a selection that is not composed here: only the artifact of this component is made local.
        return self.backend.resolve_artifact(selection.components[name])

    def materialize_component_files(
        self, ref: SnapshotRef, name: str | None, paths: tuple[str, ...] | None
    ) -> Artifact:
        """The component ``name`` of the snapshot ``ref`` with at least the files ``paths`` local
        (``ArtifactRepository.materialize_component_files``). A snapshot of this host is local as a whole; of a
        selection view, the artifact of ``name`` is read; of a stored snapshot, the backend reads only ``paths``
        where it can."""
        if ref.revision in self._local_artifacts:
            local = self.materialize(ref)
            return local if name is None else local.component(name)
        selection = self.selections.get(ref.revision)
        if selection is None:
            return self.backend.materialize_component_files(ref, name, paths)
        if name is None or name not in selection.components:
            raise ArtifactNotFoundError(f"snapshot {ref.revision!r} binds no component {name!r}")
        return self.backend.resolve_artifact(selection.components[name], paths)

    def snapshot(self, ref: SnapshotRef) -> Artifact:
        """The snapshot ``ref`` with its metadata and without its files: ``materialize`` or ``component`` of the
        result makes local what a reader asks for."""
        if ref.revision in self.selections:
            return self.select(self.selections[ref.revision])
        return Artifact(ref, self, metadata=self.snapshot_metadata(ref))

    def snapshot_metadata(self, ref: SnapshotRef) -> Mapping[str, object]:
        """The metadata of the snapshot ``ref``, read without its files: of a snapshot staged on this host, or
        the backend's ``snapshot_metadata`` of a stored one."""
        local = self._local_artifacts.get(ref.revision)
        if local is not None and local.ref == ref:
            return local.storage_metadata
        return self.backend.snapshot_metadata(ref)

    def selection(self, ref: SnapshotRef, names: tuple[str, ...]) -> ArtifactSelection:
        """The artifact ID of each of ``names`` in the snapshot ``ref``.

        A selection view answers from the selection it was composed from. Any other snapshot answers from its
        metadata (``snapshot_metadata``), and no file is read: a snapshot of one component gives its artifact for
        every name. Raise ``ArtifactNotFoundError`` for a name that a snapshot of several components does not
        bind.
        """
        selected = self.selections.get(ref.revision)
        if selected is not None:
            return selected
        snapshot = Artifact(ref, None, metadata=self.snapshot_metadata(ref))
        manifest = snapshot.components
        if manifest is None or manifest.single:
            return ArtifactSelection(dict.fromkeys(names, snapshot.artifact_id))
        missing = [name for name in names if name not in manifest.entries]
        if missing:
            raise ArtifactNotFoundError(f"snapshot {ref.revision!r} binds no component {missing[0]!r}")
        return ArtifactSelection({name: manifest.artifact_id(name, ref.revision) for name in names})

    def select(self, selection: ArtifactSelection) -> Artifact:
        """The view of existing component artifacts that ``selection`` names, with its manifest and no file.

        Selecting is a metadata operation: nothing is published, no storage head moves, and no file is fetched.
        A reader of the view makes local what it reads: ``Artifact.component`` one component
        (``materialize_component``), and ``materialize`` the whole view. A selection of one artifact is the stored
        snapshot of that artifact. The deterministic private cache key lets an operation record name the view
        during recovery. Its complete selection must be persisted with that record; the cache key alone cannot
        recover its files.
        """
        if len(selection.components) == 1:
            artifact_id = next(iter(selection.components.values()))
            stored = self.backend.artifact_snapshot(artifact_id)
            snapshot = Artifact(stored, self, metadata=self.backend.snapshot_metadata(stored))
            manifest = snapshot.components
            if manifest is None or manifest.single:
                return snapshot
            return self.backend.resolve_artifact(artifact_id).with_repository(self)
        with self.selection_lock:
            cache_key = f"{SELECTION_VIEW_PREFIX}{selection_digest(selection)}"
            cached = self._local_artifacts.get(cache_key)
            if cached is not None:
                return cached
            manifest = SnapshotComponents(
                {name: self.stored_entry(artifact_id) for name, artifact_id in selection.components.items()}
            )
            self.selections[cache_key] = selection
            return Artifact(SnapshotRef(cache_key, None), self, metadata={COMPONENTS_METADATA_KEY: manifest.to_dict()})

    def stored_entry(self, artifact_id: str) -> ComponentEntry:
        """The manifest entry of the stored artifact ``artifact_id``: its ID and its own metadata, read from the
        metadata of the snapshot that binds it."""
        stored = self.backend.artifact_snapshot(artifact_id)
        snapshot = Artifact(stored, None, metadata=self.backend.snapshot_metadata(stored))
        return ComponentEntry(artifact_id, content_metadata(snapshot, artifact_components(snapshot)[artifact_id]))

    def materialize_selection(self, selection: ArtifactSelection) -> Artifact:
        """The view of ``selection`` with the files of every component in one local directory, composed once."""
        if len(selection.components) == 1:
            artifact_id = next(iter(selection.components.values()))
            return self.backend.resolve_artifact(artifact_id).with_repository(self)
        # Readers can materialize the same recovered view concurrently. Only its first reader builds the tree.
        # This lock is separate from head movement, which never waits for file I/O.
        with self.selection_lock:
            digest = selection_digest(selection)
            cache_key = f"{SELECTION_VIEW_PREFIX}{digest}"
            cached = self._local_artifacts.get(cache_key)
            if cached is not None:
                return cached
            components = {
                name: self.backend.resolve_artifact(artifact_id) for name, artifact_id in selection.components.items()
            }
            destination = self.local_root / self._process_id / digest
            composed = compose_snapshot(components, directory=destination)
            artifact = Artifact(
                SnapshotRef(cache_key, None),
                self,
                local_path=destination,
                metadata=composed.metadata,
            )
            self._local_artifacts[cache_key] = artifact
            self.selections[cache_key] = selection
            return artifact

    def remember_selection(self, ref: SnapshotRef, selection: ArtifactSelection) -> None:
        """Register the recorded selection view ``ref`` without composing its files; other snapshots need nothing."""
        if not ref.selection_view:
            return
        existing = self.selections.get(ref.revision)
        if existing is not None and existing != selection:
            raise ArtifactPublicationError("stored selections disagree for the same local view")
        # select() derives filesystem paths only from the selection's component mapping.
        self.selections[ref.revision] = selection

    def stage(
        self,
        commit_number: int,
        artifact: Artifact,
        *,
        parent: SnapshotRef,
    ) -> Artifact:
        """Stage ``artifact`` as the flat snapshot of commit ``commit_number`` with the parent ``parent``.

        An artifact with a local path is copied under ``local_root``. An artifact of stored content
        (``Artifact.content``) is staged by name: nothing is copied, and ``publish`` has the backend make the
        snapshot of that content.
        """
        if artifact.content is not None:
            return self.stage_stored(
                commit_number, StoredFiles(content=artifact.content), artifact.storage_metadata, parent=parent
            )
        if artifact.local_path is None:
            raise ArtifactPublicationError("staged artifact requires local_path")
        destination = self.local_root / self._process_id / uuid.uuid4().hex
        try:
            shutil.copytree(artifact.local_path, destination)
        except OSError as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise ArtifactPublicationError(f"failed to stage artifact from {artifact.local_path}: {exc}") from exc
        return self.register_staged(commit_number, artifact, destination, parent=parent)

    def stage_stored(
        self,
        commit_number: int,
        files: StoredTree,
        metadata: Mapping[str, object],
        *,
        parent: SnapshotRef,
        owned_content: str | None = None,
    ) -> Artifact:
        """Stage the snapshot of commit ``commit_number`` whose files the backend holds already (``files``), with
        the storage metadata ``metadata`` and the parent ``parent``.

        The result has no local path. ``materialize`` makes it local for a reader that asks, ``publish`` stores it
        with ``RepositoryBackend.publish_stored``, and ``discard`` forgets it. ``owned_content`` names stored
        content that exists for this snapshot alone: ``discard`` removes it, also after the publication.
        """
        staged = Artifact(
            SnapshotRef(
                revision=f"{LOCAL_SNAPSHOT_PREFIX}{self._process_id}:{uuid.uuid4().hex}:{commit_number}",
                parent_revision=parent.revision,
            ),
            self,
            metadata=metadata,
        )
        self.stored_trees[staged.ref.revision] = files
        self._local_artifacts[staged.ref.revision] = staged
        if owned_content is not None:
            self.owned_content[staged.ref.revision] = owned_content
        return staged

    def register_staged(
        self, commit_number: int, source: Artifact, destination: Path, *, parent: SnapshotRef
    ) -> Artifact:
        """Record the tree staged at ``destination`` as a local snapshot with ``source``'s artifact IDs and metadata."""
        staged = Artifact(
            SnapshotRef(
                revision=f"{LOCAL_SNAPSHOT_PREFIX}{self._process_id}:{uuid.uuid4().hex}:{commit_number}",
                parent_revision=parent.revision,
            ),
            self,
            local_path=destination,
            metadata=source.storage_metadata,
        )
        self._local_artifacts[staged.ref.revision] = staged
        return staged

    def publish(
        self,
        artifact: Artifact,
        *,
        expected_parent: SnapshotRef,
        metadata: Mapping[str, object] | None = None,
        advance_heads: bool = True,
        activate: bool = True,
    ) -> SnapshotRef:
        """Publish ``artifact`` as the next snapshot.

        ``advance_heads`` moves the chain tip, the checkpoint ref, and the one
        storage head the backend owns (``RepositoryBackend.publish``'s
        ``advance_head``); ``activate`` then also makes the snapshot the active
        artifact (``current``). ``advance_heads=False`` mints a snapshot beside
        the chain that no head moves to.
        """
        stored = self.stored_trees.get(artifact.ref.revision)
        if stored is None:
            ref = self._backend.publish(
                (
                    artifact.with_repository(self)
                    if metadata is None
                    else Artifact(
                        artifact.ref,
                        self,
                        local_path=artifact.local_path,
                        metadata={**artifact.storage_metadata, **dict(metadata)},
                    )
                ),
                expected_parent=expected_parent,
                advance_head=advance_heads,
            )
        else:
            ref = self._backend.publish_stored(
                stored,
                metadata={**artifact.storage_metadata, **dict(metadata or {})},
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
        """Forget the staged snapshot or the view of a selection ``artifact``, and remove its local files.

        A view of a selection is named by its revision, whatever object holds it: the view that ``select``
        returned has no file, and a reader that made it local registered another object for the same view.
        """
        if artifact.ref.selection_view:
            composed = self._local_artifacts.pop(artifact.ref.revision, None)
            self.selections.pop(artifact.ref.revision, None)
            if composed is not None:
                composed.discard()
            return
        if self._local_artifacts.get(artifact.ref.revision) is artifact:
            self._local_artifacts.pop(artifact.ref.revision, None)
            self.selections.pop(artifact.ref.revision, None)
            self.stored_trees.pop(artifact.ref.revision, None)
            composed = self.materialized_stored.pop(artifact.ref.revision, None)
            if composed is not None:
                composed.discard()
            owned = self.owned_content.pop(artifact.ref.revision, None)
            if owned is not None:
                # The snapshot keeps the files. A failure leaves content that no job item names.
                try:
                    self.backend.discard_content(owned)
                except ArtifactError:
                    logger.warning("could not discard the stored content %s", owned)
        path = artifact.local_path
        artifact.discard()
        if path is not None:
            with contextlib.suppress(OSError):
                path.parent.rmdir()
