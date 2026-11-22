"""Storage contracts and values for the published Harbor tasks of a scenario.

A task store holds published tasks, each in the train or eval split. A
record-to-sample stage generates them. The stage saves each generation job in
the store before any rollout. A generation holds its source record ids and the
directories of its candidate tasks. Each candidate then gets one outcome: the
stage publishes it to a split, or rejects it with a reason.

The stage chooses the split. The store keeps it and refuses writes that
conflict with it.

A task is a Harbor task directory. Its files can be text or binary, so they
are not in the database. The database keeps the state of the store and
the digest of each task directory. A ``TaskFileStore`` keeps the files under
that digest. The same files always give the same digest, so a repeated save
changes nothing. A deployment selects the file store in
``storage.task_file_backend``: ``local`` selects :class:`LocalTaskFileStore`,
and a plugin reference selects another subclass of :class:`TaskFileStore`.

A client that writes a task directory into the task file store itself computes
the digest with :class:`TaskDirectoryDigest`, one file at a time, in the order of
``task_file_order``. Every client checks a task's files against the same limits
(``task_limit_errors``): ``MAX_TASK_FILE_BYTES`` for one file,
``MAX_TASK_DIRECTORY_ENTRIES`` for the entries of one directory, and
``MAX_TASK_FILES`` for the files of one task.

A store also keeps documents that the stage writing the store saves under
keys of its own. The store keeps each
document as text and does not read it. A document is written once. A state
document is a second kind of document: its key can be written again and
removed. The stage keeps there the state that changes in place, such as the
state of its task generator.

A task store belongs to a scenario session. Archiving the scenario retires its
task stores and task files with its records and commits. Record retention
does not delete tasks.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from reef.core.errors import ReefError
from reef.core.job_data import FILES_DIGEST_PATTERN, TaskFiles
from reef.core.plugins import NoSettings, Plugin

SplitName = Literal["train", "eval"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "eval")

#: The most bytes of one file of a task: the hard limit of one file on the Hugging Face Hub (500 GB).
MAX_TASK_FILE_BYTES = 500 * 1000**3
#: The most entries, files and directories, directly in one directory of a task: the Hugging Face Hub's limit.
MAX_TASK_DIRECTORY_ENTRIES = 10_000
#: The most files in one task.
MAX_TASK_FILES = 100_000
#: The bytes that ``files_digest`` reads from a file at a time.
DIGEST_CHUNK_BYTES = 1024 * 1024
#: The kinds of ``TaskLimitError``, in the order that ``task_limit_errors`` returns them.
TaskLimitCode = Literal["file_too_large", "too_many_entries", "too_many_files"]
TASK_LIMIT_CODES: tuple[TaskLimitCode, ...] = ("file_too_large", "too_many_entries", "too_many_files")


class TaskStoreError(ReefError):
    """A write conflicts with the stored state, or it names state that the store does not hold."""


class TaskConflictError(TaskStoreError):
    """A task name is already published with different files or in the other split."""


class TaskFilesNotFoundError(TaskStoreError):
    """A partition of the task file store holds no complete task directory of a digest and a name."""


@dataclass(frozen=True)
class TaskCandidate:
    """One candidate task of a generation, and its outcome.

    Values: ``(name, files_digest, split, error)`` is a candidate if and only if
    ``name`` is a non-empty string, ``files_digest`` is the digest of the
    candidate's task directory in the task file store, ``split`` is ``train``,
    ``eval`` or ``None``, and ``split`` and ``error`` are not both set. A
    published candidate has a ``split``. A rejected candidate has a non-empty
    ``error``, the reason. A pending candidate has neither.

    Purpose: ``TaskStore`` makes it in ``GenerationState.candidates``. The
    record-to-sample stage reads the task directory (``TaskStore.path``) and the
    outcome, to validate, publish or run the candidate, also after a restart.
    """

    name: str
    files_digest: str
    split: SplitName | None = None
    error: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.files_digest:
            raise ValueError("a candidate needs a name and a files digest")
        if self.split is not None and self.split not in SPLIT_NAMES:
            raise ValueError("split must be 'train', 'eval' or None")
        if self.split is not None and self.error:
            raise ValueError("a published candidate has no error")

    @property
    def is_pending(self) -> bool:
        return self.split is None and not self.error


@dataclass(frozen=True)
class GenerationState:
    """The checkpoint of one generation job.

    ``candidates`` is ``None`` until the stage saves them. An empty tuple
    means that the job produced no candidate.
    """

    generation: int
    source_record_ids: tuple[str, ...]
    candidates: tuple[TaskCandidate, ...] | None = None
    is_complete: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        record_ids = self.source_record_ids
        if (
            not isinstance(record_ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError("source_record_ids must be a tuple of distinct non-empty strings")
        if self.is_complete and (self.candidates is None or self.pending_indices):
            raise ValueError("a complete generation has an outcome for each candidate")

    @property
    def pending_indices(self) -> tuple[int, ...]:
        """The indices of the candidates that have no outcome."""
        return tuple(index for index, candidate in enumerate(self.candidates or ()) if candidate.is_pending)


@dataclass(frozen=True)
class StoredTask:
    """A published task and its split: the first candidate that published its name.

    Values: ``(name, split, files_digest)`` is a stored task if and only if
    ``name`` is a non-empty string, ``split`` is ``train`` or ``eval``, and
    ``files_digest`` is the digest of the task directory in the task file
    store. The directory has the name ``name``.

    Purpose: ``TaskStore.tasks`` makes it from the candidates. The caller reads ``split``, and gives
    ``files_digest`` to ``TaskStore.path`` to get the task directory.
    """

    name: str
    split: SplitName
    files_digest: str

    def __post_init__(self) -> None:
        if not self.name or self.split not in SPLIT_NAMES or not self.files_digest:
            raise ValueError("a stored task needs a name, the split 'train' or 'eval', and a files digest")


class TaskStore(ABC):
    """The generations, published tasks and documents of one named store of a scenario.

    Values: the state of a store is a triple ``(G, D, S)``. ``G`` is a finite
    sequence of generations in increasing ``generation``, each with its
    candidates and their outcomes. ``D : K ⇀ Str`` maps non-empty keys to
    documents, each key written once. Nothing is removed from ``G`` or ``D``.
    ``S : K ⇀ Str`` maps non-empty keys to state documents. A key of ``S``
    can be written again and removed. ``D`` and ``S`` have separate keys: a
    write to one does not change the other.
    The published tasks are the candidates with a split, in generation and
    candidate order, each name once: the first candidate that published it.
    Candidates that publish one name have the same files and split.

    Purpose: a record-to-sample stage writes its generations and publishes their
    candidates. Readers take the published tasks of a split (``tasks``) and their
    directories (``path``). The stage that writes the store keeps its own
    durable state about its tasks in ``D``. It keeps the state that changes in
    place in ``S``, such as the state of its task generator. The store does
    not interpret a document of ``D`` or ``S``.

    The task files of the scenario are not part of the triple. They hold the
    directory of each candidate, and each directory that ``write_files`` saved,
    by the digest of its files.

    A store is part of a scenario session and closes with it. Each write is
    durable when it returns. An identical retry of a write changes nothing.
    Each write is atomic, so writers may run on several threads.
    """

    @abstractmethod
    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        """Create the checkpoint of a generation, or return the checkpoint on a retry.

        Raises ``TaskStoreError`` if the generation exists with other source record ids.
        """

    @abstractmethod
    def write_candidates(self, *, generation: int, directories: tuple[Path, ...]) -> GenerationState:
        """Save a copy of each candidate task directory of a generation, in order, before validation or rollouts.

        The name of a candidate is the name of its directory. Raises
        ``TaskStoreError`` if the generation does not exist or has other
        candidates, or if a directory holds an entry that is not a file or a
        directory.
        """

    @abstractmethod
    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        """Publish one candidate to ``split`` and return the published task.

        Each task name is published once. A later candidate with the same
        files must give the same split. The method then returns the published
        task.

        Raises ``TaskConflictError`` if the name is published with other files or
        in the other split. Raises ``TaskStoreError`` if the candidate does not
        exist or is rejected.
        """

    @abstractmethod
    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        """Save the rejection of one candidate.

        Raises ``TaskStoreError`` if the candidate does not exist or has another outcome.
        """

    @abstractmethod
    def complete_generation(self, *, generation: int) -> GenerationState:
        """Mark a generation complete. Raises ``TaskStoreError`` if a candidate has no outcome."""

    @abstractmethod
    def generation(self, generation: int) -> GenerationState | None:
        """Return the checkpoint of one generation, or ``None`` if the generation does not exist."""

    @abstractmethod
    def generations(self) -> tuple[GenerationState, ...]:
        """Return all generation checkpoints in generation order."""

    @abstractmethod
    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        """Return the published tasks in generation and candidate order, optionally of one split."""

    @abstractmethod
    def write_document(self, *, key: str, document: str) -> None:
        """Save ``document`` under ``key``: after the call, ``D(key) = document``.

        An identical retry changes nothing. Raises ``TaskStoreError`` if ``key``
        is empty, or if ``D(key)`` is already another document.
        """

    @abstractmethod
    def documents(self) -> Mapping[str, str]:
        """Return ``D``: every saved document by its key."""

    @abstractmethod
    def write_state_document(self, *, key: str, document: str) -> None:
        """Save ``document`` under ``key`` in ``S``: after the call, ``S(key) = document``.

        The call replaces an earlier document of ``key``. Raises
        ``TaskStoreError`` if ``key`` is empty.
        """

    @abstractmethod
    def delete_state_document(self, *, key: str) -> None:
        """Remove ``key`` from ``S``: after the call, ``S`` has no document of ``key``.

        A key that ``S`` does not hold changes nothing.
        """

    @abstractmethod
    def state_document(self, key: str) -> str | None:
        """Return ``S(key)``, or ``None`` if ``S`` has no document of ``key``."""

    @abstractmethod
    def state_document_keys(self, *, prefix: str = "") -> tuple[str, ...]:
        """Return the keys of ``S`` that start with ``prefix``, in increasing string order."""

    @abstractmethod
    def write_files(self, directory: Path) -> str:
        """Save a copy of ``directory`` in the task files of the scenario and return the digest of its files.

        ``path`` returns the copy by this digest. The call does not change
        ``(G, D, S)``: the caller keeps the digest, for example in a state
        document. The copy is durable when the method returns, and a repeated
        save changes nothing. Raises ``TaskStoreError`` if ``directory`` holds
        an entry that is not a file or a directory.
        """

    @abstractmethod
    def path(self, files_digest: str) -> Path:
        """Return the local directory of a candidate, a published task or a ``write_files`` call, by its ``files_digest``.

        The caller must not change the directory. Raises ``TaskStoreError`` if
        the scenario holds no files of ``files_digest``.
        """

    @abstractmethod
    def task_files(self, files_digest: str) -> TaskFiles:
        """Return the name in the task file store of the directory that ``path`` returns for ``files_digest``.

        A job data directory names a task by it. The call does not check that the store holds the digest.
        """


class TaskStorage(ABC):
    """Opens the task stores of one scenario by name."""

    @abstractmethod
    def open(self, name: str) -> TaskStore:
        """Open the task store ``name``. A store that holds nothing yet is empty."""


class TaskFileStore(Plugin, ABC):
    """Keeps the directory of each saved candidate, published task and imported task, addressed by the digest of its files.

    The store also keeps each directory that ``TaskStore.write_files`` saved.
    It keeps the files of each scenario apart, so that ``archive`` retires
    them with the scenario. The scenario storage owns the store and closes
    it.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        """Build the store that a deployment names.

        ``config`` is the configuration of the plugin. ``directory`` is
        ``storage.task_file_dir``. A local store keeps the files there. A
        remote store can use it as a local cache.
        """

    @abstractmethod
    def write(self, scenario: str, source: Path) -> str:
        """Save a copy of the task directory ``source`` and return the digest of its files.

        The copy is durable when the method returns. Raises ``TaskStoreError``
        if ``source`` holds an entry that is not a file or a directory.
        """

    @abstractmethod
    def path(self, scenario: str, digest: str) -> Path:
        """Return a local task directory with the files of ``digest``.

        The directory has the name of the saved source directory. The caller
        must not change it. Raises ``TaskStoreError`` if the store does not
        hold the digest.
        """

    @abstractmethod
    def contains(self, partition: str, digest: str, name: str) -> bool:
        """Return whether ``partition`` holds a complete task directory named ``name`` with the files of ``digest``.

        ``partition`` is a scenario, or the partition of a data owner (``reef.storage.data.item_kinds.task_partition``).
        A client can write a directory into the store itself, in the layout of the store: the directory is complete
        when the client has written its last part. When the call returns True, ``path(partition, digest)``
        returns the directory. The call does not read the files or check their digest. Raises ``ValueError`` if
        ``digest`` does not match ``FILES_DIGEST_PATTERN``.
        """

    @abstractmethod
    def archive(self, scenario: str) -> tuple[str, ...]:
        """Retire the files of a scenario and return the archive locations."""

    @abstractmethod
    def close(self) -> None:
        """Release the resources of the store. A second call does nothing."""


def task_file_order(relative_path: str) -> tuple[str, ...]:
    """Return the sort key of the file ``relative_path`` of a task directory: the parts of the path.

    ``relative_path`` is relative to the task directory, with ``/`` between its parts. ``files_digest`` takes the
    files in increasing key order. The key compares the paths part by part, not as whole strings: ``a/b`` comes
    before ``a-b``, though ``-`` comes before ``/`` in a string. Raises ``ValueError`` unless ``relative_path`` is a
    non-empty relative path whose parts are not empty, ``.`` or ``..``.
    """
    parts = tuple(relative_path.split("/")) if isinstance(relative_path, str) else ()
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{relative_path!r} is not a relative path in a task directory")
    return parts


class FilesDigest:
    """The legacy content-only digest of a task directory, from its files given one at a time.

    Values: a digest is the triple ``(H, last, remaining)``. ``H`` is the SHA-256 state of the bytes of the directory
    name and of the files given so far. ``last`` is the ``task_file_order`` key of the last file given, or ``None``
    before the first file. ``remaining`` is the count of the bytes of the last file that have not come yet; it is 0
    before the first file. The files come in strictly increasing ``task_file_order``.

    Purpose: a client that writes a task directory into the task file store computes its digest with bounded memory,
    from a stream of each file. ``TaskDirectoryDigest`` uses it to hash file paths and bytes.

    Operations:

    * ``FilesDigest(directory_name)`` begins the digest of a directory named ``directory_name``: ``H`` holds the
      UTF-8 bytes of the name. Raises ``ValueError`` for an empty name.
    * ``add_file(relative_path, size_bytes)`` begins the next file. ``H`` gets the length and the UTF-8 bytes of
      ``relative_path``, and the length ``size_bytes``. Then ``last`` is the key of ``relative_path`` and
      ``remaining`` is ``size_bytes``. Raises ``TaskStoreError`` while ``remaining`` is not 0, for a path that
      ``task_file_order`` refuses or that does not come after ``last``, and for a negative size.
    * ``update(chunk)`` gives the next bytes of the file: ``H`` gets them, and ``remaining`` goes down by their
      count. Raises ``TaskStoreError`` if ``chunk`` holds more than ``remaining`` bytes.
    * ``hexdigest()`` returns the hex digest of ``H``, and changes nothing. Raises ``TaskStoreError`` while
      ``remaining`` is not 0.

    The name of a directory and the files of its ``rglob``, given in ``task_file_order``, give the legacy digest of the
    directory. A directory is not given, so an empty directory does not change the digest.
    """

    def __init__(self, directory_name: str) -> None:
        if not isinstance(directory_name, str) or not directory_name:
            raise ValueError("a task directory has a non-empty name")
        self.hash = hashlib.sha256(directory_name.encode("utf-8"))
        self.last: tuple[str, ...] | None = None
        self.remaining = 0

    def add_file(self, relative_path: str, size_bytes: int) -> None:
        if self.remaining and self.last is not None:
            raise TaskStoreError(f"{'/'.join(self.last)} ended {self.remaining} bytes before its size")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise TaskStoreError(f"the size of {relative_path} is not a non-negative integer")
        try:
            key = task_file_order(relative_path)
        except ValueError as exc:
            raise TaskStoreError(str(exc)) from exc
        if self.last is not None and key <= self.last:
            raise TaskStoreError(f"{relative_path} does not come after {'/'.join(self.last)} in task_file_order")
        name = relative_path.encode("utf-8")
        # Length prefixes keep two different file lists from giving the same bytes.
        self.hash.update(len(name).to_bytes(8, "big") + name + size_bytes.to_bytes(8, "big"))
        self.last, self.remaining = key, size_bytes

    def update(self, chunk: bytes) -> None:
        if len(chunk) > self.remaining:
            raise TaskStoreError("a chunk goes past the size of the current file")
        self.hash.update(chunk)
        self.remaining -= len(chunk)

    def hexdigest(self) -> str:
        if self.remaining and self.last is not None:
            raise TaskStoreError(f"{'/'.join(self.last)} ended {self.remaining} bytes before its size")
        return self.hash.hexdigest()


class TaskDirectoryDigest:
    """The digest of a task directory's name, file paths, bytes, and executable flags.

    A value holds a ``FilesDigest`` and the SHA-256 state of the digest prefix.
    The prefix starts with the ASCII text ``reef-task-directory-v2`` and a zero byte.
    Then it holds one byte per regular file, in file order.
    The byte is 1 if any executable bit is set and 0 otherwise.

    ``add_file`` starts a file in both states, using its Unix mode for the flag.
    ``update`` adds the next chunk to the content state. Both enforce the order
    and sizes of ``FilesDigest``. ``hexdigest`` hashes the prefix followed by the
    32 content digest bytes. Legacy writers can read ``content.hexdigest()``.
    """

    def __init__(self, directory_name: str) -> None:
        self.content = FilesDigest(directory_name)
        self.flags = hashlib.sha256(b"reef-task-directory-v2\x00")

    def add_file(self, relative_path: str, size_bytes: int, mode: int) -> None:
        self.content.add_file(relative_path, size_bytes)
        self.flags.update(b"\x01" if mode & 0o111 else b"\x00")

    def update(self, chunk: bytes) -> None:
        self.content.update(chunk)

    def hexdigest(self) -> str:
        digest = self.flags.copy()
        digest.update(bytes.fromhex(self.content.hexdigest()))
        return digest.hexdigest()


def files_digest(source: Path) -> str:
    """Return the SHA-256 digest of a task directory's name, file paths, bytes, and executable flags."""
    return task_directory_digest(source).hexdigest()


def task_directory_digest(source: Path) -> TaskDirectoryDigest:
    """Read a task directory into a digest that also gives its legacy digest.

    The files come in ``task_file_order`` (``TaskDirectoryDigest``), and each is read in chunks of ``DIGEST_CHUNK_BYTES``.
    Raises ``TaskStoreError`` if the directory holds a symbolic link or a special file, or if a file changes size
    while it is read.
    """
    digest = TaskDirectoryDigest(source.name)
    files = ((path, path.relative_to(source).as_posix()) for path in source.rglob("*"))
    for path, relative_path in sorted(files, key=lambda file: task_file_order(file[1])):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise TaskStoreError(f"task directory {source} holds {relative_path}, which is not a file")
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            digest.add_file(relative_path, info.st_size, info.st_mode)
            while chunk := stream.read(DIGEST_CHUNK_BYTES):
                digest.update(chunk)
    return digest


def task_file_executable_flags(source: Path) -> dict[str, bool]:
    """Return the executable flag of each regular file by its path in the directory."""
    flags = {}
    for path in source.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            flags[path.relative_to(source).as_posix()] = bool(mode & 0o111)
    return flags


@dataclass(frozen=True)
class TaskLimitError:
    """One limit that a task breaks.

    Values: ``(code, path, message)`` is a limit error if and only if ``code`` is in ``TASK_LIMIT_CODES``, ``path``
    is a relative path in the task directory or ``""`` for the task directory itself, and ``message`` is a non-empty
    English sentence that names the path and the limit. ``file_too_large`` names a file larger than
    ``MAX_TASK_FILE_BYTES``. ``too_many_entries`` names a directory with more than ``MAX_TASK_DIRECTORY_ENTRIES``
    entries. ``too_many_files`` names the task directory, which holds more than ``MAX_TASK_FILES`` files.

    Purpose: ``task_limit_errors`` makes it. A client shows ``code`` and ``message`` for the task, or refuses the
    task with them.
    """

    code: TaskLimitCode
    path: str
    message: str

    def __post_init__(self) -> None:
        if self.code not in TASK_LIMIT_CODES:
            raise ValueError(f"a task limit code is one of {', '.join(TASK_LIMIT_CODES)}")
        if self.path:
            task_file_order(self.path)
        if not self.message:
            raise ValueError("a task limit error has a message")


def task_limit_errors(entries: Iterable[tuple[str, int | None]]) -> list[TaskLimitError]:
    """Return the limits that a task breaks, from the list of its entries.

    Each entry is ``(relative_path, size_bytes)``: the path of a file or a directory relative to the task
    directory, with ``/`` between its parts, and the size of a file or ``None`` for a directory. The parent
    directories of a path count as entries also when the list does not give them, and a path given twice counts
    once. The errors come in the order of ``TASK_LIMIT_CODES``, and in ``task_file_order`` of their paths for one
    code: each file larger than ``MAX_TASK_FILE_BYTES``, each directory (``""`` for the task directory) with more
    than ``MAX_TASK_DIRECTORY_ENTRIES`` entries, and the task directory if it holds more than ``MAX_TASK_FILES``
    files. An empty list means that the task is within the limits. Raises ``ValueError`` for a path that
    ``task_file_order`` refuses, or for a size that is not ``None`` or a non-negative integer.
    """
    sizes: dict[tuple[str, ...], int | None] = {}
    for relative_path, size_bytes in entries:
        parts = task_file_order(relative_path)
        if size_bytes is not None and (
            isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0
        ):
            raise ValueError(f"the size of {relative_path} is not None or a non-negative integer")
        sizes[parts] = size_bytes
    children: dict[tuple[str, ...], set[str]] = {}
    for parts in sizes:
        for depth in range(len(parts)):
            children.setdefault(parts[:depth], set()).add(parts[depth])
    errors = [
        TaskLimitError(
            "file_too_large",
            "/".join(parts),
            f"{'/'.join(parts)} has {size_bytes:,} bytes. The limit is {MAX_TASK_FILE_BYTES:,} bytes.",
        )
        for parts, size_bytes in sorted(sizes.items())
        if size_bytes is not None and size_bytes > MAX_TASK_FILE_BYTES
    ]
    for parts, names in sorted(children.items()):
        if len(names) > MAX_TASK_DIRECTORY_ENTRIES:
            directory = "/".join(parts) or "The task directory"
            errors.append(
                TaskLimitError(
                    "too_many_entries",
                    "/".join(parts),
                    f"{directory} has {len(names):,} entries. The limit is {MAX_TASK_DIRECTORY_ENTRIES:,}.",
                )
            )
    file_count = sum(1 for size_bytes in sizes.values() if size_bytes is not None)
    if file_count > MAX_TASK_FILES:
        errors.append(
            TaskLimitError(
                "too_many_files", "", f"The task has {file_count:,} files. The limit is {MAX_TASK_FILES:,}."
            )
        )
    return errors


class LocalTaskFileStore(TaskFileStore):
    """Keeps the task directories in a local directory.

    The layout is ``<directory>/<scenario key>/<digest>/<task name>/``. The
    store writes a copy under ``<directory>/.staging`` first, and then renames
    it into place. The store never reads ``.staging``. An interrupted copy can
    stay there. An operator can delete it while no Reef service runs. Without
    a directory, the store uses a temporary directory and deletes it on
    ``close``. The store creates its directories on the first ``write``.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.temporary: tempfile.TemporaryDirectory[str] | None
        if directory is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="reef-task-files-")
            self.directory = Path(self.temporary.name)
        else:
            self.temporary = None
            self.directory = Path(directory)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(directory)

    def write(self, scenario: str, source: Path) -> str:
        computed = task_directory_digest(source)
        scenario_dir = self.scenario_directory(scenario)
        legacy_digest = computed.content.hexdigest()
        legacy = scenario_dir / legacy_digest / source.name
        if legacy.is_dir() and task_file_executable_flags(legacy) == task_file_executable_flags(source):
            return legacy_digest
        digest = computed.hexdigest()
        target = scenario_dir / digest
        if target.exists():
            return digest
        staging = self.directory / ".staging" / uuid.uuid4().hex
        try:
            shutil.copytree(source, staging / source.name)
            sync_tree(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
            sync_directory(target.parent)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return digest

    def path(self, scenario: str, digest: str) -> Path:
        target = self.scenario_directory(scenario) / digest
        entries = list(target.iterdir()) if target.is_dir() else []
        if len(entries) != 1:
            raise TaskStoreError(f"the task file store holds no files with digest {digest}")
        return entries[0]

    def contains(self, partition: str, digest: str, name: str) -> bool:
        if not isinstance(digest, str) or not FILES_DIGEST_PATTERN.fullmatch(digest):
            raise ValueError(f"{digest!r} is not a files digest")
        target = self.scenario_directory(partition) / digest
        # ``write`` renames a complete copy into place, so a directory in place is complete.
        return target.is_dir() and [entry.name for entry in target.iterdir()] == [name] and (target / name).is_dir()

    def archive(self, scenario: str) -> tuple[str, ...]:
        source = self.scenario_directory(scenario)
        if not source.exists():
            return ()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        destination = self.directory / "archived" / f"{source.name}-{stamp}-{uuid.uuid4().hex}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return (str(destination),)

    def close(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()

    def scenario_directory(self, scenario: str) -> Path:
        """Return the directory that holds the task files of one scenario."""
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        return self.directory / hashlib.sha256(scenario.encode("utf-8")).hexdigest()


def sync_tree(root: Path) -> None:
    """Flush each file and directory under ``root``, the deepest entries first."""
    for path in sorted(root.rglob("*"), key=lambda entry: len(entry.parts), reverse=True):
        if path.is_dir():
            sync_directory(path)
        else:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    sync_directory(root)


def sync_directory(path: Path) -> None:
    """Flush the entries of one directory."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FILES_DIGEST_PATTERN",
    "MAX_TASK_DIRECTORY_ENTRIES",
    "MAX_TASK_FILES",
    "MAX_TASK_FILE_BYTES",
    "TASK_LIMIT_CODES",
    "FilesDigest",
    "GenerationState",
    "LocalTaskFileStore",
    "SplitName",
    "StoredTask",
    "TaskCandidate",
    "TaskConflictError",
    "TaskDirectoryDigest",
    "TaskFileStore",
    "TaskFilesNotFoundError",
    "TaskLimitCode",
    "TaskLimitError",
    "TaskStorage",
    "TaskStore",
    "TaskStoreError",
    "files_digest",
    "task_directory_digest",
    "task_file_executable_flags",
    "task_file_order",
    "task_limit_errors",
]
