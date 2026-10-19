"""Storage contracts and values for the tasks that a record-to-task stage generates.

A stage saves each generation job in a task store before any rollout. A
generation holds its source record ids and its candidate tasks. Each candidate
then gets one outcome: the stage publishes it to the train or eval split, or
rejects it with a reason. The stage chooses the split. The store keeps it and
refuses writes that conflict with it.

A published task is a Harbor task directory. Its files can be text or binary,
so they are not in the database. The database keeps the state of the store and
the digest of each task directory. A ``TaskFileStore`` keeps the files under
that digest. The same files always give the same digest, so a repeated save
changes nothing. A deployment selects the file store in
``storage.task_file_backend``: ``local`` selects :class:`LocalTaskFileStore`,
and a plugin reference selects another subclass of :class:`TaskFileStore`.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from reef.core.errors import ReefError
from reef.core.plugins import NoSettings, Plugin
from reef.core.tasks import GeneratedTask

SplitName = Literal["train", "eval"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "eval")


class TaskStoreError(ReefError):
    """A write conflicts with the stored state, or it names state that the store does not hold."""


class TaskConflict(TaskStoreError):
    """A task name is already published with different files or in the other split."""


@dataclass(frozen=True)
class SplitSettings:
    """The split settings of a task store. They do not change for the life of the store."""

    eval_fraction: float
    seed: int

    def __post_init__(self) -> None:
        fraction = self.eval_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise ValueError("eval_fraction must be a number between 0 and 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "eval_fraction", float(fraction))


@dataclass(frozen=True)
class TaskCandidate:
    """One candidate of a generation and its outcome.

    A published candidate has a ``split``. A rejected candidate has an
    ``error``. A pending candidate has neither.
    """

    generated: GeneratedTask
    split: SplitName | None = None
    error: str = ""

    def __post_init__(self) -> None:
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
    """A published task and its split.

    ``generation`` and ``index`` identify the candidate that published the task
    first. ``files_digest`` identifies its directory in the task file store.
    """

    generated: GeneratedTask
    split: SplitName
    generation: int
    index: int
    files_digest: str


class TaskStore(ABC):
    """The generations and published tasks of one record-to-task stage.

    A store is part of a scenario session and closes with it. Each write is
    durable when it returns. An identical retry of a write changes nothing.
    The caller serializes the writers of one store.
    """

    @abstractmethod
    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        """Create the checkpoint of a generation, or return the checkpoint on a retry.

        Raises ``TaskStoreError`` if the generation exists with other source record ids.
        """

    @abstractmethod
    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        """Save the candidates of a generation before validation or play.

        Raises ``TaskStoreError`` if the generation does not exist or has other candidates.
        """

    @abstractmethod
    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        """Publish one candidate to ``split`` and return the published task.

        The method saves the task files before it records the publication.
        Each task name is published once. A later candidate with the same
        files must give the same split. The method then returns the published
        task.

        Raises ``TaskConflict`` if the name is published with other files or
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
    def path(self, task: StoredTask) -> Path:
        """Return the local Harbor task directory of a published task. The caller must not change it."""


class TaskStorage(ABC):
    """Opens the task stores of one scenario. Each store has a name and its own split settings."""

    @abstractmethod
    def open(self, name: str, *, split: SplitSettings) -> TaskStore:
        """Open the task store ``name``.

        The first open saves ``split``. Raises ``TaskStoreError`` if a later
        open gives different split settings.
        """


class TaskFileStore(Plugin, ABC):
    """Keeps the directory of each published task, addressed by the digest of its files.

    The store keeps the files of each scenario apart, so that ``archive``
    retires them with the scenario. The scenario storage owns the store and
    closes it.
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
    def put(self, scenario: str, source: Path) -> str:
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
    def archive(self, scenario: str) -> tuple[str, ...]:
        """Retire the files of a scenario and return the archive locations."""

    @abstractmethod
    def close(self) -> None:
        """Release the resources of the store. A second call does nothing."""


def files_digest(source: Path) -> str:
    """Return the sha256 digest of a task directory: its name, and the path and bytes of each file.

    Raises ``TaskStoreError`` if the directory holds a symbolic link or a special file.
    """
    digest = hashlib.sha256(source.name.encode("utf-8"))
    for path in sorted(source.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise TaskStoreError(f"task directory {source} holds {path.relative_to(source)}, which is not a file")
        name = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        # Length prefixes keep two different file lists from giving the same bytes.
        digest.update(len(name).to_bytes(8, "big") + name + len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


class LocalTaskFileStore(TaskFileStore):
    """Keeps the task directories in a local directory.

    The layout is ``<directory>/<scenario key>/<digest>/<task name>/``. The
    store writes a copy under ``<directory>/.staging`` first, and then renames
    it into place. The store never reads ``.staging``. An interrupted copy can
    stay there. An operator can delete it while no Reef service runs. Without
    a directory, the store uses a temporary directory and deletes it on
    ``close``. The store creates its directories on the first ``put``.
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

    def put(self, scenario: str, source: Path) -> str:
        digest = files_digest(source)
        target = self.scenario_directory(scenario) / digest
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
    "GenerationState",
    "LocalTaskFileStore",
    "SplitName",
    "SplitSettings",
    "StoredTask",
    "TaskCandidate",
    "TaskConflict",
    "TaskFileStore",
    "TaskStorage",
    "TaskStore",
    "TaskStoreError",
    "files_digest",
]
