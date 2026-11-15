"""Job data directories: the data of a training job as read-only files that its training component reads.

Reef selects the samples of a job, writes them as a job data directory, and gives the training component the
directory (``TrainingJob.data_dir``) and its URI (``TrainingJob.data_uri``). A component that runs the job elsewhere,
such as in a container, mounts or copies the URI there. The directory holds no training request: Reef keeps the
request for its own accounting, and a component reads only the data and the job's parameters.

Format 2 of a directory:

- ``job.json``: ``{"format": 2, "key": str, "sample_count": int, "kind": "trajectory" | "task" | null}``.
- ``samples.jsonl``: one sample per line, in job order, as ``JobData.to_dict`` gives it, except that a task has no
  ``task_path``, and its ``files`` holds ``{"partition", "digest", "name"}``: the directory's name in the task file
  store (``TaskFiles``), and the name of the directory itself.

The directory holds no task files. The task file store keeps the one copy of each task's directory
(``reef.storage.tasks.TaskFileStore``).

A job holds a reference to its directory (``JobDataRef``), and the reference copies nothing. A component that trains
in Reef's own process or host reads ``JobDataRef.path`` and ``JobDataRef.task_paths``, or ``JobDataRef.read``. That
read makes the local copy of the directory, and takes each task's local directory from the task file store. A
component that runs the job elsewhere mounts or copies the URI there and takes the task directories from its own
access to the store, by the ``files`` of each line. It reads neither path, so Reef makes no local copy for it.

``key`` is ``job_data_key`` of the samples, so two jobs of the same samples share one directory. A directory does
not change after it is written. The data store keeps the directories (``DataStore.write_job_directory``), and deletes a
directory that no job used for the time to live.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from reef.core.job_data import JobData, TaskFiles, TaskSample

#: The format of the directories that ``write_job_data`` writes.
JOB_DATA_FORMAT = 2
JOB_FILE = "job.json"
SAMPLES_FILE = "samples.jsonl"
#: The data store's segment store keeps the directory of the key ``k`` as ``job-data/<k>``.
JOB_DATA_PREFIX = "job-data"
JobDataKind = Literal["trajectory", "task"]


def job_data_kind(data: JobData) -> JobDataKind | None:
    """The kind of the samples of ``data``, or None for data without samples; ``ValueError`` for mixed kinds."""
    kinds = {"task" if isinstance(sample, TaskSample) else "trajectory" for sample in data.samples}
    if len(kinds) > 1:
        raise ValueError(f"job data {data.data_id!r} holds both trajectories and tasks")
    if not kinds:
        return None
    return "task" if "task" in kinds else "trajectory"


def sample_lines(data: JobData) -> Iterator[dict[str, Any]]:
    """The lines of ``samples.jsonl`` for ``data``, in job order. A task's line names its directory in the task file
    store in place of its local path. ``ValueError`` for a task whose directory the store does not keep."""
    for entry, sample in zip(data.to_dict()["samples"], data.samples, strict=True):
        if isinstance(sample, TaskSample):
            if sample.files is None:
                raise ValueError(f"the task file store does not keep the task directory {sample.task_path}")
            entry = {key: value for key, value in entry.items() if key != "task_path"}
            entry["files"] = {**entry["files"], "name": sample.task_path.name}
        yield entry


def job_data_key(data: JobData) -> str:
    """The SHA-256 of the lines of ``data`` (``sample_lines``): equal for two jobs of the same samples, whatever
    their data IDs and requests. A task's digest stands for its files, so the key reads no task file."""
    digest = hashlib.sha256()
    for entry in sample_lines(data):
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def write_job_data(data: JobData, directory: Path) -> None:
    """Write ``data`` as a job data directory at ``directory``, which must not exist.

    The function writes a sibling temporary directory and renames it, so a reader never sees part of a directory.
    ``ValueError`` for data with both kinds of samples, or with a task whose directory the task file store does not
    keep.
    """
    kind = job_data_kind(data)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        # One line at a time: the lines of a job of many trajectories are gigabytes.
        with (staged / SAMPLES_FILE).open("w", encoding="utf-8") as lines:
            for entry in sample_lines(data):
                lines.write(json.dumps(entry, allow_nan=False, separators=(",", ":")) + "\n")
        header = {
            "format": JOB_DATA_FORMAT,
            "key": job_data_key(data),
            "sample_count": len(data.samples),
            "kind": kind,
        }
        (staged / JOB_FILE).write_text(json.dumps(header), encoding="utf-8")
        os.replace(staged, directory)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def job_header(directory: Path) -> Mapping[str, Any]:
    """The content of ``job.json`` of the job data directory ``directory``; ``ValueError`` for another format."""
    header = json.loads((directory / JOB_FILE).read_text(encoding="utf-8"))
    if not isinstance(header, Mapping) or header.get("format") != JOB_DATA_FORMAT:
        raise ValueError(f"{directory} is no job data directory of format {JOB_DATA_FORMAT}")
    return header


def job_task_files(directory: Path) -> tuple[TaskFiles, ...]:
    """The name in the task file store of each task of the job data directory ``directory``, in job order; empty
    for a job of trajectories. ``ValueError`` for another format."""
    if job_header(directory)["kind"] != "task":
        return ()
    with (directory / SAMPLES_FILE).open(encoding="utf-8") as lines:
        entries = [json.loads(line)["files"] for line in lines]
    return tuple(TaskFiles(entry["partition"], entry["digest"]) for entry in entries)


def read_job_data(directory: Path, task_paths: Sequence[Path], data_id: str | None = None) -> JobData:
    """The ``JobData`` of the job data directory ``directory``, without a request. ``task_paths`` holds the local
    directory of each task, in job order (``JobDataRef.task_paths``). The data's ID is ``data_id``, or the
    directory's key.

    ``ValueError`` for another format, or for a count of ``task_paths`` that is not the count of the tasks.
    """
    header = job_header(directory)
    paths = iter(task_paths)
    samples = []
    with (directory / SAMPLES_FILE).open(encoding="utf-8") as lines:
        for line in lines:
            entry = json.loads(line)
            if entry.get("kind") == "task":
                path = next(paths, None)
                if path is None:
                    raise ValueError(f"{directory} holds more tasks than the {len(task_paths)} task paths")
                files = {key: value for key, value in entry["files"].items() if key != "name"}
                entry = {**entry, "files": files, "task_path": str(path)}
            samples.append(entry)
    if next(paths, None) is not None:
        raise ValueError(f"{directory} holds fewer tasks than the {len(task_paths)} task paths")
    return JobData.from_dict({"data_id": data_id or header["key"], "samples": samples, "request": None})


class JobDataReader(ABC):
    """Gives the local copies of what a job data reference names.

    Operations:

    * ``job_directory(uri)`` returns a local directory with the files of the job data directory at ``uri``.
      A reader of a remote store copies them; a second call for the same URI copies nothing.
      ``FileNotFoundError`` when no complete directory is at ``uri``.
    * ``job_task_paths(directory)`` returns the local directory of each task of the job data directory
      ``directory``, in job order, and nothing for a directory of trajectories. A reader of a remote store copies
      the directories that it does not hold. ``TaskStoreError`` for a task whose files the store does not hold.

    Purpose: ``JobDataRef`` holds the reader of its directory. ``DataStore`` is the reader of the references that
    it makes, and ``LocalJobData`` the reader in a process that has no data store.
    """

    @abstractmethod
    def job_directory(self, uri: str) -> Path: ...

    @abstractmethod
    def job_task_paths(self, directory: Path) -> tuple[Path, ...]: ...


@dataclass(frozen=True)
class LocalJobData(JobDataReader):
    """The reader of one job data directory whose local copies exist.

    Values: ``path`` is the local job data directory, and ``task_paths`` the local directory of each of its tasks,
    in job order. Both operations return these for every argument.
    Purpose: a job process that has no data store reads its job with it; the component that started the process
    wrote both in the job's file (``reef.train.local_jobs``).
    """

    path: Path
    task_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("LocalJobData.path must be a Path")
        if not isinstance(self.task_paths, tuple) or any(not isinstance(path, Path) for path in self.task_paths):
            raise TypeError("LocalJobData.task_paths must be a tuple of Path")

    def job_directory(self, uri: str) -> Path:
        return self.path

    def job_task_paths(self, directory: Path) -> tuple[Path, ...]:
        return self.task_paths


@dataclass(frozen=True)
class JobDataRef:
    """The job data directory of one training job, by reference.

    Values: ``uri`` is the URI that the store gives the directory. ``key`` is the directory's ``job_data_key``, by
    which the store records the directory's use. ``data_id`` is the ``JobData.data_id`` of the job's data,
    ``sample_count`` its number of samples, and ``kind`` the kind of its samples, or None without samples.
    ``reader`` gives the local copies; two references that differ only in it are equal.

    Operations: a value copies nothing when it is made. ``path`` is a local directory with the files of the
    directory, from ``reader.job_directory(uri)``. ``task_paths`` holds the local directory of each task of the
    data, in job order, from ``reader.job_task_paths(path)``; it is empty for data without tasks. ``read`` gives
    the ``JobData`` of both. Each of the three can copy files from a remote store, and raises what the reader raises.

    Purpose: ``TrainingJob.data`` holds it. The jobs queue keeps it as ``to_dict`` gives it. A component that runs
    the job elsewhere reads ``uri`` only.
    """

    uri: str
    key: str
    data_id: str
    sample_count: int
    kind: JobDataKind | None
    reader: JobDataReader = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("JobDataRef.uri must be a non-empty string")
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("JobDataRef.key must be a non-empty string")
        if not isinstance(self.reader, JobDataReader):
            raise TypeError("JobDataRef.reader must be a JobDataReader")
        if not isinstance(self.data_id, str) or not self.data_id:
            raise ValueError("JobDataRef.data_id must be a non-empty string")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("JobDataRef.sample_count must be a non-negative int")
        if self.kind is not None and self.kind not in get_args(JobDataKind):
            raise ValueError("JobDataRef.kind must be trajectory, task or None")

    @property
    def path(self) -> Path:
        """A local directory with the files of the job data directory."""
        return self.reader.job_directory(self.uri)

    @property
    def task_paths(self) -> tuple[Path, ...]:
        """The local directory of each task of the data, in job order."""
        return self.reader.job_task_paths(self.path)

    def read(self) -> JobData:
        """The job's data, with each task at its local directory (``read_job_data``)."""
        path = self.path
        return read_job_data(path, self.reader.job_task_paths(path), self.data_id)

    def to_dict(self) -> dict[str, Any]:
        """``{data_id, uri, key, sample_count, kind}``: the reference without its reader."""
        return {
            "data_id": self.data_id,
            "uri": self.uri,
            "key": self.key,
            "sample_count": self.sample_count,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], reader: JobDataReader) -> JobDataRef:
        """The reference that ``to_dict`` wrote, with ``reader``; ``ValueError`` for another value."""
        try:
            return cls(value["uri"], value["key"], value["data_id"], value["sample_count"], value["kind"], reader)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed job data reference: {exc!r}") from exc


__all__ = [
    "JOB_DATA_FORMAT",
    "JOB_DATA_PREFIX",
    "JOB_FILE",
    "SAMPLES_FILE",
    "JobDataKind",
    "JobDataReader",
    "JobDataRef",
    "LocalJobData",
    "job_data_key",
    "job_data_kind",
    "job_task_files",
    "read_job_data",
    "write_job_data",
]
