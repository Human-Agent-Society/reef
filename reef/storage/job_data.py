"""Job data directories: the data of a training job as read-only files that its training component reads.

Reef selects the samples of a job, writes them as a job data directory, and gives the training component the
directory (``TrainingJob.data_dir``) and its URI (``TrainingJob.data_uri``). A component that runs the job elsewhere,
such as in a container, mounts or copies the URI there. The directory holds no training request: Reef keeps the
request for its own accounting, and a component reads only the data and the job's parameters.

Format 1 of a directory:

- ``job.json``: ``{"format": 1, "key": str, "sample_count": int, "kind": "trajectory" | "task" | null}``.
- ``samples.jsonl``: one sample per line, in job order, as ``JobData.to_dict`` gives it, except that a task's
  ``task_path`` is relative to the directory: ``tasks/<n>/<directory name>``.
- ``tasks/<n>/<directory name>/``: the directory of the ``n``-th task, where ``n`` counts the tasks from 0.

``key`` is ``job_data_key`` of the samples, so two jobs of the same samples share one directory. A directory does
not change after it is written. The data store keeps the directories (``JobDataStore``), and deletes a directory
that no job used for the time to live.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from reef.core.job_data import JobData, TaskSample
from reef.storage.segments import LocalSegmentStore, SegmentStore

#: The format of the directories that ``write_job_data`` writes.
JOB_DATA_FORMAT = 1
JOB_FILE = "job.json"
SAMPLES_FILE = "samples.jsonl"
TASKS_DIR = "tasks"
#: The segment store keeps the directory of the key ``k`` as ``job-data/<k>``.
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


def job_data_key(data: JobData) -> str:
    """The SHA-256 of the samples of ``data`` and of the files of its tasks: equal for two jobs of the same samples,
    whatever their data IDs and requests."""
    digest = hashlib.sha256()
    for entry, sample in zip(data.to_dict()["samples"], data.samples, strict=True):
        if isinstance(sample, TaskSample):
            entry = {**entry, "task_path": sample.task_path.name}
            for path in sorted(item for item in sample.task_path.rglob("*") if item.is_file()):
                digest.update(path.relative_to(sample.task_path).as_posix().encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def write_job_data(data: JobData, directory: Path) -> None:
    """Write ``data`` as a job data directory at ``directory``, which must not exist.

    The function writes a sibling temporary directory and renames it, so a reader never sees part of a directory.
    It copies each task's directory. ``ValueError`` for data with both kinds of samples.
    """
    kind = job_data_kind(data)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        lines: list[str] = []
        tasks = 0
        for entry, sample in zip(data.to_dict()["samples"], data.samples, strict=True):
            if isinstance(sample, TaskSample):
                relative = f"{TASKS_DIR}/{tasks}/{sample.task_path.name}"
                shutil.copytree(sample.task_path, staged / relative)
                entry = {**entry, "task_path": relative}
                tasks += 1
            lines.append(json.dumps(entry, allow_nan=False, separators=(",", ":")))
        (staged / SAMPLES_FILE).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        header = {"format": JOB_DATA_FORMAT, "key": job_data_key(data), "sample_count": len(lines), "kind": kind}
        (staged / JOB_FILE).write_text(json.dumps(header), encoding="utf-8")
        os.replace(staged, directory)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def read_job_data(directory: Path, data_id: str | None = None) -> JobData:
    """The ``JobData`` of the job data directory ``directory``, without a request; each task path is absolute,
    under ``directory``. Its ID is ``data_id``, or the directory's key. ``ValueError`` for another format."""
    header = json.loads((directory / JOB_FILE).read_text(encoding="utf-8"))
    if not isinstance(header, Mapping) or header.get("format") != JOB_DATA_FORMAT:
        raise ValueError(f"{directory} is no job data directory of format {JOB_DATA_FORMAT}")
    samples = []
    with (directory / SAMPLES_FILE).open(encoding="utf-8") as lines:
        for line in lines:
            entry = json.loads(line)
            if entry.get("kind") == "task":
                entry = {**entry, "task_path": str(directory / entry["task_path"])}
            samples.append(entry)
    return JobData.from_dict({"data_id": data_id or header["key"], "samples": samples, "request": None})


@dataclass(frozen=True)
class JobDataRef:
    """The job data directory of one training job.

    Values: ``uri`` is the URI that the store gives the directory. ``path`` is a local directory with its content,
    which ``JobDataStore.open_job_data`` gives. ``data_id`` is the ``JobData.data_id`` of the job's data,
    ``sample_count`` its number of samples, and ``kind`` the kind of its samples, or None without samples.
    Purpose: ``TrainingJob.data`` holds it. The jobs queue keeps it without ``path``, as ``to_dict`` gives it.
    """

    uri: str
    path: Path
    data_id: str
    sample_count: int
    kind: JobDataKind | None

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("JobDataRef.uri must be a non-empty string")
        if not isinstance(self.path, Path):
            raise TypeError("JobDataRef.path must be a Path")
        if not isinstance(self.data_id, str) or not self.data_id:
            raise ValueError("JobDataRef.data_id must be a non-empty string")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("JobDataRef.sample_count must be a non-negative int")
        if self.kind is not None and self.kind not in get_args(JobDataKind):
            raise ValueError("JobDataRef.kind must be trajectory, task or None")

    def to_dict(self) -> dict[str, Any]:
        """``{data_id, uri, sample_count, kind}``: the reference without its local path."""
        return {"data_id": self.data_id, "uri": self.uri, "sample_count": self.sample_count, "kind": self.kind}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: Path) -> JobDataRef:
        """The reference that ``to_dict`` wrote, with the local directory ``path``; ``ValueError`` for another
        value."""
        try:
            return cls(value["uri"], path, value["data_id"], value["sample_count"], value["kind"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed job data reference: {exc!r}") from exc


class JobDataStore(ABC):
    """Keeps the job data directories in a segment store, one per key, and deletes a directory that no job used for
    the time to live.

    Values: ``job_files`` is the segment store of the directories. A store records when a job last used each key:
    ``use_job_data(key, now)``, ``unused_job_data(before)`` (the keys last used before it) and
    ``forget_job_data(key)``.

    Operations:

    * ``write_job_data(data)`` writes the directory of ``data`` unless the store holds it, records the use, and
      returns its reference.
    * ``open_job_data(value)`` is the reference that ``JobDataRef.to_dict`` wrote, with a local directory of its
      content; it records the use.
    * ``expire_job_data(before)`` deletes each directory that no job used since ``before``; it returns the count.

    Purpose: the data store is one (``reef.storage.data.DataStore``). Control writes the directory of each job that
    it submits, the training service opens it, and control expires the directories.
    """

    job_files: SegmentStore

    @abstractmethod
    def use_job_data(self, key: str, now: float) -> None: ...

    @abstractmethod
    def unused_job_data(self, before: float) -> tuple[str, ...]: ...

    @abstractmethod
    def forget_job_data(self, key: str) -> None: ...

    def write_job_data(self, data: JobData) -> JobDataRef:
        key = job_data_key(data)
        name = f"{JOB_DATA_PREFIX}/{key}"
        uri = self.job_files.find_directory(name)
        if uri is None:
            with tempfile.TemporaryDirectory() as staging:
                directory = Path(staging) / key
                write_job_data(data, directory)
                uri = self.job_files.put_directory(name, directory)
        self.use_job_data(key, time.time())
        return JobDataRef(
            uri, self.job_files.open_directory(uri), data.data_id, len(data.samples), job_data_kind(data)
        )

    def open_job_data(self, value: Mapping[str, Any]) -> JobDataRef:
        uri = value.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError("a job data reference has a uri")
        path = self.job_files.open_directory(uri)
        header = json.loads((path / JOB_FILE).read_text(encoding="utf-8"))
        self.use_job_data(header["key"], time.time())
        return JobDataRef.from_dict(value, path)

    def expire_job_data(self, before: float) -> int:
        expired = self.unused_job_data(before)
        for key in expired:
            self.job_files.delete_directory(f"{JOB_DATA_PREFIX}/{key}")
            self.forget_job_data(key)
        return len(expired)


class LocalJobDataStore(JobDataStore):
    """Job data directories in a local segment store at ``directory``, with the times of use in memory.

    Purpose: the store of a process that has no data store, and of tests.
    """

    def __init__(self, directory: Path | None = None) -> None:
        root = Path(tempfile.mkdtemp(prefix="reef-job-data-")) if directory is None else Path(directory)
        self.job_files = LocalSegmentStore(root)
        self.used: dict[str, float] = {}

    def use_job_data(self, key: str, now: float) -> None:
        self.used[key] = now

    def unused_job_data(self, before: float) -> tuple[str, ...]:
        return tuple(key for key, used in self.used.items() if used < before)

    def forget_job_data(self, key: str) -> None:
        self.used.pop(key, None)


__all__ = [
    "JOB_DATA_FORMAT",
    "JOB_DATA_PREFIX",
    "JOB_FILE",
    "SAMPLES_FILE",
    "TASKS_DIR",
    "JobDataKind",
    "JobDataRef",
    "JobDataStore",
    "LocalJobDataStore",
    "job_data_key",
    "job_data_kind",
    "read_job_data",
    "write_job_data",
]
