"""Job data directories: the data of one training job as read-only files that its training component reads.

Reef selects the samples of a job, writes them as a job data directory, and gives the training component the
directory (``TrainingJob.data_dir``) and its URI (``TrainingJob.data_uri``). A component that runs the job in a
container mounts the URI there. The directory holds no training request: Reef keeps the request for its own
accounting, and a component reads only the data and the job's parameters.

Format 1 of a directory:

- ``job.json``: ``{"format": 1, "data_id": str, "sample_count": int, "kind": "trajectory" | "task" | null}``.
- ``samples.jsonl``: one sample per line, in job order, as ``JobData.to_dict`` gives it, except that a task's
  ``task_path`` is relative to the directory: ``tasks/<n>/<directory name>``.
- ``tasks/<n>/<directory name>/``: the directory of the ``n``-th task, where ``n`` counts the tasks from 0.

A directory does not change after it is written. A store names it by the ID of its job.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, get_args
from urllib.parse import unquote, urlparse

from reef.core.job_data import JobData, TaskSample
from reef.core.plugins import NoSettings, Plugin

#: The format of the directories that ``write_job_data`` writes.
JOB_DATA_FORMAT = 1
JOB_FILE = "job.json"
SAMPLES_FILE = "samples.jsonl"
TASKS_DIR = "tasks"
JobDataKind = Literal["trajectory", "task"]
#: A job ID that names a directory: no path separator, and not "." or "..".
JOB_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def job_data_kind(data: JobData) -> JobDataKind | None:
    """The kind of the samples of ``data``, or None for data without samples; ``ValueError`` for mixed kinds."""
    kinds = {"task" if isinstance(sample, TaskSample) else "trajectory" for sample in data.samples}
    if len(kinds) > 1:
        raise ValueError(f"job data {data.data_id!r} holds both trajectories and tasks")
    if not kinds:
        return None
    return "task" if "task" in kinds else "trajectory"


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
        header = {"format": JOB_DATA_FORMAT, "data_id": data.data_id, "sample_count": len(lines), "kind": kind}
        (staged / JOB_FILE).write_text(json.dumps(header), encoding="utf-8")
        os.replace(staged, directory)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def read_job_data(directory: Path) -> JobData:
    """The ``JobData`` of the job data directory ``directory``, without a request; each task path is absolute,
    under ``directory``. ``ValueError`` for a directory of another format."""
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
    return JobData.from_dict({"data_id": header["data_id"], "samples": samples, "request": None})


@dataclass(frozen=True)
class JobDataRef:
    """The job data directory of one training job.

    Values: ``uri`` is the URI that the store gives the directory. ``path`` is a local directory with its content,
    which ``JobDataStore.open`` gives. ``data_id`` is the ``JobData.data_id`` of the data, ``sample_count`` its
    number of samples, and ``kind`` the kind of its samples, or None without samples.
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


class JobDataStore(Plugin, ABC):
    """Keeps the job data directory of each training job.

    Operations:

    * ``write(job_id, data)`` writes ``data`` as the directory of the job ``job_id`` and returns its URI. A second
      write of the same job ID changes nothing and returns the same URI.
    * ``open(uri)`` returns a local directory with the content of the directory at ``uri``. The directory stays
      readable while the job runs, also after the process restarts. A store that keeps the data elsewhere copies it
      into a local cache.
    * ``delete(job_id)`` removes the job's directory. A missing directory is not an error.

    Purpose: a deployment names a subclass in ``storage.job_data_backend``. Control writes the directory of each
    job that it submits. The training service opens it before the component runs the job. Control deletes it when
    it deletes the job's item from the jobs queue.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        """Build the store that a deployment names. ``directory`` is ``storage.job_data_dir``: a local store keeps
        the directories there, and a remote store can use it as its local cache."""

    @abstractmethod
    def write(self, job_id: str, data: JobData) -> str: ...

    @abstractmethod
    def open(self, uri: str) -> Path: ...

    @abstractmethod
    def delete(self, job_id: str) -> None: ...

    def ref(self, job_id: str, data: JobData) -> JobDataRef:
        """Write ``data`` as the directory of ``job_id`` and return its reference."""
        uri = self.write(job_id, data)
        return JobDataRef(uri, self.open(uri), data.data_id, len(data.samples), job_data_kind(data))

    def close(self) -> None:
        """Release the resources of the store. The default holds none."""


def job_directory_name(job_id: str) -> str:
    """``job_id`` as a directory name; ``ValueError`` for an ID that is no safe name."""
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError(f"job ID {job_id!r} is no safe directory name")
    return job_id


class LocalJobDataStore(JobDataStore):
    """Keeps each job's directory at ``<directory>/<job_id>``, with the URI ``file://<absolute path>``.

    The processes that share ``directory``, such as control and the training service on one volume, share the store.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(directory)

    def write(self, job_id: str, data: JobData) -> str:
        target = self.directory / job_directory_name(job_id)
        if not (target / JOB_FILE).is_file():
            try:
                write_job_data(data, target)
            except OSError:
                # Another writer of the same job renamed its directory into place first.
                if not (target / JOB_FILE).is_file():
                    raise
        return target.as_uri()

    def open(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"the local job data store opens file:// URIs, not {uri!r}")
        path = Path(unquote(parsed.path))
        if not (path / JOB_FILE).is_file():
            raise FileNotFoundError(f"no job data directory at {uri}")
        return path

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.directory / job_directory_name(job_id), ignore_errors=True)


__all__ = [
    "JOB_DATA_FORMAT",
    "JOB_FILE",
    "SAMPLES_FILE",
    "TASKS_DIR",
    "JobDataKind",
    "JobDataRef",
    "JobDataStore",
    "LocalJobDataStore",
    "job_data_kind",
    "read_job_data",
    "write_job_data",
]
