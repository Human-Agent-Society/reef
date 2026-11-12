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
- ``delayed.jsonl``, only for data with delayed samples (``JobData.delayed``): one delayed sample per line, in order,
  as the ``delayed`` entries of ``JobData.to_dict``.

``key`` is ``job_data_key`` of the samples and the delayed samples, so two jobs of the same data share one directory.
A directory does not change after it is written. The data store keeps the directories
(``DataStore.write_job_directory``), and deletes a directory that no job used for the time to live.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from reef.core.job_data import JobData, TaskSample

#: The format of the directories that ``write_job_data`` writes.
JOB_DATA_FORMAT = 1
JOB_FILE = "job.json"
SAMPLES_FILE = "samples.jsonl"
DELAYED_FILE = "delayed.jsonl"
TASKS_DIR = "tasks"
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


def job_data_key(data: JobData) -> str:
    """The SHA-256 of the samples of ``data``, of the files of its tasks and of its delayed samples: equal for two jobs
    of the same data, whatever their data IDs and requests. Data without delayed samples has the key of its samples
    alone."""
    value = data.to_dict()
    digest = hashlib.sha256()
    for entry, sample in zip(value["samples"], data.samples, strict=True):
        if isinstance(sample, TaskSample):
            entry = {**entry, "task_path": sample.task_path.name}
            for path in sorted(item for item in sample.task_path.rglob("*") if item.is_file()):
                digest.update(path.relative_to(sample.task_path).as_posix().encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    if data.delayed:
        digest.update(json.dumps({"delayed": value["delayed"]}, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def write_job_data(data: JobData, directory: Path) -> None:
    """Write ``data`` as a job data directory at ``directory``, which must not exist.

    The function writes a sibling temporary directory and renames it, so a reader never sees part of a directory.
    It copies each task's directory, and writes ``DELAYED_FILE`` only for data with delayed samples. ``ValueError``
    for data with both kinds of samples.
    """
    kind = job_data_kind(data)
    value = data.to_dict()
    directory.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        tasks = 0
        # One line at a time: the lines of a job of many trajectories are gigabytes.
        with (staged / SAMPLES_FILE).open("w", encoding="utf-8") as lines:
            for entry, sample in zip(value["samples"], data.samples, strict=True):
                if isinstance(sample, TaskSample):
                    relative = f"{TASKS_DIR}/{tasks}/{sample.task_path.name}"
                    shutil.copytree(sample.task_path, staged / relative)
                    entry = {**entry, "task_path": relative}
                    tasks += 1
                lines.write(json.dumps(entry, allow_nan=False, separators=(",", ":")) + "\n")
        if data.delayed:
            with (staged / DELAYED_FILE).open("w", encoding="utf-8") as lines:
                for entry in value["delayed"]:
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


def read_job_data(directory: Path, data_id: str | None = None) -> JobData:
    """The ``JobData`` of the job data directory ``directory``, without a request; each task path is absolute,
    under ``directory``. Its ID is ``data_id``, or the directory's key. Its delayed samples are those of
    ``DELAYED_FILE``, or none without that file. ``ValueError`` for another format."""
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
    value: dict[str, Any] = {"data_id": data_id or header["key"], "samples": samples, "request": None}
    if (directory / DELAYED_FILE).is_file():
        value["delayed"] = [
            json.loads(line) for line in (directory / DELAYED_FILE).read_text(encoding="utf-8").splitlines()
        ]
    return JobData.from_dict(value)


@dataclass(frozen=True)
class JobDataRef:
    """The job data directory of one training job.

    Values: ``uri`` is the URI that the store gives the directory. ``path`` is a local directory with its content,
    which ``DataStore.open_job_directory`` gives. ``data_id`` is the ``JobData.data_id`` of the job's data,
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


__all__ = [
    "DELAYED_FILE",
    "JOB_DATA_FORMAT",
    "JOB_DATA_PREFIX",
    "JOB_FILE",
    "SAMPLES_FILE",
    "TASKS_DIR",
    "JobDataKind",
    "JobDataRef",
    "job_data_key",
    "job_data_kind",
    "read_job_data",
    "write_job_data",
]
