"""The data of one training job, the trajectories and tasks that Reef gives a training component, and its outcome.

Reef submits one training job for each ``JobData`` value. The training
framework cuts the value's items into its own optimizer batches. When the job
ends, its ``JobOutcome`` is what the training pipeline and the recipe's job plan
learn about it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from reef.core.training_request import TrainingRequest


@dataclass(frozen=True)
class TrajectorySample:
    """An ATIF document and its grouping within one job's data and record references.

    Values: ``trajectory`` is a mapping with an ``ATIF-v`` ``schema_version``,
    an ``agent`` object and a non-empty ``steps`` list. ``group_id`` is None or
    a string. ``source_agent_record_ids`` is a tuple of the non-empty IDs of
    the agent records the trajectory came from. Purpose: the record-to-sample
    stage makes the value, and the training component reads it.

    The document is the single source of trajectory data. Reef-specific feedback,
    captured provider records, and exact training tensors live in ``extra.reef``.
    ``training`` reads its ``training`` object without reconstructing tokens.
    Full ATIF validation and algorithm-required fields belong to the consumer.
    """

    trajectory: Mapping[str, Any]
    group_id: str | None = None
    source_agent_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, Mapping):
            raise TypeError("trajectory must be an ATIF document")
        version = self.trajectory.get("schema_version")
        if not isinstance(version, str) or not version.startswith("ATIF-v"):
            raise ValueError("trajectory must have an ATIF schema_version")
        if not isinstance(self.trajectory.get("agent"), Mapping):
            raise ValueError("ATIF trajectories must contain an agent object")
        steps = self.trajectory.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("ATIF trajectories must contain a non-empty steps list")
        if self.group_id is not None and not isinstance(self.group_id, str):
            raise TypeError("trajectory group_id must be a string or None")
        check_record_ids(self.source_agent_record_ids)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Reef's JSON metadata within the ATIF extension object."""
        extra = self.trajectory.get("extra") or {}
        if not isinstance(extra, Mapping):
            raise ValueError("ATIF extra must be an object")
        metadata = extra.get("reef", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("ATIF extra.reef must be an object")
        return metadata

    @property
    def training(self) -> Mapping[str, Any]:
        """Captured training fields; absence does not authorize re-tokenization."""
        training = self.metadata.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("ATIF extra.reef.training must be an object")
        return training

    def with_metadata(self, **fields: Any) -> TrajectorySample:
        """Return a new document with updated Reef metadata, preserving other extensions."""
        metadata = {**self.metadata, **fields}
        extra = {**(self.trajectory.get("extra") or {}), "reef": metadata}
        return replace(self, trajectory={**self.trajectory, "extra": extra})

    def with_training(self, **fields: Any) -> TrajectorySample:
        """Return a new document with updated captured training fields."""
        return self.with_metadata(training={**self.training, **fields})


@dataclass(frozen=True)
class TaskSample:
    """A Harbor task directory containing its instruction, environment and verifier.

    Values: ``task_path`` is a ``Path``, ``source_agent_record_ids`` a tuple of
    the non-empty IDs of the agent records the task came from, and
    ``metadata`` a mapping. Purpose: the record-to-sample stage makes the value,
    and a training component that runs task jobs reads it.

    The path is resolved in the consuming algorithm's execution environment;
    constructing a value neither reads files nor launches a rollout. Algorithms
    that support tasks own task validation, rollout, and conversion to trajectories.
    """

    task_path: Path
    source_agent_record_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_path, Path):
            raise TypeError("task_path must be a pathlib.Path to a Harbor task directory")
        check_record_ids(self.source_agent_record_ids)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("task metadata must be a mapping")


#: One sample of job data: a trajectory that trains as it is, or a Harbor task that the training component rolls
#: out. The record-to-sample stage makes samples, ``job_samples`` and the sample buffer count them, and
#: ``JobData.samples`` holds them.
Sample = TrajectorySample | TaskSample


def check_record_ids(record_ids: tuple[str, ...]) -> None:
    """Raise ``TypeError`` unless ``record_ids`` is a tuple of non-empty strings."""
    if not isinstance(record_ids, tuple) or not all(isinstance(value, str) and value for value in record_ids):
        raise TypeError("source_agent_record_ids must be a tuple of non-empty strings")


@dataclass(frozen=True)
class JobData:
    """The data of one training job: an ID, the samples in order, and the training instruction, if any.

    Values: ``data_id`` is the ID that the training pipeline assigns,
    ``<scenario>:data:<number>`` for automatic data and
    ``<scenario>:instruction:<request id>`` for an instruction's data.
    ``samples`` is a tuple of ``TrajectorySample`` and ``TaskSample`` values in
    the order of the sample-to-job stage; it can be empty, for a harness step
    that runs an instruction alone. ``request`` is the training instruction that
    the data carries, or None. The constructor refuses other sample types.
    Purpose: ``Sample2Job.make_job_data`` makes it, and the training component
    (``TrainingJob.data``) or the harness backend reads it. The consuming
    algorithm decides which sample kinds it supports and refuses the others
    explicitly.
    """

    data_id: str
    samples: tuple[Sample, ...] = ()
    request: TrainingRequest | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.data_id, str) or not self.data_id:
            raise ValueError("JobData.data_id must be a non-empty string")
        if self.request is not None and not isinstance(self.request, TrainingRequest):
            raise TypeError("JobData.request must be a TrainingRequest or None")
        if not isinstance(self.samples, tuple):
            raise TypeError("JobData.samples must be a tuple")
        for index, sample in enumerate(self.samples):
            if not isinstance(sample, (TrajectorySample, TaskSample)):
                raise TypeError(f"JobData.samples[{index}] must be a TrajectorySample or TaskSample")

    def to_dict(self) -> dict[str, Any]:
        """The value as JSON values; ``from_dict`` reads them back to an equal value.

        Precondition: every trajectory, every task's ``metadata`` and the
        request hold only JSON values (strings, finite numbers, booleans,
        None, lists, and objects with string keys). The result holds the same
        values, so it is JSON exactly when the precondition holds.
        """
        samples: list[dict[str, Any]] = []
        for sample in self.samples:
            if isinstance(sample, TrajectorySample):
                samples.append(
                    {
                        "kind": "trajectory",
                        "trajectory": dict(sample.trajectory),
                        "group_id": sample.group_id,
                        "source_agent_record_ids": list(sample.source_agent_record_ids),
                    }
                )
            else:
                samples.append(
                    {
                        "kind": "task",
                        "task_path": str(sample.task_path),
                        "source_agent_record_ids": list(sample.source_agent_record_ids),
                        "metadata": dict(sample.metadata),
                    }
                )
        request = None if self.request is None else {"id": self.request.id, **self.request.to_dict()}
        return {"data_id": self.data_id, "samples": samples, "request": request}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobData:
        """Read the value that ``to_dict`` wrote.

        Raise ``ValueError`` for a malformed value: a missing key, a value of
        the wrong type, an unknown sample kind, or a sample or request that fails
        its own checks.
        """
        try:
            samples: list[Sample] = []
            for entry in value["samples"]:
                if entry["kind"] == "trajectory":
                    samples.append(
                        TrajectorySample(
                            entry["trajectory"], entry["group_id"], tuple(entry["source_agent_record_ids"])
                        )
                    )
                elif entry["kind"] == "task":
                    samples.append(
                        TaskSample(
                            Path(entry["task_path"]), tuple(entry["source_agent_record_ids"]), entry["metadata"]
                        )
                    )
                else:
                    raise ValueError(f"unknown job data sample kind {entry['kind']!r}")
            request_value = value["request"]
            request = (
                None
                if request_value is None
                else TrainingRequest.from_dict(
                    {key: field_value for key, field_value in request_value.items() if key != "id"},
                    request_id=request_value["id"],
                )
            )
            return cls(value["data_id"], tuple(samples), request=request)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"malformed job data: {exc!r}") from exc


def trajectories(data: JobData) -> tuple[TrajectorySample, ...]:
    """Read the ATIF trajectory samples in the data's order; task samples require a rollout-capable algorithm."""
    found: list[TrajectorySample] = []
    for index, sample in enumerate(data.samples):
        if not isinstance(sample, TrajectorySample):
            raise TypeError(
                f"this algorithm requires trajectories; unsupported sample {index}: {type(sample).__name__}"
            )
        found.append(sample)
    return tuple(found)


def trajectory_groups(data: JobData) -> tuple[tuple[TrajectorySample, ...], ...]:
    """Read explicit, contiguous comparison groups without reordering trajectory rows."""
    groups: dict[str, list[TrajectorySample]] = {}
    previous: str | None = None
    for item in trajectories(data):
        if item.group_id is None:
            raise ValueError("group-relative training requires a group_id on every trajectory")
        if item.group_id in groups and item.group_id != previous:
            raise ValueError("comparison group trajectories must be contiguous")
        groups.setdefault(item.group_id, []).append(item)
        previous = item.group_id
    return tuple(tuple(group) for group in groups.values())


@dataclass(frozen=True)
class JobOutcome:
    """How one training job ended.

    Values: ``job_id`` and ``backend`` (the training component's name) are non-empty strings; ``event`` is
    ``finished`` or ``failed``; ``metrics`` is the component's telemetry for the whole job, a mapping that
    encodes as JSON and that Reef records without reading, such as a task job's ``rollout_rewards``;
    ``error`` is a non-empty string exactly when ``event`` is ``failed``. Purpose: the weight trainer's job
    loop makes it when it commits a job's end and keeps the last one in its state; the recipe's job plan
    (``JobPlanContext.last_job``) and the sample-to-job stage (``Sample2Job.job_ended``) read it.
    """

    job_id: str
    backend: str
    event: Literal["finished", "failed"]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("job_id", self.job_id), ("backend", self.backend)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"JobOutcome.{name} must be a non-empty string")
        if self.event not in ("finished", "failed"):
            raise ValueError(f"JobOutcome.event must be finished or failed, got {self.event!r}")
        if (self.error is not None) != (self.event == "failed"):
            raise ValueError("JobOutcome.error must be set exactly when the job failed")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("JobOutcome.error must be a non-empty string")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("JobOutcome.metrics must be a mapping")
        metrics = dict(self.metrics)
        json.dumps(metrics, allow_nan=False)
        object.__setattr__(self, "metrics", metrics)

    def to_dict(self) -> dict[str, Any]:
        """The value as a JSON object; ``from_dict`` of the result is an equal value."""
        value: dict[str, Any] = {
            "job_id": self.job_id,
            "backend": self.backend,
            "event": self.event,
            "metrics": dict(self.metrics),
        }
        if self.error is not None:
            value["error"] = self.error
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobOutcome:
        """Read ``to_dict``'s object; ``ValueError`` for another key set or a malformed value."""
        if not isinstance(value, Mapping):
            raise ValueError("a job outcome must be an object")
        keys = {"job_id", "backend", "event", "metrics"} | ({"error"} if "error" in value else set())
        if set(value) != keys:
            raise ValueError(f"a job outcome has the keys {sorted(keys)}, not {sorted(value)}")
        return cls(value["job_id"], value["backend"], value["event"], value["metrics"], value.get("error"))
