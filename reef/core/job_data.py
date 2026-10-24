"""The data of one training job: the trajectories and tasks that Reef gives a training component.

Reef submits one training job for each ``JobData`` value. The training
framework cuts the value's items into its own optimizer batches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from reef.core.training_request import TrainingRequest


@dataclass(frozen=True)
class TrajectoryItem:
    """An ATIF document and its grouping within one job's data and record references.

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

    def with_metadata(self, **fields: Any) -> TrajectoryItem:
        """Return a new document with updated Reef metadata, preserving other extensions."""
        metadata = {**self.metadata, **fields}
        extra = {**(self.trajectory.get("extra") or {}), "reef": metadata}
        return replace(self, trajectory={**self.trajectory, "extra": extra})

    def with_training(self, **fields: Any) -> TrajectoryItem:
        """Return a new document with updated captured training fields."""
        return self.with_metadata(training={**self.training, **fields})


@dataclass(frozen=True)
class TaskItem:
    """A Harbor task directory containing its instruction, environment and verifier.

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


TrainDataItem = TrajectoryItem | TaskItem


@dataclass(frozen=True)
class JobData:
    """The data of one training job: an ID, the items in order, and the training instruction, if any.

    Values: ``data_id`` is the ID that the training pipeline assigns,
    ``<scenario>:data:<number>`` for automatic data and
    ``<scenario>:instruction:<request id>`` for an instruction's data.
    ``items`` is a tuple of ``TrajectoryItem`` and ``TaskItem`` values in the
    order of the task-to-job stage; it can be empty, for a harness step that
    runs an instruction alone. ``request`` is the training instruction that the
    data carries, or None. The constructor refuses other item types.
    Purpose: ``Task2Job.make_job_data`` makes it, and the training component
    (``TrainingJob.data``) or the harness backend reads it. The consuming
    algorithm decides which item kinds it supports and refuses the others
    explicitly.
    """

    data_id: str
    items: tuple[TrainDataItem, ...] = ()
    request: TrainingRequest | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.data_id, str) or not self.data_id:
            raise ValueError("JobData.data_id must be a non-empty string")
        if self.request is not None and not isinstance(self.request, TrainingRequest):
            raise TypeError("JobData.request must be a TrainingRequest or None")
        if not isinstance(self.items, tuple):
            raise TypeError("JobData.items must be a tuple")
        for index, item in enumerate(self.items):
            if not isinstance(item, (TrajectoryItem, TaskItem)):
                raise TypeError(f"JobData.items[{index}] must be a TrajectoryItem or TaskItem")

    def to_dict(self) -> dict[str, Any]:
        """The value as JSON values; ``from_dict`` reads them back to an equal value.

        Precondition: every trajectory, every task's ``metadata`` and the
        request hold only JSON values (strings, finite numbers, booleans,
        None, lists, and objects with string keys). The result holds the same
        values, so it is JSON exactly when the precondition holds.
        """
        items: list[dict[str, Any]] = []
        for item in self.items:
            if isinstance(item, TrajectoryItem):
                items.append(
                    {
                        "kind": "trajectory",
                        "trajectory": dict(item.trajectory),
                        "group_id": item.group_id,
                        "source_agent_record_ids": list(item.source_agent_record_ids),
                    }
                )
            else:
                items.append(
                    {
                        "kind": "task",
                        "task_path": str(item.task_path),
                        "source_agent_record_ids": list(item.source_agent_record_ids),
                        "metadata": dict(item.metadata),
                    }
                )
        request = None if self.request is None else {"id": self.request.id, **self.request.to_dict()}
        return {"data_id": self.data_id, "items": items, "request": request}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobData:
        """Read the value that ``to_dict`` wrote.

        Raise ``ValueError`` for a malformed value: a missing key, a value of
        the wrong type, an unknown item kind, or an item or request that fails
        its own checks.
        """
        try:
            items: list[TrainDataItem] = []
            for item in value["items"]:
                if item["kind"] == "trajectory":
                    items.append(
                        TrajectoryItem(item["trajectory"], item["group_id"], tuple(item["source_agent_record_ids"]))
                    )
                elif item["kind"] == "task":
                    items.append(
                        TaskItem(Path(item["task_path"]), tuple(item["source_agent_record_ids"]), item["metadata"])
                    )
                else:
                    raise ValueError(f"unknown job data item kind {item['kind']!r}")
            request_value = value["request"]
            request = (
                None
                if request_value is None
                else TrainingRequest.from_dict(
                    {key: field_value for key, field_value in request_value.items() if key != "id"},
                    request_id=request_value["id"],
                )
            )
            return cls(value["data_id"], tuple(items), request=request)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"malformed job data: {exc!r}") from exc


def trajectories(data: JobData) -> tuple[TrajectoryItem, ...]:
    """Read ATIF items in the data's order; tasks require a rollout-capable algorithm."""
    items: list[TrajectoryItem] = []
    for index, item in enumerate(data.items):
        if not isinstance(item, TrajectoryItem):
            raise TypeError(f"this algorithm requires trajectories; unsupported item {index}: {type(item).__name__}")
        items.append(item)
    return tuple(items)


def trajectory_groups(data: JobData) -> tuple[tuple[TrajectoryItem, ...], ...]:
    """Read explicit, contiguous comparison groups without reordering trajectory rows."""
    groups: dict[str, list[TrajectoryItem]] = {}
    previous: str | None = None
    for item in trajectories(data):
        if item.group_id is None:
            raise ValueError("group-relative training requires a group_id on every trajectory")
        if item.group_id in groups and item.group_id != previous:
            raise ValueError("comparison group trajectories must be contiguous")
        groups.setdefault(item.group_id, []).append(item)
        previous = item.group_id
    return tuple(tuple(group) for group in groups.values())
