from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from reef.core.training_request import TrainingRequest


@dataclass(frozen=True)
class TrajectoryItem:
    """An ATIF document and its batch-local grouping and record references.

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
class TrainingBatch:
    """One reserved processor output, possibly mixing trajectories and tasks.

    Items stay in processor order. The consuming algorithm decides which item
    kinds and trajectory representations it supports, and must reject unsupported
    input explicitly. Empty batches support request-only harness evolution.
    """

    batch_id: str
    items: tuple[TrainDataItem, ...] = ()
    request: TrainingRequest | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("TrainingBatch.items must be a tuple")
        for index, item in enumerate(self.items):
            if not isinstance(item, (TrajectoryItem, TaskItem)):
                raise TypeError(f"TrainingBatch.items[{index}] must be a TrajectoryItem or TaskItem")


@dataclass(frozen=True, slots=True)
class StepScheduling:
    """How the training runtime cuts one reserved batch into optimizer steps.

    A recipe binds this next to its objective in ``WeightTrainingSpec``; the
    runtime carries it to the backend with the objective reference. A
    processor hands the objective one batch; the runtime may run several
    optimizer steps over it. The *rollout* is the unit that is never split
    across steps — a comparison set by default, or one sample with
    ``unit="sample"``.

    ``batch_size`` is rollouts per optimizer step:

    * ``"configured"`` — the backend's own setting (Slime ``--global-batch-size``);
    * ``"actual"`` — every rollout in the batch, i.e. one step per batch;
    * an ``int`` — an explicit step size, so a batch of 1024 rollouts with
      ``batch_size=64`` trains sixteen steps.

    ``epochs`` repeats the whole batch that many passes; every pass is its own
    run of steps, and the reference / old-policy log-probs are computed once
    before the first, so multi-epoch training is PPO-style off-policy after
    the first pass — only an objective declaring ``supports_multiple_epochs``
    accepts it. ``shuffle`` reorders rollouts independently in every epoch,
    deterministically from the batch id.

    ``remainder`` decides what happens when an epoch's rollout count is not
    a multiple of the step size: ``"partial"`` (default) trains the tail as
    one smaller final step — what SFT trainers do with ``drop_last=False`` —
    ``"drop"`` leaves the tail out, ``"error"`` refuses the batch. A partial
    step needs at least one sample per data-parallel rank and, under static
    micro-batching, a sample count divisible by ``dp_size * micro_batch_size``;
    dynamic batching splits its bins to fit.
    """

    unit: Literal["comparison_set", "sample"] = "comparison_set"
    batch_size: Literal["configured", "actual"] | int = "configured"
    epochs: int = 1
    shuffle: bool = False
    remainder: Literal["partial", "drop", "error"] = "partial"

    def __post_init__(self) -> None:
        if self.unit not in ("comparison_set", "sample"):
            raise ValueError(f"StepScheduling.unit must be 'comparison_set' or 'sample', got {self.unit!r}")
        if isinstance(self.batch_size, bool) or (
            not isinstance(self.batch_size, int) and self.batch_size not in ("configured", "actual")
        ):
            raise ValueError(
                f"StepScheduling.batch_size must be 'configured', 'actual' or a positive int, got {self.batch_size!r}"
            )
        if isinstance(self.batch_size, int) and self.batch_size <= 0:
            raise ValueError(f"StepScheduling.batch_size must be positive, got {self.batch_size}")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"StepScheduling.epochs must be a positive int, got {self.epochs!r}")
        if not isinstance(self.shuffle, bool):
            raise ValueError(f"StepScheduling.shuffle must be a bool, got {self.shuffle!r}")
        if self.remainder not in ("partial", "drop", "error"):
            raise ValueError(f"StepScheduling.remainder must be 'partial', 'drop' or 'error', got {self.remainder!r}")
        if self.remainder == "drop" and not isinstance(self.batch_size, int):
            raise ValueError("StepScheduling.remainder='drop' requires an explicit integer batch_size")


def trajectories(batch: TrainingBatch) -> tuple[TrajectoryItem, ...]:
    """Read ATIF items in batch order; tasks require a rollout-capable algorithm."""
    items: list[TrajectoryItem] = []
    for index, item in enumerate(batch.items):
        if not isinstance(item, TrajectoryItem):
            raise TypeError(f"this algorithm requires trajectories; unsupported item {index}: {type(item).__name__}")
        items.append(item)
    return tuple(items)


def trajectory_groups(batch: TrainingBatch) -> tuple[tuple[TrajectoryItem, ...], ...]:
    """Read explicit, contiguous comparison groups without reordering trajectory rows."""
    groups: dict[str, list[TrajectoryItem]] = {}
    previous: str | None = None
    for item in trajectories(batch):
        if item.group_id is None:
            raise ValueError("group-relative training requires a group_id on every trajectory")
        if item.group_id in groups and item.group_id != previous:
            raise ValueError("comparison group trajectories must be contiguous")
        groups.setdefault(item.group_id, []).append(item)
        previous = item.group_id
    return tuple(tuple(group) for group in groups.values())
