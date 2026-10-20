"""The record-to-task contract: each record set from the record-to-record stage becomes tasks.

A task is a ``TrainDataItem``: a ``TrajectoryItem`` that trains directly, or a
``TaskItem`` that the algorithm rolls out. The training pipeline pushes each
record set that the record stage forms (``push``) and pulls the task results
(``pull``). It gives each result back to the record stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.batches import TrainDataItem
from reef.core.plugins import NoSettings, Plugin, PluginConfigError
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, RecordSet, TaskStatus


@dataclass(frozen=True)
class TaskResult:
    """The tasks made from one record set, and what the record stage does with the set.

    Values: a ``READY`` result holds one or more tasks. A ``SKIPPED`` result holds no task.
    """

    status: TaskStatus
    tasks: tuple[TrainDataItem, ...] = ()

    def __post_init__(self) -> None:
        if (self.status is TaskStatus.READY) != bool(self.tasks):
            raise ValueError("a READY result holds one or more tasks, and a SKIPPED result holds no task")

    @classmethod
    def ready(cls, tasks: tuple[TrainDataItem, ...]) -> TaskResult:
        """The set trains as ``tasks``."""
        return cls(TaskStatus.READY, tasks)

    @classmethod
    def skipped(cls) -> TaskResult:
        """The set cannot train. The record stage ends its records."""
        return cls(TaskStatus.SKIPPED)


class Record2Task(ABC, Plugin):
    """The record-to-task stage: record sets in, task results out.

    The pipeline pushes each record set one time. ``pull`` returns each pushed
    set's result exactly one time, by set ID, as soon as it is ready. Both run
    on the trainer thread and must not block on network or model calls. A
    stage with slow work runs it on a worker that it owns, and ``close`` stops
    the worker. Raise on a broken training contract. Most stages make the
    tasks of a set at once: they inherit ``ImmediateRecord2Task``. A
    deployment names a stage class in ``pipeline.record2task`` and gives its
    config. The class parses the config into its settings dataclass and builds
    the stage in ``from_config``.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the stage of one scenario from its config.

        ``record2record`` is the scenario's record-to-record stage, built first.
        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    @classmethod
    def scenario_state_dirs(cls, config: Mapping[str, Any], scenario: str) -> tuple[Path, ...]:
        """The directories under which a stage built from ``config`` keeps a scenario's state, for archiving."""
        return ()

    @abstractmethod
    def push(self, record_set: RecordSet) -> None:
        """Take one record set to make its tasks."""

    @abstractmethod
    def pull(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        """Return the results that are ready since the last call, by set ID."""

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class ImmediateRecord2Task(Record2Task):
    """A record-to-task stage that makes the tasks of a set when the pipeline pushes it.

    A subclass implements ``make_tasks``. A subclass with its own ``__init__``
    calls ``super().__init__()``.
    """

    def __init__(self) -> None:
        self.results: list[tuple[Hashable, TaskResult]] = []

    @abstractmethod
    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        """Return the tasks of ``record_set``, or a skip."""

    def push(self, record_set: RecordSet) -> None:
        self.results.append((record_set.set_id, self.make_tasks(record_set)))

    def pull(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        results, self.results = tuple(self.results), []
        return results


class NoTasks(Record2Task):
    """The record-to-task stage of a pipeline that trains nothing. Its record stage, ``NoRecordSets``, forms no set."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def push(self, record_set: RecordSet) -> None:
        raise TypeError(f"{type(self).__name__} makes no tasks: pair it with NoRecordSets")

    def pull(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        return ()
