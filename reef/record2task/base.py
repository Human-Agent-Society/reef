"""The record-to-task contract: one record set from the record-to-record stage becomes tasks.

A task is a ``TrainDataItem``: a ``TrajectoryItem`` that trains directly, or a
``TaskItem`` that the algorithm rolls out. The training pipeline calls ``make_tasks``
for each set that the record stage forms and gives the ``TaskResult`` back to
the stage. A stage that needs slow work returns ``TaskResult.pending()`` and
hands the finished result back later through ``take_finished``.
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
    """The tasks made from one record set, and what the record stage does with the set."""

    status: TaskStatus
    tasks: tuple[TrainDataItem, ...] = ()

    @classmethod
    def ready(cls, tasks: tuple[TrainDataItem, ...]) -> TaskResult:
        """The set trains as ``tasks``."""
        return cls(TaskStatus.READY, tasks)

    @classmethod
    def pending(cls) -> TaskResult:
        """The stage is still working on the set. ``take_finished`` hands back the finished result."""
        return cls(TaskStatus.PENDING)

    @classmethod
    def skipped(cls) -> TaskResult:
        """The set cannot train. The record stage ends its records."""
        return cls(TaskStatus.SKIPPED)


class Record2Task(ABC, Plugin):
    """The record-to-task stage: turn one record set into tasks.

    Runs on the trainer thread without network or model calls. Raise on a
    broken training contract. Slow work runs on a worker that the stage owns:
    ``make_tasks`` submits it and returns ``TaskResult.pending()``, and
    ``take_finished`` hands back the result. ``close`` stops the worker. A
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
    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        """Return the tasks of ``record_set``, a skip, or ``TaskResult.pending()`` while slow work runs."""

    def take_finished(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        """Return the results of pending sets that finished since the last call, by set ID.

        The pipeline calls it each time it looks for a batch and settles each set once.
        A result must not be pending.
        """
        return ()

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class NoTasks(Record2Task):
    """Make no tasks. The record stage forgets each set; the no-update default."""

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

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        return TaskResult(TaskStatus.INGESTED)
