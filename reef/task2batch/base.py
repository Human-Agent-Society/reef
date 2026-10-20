"""The task-to-batch contract: the tasks of the selected batch units become one batch.

The training pipeline (:class:`~reef.train.pipeline.TrainingPipeline`) selects
the batch units of the next batch from the record-to-record stage, takes the
tasks that the record-to-task stage made from them, and gives the tasks to a
recipe's ``Task2Batch``. A task is a ``TrainDataItem``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.batches import TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem
from reef.core.config import config_option
from reef.core.plugins import NoSettings, Plugin, PluginConfigError, PluginSettings
from reef.core.records_types import RequestType
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import Record2Task


@dataclass(frozen=True)
class BatchSettings(PluginSettings):
    """The config of a task-to-batch stage that batches ``batch_size`` batch units."""

    batch_size: int = config_option(1, help="The batch units in one batch.")

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class Task2Batch(ABC, Plugin):
    """The task-to-batch stage: shape the tasks of the selected batch units into one batch.

    A recipe implements ``make_batch``. The class attributes declare what the
    pipeline and the trainer need to know about the recipe's batches.
    ``make_batch`` runs on the trainer thread without network or model calls.
    A deployment names a stage class in ``pipeline.task2batch`` and gives its
    config. The class parses the config into its settings dataclass and
    builds the stage in ``from_config``.
    """

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: The training modes that the recipe's batches support. ``manual`` and
    #: ``hybrid`` batch on a training instruction.
    supported_training_modes: frozenset[str] = frozenset({"auto"})
    #: The kinds of record that the trainer gives the pipeline.
    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    #: The kinds of uploaded data item that the recipe trains on. Admission refuses an
    #: uploaded data item whose kind no trainer of the scenario accepts.
    accepted_items: frozenset[type[TrajectoryItem] | type[TaskItem]] = frozenset()

    def __init__(self, context: PipelineContext, batch_size: int) -> None:
        self.context = context
        self.scenario = context.scenario
        #: The number of batch units in one batch.
        self.batch_size = batch_size

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the stage of one scenario from its config.

        ``record2record`` and ``record2task`` are the scenario's earlier stages, built first.
        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    @abstractmethod
    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        """Build a batch from the tasks of the selected record sets, in batch order.

        Use ``batch_id`` as the ID of the returned batch. The pipeline assigns
        it. An automatic batch gets ``<scenario>:batch:<number>``. An
        instruction's batch gets ``<scenario>:instruction:<request id>``. The
        number counts the batches of this pipeline and starts again when the
        pipeline restarts. It is not a training step.
        The pipeline tracks consumption by record set, including the tasks
        that the recipe leaves out of the batch. In ``manual`` mode an
        instruction's batch has no tasks.
        """


class NoBatches(Task2Batch):
    """Make no batch: the no-update default. A pipeline with it never becomes ready."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, 1)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        raise RuntimeError(f"scenario {self.scenario!r} trains no weights and never produces a training batch")
