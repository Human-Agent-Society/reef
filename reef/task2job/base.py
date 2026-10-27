"""The task-to-job contract: a buffer of samples, and the samples of the selected record set units become one job's data.

The tasks of a record set enter the stage as samples when the record-to-task
stage makes them. The buffer holds the samples of the record sets that wait
for a job, at most ``buffer_size``. When the buffer holds at least
``job_samples`` samples, the trainer asks the training pipeline
(:class:`~reef.train.pipeline.TrainingPipeline`) for the next job's data. The
pipeline selects whole record set units from the record-to-record stage and
gives their samples to a recipe's ``Task2Job``. A task is a ``TrainDataItem``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.job_data import JobData, JobOutcome, TrainDataItem
from reef.core.plugins import NoSettings, Plugin, PluginConfigError, PluginSettings
from reef.core.records_types import RequestType
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import Record2Task

#: The jobs whose data the sample buffer holds when ``pipeline.task2job.config.buffer_size`` is not set.
DEFAULT_BUFFER_JOBS = 4


def require_optional_positive(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is None or a positive int (not a bool)."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} must be None or a positive int, got {value!r}")


@dataclass(frozen=True)
class Task2JobSettings(PluginSettings):
    """The config of a task-to-job stage: the size of its sample buffer and the fewest samples of one job.

    Values: ``buffer_size`` and ``job_samples`` are each None or a positive
    int. The constructor refuses other values.
    Purpose: a stage's ``from_config`` parses it from
    ``pipeline.task2job.config`` and gives both fields to
    ``Task2Job.__init__``, where None means the default. A stage settings
    class that has more keys extends it.
    """

    buffer_size: int | None = config_option(
        None,
        help="The most samples the buffer holds; at least job_samples. Omitted holds the data of four jobs.",
    )
    job_samples: int | None = config_option(
        None,
        help=(
            "The fewest samples one job trains. Required for weight training; harness evolution "
            "defaults it to evolution.batch_size."
        ),
    )

    def __post_init__(self) -> None:
        require_optional_positive(self.buffer_size, "buffer_size")
        require_optional_positive(self.job_samples, "job_samples")


class Task2Job(ABC, Plugin):
    """The task-to-job stage: bound the sample buffer, and shape the samples of the selected record set units into one job's data.

    Values: for a stage that ``trains_own_job_data_only``, ``job_samples`` and
    ``buffer_size`` are None: its jobs are its own samples, which wait in the
    stage, not in a buffer. For any other stage, ``job_samples`` is None or a
    positive int: the configured value, or else ``context.default_job_samples``;
    it is None only for a trainer that trains nothing. ``buffer_size`` is None
    or a positive int: the configured size, or else ``DEFAULT_BUFFER_JOBS *
    job_samples``; None holds every sample. ``max_job_samples`` is None or a
    positive int, the most samples one job's data may hold; None sets no limit.
    ``buffer_size`` is at least ``job_samples``, and ``job_samples`` at most
    ``max_job_samples``, when both are set. The constructor refuses other
    values with ``PluginConfigError``.
    Purpose: the pipeline is ready when its held samples are at least
    ``job_samples``, and it keeps at most ``buffer_size`` samples of held
    record sets. The pipeline takes whole record set units, so one job's data
    can hold more than ``job_samples``.
    A recipe implements ``make_job_data``. It runs on the trainer thread
    without network or model calls. The class attributes declare what the
    pipeline and the trainer need to know about the recipe's job data.
    A deployment names a stage class in ``pipeline.task2job`` and gives its
    config. The class parses the config into its settings dataclass and
    builds the stage in ``from_config``.
    """

    #: The job data type ``make_job_data`` returns; the trainer validates it.
    output_schema: type[JobData] = JobData
    #: The training modes that the recipe's job data supports. ``manual`` and
    #: ``hybrid`` make job data for a training instruction.
    supported_training_modes: frozenset[str] = frozenset({"auto"})
    #: The kinds of record that the trainer gives the pipeline.
    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    #: Whether the recipe trains on uploaded trajectories. Admission refuses an upload
    #: when no trainer of the scenario accepts uploaded trajectories.
    accepts_trajectories: bool = False
    #: Whether the recipe reads the scenario's imported tasks (the ``imported-tasks`` store). Admission refuses a
    #: task upload (``POST /reef/scenarios/{scenario}/tasks``) when no trainer of the scenario reads them.
    accepts_imported_tasks: bool = False
    #: Whether every job of the stage trains only the samples it makes itself (``own_job_data``), so it has no
    #: buffer and no ``job_samples``.
    trains_own_job_data_only: bool = False

    def __init__(
        self,
        context: PipelineContext,
        buffer_size: int | None,
        job_samples: int | None,
        *,
        max_job_samples: int | None = None,
    ) -> None:
        require_optional_positive(buffer_size, "buffer_size")
        require_optional_positive(job_samples, "job_samples")
        require_optional_positive(max_job_samples, "max_job_samples")
        self.context = context
        self.scenario = context.scenario
        stage = type(self).__name__
        #: The fewest samples one job trains; None for a stage that trains only its own job data, and for a
        #: trainer that trains nothing.
        self.job_samples: int | None
        #: The most samples that the buffer holds; None holds every sample.
        self.buffer_size: int | None
        if self.trains_own_job_data_only:
            if buffer_size is not None or job_samples is not None:
                raise PluginConfigError(
                    f"{stage} trains only its own job data: it takes no buffer_size or job_samples"
                )
            self.job_samples = None
            self.buffer_size = None
        else:
            self.job_samples = context.default_job_samples if job_samples is None else job_samples
            if buffer_size is not None:
                self.buffer_size = buffer_size
            elif self.job_samples is not None:
                self.buffer_size = DEFAULT_BUFFER_JOBS * self.job_samples
            else:
                self.buffer_size = None
        #: The most samples one job's data may hold; None sets no limit. A stage whose data is one fixed unit sets it.
        self.max_job_samples = max_job_samples
        if self.job_samples is not None and self.buffer_size is not None and self.buffer_size < self.job_samples:
            raise PluginConfigError(
                f"{stage} buffer_size {self.buffer_size} is smaller than job_samples {self.job_samples}"
            )
        if self.job_samples is not None and max_job_samples is not None and self.job_samples > max_job_samples:
            raise PluginConfigError(
                f"{stage} holds at most {max_job_samples} samples in one job's data, not job_samples {self.job_samples}"
            )

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
    def make_job_data(self, tasks: tuple[TrainDataItem, ...], data_id: str) -> JobData:
        """Build one job's data from the tasks of the selected record sets, in selection order.

        Use ``data_id`` as the ID of the returned value. The pipeline assigns
        it. Automatic job data gets ``<scenario>:data:<number>``. An
        instruction's data gets ``<scenario>:instruction:<request id>``. The
        number counts the job data of this pipeline and starts again when the
        pipeline restarts. It is not a training step.
        The pipeline tracks consumption by record set, including the tasks
        that the recipe leaves out of the job data. In ``manual`` mode an
        instruction's data has no tasks.
        """

    # A stage may also make samples itself, from no record, and train them as one job of its own: the
    # pipeline asks ``own_job_data`` before it selects record sets. The defaults make no such samples.

    def advance(self) -> None:
        """Advance the stage's own work on the trainer thread, such as starting the next rollout. The default does nothing."""
        return

    def own_job_data(self) -> tuple[TrainDataItem, ...] | None:
        """Return samples that the stage made itself and that train now as one job, or None.

        While the value is not None, the pipeline is ready and its next job data holds exactly these samples. It
        stays the same until ``own_job_data_submitted``.
        """
        return None

    def own_job_data_submitted(self) -> frozenset[str]:
        """The trainer prepared the commit that submits the stage's own job data as a job.

        Return the IDs that the commit lists for it; the pipeline gives them back to ``release_own_ids`` once the
        commit is durable. The job trains after this; its end arrives through ``job_ended``.
        """
        return frozenset()

    def release_own_ids(self, ids: frozenset[str]) -> None:
        """The commit that listed these IDs of ``own_job_data_submitted`` is durable. The default does nothing."""
        return

    def job_ended(self, outcome: JobOutcome) -> None:
        """A training job ended and the commit of its end is durable; the default does nothing.

        A stage that hands out tasks reads their results from ``outcome``. A trainer that starts with no running
        job calls it again for the last job that ended, before any other call of a new stage, so a stage that
        starts in that process takes it as the end of the last job it may have handed out.
        """
        return

    def background_jobs_running(self) -> bool:
        """Whether the stage's own work runs, so that the trainer polls instead of waiting for a record."""
        return False

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def close(self) -> None:
        """Stop the stage's own work. Calling it more than one time is safe."""
        return


class NoJobData(Task2Job):
    """Make no job data: the no-update default. A pipeline with it never becomes ready."""

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
        super().__init__(context, None, None)

    def make_job_data(self, tasks: tuple[TrainDataItem, ...], data_id: str) -> JobData:
        raise RuntimeError(f"scenario {self.scenario!r} trains no weights and never produces job data")
