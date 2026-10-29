"""The sample-to-job contract: the size of a job, and the samples of the buffer become one job's data.

The record-to-sample stage makes the samples of each record set, and the
pipeline stores the samples of each ready unit in the scenario's sample store.
The buffer is the untrained samples that the next job takes: the first
``buffer_size`` of them in store order, and the rest of each group that those
touch. When the buffer holds at least ``job_samples`` samples, the trainer
asks the training pipeline (:class:`~reef.train.pipeline.TrainingPipeline`)
for the next job's data, and the pipeline gives the buffer's samples to a
recipe's ``Sample2Job``. A sample is a ``Sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.job_data import JobData, JobOutcome, Sample, TaskSample
from reef.core.plugins import NoSettings, Plugin, PluginConfigError, PluginSettings
from reef.core.records_types import RequestType
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2sample import Record2Sample
from reef.storage.samples import SampleKind


def require_optional_positive(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is None or a positive int (not a bool)."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} must be None or a positive int, got {value!r}")


@dataclass(frozen=True)
class Sample2JobSettings(PluginSettings):
    """The config of a sample-to-job stage: the most samples of one automatic job, and the fewest.

    Values: ``buffer_size`` and ``job_samples`` are each None or a positive
    int. The constructor refuses other values.
    Purpose: a stage's ``from_config`` parses it from
    ``pipeline.sample2job.config`` and gives both fields to
    ``Sample2Job.__init__``, where None means the default. A stage settings
    class that has more keys extends it.
    """

    buffer_size: int | None = config_option(
        None,
        help=(
            "The most samples that one job takes from the buffer, beside the rest of a group; at least "
            "job_samples. Omitted takes job_samples."
        ),
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


class Sample2Job(ABC, Plugin):
    """The sample-to-job stage: the size of each job, and the job's data and parameters from the buffer's samples.

    Values: for a stage that ``trains_own_job_data_only``, ``job_samples`` and
    ``buffer_size`` are None: its jobs are its own samples, which wait in the
    stage, not in a buffer. For any other stage, ``job_samples`` is None or a
    positive int: the configured value, or else ``context.default_job_samples``;
    it is None only for a trainer that trains nothing. ``buffer_size`` is None
    or a positive int: the configured size, or else ``job_samples``; None
    takes every untrained sample. ``max_job_samples`` is None or a
    positive int, the most samples one job's data may hold; None sets no limit.
    ``buffer_size`` is at least ``job_samples``, and ``job_samples`` at most
    ``max_job_samples``, and ``buffer_size`` at most ``max_job_samples``,
    when both are set. The constructor refuses other
    values with ``PluginConfigError``.
    Purpose: the pipeline is ready when its buffer holds at least
    ``job_samples`` samples. The buffer is the first ``buffer_size`` untrained
    samples and the rest of each group that those touch, so one job's data can
    hold more than ``buffer_size``.
    A recipe implements ``make_job_data``. It runs on the trainer thread
    without network or model calls. The class attributes declare what the
    pipeline and the trainer need to know about the recipe's job data.
    A deployment names a stage class in ``pipeline.sample2job`` and gives its
    config. The class parses the config into its settings dataclass and
    builds the stage in ``from_config``.
    """

    #: The job data type ``make_job_data`` returns; the trainer validates it.
    output_schema: type[JobData] = JobData
    #: The training modes that the recipe's job data supports. ``auto`` makes job data when
    #: ``job_samples`` samples are held; ``manual`` makes it for a queued training instruction.
    supported_training_modes: frozenset[str] = frozenset({"auto", "manual"})
    #: Whether the job reads the instruction in its data, as a harness proposer does. In ``manual`` such a stage
    #: runs an instruction with fewer than ``job_samples`` samples, or none. Another stage takes the instruction as
    #: a request to train and waits until ``job_samples`` samples are held.
    reads_instructions: bool = False
    #: The kinds of record that the trainer gives the pipeline.
    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    #: Whether the recipe trains on uploaded trajectories. Admission refuses an upload
    #: when no trainer of the scenario accepts uploaded trajectories.
    accepts_trajectories: bool = False
    #: Whether the recipe reads the scenario's imported tasks (its sample store's tasks). Admission refuses a
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
                self.buffer_size = self.job_samples
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
        if self.buffer_size is not None and max_job_samples is not None and self.buffer_size > max_job_samples:
            raise PluginConfigError(
                f"{stage} holds at most {max_job_samples} samples in one job's data, not buffer_size {self.buffer_size}"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the stage of one scenario from its config.

        ``record2record`` and ``record2sample`` are the scenario's earlier stages, built first.
        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    @abstractmethod
    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        """Build one job's data from the samples of the buffer, in store order.

        Use ``data_id`` as the ID of the returned value. The pipeline assigns
        it. Automatic job data gets ``<scenario>:data:<number>``. An
        instruction's data gets ``<scenario>:instruction:<request id>``. The
        number counts the job data of this pipeline and starts again when the
        pipeline restarts. It is not a training step.
        The job's commit consumes every sample of ``samples``, including the
        samples that the recipe leaves out of the job data. In ``manual`` mode an
        instruction's data holds the samples that automatic job data would take next: the buffer, which holds at
        least ``job_samples`` samples unless the stage ``reads_instructions``.
        """

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        """The parameters of the training job that trains ``data``: ``context.job_parameters`` of its samples' kind.

        Together with ``make_job_data``, this is the stage's function from the selected samples to the job: the
        data, and the parameters that the job loop submits with it. The result depends only on ``data``, since a
        resubmission after a restart must name the same job. Raise ``ValueError`` for data with both trajectories
        and tasks, and for a kind that the recipe gives no parameters. Data with no sample has no parameters: a
        harness step trains such data, and a harness step takes none.
        """
        kinds: set[SampleKind] = {
            "task" if isinstance(sample, TaskSample) else "trajectory" for sample in data.samples
        }
        if len(kinds) > 1:
            raise ValueError(f"job data {data.data_id!r} holds both trajectories and tasks")
        if not kinds:
            return {}
        (kind,) = kinds
        parameters = self.context.job_parameters.get(kind)
        if parameters is None:
            raise ValueError(f"the recipe gives no job parameters for a job of {kind} samples")
        return dict(parameters)

    def experiment_config(self) -> Mapping[str, Any]:
        """A non-secret description of the jobs that the stage makes, which experiment runs record."""
        return {"job_parameters": {kind: dict(parameters) for kind, parameters in self.context.job_parameters.items()}}

    # A stage may also make samples itself, from no record, and train them as one job of its own: the
    # pipeline asks ``own_job_data`` before it selects record sets. The defaults make no such samples.

    def advance(self) -> None:
        """Advance the stage's own work on the trainer thread, such as starting the next rollout. The default does nothing."""
        return

    def own_job_data(self) -> tuple[Sample, ...] | None:
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


class NoJobData(Sample2Job):
    """Make no job data: the no-update default. A pipeline with it never becomes ready."""

    #: A pipeline that trains nothing takes no instruction.
    supported_training_modes = frozenset({"auto"})

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, None, None)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        raise RuntimeError(f"scenario {self.scenario!r} trains no weights and never produces job data")
