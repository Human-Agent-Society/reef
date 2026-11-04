"""The sample-to-job contract: the size of a job, and the samples of the next job become its data.

The record-to-sample stage makes the samples of each record set, and the
pipeline saves the samples of each ready unit as data items that the scenario uses.
In ``auto``, the next job takes the untrained samples: the first
``buffer_size`` of them in store order, and the rest of each group that those
touch. When at least ``job_samples`` samples wait, the trainer asks the
training pipeline (:class:`~reef.train.pipeline.TrainingPipeline`) for the
next job's data. In ``manual``, a training request names the samples of
its job. The pipeline gives the job's samples to a recipe's ``Sample2Job``. A
sample is a ``Sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.job_data import JobData, Sample, TaskSample
from reef.core.plugins import NoSettings, Plugin, PluginConfigError, PluginSettings
from reef.core.records_types import RequestType
from reef.core.values import deep_merge
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
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
            "The most samples that one automatic job takes, beside the rest of a group; at least "
            "job_samples. Omitted takes job_samples."
        ),
    )
    job_samples: int | None = config_option(
        None,
        help=(
            "The fewest samples one job trains. Required for weight training; a harness evolution "
            "stage defaults it to 1."
        ),
    )

    def __post_init__(self) -> None:
        require_optional_positive(self.buffer_size, "buffer_size")
        require_optional_positive(self.job_samples, "job_samples")


class Sample2Job(ABC, Plugin):
    """The sample-to-job stage: the size of each job, and the job's data and parameters from its samples.

    Values: ``job_samples`` is None or a positive int: the configured value,
    or else the stage's own default. ``Trainer.build`` refuses None for a
    trainer that trains. ``buffer_size`` is None
    or a positive int: the configured size, or else ``job_samples``; None
    takes every untrained sample. ``max_job_samples`` is None or a
    positive int, the most samples one job's data may hold; None sets no limit.
    ``buffer_size`` is at least ``job_samples``, and ``job_samples`` at most
    ``max_job_samples``, and ``buffer_size`` at most ``max_job_samples``,
    when both are set. The constructor refuses other
    values with ``PluginConfigError``.
    Purpose: in ``auto``, the pipeline is ready when the next job holds at
    least ``job_samples`` samples. The next job takes the first ``buffer_size``
    untrained samples and the rest of each group that those touch, so one job's
    data can hold more than ``buffer_size``.
    A recipe implements ``make_job_data``. It runs on the trainer thread
    without network or model calls. The class attributes declare what the
    pipeline and the trainer need to know about the recipe's job data.
    A deployment names a stage class in ``pipeline.sample2job`` and gives its
    config. The class parses the config into its settings dataclass and
    builds the stage in ``from_config``.
    """

    #: The training modes that the recipe's job data supports. ``auto`` makes job data when ``job_samples``
    #: samples wait; ``manual`` makes it of the samples that a queued training request names.
    supported_training_modes: frozenset[str] = frozenset({"auto", "manual"})
    #: Whether the job reads its training request, as a harness proposer does: its job parameters carry the request.
    #: In ``manual`` such a stage runs a request that names no sample too. Another stage needs a request that names
    #: samples.
    reads_requests: bool = False
    #: The kinds of record that the trainer gives the pipeline.
    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    #: Whether the recipe trains on uploaded trajectories. Admission refuses an upload
    #: when no trainer of the scenario accepts uploaded trajectories.
    accepts_trajectories: bool = False
    #: Whether the recipe trains the scenario's imported ``train`` tasks, as it trains uploaded trajectories. Admission refuses an import to the ``train`` split
    #: (``POST /reef/scenarios/{scenario}/tasks``, ``task-import``) when no trainer of the scenario trains them.
    accepts_imported_tasks: bool = False

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
        #: The fewest samples one job trains; None when the stage sets no job size (a trainer that trains nothing).
        self.job_samples = job_samples
        #: The most samples that one automatic job takes; None takes every sample.
        self.buffer_size = self.job_samples if buffer_size is None else buffer_size
        #: The most samples one job's data may hold; None sets no limit. A stage whose data is one fixed grid sets it,
        #: and a request names no more (``TrainingPipeline.request_entries``).
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
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the stage of one scenario from its config.

        ``record2sample`` is the scenario's record-to-sample stage, built first.
        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    @abstractmethod
    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        """Build one job's data from the samples of the next job, in store order.

        Use ``data_id`` as the ID of the returned value. The pipeline assigns
        it. Automatic job data gets ``<scenario>:data:<number>``. A
        request's data gets ``<scenario>:request:<request id>``. The
        number counts the job data of this pipeline and starts again when the
        pipeline restarts. It is not a training step.
        The job's commit consumes every sample of ``samples``, including the
        samples that the recipe leaves out of the job data. In ``manual`` mode a
        request's data holds the samples that the request names, at most ``max_job_samples``: one or more,
        or none for a stage that ``reads_requests``.
        """

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        """The parameters of the training job that trains ``data``: ``context.job_parameters`` of its samples' kind,
        with its request's ``parameters`` merged in (``deep_merge``): nested objects merge key by key, and every
        other value of the request wins.

        Together with ``make_job_data``, this is the stage's function from the selected samples to the job: the
        data, and the parameters that the job loop submits with it. The result depends only on ``data``, which
        holds the request, since a resubmission after a restart must name the same job. The training
        component checks the result. Raise ``ValueError`` for data with both trajectories and tasks, and for a
        kind that the recipe gives no parameters when no request gives any. Data with no sample has no
        parameters: a harness step trains such data, and a harness step takes none.
        """
        kinds: set[SampleKind] = {
            "task" if isinstance(sample, TaskSample) else "trajectory" for sample in data.samples
        }
        if len(kinds) > 1:
            raise ValueError(f"job data {data.data_id!r} holds both trajectories and tasks")
        if not kinds:
            return {}
        (kind,) = kinds
        defaults = self.context.job_parameters.get(kind)
        overrides = {} if data.request is None else data.request.parameters
        if defaults is None and not overrides:
            raise ValueError(f"the recipe gives no job parameters for a job of {kind} samples")
        return deep_merge(defaults or {}, overrides)

    def experiment_config(self) -> Mapping[str, Any]:
        """A non-secret description of the jobs that the stage makes, which experiment runs record."""
        return {"job_parameters": {kind: dict(parameters) for kind, parameters in self.context.job_parameters.items()}}

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def close(self) -> None:
        """Stop the stage's own work. Calling it more than one time is safe."""
        return


class NoJobData(Sample2Job):
    """Make no job data: the no-update default. A pipeline with it never becomes ready."""

    #: A pipeline that trains nothing takes no request.
    supported_training_modes = frozenset({"auto"})

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, None, None)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        raise RuntimeError(f"scenario {self.scenario!r} trains no weights and never produces job data")
