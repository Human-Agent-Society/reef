"""Job data of harness evolution: trace samples as they come, and offline runs of the imported train tasks.

``TraceJobData`` puts the trace samples of the next job into one job's data.
``TrainTaskJobData`` also trains the scenario's imported train tasks: they
are samples of the next job, and the harness backend rolls a job's tasks out on the
head tree as the step's material. That is offline harness evolution: a step
needs no traffic, so uploaded train tasks start a run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core import RequestType
from reef.core.config import config_option
from reef.core.job_data import JobData, TaskSample
from reef.core.training_request import request_parameter
from reef.pipeline_context import PipelineContext
from reef.record2sample import Record2Sample
from reef.sample2job.base import Sample2JobSettings
from reef.sample2job.distill import DistillJobData


@dataclass(frozen=True)
class TraceJobDataSettings(Sample2JobSettings):
    """The config of a harness stage: ``Sample2JobSettings``, and whether each job names the eval split."""

    evaluation_split: bool = config_option(
        False,
        help=(
            "Whether each job names the task paths of the scenario's eval split, which a harness step evaluates on. "
            "A harness recipe sets it from evolution.evaluation.imported_tasks."
        ),
    )


class TraceJobData(DistillJobData):
    """Put trace samples into job data unmodified; the backends consume them without more processing.

    Values: as ``DistillJobData``, except that ``job_samples`` defaults to 1,
    so each trace starts a step unless the deployment sets
    ``pipeline.sample2job.config.job_samples``.
    In ``manual`` a queued request takes the samples that it names,
    possibly none, so the method reads the request beside them.

    ``job_parameters(data)`` adds to the stage's parameters what a harness step reads beside its data:
    ``request``, the job's training request (``request_parameter``), or None; and, with ``evaluation_split``,
    ``evaluation_tasks``, the task paths of the eval split of the scenario's imported tasks in arrival order, which
    the step evaluates on instead of its configured tasks.
    """

    reads_requests = True
    required_request_types = frozenset(RequestType)

    def __init__(
        self,
        context: PipelineContext,
        buffer_size: int | None,
        job_samples: int | None,
        *,
        max_job_samples: int | None = None,
        evaluation_split: bool = False,
    ) -> None:
        super().__init__(
            context, buffer_size, 1 if job_samples is None else job_samples, max_job_samples=max_job_samples
        )
        self.evaluation_split = evaluation_split

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = TraceJobDataSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples, evaluation_split=settings.evaluation_split)

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        parameters = {
            **super().job_parameters(data),
            "request": None if data.request is None else request_parameter(data.request),
        }
        if self.evaluation_split:
            samples = self.context.samples
            stored = () if samples is None else samples.imported_tasks(split="eval")
            parameters["evaluation_tasks"] = [
                str(item.sample.task_path) for item in stored if isinstance(item.sample, TaskSample)
            ]
        return parameters


class TrainTaskJobData(TraceJobData):
    """``TraceJobData`` that also trains the scenario's imported train tasks: offline harness evolution.

    Values: as ``TraceJobData``, and the stage accepts imported tasks
    (``accepts_imported_tasks``). Each imported ``train`` task is a buffer
    sample, in upload order. One job's data holds samples of one kind, so a
    step either reads traces or rolls out train tasks; ``job_samples`` and
    ``buffer_size`` set how many tasks one step rolls out. The scenario's training
    state counts how many jobs held each task, and ``auto`` trains a task once.
    Purpose: a deployment names the class in ``pipeline.sample2job`` for an
    offline run. The harness backend rolls a job's tasks out on the head tree
    before the method runs, and the method reads the runs as traces.
    """

    accepts_imported_tasks = True


__all__ = ["TraceJobData", "TraceJobDataSettings", "TrainTaskJobData"]
