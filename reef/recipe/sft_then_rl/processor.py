"""The job data of the SFT-then-RL recipe: SFT jobs on held samples, RL task jobs on imported tasks.

``SFTThenRLJobData`` holds two kinds of job data:

1. Held samples: the SFT samples of reports and of uploaded ATIF trajectories, as ``SFTJobData`` holds them.
2. Its own task jobs: the train tasks of the scenario's sample store that have not trained, up to
   ``tasks_per_job`` of them in one job. The tasks arrive by ``POST /reef/scenarios/{scenario}/tasks`` or by
   the deployment's ``task-import`` section. The training component rolls them out on its own sampler and
   trains on their rollouts.

The training pipeline gives held samples priority, so a scenario trains the trajectories it holds before it
trains tasks. The recipe gives a job of trajectories the SFT parameters and a task job the RL parameters
(``SFTThenRLRecipe.parameters_by_kind``). Each job starts from the last checkpoint of the job before it, so an RL job starts from the weights
that SFT trained. A task trains in one task job of any trainer of the scenario: the commit that submits the job
lists ``sample:<task name>`` for each task, and the sample store counts it in the transaction of that commit.
Eval tasks never train.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.job_data import JobOutcome, Sample, TaskSample
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks.rollout import HandedOutTasks, PooledTask, TaskPool
from reef.recipe.sft.processor import SFTJobData
from reef.record2record import Record2Record
from reef.record2sample import Record2Sample
from reef.sample2job import Sample2JobSettings
from reef.storage.samples import SampleStore, consumed_sample_id

#: The imported train tasks of one task job when ``pipeline.sample2job.config.tasks_per_job`` is not set.
DEFAULT_TASKS_PER_JOB = 8


@dataclass(frozen=True)
class SFTThenRLJobSettings(Sample2JobSettings):
    """The config of ``SFTThenRLJobData``: the sample buffer of ``Sample2JobSettings``, and ``tasks_per_job``.

    Values: ``tasks_per_job`` is a positive int, the most imported train tasks of one task job. Purpose:
    ``SFTThenRLJobData.from_config`` parses it from ``pipeline.sample2job.config``.
    """

    tasks_per_job: int = config_option(DEFAULT_TASKS_PER_JOB, help="The most imported train tasks of one task job.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.tasks_per_job, bool) or not isinstance(self.tasks_per_job, int) or self.tasks_per_job <= 0:
            raise ValueError("tasks_per_job must be a positive int")


class SFTThenRLJobData(SFTJobData):
    """``SFTJobData`` that also hands the scenario's imported train tasks to the training component as task jobs.

    Values: ``tasks_per_job`` is a positive int. ``samples`` is the scenario's sample store, or None for a
    trainer without a scenario session, which makes no task job. ``task_pool`` holds the train tasks that wait
    for a task job and the tasks of the task jobs submitted in this process that did not end; ``handed_out`` is
    the tasks of the next task job, from hand-out until the trainer commits its submission.

    Operations: ``advance`` with nothing handed out adds the store's train tasks that no committed job of any
    trainer trained (no train count) and that the pool does not hold, then hands out the oldest
    ``tasks_per_job`` of them. ``own_job_data`` is the handed-out tasks. ``own_job_data_submitted`` returns
    ``sample:<task name>`` for each, so the commit that submits the job counts them. ``job_ended`` drops the
    ended job's tasks from the pool. A restart rebuilds the pool from the store, so a task whose submission did
    not commit trains again.

    Purpose: the SFT-then-RL recipe's sample-to-job stage (``pipeline.sample2job``).
    """

    accepts_imported_tasks = True

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
        return cls(context, SFTThenRLJobSettings.parse(config, plugin_class=cls))

    def __init__(self, context: PipelineContext, settings: SFTThenRLJobSettings) -> None:
        super().__init__(context, settings.buffer_size, settings.job_samples)
        self.tasks_per_job = settings.tasks_per_job
        self.samples: SampleStore | None = context.samples
        self.task_pool = TaskPool()
        self.handed_out: HandedOutTasks | None = None
        #: The tasks of the task jobs submitted in this process that did not end.
        self.submitted_tasks: tuple[PooledTask, ...] = ()

    def advance(self) -> None:
        """Hand out the next task job's tasks when none are handed out and untrained train tasks wait."""
        super().advance()
        if self.handed_out is not None or self.samples is None:
            return
        pooled = {*self.task_pool.waiting, *self.task_pool.taken}
        for stored in self.samples.tasks(split="train"):
            task = stored.sample
            if isinstance(task, TaskSample) and not stored.train_counts and stored.sample_id not in pooled:
                self.task_pool.add(stored.sample_id, task)
        taken = self.task_pool.take(self.tasks_per_job)
        if not taken:
            return
        batch = f"{self.scenario}:rollout:{uuid.uuid4().hex[:12]}"
        self.handed_out = HandedOutTasks(batch, tuple(f"{batch}/{task.task_id}" for task in taken), taken)

    def own_job_data(self) -> tuple[Sample, ...] | None:
        if self.handed_out is None:
            return None
        return tuple(task.item for task in self.handed_out.tasks)

    def own_job_data_submitted(self) -> frozenset[str]:
        handed_out = self.handed_out
        if handed_out is None:
            raise RuntimeError("no imported tasks were handed out")
        self.submitted_tasks = (*self.submitted_tasks, *handed_out.tasks)
        self.handed_out = None
        return frozenset(consumed_sample_id(str(task.task_id)) for task in handed_out.tasks)

    def job_ended(self, outcome: JobOutcome) -> None:
        """The ended job's imported tasks leave the pool; they trained, and their rewards are in its metrics."""
        if self.submitted_tasks:
            self.task_pool.drop(self.submitted_tasks)
            self.submitted_tasks = ()

    def status(self) -> Mapping[str, Any]:
        handed_out = self.handed_out
        return {
            **super().status(),
            "imported_task_pool": len(self.task_pool),
            "handed_out_tasks": 0 if handed_out is None else len(handed_out.tasks),
        }
