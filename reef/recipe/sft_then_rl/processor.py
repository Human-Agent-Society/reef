"""The job data and the job plan of the SFT-then-RL recipe: SFT jobs on held samples, RL task jobs on imported tasks.

``SFTThenRLJobData`` holds two kinds of job data:

1. Held samples: the SFT samples of reports and of uploaded ATIF trajectories, as ``SFTJobData`` holds them.
2. Its own task jobs: the train tasks of the scenario's ``imported-tasks`` store that have not trained, up to
   ``tasks_per_job`` of them in one job. The tasks arrive by ``POST /reef/scenarios/{scenario}/tasks`` or by
   the deployment's ``task-import`` section. The training component rolls them out on its own sampler and
   trains on their rollouts.

``ImportedTaskEvaluation`` evaluates every artifact that the trainer publishes on the store's eval tasks: an
evaluation job from the artifact's checkpoint, which rolls the tasks out and trains nothing.

The training pipeline gives held samples priority, so a scenario trains the trajectories it holds before it
trains tasks. ``SFTThenRLJobPlan`` gives a job of trajectories the SFT parameters and a task job the RL
parameters. Each job starts from the last checkpoint of the job before it, so an RL job starts from the weights
that SFT trained. A task trains in one task job: the commit that submits the job lists ``rollout:<task use>``
for each task, and the store marks the tasks trained once that commit is durable. Eval tasks never train.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.evaluation import CheckpointCandidate
from reef.core.job_data import JobData, JobOutcome, Sample, TaskSample
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks.rollout import (
    HandedOutTasks,
    PooledTask,
    SubmittedTaskJob,
    TaskPool,
    consumed_rollout_id,
)
from reef.recipe.sft.processor import SFTJobData
from reef.record2record import Record2Record
from reef.record2sample import Record2Sample
from reef.sample2job import Sample2JobSettings
from reef.storage.tasks import IMPORTED_TASK_STORE, StoredTask, TaskStorage, TaskStore
from reef.train.jobs import ArtifactEvaluation, EvaluationJob, JobPlan, JobPlanContext, JobRequest

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


def imported_task_item(store: TaskStore, stored: StoredTask) -> TaskSample:
    """The task that a task job rolls out for the imported task ``stored``: its directory, named by its task name."""
    return TaskSample(
        store.path(stored.files_digest),
        metadata={"task": {"name": stored.name, "files_digest": stored.files_digest, "split": stored.split}},
    )


class SFTThenRLJobData(SFTJobData):
    """``SFTJobData`` that also hands the scenario's imported train tasks to the training component as task jobs.

    Values: ``tasks_per_job`` is a positive int. ``task_store`` is the scenario's ``imported-tasks`` store, or
    None for a trainer without task storage, which makes no task job. ``task_pool`` holds the train tasks that
    wait for a task job and the tasks of the task jobs submitted in this process that did not end;
    ``handed_out`` is the tasks of the next task job, from hand-out until the trainer commits its submission.

    Operations: ``advance`` with nothing handed out adds the store's untrained train tasks that the pool does not
    hold, then hands out the oldest ``tasks_per_job`` of them. ``own_job_data`` is the handed-out tasks.
    ``own_job_data_submitted`` returns ``rollout:<task use>`` for each, and ``release_own_ids`` marks them
    trained in the store once one of those IDs is durable. ``job_ended`` drops the ended job's tasks from the
    pool. A restart rebuilds the pool from the store, so a task whose submission did not commit trains again.

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
        self.task_store = None if context.task_storage is None else context.task_storage.open(IMPORTED_TASK_STORE)
        self.task_pool = TaskPool()
        self.handed_out: HandedOutTasks | None = None
        #: The submitted task jobs whose submission's commit is not yet durable.
        self.trained_rollouts: list[SubmittedTaskJob] = []
        #: The tasks of the task jobs submitted in this process that did not end.
        self.submitted_tasks: tuple[PooledTask, ...] = ()

    def advance(self) -> None:
        """Hand out the next task job's tasks when none are handed out and untrained train tasks wait."""
        super().advance()
        if self.handed_out is not None or self.task_store is None:
            return
        pooled = {*self.task_pool.waiting, *self.task_pool.taken}
        trained = self.task_store.trained_task_names()
        for stored in self.task_store.tasks(split="train"):
            if stored.name not in trained and stored.name not in pooled:
                self.task_pool.add(stored.name, imported_task_item(self.task_store, stored))
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
        ids = frozenset(consumed_rollout_id(task_use) for task_use in handed_out.task_uses)
        self.trained_rollouts.append(SubmittedTaskJob(tuple(str(task.task_id) for task in handed_out.tasks), ids))
        self.submitted_tasks = (*self.submitted_tasks, *handed_out.tasks)
        self.handed_out = None
        return ids

    def release_own_ids(self, ids: frozenset[str]) -> None:
        durable = [job for job in self.trained_rollouts if job.consumed_ids & ids]
        if self.task_store is not None:
            for job in durable:
                self.task_store.mark_trained(task_names=job.task_names)
        self.trained_rollouts = [job for job in self.trained_rollouts if job not in durable]

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


class SFTThenRLJobPlan(JobPlan):
    """Give a job of trajectories the ``sft`` request and a task job the ``rl`` request.

    Values: ``sft`` and ``rl`` are ``JobRequest`` values. Operation: ``next_job(context)`` is ``rl`` when
    ``context.data`` holds one or more items and every item is a ``TaskSample``, and ``sft`` when no item is a
    ``TaskSample``; data that holds both kinds raises ``ValueError``. Purpose: ``SFTThenRLRecipe.job_plan`` makes
    it, and the weight trainer's job loop asks it for each job.
    """

    def __init__(self, sft: JobRequest, rl: JobRequest) -> None:
        if not isinstance(sft, JobRequest) or not isinstance(rl, JobRequest):
            raise TypeError("SFTThenRLJobPlan takes two JobRequest values")
        self.sft = sft
        self.rl = rl

    def next_job(self, context: JobPlanContext) -> JobRequest:
        tasks = sum(isinstance(item, TaskSample) for item in context.data.samples)
        if tasks == 0:
            return self.sft
        if tasks == len(context.data.samples):
            return self.rl
        raise ValueError(f"job data {context.data.data_id!r} holds both trajectories and tasks")

    def experiment_config(self) -> Mapping[str, Any]:
        return {
            "job_plan": "sft-then-rl",
            "sft_parameters": dict(self.sft.parameters),
            "rl_parameters": dict(self.rl.parameters),
        }


class ImportedTaskEvaluation(ArtifactEvaluation):
    """Evaluate every artifact on the eval tasks of the scenario's ``imported-tasks`` store.

    Values: ``task_store`` is that store, and ``parameters`` the RL task job parameters (with ``rollout``),
    a mapping. Operation: ``evaluation_job(candidate)`` is ``None`` while the store holds no eval task, and else
    the job of every eval task, in publication order, with ``parameters`` and ``evaluation_of`` set to the
    candidate's ``candidate_id``, which the training component scores without training. Purpose:
    ``SFTThenRLRecipe.artifact_evaluation`` makes it, and the weight trainer's job loop asks it at each decision.
    """

    def __init__(self, task_store: TaskStore, parameters: Mapping[str, Any]) -> None:
        self.task_store = task_store
        self.parameters = dict(parameters)

    @classmethod
    def for_storage(
        cls, task_storage: TaskStorage | None, parameters: Mapping[str, Any]
    ) -> ImportedTaskEvaluation | None:
        """The evaluation over ``task_storage``'s imported tasks, or ``None`` without task storage."""
        return None if task_storage is None else cls(task_storage.open(IMPORTED_TASK_STORE), parameters)

    def evaluation_job(self, candidate: CheckpointCandidate) -> EvaluationJob | None:
        held_out = self.task_store.tasks(split="eval")
        if not held_out:
            return None
        items = tuple(imported_task_item(self.task_store, stored) for stored in held_out)
        return EvaluationJob(
            JobData(f"{candidate.candidate_id}:evaluation", items),
            {**self.parameters, "evaluation_of": candidate.candidate_id},
        )
