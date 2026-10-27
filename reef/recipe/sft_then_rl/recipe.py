"""The SFT-then-RL recipe: SFT on reports and uploaded trajectories, then RL on the scenario's imported tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.sft.processor import SFTRecord2Record, SFTTasks
from reef.recipe.sft.recipe import SFTRecipe
from reef.recipe.sft_then_rl.processor import ImportedTaskEvaluation, SFTThenRLJobData, SFTThenRLJobPlan
from reef.runtime.interfaces import TrainingMethod
from reef.storage.tasks import TaskStorage
from reef.train.jobs import ArtifactEvaluation, JobPlan, JobRequest
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SFTThenRLRecipe(SFTRecipe):
    """SFT jobs on held samples, and RL task jobs on the scenario's imported train tasks, from the SFT weights.

    The recipe trains what the scenario holds, in this order:

    * SFT jobs, as ``SFTRecipe`` trains them: the samples of ``TeacherContextReport`` reports and of uploaded
      ATIF trajectories (``POST /reef/trajectories``), with the objective ``SftObjective`` and
      ``recipe.config.job_parameters``.
    * RL task jobs on the train tasks of the scenario's ``imported-tasks`` store
      (``POST /reef/scenarios/{scenario}/tasks``, or the deployment's ``task-import`` section), up to
      ``pipeline.task2job.config.tasks_per_job`` tasks in one job, with the objective ``task_group`` (group
      relative advantages per task) and ``recipe.config.task_job_parameters``. The training component rolls
      the tasks out on its own sampler, as ``task_job_parameters.rollout`` says.

    Held samples train first. Each job starts from the last checkpoint of the job before it, so RL starts from
    the SFT weights. Each train task trains in one task job; eval tasks never train. Every artifact that the
    trainer publishes is evaluated on the eval tasks (``ImportedTaskEvaluation``), with the RL task job
    parameters, by an evaluation job from the artifact's checkpoint. The evaluation decides nothing: the
    ``evaluation`` section decides which artifact serves. The training component must
    run task jobs (``TrainingComponent.check_task_jobs``), such as Tinker with a sampler.
    """

    name: str = "sft-then-rl"
    task_job_parameters: Mapping[str, Any] | None = config_field(
        None,
        help=(
            "The parameters of every RL task job, beside method: rollout (required) and the component's own "
            "parameters, such as learning_rate."
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.task_job_parameters is None:
            raise RecipeConfigError("the SFT-then-RL recipe needs recipe.config.task_job_parameters with rollout")
        if "method" in self.task_job_parameters:
            raise RecipeConfigError("task_job_parameters cannot set method: RL task jobs train task_group")
        try:
            self.training_component.check_task_jobs(self.task_job_parameters)
        except ValueError as exc:
            raise RecipeConfigError(f"the training component cannot run the recipe's RL task jobs: {exc}") from exc

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=SFTRecord2Record, record2task=SFTTasks, task2job=SFTThenRLJobData)

    def artifact_evaluation(self, task_storage: TaskStorage | None) -> ArtifactEvaluation | None:
        rl = {**(self.task_job_parameters or {}), "method": TrainingMethod(TaskGroupObjective.name).to_dict()}
        return ImportedTaskEvaluation.for_storage(task_storage, rl)

    def job_plan(self) -> JobPlan:
        sft = {**(self.job_parameters or {}), "method": TrainingMethod(type(self).training_spec().objective).to_dict()}
        rl = {**(self.task_job_parameters or {}), "method": TrainingMethod(TaskGroupObjective.name).to_dict()}
        return SFTThenRLJobPlan(JobRequest(sft), JobRequest(rl))
