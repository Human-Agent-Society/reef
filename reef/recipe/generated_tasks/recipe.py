"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskJobData,
    GeneratedTaskRecord2Record,
    GeneratedTaskSamples,
)
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class GeneratedTaskRecipe(WeightTrainingRecipe):
    """Group relative advantages per generated task, on Tinker's importance sampling loss.

    The recipe binds the objective and the report contract. Its stages are
    ``GeneratedTaskRecord2Record``, ``GeneratedTaskSamples``, and
    ``GeneratedTaskJobData``; the deployment's ``pipeline`` section gives
    their config: the rounds, the task generator, and the generator service
    in ``record2sample``, the source sessions and the pace of the rounds in
    ``record2record``, and the tasks of one job in ``sample2job``. Without that
    section, no round runs and nothing trains. Each job is a task job: the
    training component rolls its tasks out, as ``recipe.config.job_parameters.rollout``
    says (the Tinker component's ``RolloutSettings``), which the recipe
    requires.

    Enabled rounds save each task and its train/eval split in the scenario's task store before their rollouts run.
    Eval tasks stay in the store without rollouts or training feedback.
    """

    name: str = "generated-tasks"

    @property
    def report_type(self) -> type[ReportBase]:
        # A scored report: what a source session ends with. The recipe trains on no report.
        return ScoredRolloutReport

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            self.training_component.check_task_jobs(self.job_parameters or {})
        except ValueError as exc:
            raise RecipeConfigError(
                "the generated-task recipe trains task jobs, whose recipe.config.job_parameters.rollout names the "
                f"rollouts of each task and the agent that runs them: {exc}"
            ) from exc

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective=TaskGroupObjective.name)

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=GeneratedTaskRecord2Record,
            record2sample=GeneratedTaskSamples,
            sample2job=GeneratedTaskJobData,
        )
