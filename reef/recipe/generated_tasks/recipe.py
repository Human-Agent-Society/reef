"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks.processor import GeneratedTaskJobData, GeneratedTaskSamples
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class GeneratedTaskRecipe(WeightTrainingRecipe):
    """Train on generated tasks, with the objective that the deployment provides.

    The recipe binds the report contract and names no objective. Its stages are
    ``GeneratedTaskSamples`` and ``GeneratedTaskJobData``; the deployment's
    ``pipeline`` section gives their config: the rounds, their pace, the source
    sessions, the task generator, and the generator service in
    ``record2sample``, and the tasks of one job in ``sample2job``. Without
    ``rounds``, no round runs, and the recipe trains only the train tasks that
    earlier generations saved. Each job is a task job: the
    training component rolls its tasks out, as ``recipe.config.job_parameters``
    says in the backend's own shape: ``objective``, a dotted reference to a method
    that the deployment provides, and the rollouts (``rollout`` on Tinker, ``harbor``
    on Slime). The recipe requires both.

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
            self.training_component.check_task_jobs(self.parameters_by_kind()["task"])
        except ValueError as exc:
            raise RecipeConfigError(
                "the generated-task recipe trains task jobs, whose recipe.config.job_parameters name the "
                f"objective, the rollouts of each task and the agent that runs them: {exc}"
            ) from exc

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=GeneratedTaskSamples, sample2job=GeneratedTaskJobData)
