"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import GeneratedTaskBatches, GeneratedTaskRecord2Record, GeneratedTaskTasks
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class GeneratedTaskRecipe(WeightTrainingRecipe):
    """Group relative advantages per generated task, on Tinker's importance sampling loss.

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``GeneratedTaskRecord2Record``,
    ``GeneratedTaskTasks``, and ``GeneratedTaskBatches``; the deployment's
    ``pipeline`` section gives their config: the rounds, the task
    generator, and the generator service in ``record2task``, the source
    sessions and the pace of the rounds in ``record2record``, and the rollout
    batches and their task groups in ``task2batch``. Without that section, no round runs, no task rolls out,
    and nothing trains.

    Enabled rounds save each task and its train/eval split in the scenario's task store before their episodes run.
    Eval tasks stay in the store without episodes or training feedback.
    """

    name: str = "generated-tasks"

    @property
    def report_type(self) -> type[ReportBase]:
        # A scored report: what a source session ends with. The recipe trains on no report.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective=TaskGroupObjective.name, scheduling=StepScheduling(unit="sample"))

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=GeneratedTaskRecord2Record, record2task=GeneratedTaskTasks, task2batch=GeneratedTaskBatches
        )
