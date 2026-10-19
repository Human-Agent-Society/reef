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
    generator, and the generator service in ``record2task``, the task groups
    and the source sessions in ``record2record``, and the batch size in
    ``task2batch``. Without that section, no round runs and the recipe trains
    on the training arm's reports only.

    Enabled rounds save each task and its train/eval split in the scenario's task store before play.
    Eval tasks stay in the store without plays or training feedback.
    """

    name: str = "generated-tasks"

    @property
    def report_type(self) -> type[ReportBase]:
        # The task player's report: the verifier reward as the score; the task under metadata.task.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective=TaskGroupObjective.name, scheduling=StepScheduling(unit="sample", batch_size="actual")
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=GeneratedTaskRecord2Record, record2task=GeneratedTaskTasks, task2batch=GeneratedTaskBatches
        )
