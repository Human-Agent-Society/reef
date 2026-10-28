"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskJobData,
    GeneratedTaskRecord2Record,
    GeneratedTaskSamples,
)
from reef.recipe.imported_tasks import ImportedTaskEvaluation
from reef.runtime.interfaces import TrainingMethod
from reef.storage.tasks import TaskStorage
from reef.train.jobs import ArtifactEvaluation
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

    With ``recipe.config.checkpoint_eval_rollouts_per_task`` set to k, every artifact that the trainer publishes
    is evaluated on the eval tasks of the scenario's ``imported-tasks`` store (``ImportedTaskEvaluation``): an
    evaluation job from the artifact's checkpoint rolls each task out k times, with ``job_parameters`` and k in
    place of the rollout's ``rollouts_per_task``, and trains nothing. Task uploads to that store's eval split are
    then admitted. The evaluation decides nothing: the ``evaluation`` section decides which artifact serves.

    Values: ``checkpoint_eval_rollouts_per_task`` is None (no evaluation) or a positive int that the training
    component can run as the rollouts of each task (``check_task_jobs``); construction checks both. Purpose:
    deployment assembly builds the recipe from ``recipe.config``; the weight trainer reads its job plan and its
    artifact evaluation.
    """

    name: str = "generated-tasks"
    checkpoint_eval_rollouts_per_task: int | None = config_field(
        None,
        help=(
            "Rollouts of each imported eval task in the evaluation job of every published checkpoint; "
            "none evaluates no checkpoint."
        ),
    )

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
        rollouts = self.checkpoint_eval_rollouts_per_task
        if rollouts is None:
            return
        if isinstance(rollouts, bool) or not isinstance(rollouts, int) or rollouts < 1:
            raise RecipeConfigError(
                f"recipe.config.checkpoint_eval_rollouts_per_task must be a positive int or null, got {rollouts!r}"
            )
        try:
            self.training_component.check_task_jobs(self.evaluation_parameters(rollouts))
        except ValueError as exc:
            raise RecipeConfigError(
                "the training component cannot run the checkpoint evaluation of "
                f"recipe.config.checkpoint_eval_rollouts_per_task: {exc}"
            ) from exc

    def evaluation_parameters(self, rollouts: int) -> dict[str, object]:
        """The parameters of a checkpoint's evaluation job beside ``method``.

        They are a copy of ``job_parameters`` whose ``rollout.rollouts_per_task`` is ``rollouts``;
        ``job_parameters`` does not change.
        """
        parameters: dict[str, object] = dict(self.job_parameters or {})
        rollout = parameters.get("rollout")
        settings = dict(rollout) if isinstance(rollout, Mapping) else {}
        # The component reads a rollout key in either spelling, so the training value leaves in both spellings.
        kept = {name: value for name, value in settings.items() if str(name).replace("-", "_") != "rollouts_per_task"}
        parameters["rollout"] = {**kept, "rollouts_per_task": rollouts}
        return parameters

    def artifact_evaluation(self, task_storage: TaskStorage | None) -> ArtifactEvaluation | None:
        rollouts = self.checkpoint_eval_rollouts_per_task
        if rollouts is None:
            return None
        method = TrainingMethod(TaskGroupObjective.name).to_dict()
        return ImportedTaskEvaluation.for_storage(
            task_storage, {**self.evaluation_parameters(rollouts), "method": method}
        )

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
