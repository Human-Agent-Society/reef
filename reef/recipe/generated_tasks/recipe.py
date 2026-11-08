"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from dataclasses import dataclass, replace

from reef.core.job_data import DelayedSampleLimit
from reef.core.plugins import PluginConfigError
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks.processor import GeneratedTaskJobData, GeneratedTaskSamples, GeneratedTaskSettings
from reef.record2sample import Record2Sample
from reef.train.pipeline import PipelineSpec, TrainingPipeline


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
    on Slime). The recipe requires both. With ``designer_delay`` above 0, the task
    jobs also train the Designer's replies, so the training component and the
    objective must train delayed samples (``TrainingComponent.delayed_sample_limit``);
    the stages build only the delayed samples that the limit keeps.

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
        self.delayed_sample_limit()

    def delayed_sample_limit(self) -> DelayedSampleLimit | None:
        """The delayed samples that a task job trains when the record-to-sample stage carries the Designer's replies
        (``designer_delay`` above 0), else None.

        Raise ``RecipeConfigError`` when the stage carries them and the training component or the objective of the
        task jobs trains no delayed samples, so the deployment fails when it loads.
        """
        stage = self.stages().record2sample
        stage_class = Record2Sample.referenced_class(stage)
        if not issubclass(stage_class, GeneratedTaskSamples):
            return None
        try:
            designer_delay = GeneratedTaskSettings.parse(stage.config, plugin_class=stage_class).designer_delay
        except PluginConfigError:
            # The stage refuses its own config when a scenario builds it.
            return None
        if designer_delay == 0:
            return None
        try:
            limit = self.training_component.delayed_sample_limit(self.parameters_by_kind()["task"])
        except ValueError as exc:
            raise RecipeConfigError(
                f"pipeline.record2sample.config.designer_delay {designer_delay} trains the Designer's replies in "
                f"task jobs, which the training component cannot run: {exc}"
            ) from exc
        if limit is None:
            raise RecipeConfigError(
                f"pipeline.record2sample.config.designer_delay {designer_delay} trains the Designer's replies in task "
                f"jobs, but the {self.training_component.name} training component trains no delayed samples with "
                "the objective of recipe.config.job_parameters"
            )
        return limit

    def build_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        """Build the pipeline of one scenario, with the delayed samples that its task jobs train in the context."""
        return super().build_pipeline(replace(context, delayed_sample_limit=self.delayed_sample_limit()))

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=GeneratedTaskSamples, sample2job=GeneratedTaskJobData)
