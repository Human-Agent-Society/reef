"""SAO with extra critic warmup passes on the first Retail batch."""

from dataclasses import dataclass

from recipes.sao.recipe import SAORecipe
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.runtime.interfaces import TrainingMethod
from reef.train.jobs import JobPlan, JobPlanContext, JobRequest


@dataclass(frozen=True)
class RetailJobPlan(JobPlan):
    """A nonnegative warmup step count, applied only to the first job's batch.

    The first job makes ``warmup_steps`` critic-only passes, then one normal
    SAO pass over the same trajectories. Later jobs make one normal pass.
    Each pass saves a checkpoint so Slime can resume between passes.
    The backend's num-critic-only-steps must equal warmup_steps.
    """

    warmup_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")

    def next_job(self, context: JobPlanContext) -> JobRequest:
        epochs = self.warmup_steps + 1 if context.finished_jobs == 0 else 1
        return JobRequest(
            {
                "method": TrainingMethod("sao").to_dict(),
                "scheduling": {"epochs": epochs, "shuffle": False, "remainder": "partial"},
                "save_interval": 1,
            }
        )

    def experiment_config(self) -> dict:
        return {"job_plan": "retail_warmup_reuse", "warmup_steps": self.warmup_steps}


@dataclass(frozen=True, kw_only=True)
class RetailSAORecipe(SAORecipe):
    """The SAO recipe with a nonnegative number of extra first-batch warmup passes.

    The example owns job scheduling; job_parameters must be unset or empty.
    SAO losses, critic cadence, and freezing remain backend settings.
    """

    warmup_steps: int = config_field(1, help="Extra critic-only passes over the first batch before actor training.")

    def __post_init__(self) -> None:
        super().__post_init__()
        RetailJobPlan(self.warmup_steps)
        if self.job_parameters:
            raise RecipeConfigError("Retail owns job scheduling; remove recipe.config.job_parameters")

    def job_plan(self) -> JobPlan:
        return RetailJobPlan(self.warmup_steps)
