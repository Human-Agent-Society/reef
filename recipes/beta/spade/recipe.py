"""The SPADE Reasoning Agent recipe: the served model trains on the tasks the Designer wrote, one task group at a time."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.beta.spade.preparer import LOSS_FAMILY, SpadePreparer
from recipes.beta.spade.processor import (
    DEFAULT_ROLLOUTS_PER_TASK,
    DEFAULT_SCAFFOLD_TOLERANCE,
    DEFAULT_TASKS_PER_STEP,
    SpadeProcessor,
)
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field


@dataclass(frozen=True, kw_only=True)
class SpadeRecipe(WeightTrainingRecipe):
    """Reports from the task player, grouped by task; group relative advantages on Tinker's importance sampling loss."""

    name: str = "spade"
    tasks_per_step: int = config_field(DEFAULT_TASKS_PER_STEP, env="REEF_SPADE_TASKS_PER_STEP")
    rollouts_per_task: int = config_field(DEFAULT_ROLLOUTS_PER_TASK, env="REEF_SPADE_ROLLOUTS_PER_TASK")
    # A thinking template's generation prompt ends with a think scaffold the history drops; the assembly may
    # realign that many masked tokens ahead of the previous response.
    scaffold_tolerance: int = config_field(DEFAULT_SCAFFOLD_TOLERANCE, env="REEF_SPADE_SCAFFOLD_TOLERANCE")

    @property
    def report_type(self) -> type[ReportBase]:
        # The task player's report: the verifier reward as the score; the task under metadata.task.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer=SpadePreparer.name, loss_family=LOSS_FAMILY, processor=SpadeProcessor)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
