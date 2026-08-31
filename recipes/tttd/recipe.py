"""TTT-Discover backend recipe."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.tttd.processor import TTTDProcessor
from recipes.tttd.report import TTTDGroupedRolloutReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field


@dataclass(frozen=True, kw_only=True)
class TTTDRecipe(WeightTrainingRecipe):
    name: str = "tttd"
    groups_per_step: int = config_field(8, env="REEF_TTTD_GROUPS_PER_STEP")
    rollouts_per_group: int = config_field(64, env="REEF_TTTD_ROLLOUTS_PER_GROUP")

    @property
    def report_type(self) -> type[ReportBase]:
        # One rollout addressed into its exact step grid; the processor adds
        # the config-relative check against the served grid.
        return TTTDGroupedRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="tttd", loss_family="tttd", processor=TTTDProcessor)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0:
            raise ValueError("groups_per_step must be positive")
        if self.rollouts_per_group < 2:
            raise ValueError("rollouts_per_group must be at least two")
