"""Standard PPO/RLHF reference-reward recipe."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.ppo_rlhf.processor import PpoRlhfProcessor
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class PpoRlhfRecipe(WeightTrainingRecipe):
    """One scored rollout per policy-training unit using stock Slime PPO."""

    name: str = "ppo_rlhf"
    batch_size: int = config_field(1, env="REEF_PPO_RLHF_BATCH_SIZE")

    @property
    def report_type(self) -> type[ReportBase]:
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="ppo_rlhf_reference_reward",
            processor=PpoRlhfProcessor,
            scheduling=StepScheduling(unit="sample"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
