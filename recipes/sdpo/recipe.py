"""Self-Distillation Policy Optimization on Reef."""

from __future__ import annotations

import math
from dataclasses import dataclass

from recipes.sdpo.processor import SDPOProcessor
from recipes.sdpo.report import SDPOReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class SDPORecipe(WeightTrainingRecipe):
    """A complete same-policy sampling grid supplies SDPO's privileged teacher context."""

    name: str = "sdpo"
    groups_per_step: int = config_field(32)
    rollouts_per_group: int = config_field(8)
    tokenizer_path: str = config_field("")
    max_teacher_prompt_tokens: int = config_field(10240)
    max_teacher_tokens: int = config_field(18432)
    success_reward_threshold: float = config_field(0.5)
    dont_reprompt_on_self_success: bool = config_field(True)
    remove_thinking_from_demonstration: bool = config_field(True)
    include_environment_feedback: bool = config_field(False)
    environment_feedback_only_without_solution: bool = config_field(True)
    enable_thinking: bool = config_field(False)

    @property
    def report_type(self) -> type[ReportBase]:
        return SDPOReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sdpo", processor=SDPOProcessor, scheduling=StepScheduling(unit="sample", batch_size="actual")
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required to render SDPO's teacher prompts")
        if self.max_teacher_prompt_tokens <= 0 or self.max_teacher_tokens < self.max_teacher_prompt_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")
