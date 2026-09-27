"""Self-Distillation Policy Optimization recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdpo.processor import DEFAULT_REPROMPT_TEMPLATE, SDPOProcessor, reprompt_template
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class SDPORecipe(WeightTrainingRecipe):
    """Distil feedback-conditioned self-teacher scores on each original rollout."""

    name: str = "sdpo"
    batch_size: int = config_field(1, env="REEF_SDPO_BATCH_SIZE")
    tokenizer_path: str = config_field("")
    max_teacher_tokens: int = config_field(0)
    enable_thinking: bool = config_field(False)
    reprompt_template: str = config_field(DEFAULT_REPROMPT_TEMPLATE)

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective="sdpo", processor=SDPOProcessor, scheduling=StepScheduling(unit="sample"))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
        reprompt_template(self.processor_config())

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError("SDPO objective options belong to the Slime backend in training.options")
