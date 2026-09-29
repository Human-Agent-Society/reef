"""Self-Distillation Fine-Tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdft.processor import DEFAULT_CONTEXT_TEMPLATE, SDFTProcessor, context_block_template
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class SDFTRecipe(WeightTrainingRecipe):
    """Self-Distillation Fine-Tuning (arXiv:2601.19897) on Reef.

    The served model is its own teacher: it reads the request with a
    demonstration added and its next-token distributions over the student's
    on-policy response become the target of a per-token KL. Each report
    carries one rollout's receipt and the demonstration as ``teacher_context``; with
    ``batch_size=1`` a report trains as soon as it arrives.

    ``tokenizer_path`` is the served model's tokenizer directory, which renders
    the teacher prompt with the chat template the engine applied.
    ``max_teacher_tokens`` skips reports whose teacher sequence would not fit
    the trainer's window (0 disables the check). The demonstration is added
    to the request's final user message as ``context_template`` (with
    ``{context}`` as its placeholder).

    Objective settings such as the KL direction belong to the training
    backend; for Slime they are ``--sdft-*`` flags in ``training.options``.
    ``batch_size`` must equal the Slime driver's ``--global-batch-size``: each
    sample is its own DP unit.
    """

    name: str = "sdft"
    batch_size: int = config_field(1, env="REEF_SDFT_BATCH_SIZE")
    tokenizer_path: str = config_field("")
    max_teacher_tokens: int = config_field(0)
    context_template: str = config_field(DEFAULT_CONTEXT_TEMPLATE)

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sdft",
            processor=SDFTProcessor,
            # Each sample is its own DP unit; the backend's configured step size applies.
            scheduling=StepScheduling(unit="sample"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
        # A bad template fails the deployment here, not the first scenario.
        context_block_template(self.processor_config())

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SDFT objective options are backend-owned; configure the Slime implementation with training.options"
            )
