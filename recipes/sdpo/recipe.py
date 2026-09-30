"""Self-Distillation Policy Optimization recipe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdpo.processor import SDPOProcessor
from recipes.sdpo.report import SDPOReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class SDPORecipe(WeightTrainingRecipe):
    """Self-Distillation Policy Optimization (arXiv:2601.20802) on Reef.

    The served model is its own teacher: it reads the question with a
    successful sibling's response, or the environment's feedback, and its
    next-token distributions over the student's on-policy response become the
    target of a per-token divergence. A step is a complete sampling grid,
    ``groups_per_step`` questions by ``rollouts_per_group`` attempts from one
    policy release; each report carries one rollout's receipt, its score and
    its coordinates, and the grid trains as one optimizer step once complete.

    ``tokenizer_path`` is the served model's tokenizer directory, which renders
    the teacher prompt with the chat template the engine applied. The rendered
    prompt is cut at ``max_teacher_prompt_tokens`` and the whole teacher
    sequence must fit ``max_teacher_tokens``, the trainer's window.
    ``success_reward_threshold`` picks the demonstrating rollouts;
    ``allow_own_success_as_demonstration`` lets a successful rollout read its
    own response (the reference does not); ``remove_thinking_from_demonstration``
    strips ``<think>`` blocks from the demonstration. ``include_environment_feedback``
    adds the report's ``teacher_context`` to the teacher's prompt, by default
    only for rollouts without a demonstration
    (``environment_feedback_only_without_solution``). ``enable_thinking`` is
    the chat template's switch, set as the engine sampled.

    Objective settings such as the divergence, the top-K and the teacher's
    update rate belong to the training backend; for Slime they are
    ``--sdpo-*`` flags in ``training.options``.
    """

    name: str = "sdpo"
    groups_per_step: int = config_field(32)
    rollouts_per_group: int = config_field(8)
    tokenizer_path: str = config_field("")
    max_teacher_prompt_tokens: int = config_field(10240)
    max_teacher_tokens: int = config_field(18432)
    success_reward_threshold: float = config_field(0.5)
    allow_own_success_as_demonstration: bool = config_field(False)
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
            objective="sdpo",
            processor=SDPOProcessor,
            # The grid is the optimizer step, however many rollouts it holds.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        if not 0 < self.max_teacher_prompt_tokens <= self.max_teacher_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SDPO objective options are backend-owned; configure the Slime implementation with training.options"
            )
