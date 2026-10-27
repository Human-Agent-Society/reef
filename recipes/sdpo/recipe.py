"""Self-Distillation Policy Optimization recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdpo.processor import SDPOJobData, SDPORecord2Record, SDPOTasks
from recipes.sdpo.report import SDPOReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.train.pipeline import PipelineSpec


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

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``SDPORecord2Record``, ``SDPOTasks``, and
    ``SDPOJobData``; the deployment's ``pipeline`` section gives their config:
    the grid in ``record2record``; the teacher's tokenizer and windows in
    ``record2task`` (``tokenizer_path`` is required); and the choice of
    demonstration and feedback in ``task2job``.

    Objective settings such as the divergence, the top-K and the teacher's
    update rate belong to the training backend; for Slime they are
    ``--sdpo-*`` flags in ``training.options``.
    """

    name: str = "sdpo"

    @property
    def report_type(self) -> type[ReportBase]:
        return SDPOReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective="sdpo")

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=SDPORecord2Record, record2task=SDPOTasks, task2job=SDPOJobData)

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SDPO objective options are backend-owned; configure the Slime implementation with training.options"
            )
