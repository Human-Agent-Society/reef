"""Self-Distillation Fine-Tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdft.processor import SDFTSamples
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.record2record import ExclusiveReportRecord2Record
from reef.sample2job import DistillJobData
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SDFTRecipe(WeightTrainingRecipe):
    """Self-Distillation Fine-Tuning (arXiv:2601.19897) on Reef.

    The served model is its own teacher: it reads the request with a
    demonstration added and its next-token distributions over the student's
    on-policy response become the target of a per-token KL. Each report
    carries one rollout's receipt and the demonstration as ``teacher_context``; with
    a Slime ``--global-batch-size`` of 1 a report trains as soon as it arrives.

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``ExclusiveReportRecord2Record``, ``SDFTSamples``, and
    ``DistillJobData``; the deployment's ``pipeline`` section gives their config.
    ``pipeline.record2sample.config`` takes ``tokenizer_path`` (required: the
    served model's tokenizer renders the teacher prompt),
    ``max_teacher_tokens``, and ``context_template``. A batch holds Slime's
    ``--global-batch-size`` of samples; each sample is its own DP
    unit.

    Objective settings such as the KL direction belong to the training
    backend; for Slime they are ``--sdft-*`` flags in ``training.options``.
    """

    name: str = "sdft"

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sdft",
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=ExclusiveReportRecord2Record, record2sample=SDFTSamples, sample2job=DistillJobData
        )

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SDFT objective options are backend-owned; configure the Slime implementation with training.options"
            )
