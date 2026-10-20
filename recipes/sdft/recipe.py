"""Self-Distillation Fine-Tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdft.processor import SDFTTasks
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.record2record import ExclusiveReportRecord2Record
from reef.task2batch import DistillBatches
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SDFTRecipe(WeightTrainingRecipe):
    """Self-Distillation Fine-Tuning (arXiv:2601.19897) on Reef.

    The served model is its own teacher: it reads the request with a
    demonstration added and its next-token distributions over the student's
    on-policy response become the target of a per-token KL. Each report
    carries one rollout's receipt and the demonstration as ``teacher_context``; with
    ``batch_size=1`` a report trains as soon as it arrives.

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``ExclusiveReportRecord2Record``, ``SDFTTasks``, and
    ``DistillBatches``; the deployment's ``pipeline`` section gives their config.
    ``pipeline.record2task.config`` takes ``tokenizer_path`` (required: the
    served model's tokenizer renders the teacher prompt),
    ``max_teacher_tokens``, and ``context_template``.
    ``pipeline.task2batch.config.batch_size`` must equal the Slime driver's
    ``--global-batch-size``: each sample is its own DP unit.

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
            # Each sample is its own DP unit; the backend's configured step size applies.
            scheduling=StepScheduling(unit="sample"),
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=ExclusiveReportRecord2Record, record2task=SDFTTasks, task2batch=DistillBatches
        )

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SDFT objective options are backend-owned; configure the Slime implementation with training.options"
            )
