"""The supervised fine-tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.recipe.sft.processor import SFTRecord2Record, SFTTasks
from reef.task2batch import DistillBatches
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SFTRecipe(WeightTrainingRecipe):
    """Supervised fine-tuning: the token cross-entropy of a target response, on Slime's ``sft_loss``.

    Each report references one recorded request. The target is the report's
    demonstration (``teacher_context``) or the recorded response, for example
    an imported dataset example. SDFT uses the same report contract. Thus the
    two methods can train on the same reports.

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``SFTRecord2Record``, ``SFTTasks``, and
    ``DistillBatches``. The ``pipeline`` section of the deployment configures
    the stages:

    * ``pipeline.record2task.config``: ``tokenizer_path`` (required),
      ``target``, and ``max_tokens``.
    * ``pipeline.task2batch.config.batch_size``: the number of samples in one
      batch. The Slime ``--global-batch-size`` is the number of samples in one
      optimizer step.

    Configure the optimizer in ``training.options``.
    """

    name: str = "sft"

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            # The Slime bridge does not import this recipe. A dotted reference imports the objective there.
            objective="reef.recipe.sft:SftObjective",
            # Each sample is one unit. The backend sets the step size.
            scheduling=StepScheduling(unit="sample"),
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=SFTRecord2Record, record2task=SFTTasks, task2batch=DistillBatches)

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError("SFT has no objective options; configure the optimizer with training.options")
