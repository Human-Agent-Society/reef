"""Supervised fine-tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sft.processor import SFTProcessor
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class SFTRecipe(WeightTrainingRecipe):
    """Supervised fine-tuning on demonstrations.

    Each report carries one rollout's receipt and a demonstration as
    ``context``, the contract the self-distillation recipes take, so the
    methods can be compared on identical feedback. The demonstration becomes the
    assistant turn of the recorded request, rendered with the served model's
    chat template (``tokenizer_path``), and Slime's stock ``sft_loss`` trains
    its tokens; the student's own response is ignored. With ``batch_size=1``
    a report trains as soon as it arrives.

    The optimizer belongs to the training backend (``training.options``).
    ``batch_size`` must equal the Slime driver's ``--global-batch-size``: each
    sample is its own DP unit.
    """

    name: str = "sft"
    batch_size: int = config_field(1, env="REEF_SFT_BATCH_SIZE")
    tokenizer_path: str = config_field("")

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sft",
            processor=SFTProcessor,
            # Each sample is its own DP unit; the backend's configured step size applies.
            scheduling=StepScheduling(unit="sample"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the sample")

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError("SFT has no objective options; the optimizer is training.options")
