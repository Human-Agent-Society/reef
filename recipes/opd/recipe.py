"""On-policy distillation recipe using a separate, frozen teacher."""

from dataclasses import dataclass

from recipes.opd.processor import OPDProcessor
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class OPDRecipe(WeightTrainingRecipe):
    """Distil a separate teacher on the student's own recorded responses.

    A report references one inference and leaves ``teacher_context`` empty.
    ``batch_size`` must match the trainer's global batch size. The caller
    waits for publication before sampling the next batch. Configure the
    teacher checkpoint and divergence using ``--opd-*`` training options.
    """

    name: str = "opd"
    batch_size: int = config_field(1)
    tokenizer_path: str = config_field("")
    max_teacher_tokens: int = config_field(0)

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective="opd", processor=OPDProcessor, scheduling=StepScheduling(unit="sample"))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: use the student's tokenizer shared by the teacher")
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
