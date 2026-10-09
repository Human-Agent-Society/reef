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
    The teacher reads the recorded prompt and response ids, so the recipe
    needs no tokenizer. ``batch_size`` responses form one optimizer step;
    the caller waits for publication before sampling the next batch. The
    teacher checkpoint and the divergence are the backend's ``--opd-*``
    training options. ``accept_multi_turn_policy_samples`` retains every
    assistant token in one exact episode history, masks tool context, and
    rejects teacher-window overflow.
    """

    name: str = "opd"
    batch_size: int = config_field(1)
    max_teacher_tokens: int = config_field(0)
    accept_multi_turn_policy_samples: bool = config_field(False)

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="opd",
            processor=OPDProcessor,
            # The batch is the optimizer step, however many responses it holds.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
        if not isinstance(self.accept_multi_turn_policy_samples, bool):
            raise ValueError("accept_multi_turn_policy_samples must be a boolean")
