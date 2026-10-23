"""On-policy distillation recipe using a separate, frozen teacher."""

from dataclasses import dataclass

from recipes.opd.processor import OPDTasks
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.record2record import ExclusiveReportRecord2Record
from reef.task2batch import DistillBatches
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class OPDRecipe(WeightTrainingRecipe):
    """Distil a separate teacher on the student's own recorded responses.

    A report references one inference and leaves ``teacher_context`` empty.
    The teacher reads the recorded prompt and response ids, so the recipe
    needs no tokenizer. Its stages are ``ExclusiveReportRecord2Record``, ``OPDTasks``,
    and ``DistillBatches``; the deployment's ``pipeline`` section gives their
    config: ``max_teacher_tokens`` in ``record2task``. The backend's batch
    size (Slime ``--global-batch-size``) of responses forms one optimizer
    step; the caller waits for publication before sampling the next batch. The teacher
    checkpoint and the divergence are the backend's ``--opd-*`` training
    options.
    """

    name: str = "opd"

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective="opd", scheduling=StepScheduling(unit="sample"))

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=ExclusiveReportRecord2Record, record2task=OPDTasks, task2batch=DistillBatches
        )
