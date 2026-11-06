"""On-policy distillation recipe using a separate, frozen teacher."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.opd.processor import OPDSamples
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe
from reef.sample2job import DistillJobData
from reef.storage.samples import SAMPLE_KINDS, SampleKind
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class OPDRecipe(WeightTrainingRecipe):
    """Distil a separate teacher on the student's own recorded responses.

    A report references one inference and leaves ``teacher_context`` empty.
    The teacher reads the recorded prompt and response ids, so the recipe
    needs no tokenizer. Its stages are ``OPDSamples`` and
    ``DistillJobData``; the deployment's ``pipeline`` section gives their
    config: ``max_teacher_tokens`` in ``record2sample``. The backend's batch
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
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {kind: {"objective": "opd"} for kind in SAMPLE_KINDS}

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=OPDSamples, sample2job=DistillJobData)
