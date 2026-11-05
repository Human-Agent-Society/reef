"""The supervised fine-tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.sft.processor import SFTJobData, SFTSamples
from reef.storage.samples import SampleKind
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SFTRecipe(WeightTrainingRecipe):
    """Supervised fine-tuning: the token cross-entropy of a target response, on Slime's ``sft_loss``.

    Each report references one recorded request. The target is the report's
    demonstration (``teacher_context``) or the recorded response, for example
    an imported dataset example. SDFT uses the same report contract. Thus the
    two methods can train on the same reports.

    The recipe binds the default objective and the report contract. Its stages are ``SFTSamples`` and
    ``SFTJobData``. ``SFTJobData`` also takes trajectory items (``POST /reef/namespaces/{owner}/data``), which train
    with every agent turn that is not an ``is_copied_context`` step. The training component renders every sample
    with the model's chat template. The ``pipeline`` section of the deployment configures the stages:

    * ``pipeline.record2sample.config``: ``target`` and ``max_tokens``.
    * ``pipeline.sample2job.config.job_samples`` (required): the fewest
      samples of one job. Slime cuts a job's data into optimizer steps of
      ``--global-batch-size`` samples.
    * ``pipeline.sample2job.config.buffer_size``: the most samples that one
      job takes; later samples wait for later jobs.
    * ``recipe.config.job_parameters``: the parameters of every trajectory job, for
      example ``scheduling.packing_length`` to count steps in packs of tokens.

    A trajectory job trains the ``sft`` objective. The recipe gives task jobs no
    parameters, so an imported train task trains only for a request. The
    request's ``parameters`` name its objective and its rollouts in the
    backend's own shape, such as an RL step after SFT.

    Configure the optimizer in ``training.options``.
    """

    name: str = "sft"

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {"trajectory": {"objective": "sft"}}

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=SFTSamples, sample2job=SFTJobData)

    @classmethod
    def configured_parameters(cls, job_parameters: Mapping[str, Any] | None) -> Mapping[SampleKind, Mapping[str, Any]]:
        """SFT trains trajectories: a task job takes its parameters from its request."""
        return {"trajectory": super().configured_parameters(job_parameters)["trajectory"]}

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError("SFT has no objective options; configure the optimizer with training.options")
