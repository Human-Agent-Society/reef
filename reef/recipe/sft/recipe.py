"""The supervised fine-tuning recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.recipe.sft.processor import SFTJobData, SFTRecord2Record, SFTSamples
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SFTRecipe(WeightTrainingRecipe):
    """Supervised fine-tuning: the token cross-entropy of a target response, on Slime's ``sft_loss``.

    Each report references one recorded request. The target is the report's
    demonstration (``teacher_context``) or the recorded response, for example
    an imported dataset example. SDFT uses the same report contract. Thus the
    two methods can train on the same reports.

    The recipe binds the objective and the report contract. Its stages are ``SFTRecord2Record``, ``SFTSamples``, and
    ``SFTJobData``. ``SFTJobData`` also takes uploaded ATIF trajectories (``POST /reef/trajectories``); ``SFTSamples``
    renders each one with every agent turn trained, except ``is_copied_context`` steps. The ``pipeline`` section of the deployment configures
    the stages:

    * ``pipeline.record2sample.config``: ``tokenizer_path`` (required),
      ``target``, and ``max_tokens``.
    * ``pipeline.sample2job.config.job_samples`` (required): the fewest
      samples of one job. Slime cuts a job's data into optimizer steps of
      ``--global-batch-size`` samples.
    * ``pipeline.sample2job.config.buffer_size``: the most samples that wait
      for a job.
    * ``recipe.config.job_parameters``: the parameters of every job, for
      example ``scheduling.packing_length`` to count steps in packs of tokens.

    Configure the optimizer in ``training.options``.
    """

    name: str = "sft"

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            # The Slime component does not import this recipe. A dotted reference imports the objective there.
            objective="reef.recipe.sft:SftObjective",
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=SFTRecord2Record, record2sample=SFTSamples, sample2job=SFTJobData)

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError("SFT has no objective options; configure the optimizer with training.options")
