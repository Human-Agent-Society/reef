"""Single-Rollout Asynchronous Optimization recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sao.processor import SAOBatches, SAOTasks
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.errors import RecipeConfigError
from reef.record2record import ExclusiveReportRecord2Record
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class SAORecipe(WeightTrainingRecipe):
    """Single-Rollout Asynchronous Optimization (arXiv:2607.07508) on reef.

    "Single rollout" is one rollout per prompt: there is no comparison group
    and no slowest-sample barrier, so each scored rollout is accepted on its
    own as it lands. ``batch_size`` of them, from ``batch_size`` different
    prompts, form one optimizer step; the paper trains with 128 (§4.1). The
    value model is what makes a single sample per prompt usable, and it needs
    that many samples per step to learn, so ``batch_size=1`` (one step per
    rollout) is a smoke setting, not the paper's estimator. The DIS ratio
    needs the rollout log-probabilities as its behaviour proxy, so SAO requires
    an inference backend that attaches engine-native tensors
    (``reef.inference_handler_factory``); reef never re-tokenizes a rollout to
    reconstruct them.

    The stages are ``ExclusiveReportRecord2Record`` (every rollout trains by
    itself), ``SAOTasks``, and ``SAOBatches``.
    The deployment sets ``batch_size`` in ``pipeline.task2batch.config``
    (default 128). It must equal the Slime driver's ``--global-batch-size``:
    each rollout sample is its own DP unit.

    Objective settings such as the clipping bounds, actor/critic cadence, and GAE
    parameters belong to the training backend. For Slime they are configured by
    ``training.options``; this recipe owns only Reef-side checkpoint cadence.
    """

    name: str = "sao"

    @property
    def report_type(self) -> type[ReportBase]:
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sao",
            # Each rollout is its own DP unit; the backend's configured step size applies.
            scheduling=StepScheduling(unit="sample"),
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=ExclusiveReportRecord2Record, record2task=SAOTasks, task2batch=SAOBatches)

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SAO objective options are backend-owned; configure the Slime implementation with training.options"
            )
