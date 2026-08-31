"""Single-Rollout Asynchronous Optimization recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sao.processor import SAOProcessor
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError


@dataclass(frozen=True, kw_only=True)
class SAORecipe(WeightTrainingRecipe):
    """Single-Rollout Asynchronous Optimization (arXiv:2607.07508) on reef.

    SAO is asynchronous by construction here: with ``batch_size=1`` the
    dispatcher runs one training step per accepted rollout, so a rollout enters
    training the moment its score arrives, with no comparison group or
    slowest-sample barrier. The DIS ratio needs the rollout log-probabilities as
    its behaviour proxy, so SAO requires an inference backend that attaches
    engine-native tensors (``reef.inference_backend_factory``); reef never
    re-tokenizes a rollout to reconstruct them.

    Objective settings such as the clipping bounds, actor/critic cadence, and GAE
    parameters belong to the training backend. For Slime they are configured by
    ``training.slime_flags``; this recipe only owns Reef-side batching and
    checkpoint cadence.

    ``batch_size`` must equal the Slime driver's ``--global-batch-size``: each
    rollout sample is its own DP unit.
    """

    name: str = "sao"
    batch_size: int = config_field(1, env="REEF_SAO_BATCH_SIZE")

    @property
    def report_type(self) -> type[ReportBase]:
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="sao", loss_family="sao", processor=SAOProcessor)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SAO objective options are backend-owned; configure the Slime implementation with training.slime_flags"
            )
