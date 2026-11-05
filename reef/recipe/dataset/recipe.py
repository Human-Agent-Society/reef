"""The dataset recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.dataset.processor import DatasetJobData
from reef.storage.samples import SAMPLE_KINDS, SampleKind
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class DatasetRecipe(WeightTrainingRecipe):
    """Slime-native training on a prompt dataset: one dataset job per training request.

    The request's ``parameters`` (or ``recipe.config.job_parameters``)
    hold ``dataset``: ``{"path": <JSONL>, "epochs": 1}``, and optionally
    ``save_interval`` in rollouts. Configure the rollout, the reward model,
    the advantage estimator and the optimizer in ``training.options``, as for
    Slime itself; see "Dataset jobs on Slime" in the configuration reference.
    """

    name: str = "dataset"

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {kind: {"objective": "slime_dataset"} for kind in SAMPLE_KINDS}

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(sample2job=DatasetJobData)
