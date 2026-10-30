"""The dataset recipe."""

from __future__ import annotations

from dataclasses import dataclass

from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.dataset.processor import DatasetJobData
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class DatasetRecipe(WeightTrainingRecipe):
    """Slime-native training on a prompt dataset: one dataset job per training instruction.

    The instruction's ``parameters`` (or ``recipe.config.job_parameters``)
    hold ``dataset``: ``{"path": <JSONL>, "epochs": 1}``, and optionally
    ``save_interval`` in rollouts. Configure the rollout, the reward model,
    the advantage estimator and the optimizer in ``training.options``, as for
    Slime itself; see "Dataset jobs on Slime" in the configuration reference.
    """

    name: str = "dataset"

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        # The Slime component does not import this recipe. A dotted reference imports the objective there.
        return WeightTrainingSpec(objective="reef.recipe.dataset:SlimeDatasetObjective")

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(sample2job=DatasetJobData)
