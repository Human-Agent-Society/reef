"""Reef recipe for the GRPO(+DIS) control arm (arXiv:2607.07508, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.sao.objective import SaoObjective
from recipes.sao.processor import SAOProcessor
from recipes.sao.recipe import SAORecipe
from reef.recipe.base import StepScheduling, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.train.algos.registry import register_objective


@register_objective
class SaoGrpoControlObjective(SaoObjective):
    """Same step signal as SAO; ``loss_family`` routes the payload to the control's Slime side."""

    name = "sao-grpo-dis"
    loss_family = "sao-grpo-dis"


@dataclass(frozen=True, kw_only=True)
class SAOGrpoControlRecipe(SAORecipe):
    """GRPO(+DIS) control: a group of ``batch_size`` rollouts of one prompt per step.

    ``batch_size`` must equal the Slime driver's ``--n-samples-per-prompt`` and
    ``--global-batch-size`` so one Reef training step is exactly one GRPO group.
    """

    name: str = "sao-grpo-dis"
    batch_size: int = config_field(4, env="REEF_SAO_CONTROL_GROUP")

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sao-grpo-dis", processor=SAOProcessor, scheduling=StepScheduling(unit="sample")
        )
