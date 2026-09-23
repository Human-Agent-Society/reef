"""Reef recipe for the GRPO(+DIS) control arm (arXiv:2607.07508, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.sao.objective import SaoObjective
from recipes.sao.processor import SAOProcessor
from recipes.sao.recipe import SAORecipe
from reef.recipe.base import StepScheduling, WeightTrainingSpec
from reef.train.algos.registry import register_objective


@register_objective
class SaoGrpoControlObjective(SaoObjective):
    """Same step signal as SAO; ``loss_family`` routes the payload to the control's Slime side."""

    name = "sao-grpo-dis"
    loss_family = "sao-grpo-dis"


@dataclass(frozen=True, kw_only=True)
class SAOGrpoControlRecipe(SAORecipe):
    """GRPO(+DIS) control: ``batch_size`` rollouts per step in complete groups of one prompt.

    The driver posts each group's ``--n-samples-per-prompt`` reports together,
    so a step of ``batch_size`` accepted rollouts holds only complete groups
    and Slime's group-relative baseline never mixes prompts. ``batch_size``
    is therefore a multiple of the group size (one group per step when the
    two are equal) and, as for SAO, must equal the driver's
    ``--global-batch-size``. It inherits SAO's default and environment key.
    """

    name: str = "sao-grpo-dis"

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sao-grpo-dis", processor=SAOProcessor, scheduling=StepScheduling(unit="sample")
        )
