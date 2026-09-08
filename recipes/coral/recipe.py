"""The CORAL weight-training recipe: what a deployment names in ``serve.yaml``.

Binds :class:`~reef_coral.processor.CoralProcessor` to the grouped
relative-reward training machinery. CORAL sibling groups have the same shape
as TTT-Discover steps — comparison sets of scored rollouts from one shared
starting point — so the recipe reuses the ``tttd`` step preparer (grouped
adaptive-entropic leave-one-out advantages) and the ``tttd`` Slime loss
family rather than duplicating either. The recipe's own configuration is the
group barrier.

Deployment reference: ``recipes.coral.recipe:CoralRecipe`` (after
``pip install -e recipes/coral`` alongside the repository cookbook, which
provides the ``tttd`` registrations this recipe reuses).
"""

from __future__ import annotations

from dataclasses import dataclass

# The tttd step preparer and loss-family reference live in the repository
# cookbook; importing them is what registers both names this recipe binds.
import recipes.tttd  # noqa: F401  (registration side effect)
from recipes.coral.processor import CoralProcessor
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field


@dataclass(frozen=True, kw_only=True)
class CoralRecipe(WeightTrainingRecipe):
    name: str = "coral"
    #: Scored siblings of one parent commit required before their group trains.
    group_size: int = config_field(4, env="REEF_CORAL_GROUP_SIZE")

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="tttd", loss_family="tttd", processor=CoralProcessor)

    def __post_init__(self) -> None:
        # Validate our own field first: the group barrier is checkable
        # without a fully constructed training runtime.
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")
        super().__post_init__()
