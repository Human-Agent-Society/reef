"""Standard PPO/RLHF reference-reward training recipe.

This package is the deliberately small owner for the adaptive reference-policy
KL path.  It uses Slime's stock clipped ``policy_loss`` and advantage pass;
the recipe does not redefine PPO or share a KL controller with the cookbook
loss families.
"""

from recipes.ppo_rlhf.objective import PpoRlhfObjective
from recipes.ppo_rlhf.processor import PpoRlhfProcessor
from recipes.ppo_rlhf.recipe import PpoRlhfRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref(
    "ppo_rlhf_reference_reward",
    "recipes.ppo_rlhf.slime:PpoRlhfAlgorithm",
)

__all__ = ["PpoRlhfObjective", "PpoRlhfProcessor", "PpoRlhfRecipe"]
