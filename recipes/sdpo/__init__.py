"""Self-Distillation Policy Optimization (arXiv:2601.20802)."""

from recipes.sdpo.objective import SdpoObjective
from recipes.sdpo.processor import SDPOProcessor
from recipes.sdpo.recipe import SDPORecipe
from recipes.sdpo.report import SDPOReport
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sdpo", "recipes.sdpo.slime:SdpoAlgorithm")

__all__ = ["SDPOProcessor", "SDPORecipe", "SDPOReport", "SdpoObjective"]
