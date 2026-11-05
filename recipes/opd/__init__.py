"""On-policy distillation from a separate teacher with the student's tokenizer.

The recipe keeps requests unchanged, reuses the shared distillation stages
and Slime backend, and trains only on the student's recorded samples.
"""

from recipes.opd.processor import OPDSamples, OPDSettings
from recipes.opd.recipe import OPDRecipe
from reef.train.algos.registry import register_loss_family_ref, register_objective_ref

register_objective_ref("opd", "recipes.opd.slime:OpdObjective")
register_loss_family_ref("opd", "recipes.opd.slime:OpdAlgorithm")

__all__ = ["OPDRecipe", "OPDSamples", "OPDSettings"]
