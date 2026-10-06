"""On-policy distillation from a separate teacher with the student's tokenizer.

The recipe keeps requests unchanged, reuses the shared distillation processor
and Slime backend, and trains only on the student's recorded samples.
"""

from recipes.opd.objective import OpdObjective
from recipes.opd.processor import OPDProcessor
from recipes.opd.recipe import OPDRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("opd", "recipes.opd.slime:OpdAlgorithm")

__all__ = ["OPDProcessor", "OPDRecipe", "OpdObjective"]
