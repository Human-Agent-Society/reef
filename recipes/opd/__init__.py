"""On-policy distillation from a separate teacher with the student's tokenizer.

The recipe keeps requests unchanged, reuses the shared distillation stages
and Slime backend, and trains only on the student's recorded samples.
"""

from recipes.opd.processor import OPDSamples, OPDSettings
from recipes.opd.recipe import OPDRecipe
from reef.train.slime_backend.objectives import register_objective

register_objective("opd", "recipes.opd.slime:OpdObjective")

__all__ = ["OPDRecipe", "OPDSamples", "OPDSettings"]
