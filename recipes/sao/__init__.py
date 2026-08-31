"""Single-Rollout Asynchronous Optimization (arXiv:2607.07508): one method, one package.

- ``recipe`` — the ``sao`` recipe kind and its ``WeightTrainingSpec``.
- ``processor`` — one scored rollout, one batch unit.
- ``preparer`` — the backend-agnostic step preparer.
- ``slime`` — the Slime loss family (DIS ratio, actor/critic cadence) and
  its torch objective. Imported by the training driver and workers only;
  this package's public surface never loads it.
"""

from recipes.sao.preparer import SaoPreparer
from recipes.sao.processor import SAOProcessor
from recipes.sao.recipe import SAORecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sao", "recipes.sao.slime:SaoAlgorithm")

__all__ = ["SAOProcessor", "SAORecipe", "SaoPreparer"]
