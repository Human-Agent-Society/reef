"""Single-Rollout Asynchronous Optimization (arXiv:2607.07508): one method, one package.

- ``recipe`` — the SAO recipe class and its default job parameters.
- ``processor`` — the stages ``SAOSamples`` and ``SAOJobData``: one scored
  rollout, one record set unit.
- ``slime`` — the Slime loss family (DIS ratio, actor/critic cadence) and
  its torch objective. Imported by the Slime training component and workers only;
  this package's public surface never loads it.
"""

from recipes.sao.processor import SAOJobData, SAOSamples
from recipes.sao.recipe import SAORecipe
from reef.train.slime_backend.objectives import register_objective

register_objective("sao", "recipes.sao.slime:SaoObjective")

__all__ = ["SAOJobData", "SAORecipe", "SAOSamples"]
