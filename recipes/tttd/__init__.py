"""TTT-Discover (arXiv:2601.16175): one method, one package.

Everything Reef-side that decides what TTT-Discover trains lives here:

- ``recipe`` — the TTT-Discover recipe class, its default job parameters, and
  its default pipeline stages.
- ``processor`` — the pipeline stages: ``TTTDSamples`` owns the step grid and
  its barrier and makes the grouped policy samples, and
  ``TTTDJobData`` makes one batch from one complete step.
- ``report`` — the method-specific grouped rollout wire contract.
- ``objective`` — the training signal that both backends' objectives train
  (adaptive-entropic grouped advantages).
- ``slime`` — the Slime objective, loss family and torch objective hooks. It is
  imported by the Slime training component and workers only; this package's
  public surface never loads it, so a service without Slime training stays
  free of the Slime stack.
- ``tinker`` — the Tinker objective on Tinker's built-in importance-sampling
  loss, imported only when that backend resolves the family.
"""

from recipes.tttd.processor import TTTDJobData, TTTDSamples, TTTDSettings
from recipes.tttd.recipe import TTTDRecipe
from recipes.tttd.report import TTTDGroupedRolloutReport
from reef.train.algos.registry import register_loss_family_ref, register_objective_ref

register_objective_ref("tttd", "recipes.tttd.slime:TttdObjective")
register_objective_ref("tttd", "recipes.tttd.tinker:TttdTinkerObjective", backend="tinker")
register_loss_family_ref("tttd", "recipes.tttd.slime:TttdAlgorithm")
register_loss_family_ref("tttd", "recipes.tttd.tinker:TttdTinkerLoss", backend="tinker")

__all__ = [
    "TTTDGroupedRolloutReport",
    "TTTDJobData",
    "TTTDRecipe",
    "TTTDSamples",
    "TTTDSettings",
]
