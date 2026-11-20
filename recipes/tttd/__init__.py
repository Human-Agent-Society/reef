"""TTT-Discover (arXiv:2601.16175): one method, one package.

Everything Reef-side that decides what TTT-Discover trains lives here:

- ``recipe`` — the TTT-Discover recipe class, its default job parameters, and
  its default pipeline stages.
- ``processor`` — the pipeline stages: ``TTTDSamples`` owns the step grid and
  its barrier and makes the grouped policy samples, and
  ``TTTDJobData`` makes one batch from one complete step.
- ``report`` — the method-specific grouped rollout wire contract.
- ``objective`` — the adaptive-entropic grouped advantages that both backends'
  objectives train. It names no backend.
- ``slime`` — the Slime objective, loss family and torch objective hooks. It is
  imported by the Slime training component and workers only; this package's
  public surface never loads it, so a service without Slime training stays
  free of the Slime stack.
- ``tinker`` — the Tinker objective and loss family, on Tinker's built-in
  importance-sampling loss. It is imported only when that backend resolves the
  objective.
"""

from recipes.tttd.processor import TTTDJobData, TTTDSamples, TTTDSettings
from recipes.tttd.recipe import TTTDRecipe
from recipes.tttd.report import TTTDGroupedRolloutReport
from reef.train.slime_backend.objectives import register_objective as register_slime_objective
from reef.train.tinker_backend.objectives import register_objective as register_tinker_objective

register_slime_objective("tttd", "recipes.tttd.slime:TttdObjective")
register_tinker_objective("tttd", "recipes.tttd.tinker:TttdTinkerObjective")

__all__ = [
    "TTTDGroupedRolloutReport",
    "TTTDJobData",
    "TTTDRecipe",
    "TTTDSamples",
    "TTTDSettings",
]
