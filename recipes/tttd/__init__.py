"""TTT-Discover (arXiv:2601.16175): one method, one package.

Everything Reef-side that decides what TTT-Discover trains lives here:

- ``recipe`` — the TTT-Discover recipe class, its config fields, and the
  ``WeightTrainingSpec`` that binds the pieces below.
- ``processor`` — the exact-step barrier and grouped policy-data processor.
- ``report`` — the method-specific grouped rollout wire contract.
- ``preparer`` — the backend-agnostic step preparer (adaptive-entropic
  grouped advantages).
- ``slime`` — the Slime loss family and its torch objective hooks. It is
  imported by the training driver and workers only; this package's public
  surface never loads it, so the service process stays free of the Slime
  stack.
"""

from recipes.tttd.preparer import TttdPreparer
from recipes.tttd.processor import TTTDProcessor
from recipes.tttd.recipe import TTTDRecipe
from recipes.tttd.report import TTTDGroupedRolloutReport
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("tttd", "recipes.tttd.slime:TttdAlgorithm")

__all__ = ["TTTDGroupedRolloutReport", "TTTDProcessor", "TTTDRecipe", "TttdPreparer"]
