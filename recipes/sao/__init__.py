"""Single-Rollout Asynchronous Optimization (arXiv:2607.07508): one method, one package.

- ``recipe`` — the SAO recipe class and its ``WeightTrainingSpec``. Its
  record stage is Reef's ``ExclusiveReportRecord2Record``.
- ``processor`` — the stages ``SAOTasks`` and ``SAOBatches``: one scored
  rollout, one batch unit.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family (DIS ratio, actor/critic cadence) and
  its torch objective. Imported by the training driver and workers only;
  this package's public surface never loads it.
"""

from recipes.sao.objective import SaoObjective
from recipes.sao.processor import SAOBatches, SAOBatchSettings, SAOTasks
from recipes.sao.recipe import SAORecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sao", "recipes.sao.slime:SaoAlgorithm")

__all__ = ["SAOBatchSettings", "SAOBatches", "SAORecipe", "SAOTasks", "SaoObjective"]
