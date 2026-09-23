"""Single-Rollout Asynchronous Optimization (arXiv:2607.07508): one method, one package.

- ``recipe`` — the SAO recipe class and its ``WeightTrainingSpec``.
- ``processor`` — one scored rollout, one batch unit.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family (DIS ratio, actor/critic cadence) and
  its torch objective, plus the GRPO(+DIS) control arm of the paper's
  comparison in ``slime.grpo_dis``. Imported by the training driver and
  workers only; this package's public surface never loads it.

The control shares SAO's recipe, processor and DIS primitive and differs in
the advantage estimate alone, so its Reef-side classes live beside SAO's in
``recipe`` and ``objective``.
"""

from recipes.sao.objective import SaoGrpoControlObjective, SaoObjective
from recipes.sao.processor import SAOProcessor
from recipes.sao.recipe import SAOGrpoControlRecipe, SAORecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sao", "recipes.sao.slime:SaoAlgorithm")
register_loss_family_ref("sao-grpo-dis", "recipes.sao.slime.grpo_dis:SaoGrpoControlAlgorithm")

__all__ = ["SAOGrpoControlRecipe", "SAOProcessor", "SAORecipe", "SaoGrpoControlObjective", "SaoObjective"]
