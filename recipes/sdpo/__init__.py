"""Self-Distillation Policy Optimization (arXiv:2601.20802): one method, one package.

- ``recipe`` — the SDPO recipe class, its ``WeightTrainingSpec`` and its
  ``PipelineSpec``. Its report contract is ``report.SDPOReport``: the shared
  :class:`reef.core.reports.TeacherContextReport` plus the rollout's
  coordinates in the sampling step.
- ``processor`` — the stages ``SDPORecord2Record`` (the grid),
  ``SDPOTasks`` (the teacher's renderer and windows) and ``SDPOBatches``, with
  a whole sampling step as one batch: the teacher's request is composed once
  the step's rollouts are known, from the reference's reprompt template.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family: a thin family on the backend's
  distillation base (``reef.train.slime_backend.distill``), which selects the
  student's top-K ids, runs the self-teacher forward pass and computes the
  per-token divergence. Imported by the training driver and workers only;
  this package's public surface never loads it.
"""

from recipes.sdpo.objective import SdpoObjective
from recipes.sdpo.processor import SDPOBatches, SDPORecord2Record, SDPOTasks
from recipes.sdpo.recipe import SDPORecipe
from recipes.sdpo.report import SDPOReport
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sdpo", "recipes.sdpo.slime:SdpoAlgorithm")

__all__ = ["SDPOBatches", "SDPORecipe", "SDPORecord2Record", "SDPOReport", "SDPOTasks", "SdpoObjective"]
