"""Self-Distillation Fine-Tuning (arXiv:2601.19897): one method, one package.

- ``recipe`` — the SDFT recipe class and its ``WeightTrainingSpec``. Its report
  contract is the shared :class:`reef.core.reports.TeacherContextReport`: a
  rollout's receipt and its ``teacher_context``.
- ``processor`` — the shared ``DistillProcessor`` with the demonstration
  appended to the recorded request, the reference's layout: one rollout with
  its demonstration, one batch unit.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family: a thin family on the backend's
  distillation base (``reef.train.slime_backend.distill``), which runs the
  self-teacher forward pass and the per-token divergence. Imported by the
  training driver and workers only; this package's public surface never
  loads it.
"""

from recipes.sdft.objective import SdftObjective
from recipes.sdft.processor import SDFTProcessor
from recipes.sdft.recipe import SDFTRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sdft", "recipes.sdft.slime:SdftAlgorithm")

__all__ = ["SDFTProcessor", "SDFTRecipe", "SdftObjective"]
