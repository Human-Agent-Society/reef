"""Self-Distillation Fine-Tuning (arXiv:2601.19897): one method, one package.

- ``recipe`` — the SDFT recipe class, its default job parameters and its
  ``PipelineSpec``. Its report contract is the shared
  :class:`reef.core.reports.TeacherContextReport`: a rollout's receipt and
  its ``teacher_context``.
- ``processor`` — ``SDFTSamples``, the shared ``DistillSamples`` with the
  demonstration appended to the recorded request, the reference's layout:
  one rollout with its demonstration, one record set unit.
- ``slime`` — the Slime loss family: a thin family on the backend's
  distillation base (``reef.train.slime_backend.distill``), which runs the
  self-teacher forward pass and the per-token divergence. Imported by the
  Slime training component and workers only; this package's public surface never
  loads it.
"""

from recipes.sdft.processor import SDFTSamples
from recipes.sdft.recipe import SDFTRecipe
from reef.train.slime_backend.objectives import register_objective

register_objective("sdft", "recipes.sdft.slime:SdftObjective")

__all__ = ["SDFTRecipe", "SDFTSamples"]
