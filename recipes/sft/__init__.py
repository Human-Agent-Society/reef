"""Supervised fine-tuning on demonstrations: one method, one package.

- ``recipe`` — the SFT recipe class and its ``WeightTrainingSpec``. Its report
  contract is the shared :class:`reef.core.reports.TeacherContextReport`: a
  rollout's receipt and the demonstration as ``context``, the report the
  self-distillation recipes take, so the methods train from identical feedback.
- ``processor`` — one report, one supervised sample: the demonstration as the
  assistant turn of the recorded request, in the served model's chat template.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family: the stock ``sft_loss`` over the sample's
  response tokens. Imported by the training driver and workers only.
"""

from recipes.sft.objective import SftObjective
from recipes.sft.processor import SFTProcessor
from recipes.sft.recipe import SFTRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sft", "recipes.sft.slime:SftAlgorithm")

__all__ = ["SFTProcessor", "SFTRecipe", "SftObjective"]
