"""OpenClaw-RL (arXiv:2603.10165): one method, one package.

- ``recipe`` — the OpenClaw-RL recipe class and its ``WeightTrainingSpec``.
- ``processor`` — the computed-feedback processor: sessions trace-matched
  from records, turns judged on a processor-private PRM worker.
- ``sessions`` / ``turns`` / ``prm`` — the derivation machinery the
  processor calls: how a session is recognized, what one turn's judgment
  gets and answers, what one PRM call does.
- ``preparer`` — the backend-agnostic step preparer.
- ``slime`` — the Slime loss family, its torch objective, and the frozen
  Megatron teacher. Imported by the training driver and workers only; this
  package's public surface never loads it.
"""

from recipes.openclawrl.preparer import OpenClawRLPreparer
from recipes.openclawrl.processor import OpenClawRLProcessor
from recipes.openclawrl.recipe import OpenClawRLRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("openclawrl", "recipes.openclawrl.slime:OpenclawrlAlgorithm")

__all__ = ["OpenClawRLPreparer", "OpenClawRLProcessor", "OpenClawRLRecipe"]
