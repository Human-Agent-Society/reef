"""OpenClaw-RL (arXiv:2603.10165): one method, one package.

- ``recipe`` — the OpenClaw-RL recipe class, its ``WeightTrainingSpec``, and
  its ``PipelineSpec``.
- ``judging`` — the correlation and the judge: sessions correlated from
  record tags or, as a fallback, matching traces; turns judged on a PRM worker.
- ``processor`` — the record-to-sample stage ``OpenClawRLSamples``, which runs
  the judging, and the sample-to-job stage: judged turns into samples and
  batches, and the PRM record file.
- ``sessions`` / ``turns`` / ``prm`` — the derivation machinery the
  judging calls: how a session is recognized, what one turn's judgment
  gets and answers, what one PRM call does.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family, its torch objective, and the frozen
  Megatron teacher. Imported by the Slime training component and workers only; this
  package's public surface never loads it.
"""

from recipes.openclawrl.judging import OpenClawRLJudgingSettings
from recipes.openclawrl.objective import OpenClawRLObjective
from recipes.openclawrl.processor import OpenClawRLJobData, OpenClawRLJobSettings, OpenClawRLSamples
from recipes.openclawrl.recipe import OpenClawRLRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("openclawrl", "recipes.openclawrl.slime:OpenclawrlAlgorithm")

__all__ = [
    "OpenClawRLJobData",
    "OpenClawRLJobSettings",
    "OpenClawRLJudgingSettings",
    "OpenClawRLObjective",
    "OpenClawRLRecipe",
    "OpenClawRLSamples",
]
