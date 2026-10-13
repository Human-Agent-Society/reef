"""OpenClaw-RL (arXiv:2603.10165): one method, one package.

- ``recipe`` — the OpenClaw-RL recipe class, its ``WeightTrainingSpec``, and
  its ``PipelineSpec``.
- ``judging`` — the record-to-record stage ``OpenClawRLRecord2Record`` and
  its correlation and judge: sessions correlated from record tags or, as a
  fallback, matching traces; turns judged on the stage's PRM worker.
- ``processor`` — the record-to-task and task-to-batch stages: judged turns
  into samples and batches, and the PRM record file.
- ``sessions`` / ``turns`` / ``prm`` — the derivation machinery the
  judging calls: how a session is recognized, what one turn's judgment
  gets and answers, what one PRM call does.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family, its torch objective, and the frozen
  Megatron teacher. Imported by the training driver and workers only; this
  package's public surface never loads it.
"""

from recipes.openclawrl.judging import OpenClawRLJudgingSettings, OpenClawRLRecord2Record
from recipes.openclawrl.objective import OpenClawRLObjective
from recipes.openclawrl.processor import OpenClawRLBatches, OpenClawRLBatchSettings, OpenClawRLTasks
from recipes.openclawrl.recipe import OpenClawRLRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("openclawrl", "recipes.openclawrl.slime:OpenclawrlAlgorithm")

__all__ = [
    "OpenClawRLBatchSettings",
    "OpenClawRLBatches",
    "OpenClawRLJudgingSettings",
    "OpenClawRLObjective",
    "OpenClawRLRecipe",
    "OpenClawRLRecord2Record",
    "OpenClawRLTasks",
]
