"""OpenClaw-RL next-state binary reward recipe."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.openclawrl.judging import OpenClawRLRecord2Record
from recipes.openclawrl.processor import OpenClawRLBatches, OpenClawRLTasks
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class OpenClawRLRecipe(WeightTrainingRecipe):
    """Reproduces upstream OpenClaw-RL's binary-RL configuration on reef.

    The whole method runs server-side. The record stage
    (``OpenClawRLRecord2Record``) correlates main-turn records into sessions:
    it prefers a harness-supplied session tag and falls back to trace
    matching. It judges each turn by its next state against the PRM's SGLang
    server. ``OpenClawRLTasks`` and ``OpenClawRLBatches`` batch the judged
    turns directly, with no external grader. The training objective then
    applies the verbatim upstream top-K select objective.

    The deployment's ``pipeline`` section configures the stages: the
    sessions and the PRM judge in ``record2record``, and the PRM record file
    in ``task2batch``. Without ``prm_url`` the recipe is correlate-only:
    sessions resolve, nothing trains, and records stay compactable. A batch
    holds the Slime driver's ``--global-batch-size`` of judged turns (each
    turn-sample is its own rollout). Teacher
    scoring is on the trainer: the Megatron teacher pass computes exact
    log-probs on the trainer's own tempered scale.
    """

    name: str = "openclawrl"

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        # The paper objective is the verbatim upstream top-K select loss
        # (recipes/openclawrl/slime), not the plain pg surrogate.
        return WeightTrainingSpec(
            objective="openclawrl",
            # Each turn-sample is its own rollout; the backend's configured step size applies.
            scheduling=StepScheduling(unit="sample"),
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(
            record2record=OpenClawRLRecord2Record, record2task=OpenClawRLTasks, task2batch=OpenClawRLBatches
        )
