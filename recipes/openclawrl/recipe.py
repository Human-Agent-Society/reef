"""OpenClaw-RL next-state binary reward recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.openclawrl.processor import OpenClawRLJobData, OpenClawRLSamples
from reef.recipe.base import WeightTrainingRecipe
from reef.storage.samples import SAMPLE_KINDS, SampleKind
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class OpenClawRLRecipe(WeightTrainingRecipe):
    """Reproduces upstream OpenClaw-RL's binary-RL configuration on reef.

    The whole method runs server-side. ``OpenClawRLSamples`` correlates
    main-turn records into sessions: it prefers a harness-supplied session tag
    and falls back to trace matching. It judges each turn by its next state
    against the PRM's SGLang server. ``OpenClawRLSamples`` and
    ``OpenClawRLJobData`` batch the judged turns directly, with no external
    grader. The training objective then
    applies the verbatim upstream top-K select objective.

    The deployment's ``pipeline`` section configures the stages: the
    sessions and the PRM judge in ``record2sample``, and the PRM record file
    in ``sample2job``. Without ``prm_url`` the recipe is correlate-only:
    sessions resolve, nothing trains, and records stay compactable. A batch
    holds Slime's ``--global-batch-size`` of judged turns (each
    turn-sample is its own rollout). Teacher
    scoring is on the trainer: the Megatron teacher pass computes exact
    log-probs on the trainer's own tempered scale.
    """

    name: str = "openclawrl"

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {kind: {"objective": "openclawrl"} for kind in SAMPLE_KINDS}

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=OpenClawRLSamples, sample2job=OpenClawRLJobData)
