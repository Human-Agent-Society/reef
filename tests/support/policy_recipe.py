"""Test-only single-sample policy recipe.

Keeps its score bar unbounded and accepts multi-turn samples, so persistence
and concurrency tests can make a batch ready after one report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any

from tests.support.threshold_processor import ThresholdJobData, ThresholdSamples

from reef.recipe.base import WeightTrainingRecipe
from reef.storage.samples import SAMPLE_KINDS, SampleKind
from reef.train.pipeline import PipelineSettings, PipelineSpec


@dataclass(frozen=True)
class TestPolicyRecipe(WeightTrainingRecipe):
    __test__ = False

    _: KW_ONLY
    min_score: float = float("-inf")
    #: The artifacts that a sample of the record stage may lag behind the served one.
    max_artifact_lag: int = 0
    name: str = "test_policy"

    def __post_init__(self) -> None:
        if self.pipeline_stages is None:
            # One report is one job, and a report of several model calls trains as one sample.
            section = {
                "record2sample": {
                    "config": {"accept_multi_turn_policy_samples": True, "max_artifact_lag": self.max_artifact_lag}
                },
                "sample2job": {"config": {"job_samples": 1}},
            }
            stages = PipelineSettings.from_setting(section, spec=self.pipeline_spec(), environ={})
            object.__setattr__(self, "pipeline_stages", stages)
        super().__post_init__()

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(ThresholdSamples, ThresholdJobData)

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {kind: {"objective": "sft"} for kind in SAMPLE_KINDS}
