"""Test-only single-sample policy recipe.

Keeps its score bar unbounded and accepts multi-turn samples, so persistence
and concurrency tests can make a batch ready after one report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any

from reef.observability import ExperimentLogger
from reef.recipe.base import WeightTrainingRecipe
from reef.records import RecordStore
from reef.runtime.base import InferenceRuntime, TrainingRuntime
from reef.train.slime_backend.backend import SlimeTrainingBackend
from reef.train.trainer import Trainer

from ._threshold_processor import ThresholdProcessor


@dataclass(frozen=True)
class TestPolicyRecipe(WeightTrainingRecipe):
    __test__ = False

    _: KW_ONLY
    batch_size: int = 1
    min_score: float = float("-inf")
    name: str = "test_policy"

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> TestPolicyRecipe:
        del environ, config
        if not isinstance(runtime, TrainingRuntime):
            raise TypeError("TestPolicyRecipe requires a TrainingRuntime")
        return cls(runtime)

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        if experiment_logger is not None:
            experiment_logger.log({"batch_size": self.batch_size}, namespace="recipe")
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: ThresholdProcessor(
                context.with_config(
                    {
                        "batch_size": self.batch_size,
                        "accept_multi_turn_policy_samples": True,
                    }
                )
            ),
            training_backend=SlimeTrainingBackend(self.runtime, "sft", scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )
