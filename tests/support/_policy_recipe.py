"""Test-only single-sample policy recipe.

Keeps its score bar unbounded and accepts multi-turn samples, so persistence
and concurrency tests can make a batch ready after one report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any

from tests.support._threshold_processor import threshold_pipeline

from reef.observability import ExperimentLogger
from reef.recipe.base import WeightTrainingRecipe
from reef.runtime.interfaces import InferenceRuntime
from reef.runtime.jobs import TrainingComponent
from reef.storage.records import RecordStore
from reef.storage.tasks import TaskStorage
from reef.train.jobs import FixedJobPlan, JobLoop, JobRequest
from reef.train.trainer import Trainer


@dataclass(frozen=True)
class TestPolicyRecipe(WeightTrainingRecipe):
    __test__ = False

    _: KW_ONLY
    min_score: float = float("-inf")
    #: The staleness limit of the record stage.
    max_staleness: int = 0
    name: str = "test_policy"

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config: Mapping[str, Any] | None = None,
        runtime: InferenceRuntime | None = None,
        training_component: TrainingComponent | None = None,
    ) -> TestPolicyRecipe:
        del environ, config
        if not isinstance(training_component, TrainingComponent):
            raise TypeError("TestPolicyRecipe requires a TrainingComponent")
        return cls(training_component, runtime=runtime)

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
    ) -> Trainer:
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: threshold_pipeline(
                context, accept_multi_turn=True, max_staleness=self.max_staleness
            ),
            job_loop=JobLoop(
                self.training_component,
                FixedJobPlan(JobRequest({"method": {"objective": "sft"}})),
                scenario=scenario,
            ),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            default_job_samples=1,
        )
