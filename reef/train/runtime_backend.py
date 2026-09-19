"""Adapt recipe candidates to Reef's training and inference runtime scheduler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.runtime.interfaces import (
    ActivatedModel,
    CandidateTrainingDeferred,
    InferenceRuntime,
    ModelCandidate,
    PreparedTrainingStep,
    RuntimeContractError,
    StaleCandidate,
    TrainingJobResult,
    TrainingRuntime,
)
from reef.runtime.scheduler import RuntimeScheduler
from reef.train.algos import StepScheduling
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.types import TrainingBatch, TrainStepResult


class RuntimeCandidateBackend(CandidateBackend):
    """Map candidate evaluation and selection onto the runtime scheduler."""

    def __init__(
        self,
        training_runtime: TrainingRuntime,
        objective: str,
        scheduling: StepScheduling,
        *,
        inference_runtime: InferenceRuntime,
        loss_family: str | None = None,
        scenario: str | None = None,
    ) -> None:
        if not objective:
            raise ValueError("objective must be non-empty")
        if not isinstance(scheduling, StepScheduling):
            raise TypeError(f"scheduling must be a StepScheduling, got {type(scheduling).__name__}")
        self.scheduler = RuntimeScheduler(training_runtime, inference_runtime)
        self.objective = objective
        self.scheduling = scheduling
        self._loss_family = loss_family
        self._scenario = scenario

    @property
    def training_runtime(self) -> TrainingRuntime:
        return self.scheduler.training_runtime

    @property
    def inference_runtime(self) -> InferenceRuntime:
        return self.scheduler.inference_runtime

    @property
    def dispatched(self) -> bool:
        return True

    def experiment_config(self) -> Mapping[str, Any]:
        return {
            "runtime": type(self.training_runtime).__name__,
            "objective": self.objective,
            "scheduling": asdict(self.scheduling),
            **({"loss_family": self._loss_family} if self._loss_family is not None else {}),
        }

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {f"runtime/{key}": value for key, value in self.scheduler.operations.snapshot().items()}

    def initial_state(self) -> Mapping[str, Any]:
        return {}

    def recover_pending_step(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
    ) -> None:
        self.scheduler.recover_pending_step(
            scenario_step,
            scenario=self._scenario,
            committed_training_job_id=committed_training_job_id,
            committed_training_without_job_id=committed_training_without_job_id,
        )

    def acknowledge_commit(self, scenario_step: int, training_job_id: str) -> None:
        self.scheduler.acknowledge_commit(scenario_step, training_job_id, scenario=self._scenario)

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        prepared = self.prepare_training_step(batch, self.objective, state, self.scheduling, scenario_step)
        next_state = dict(prepared.next_algorithm_state)
        metrics = dict(prepared.metrics)
        if prepared.action == "skip":
            return PreparedStep.skipped(state=next_state, metrics=metrics)
        if prepared.payload is None:
            raise RuntimeContractError("training runtime prepared a train step without a payload")
        runtime = self.training_runtime
        payload = dict(prepared.payload)
        if self._scenario is not None and runtime.concurrent_training_scenarios:
            # A runtime that trains several scenarios' adapters needs to know
            # whose slot this job fills; the job identity then includes it.
            payload["scenario"] = self._scenario
        try:
            candidate = self.train_candidate(payload)
        except CandidateTrainingDeferred as blocked:
            return PreparedStep.retrying(state=state, metrics=metrics, storage=blocked.storage)
        except StaleCandidate as stale:
            return PreparedStep.dropped(state=state, metrics={**metrics, **stale.metrics})
        if not isinstance(candidate, ModelCandidate):
            raise RuntimeContractError(f"{type(runtime).__name__}.train_candidate must return ModelCandidate")
        return PreparedStep.with_candidate(candidate, state=next_state, metrics=metrics)

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        """Default to training telemetry when no checkpoint plugin is configured."""
        model = self._model_candidate(candidate)
        return EvaluationResult(
            evaluator="training_runtime",
            evaluator_version="1",
            metrics=dict(model.training_metrics),
        )

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        candidate = self._prepared_candidate(prepared)
        metrics = {
            **candidate.training_metrics,
            **prepared.metrics,
            **decision.metrics,
            "selected": decision.selected,
            "selection": {"candidate_id": candidate.candidate_id, **decision.to_dict()},
        }
        if not decision.selected:
            self.reject_candidate(candidate, decision)
            return TrainStepResult(
                state=prepared.state,
                metrics=metrics,
                source_runtime_load_id=candidate.current_runtime_load_id,
            )
        activated = self.activate_candidate(candidate)
        if activated.candidate_id != candidate.candidate_id:
            raise RuntimeContractError("training runtime activated a different candidate")
        return TrainStepResult(
            state=prepared.state,
            metrics=metrics,
            runtime_load_id=activated.runtime_load_id,
            checkpoint_path=candidate.checkpoint_path,
            training_job_id=candidate.training_job_id,
            source_runtime_load_id=candidate.current_runtime_load_id,
        )

    def abort_step(self, prepared: PreparedStep) -> None:
        candidate = self._prepared_candidate(prepared)
        evaluation = EvaluationResult("reef_abort", "1", {})
        self.reject_candidate(
            candidate,
            SelectionDecision(
                "reject",
                "reef_abort",
                "1",
                "candidate processing failed before settlement",
                evaluation,
            ),
        )

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
    ) -> PreparedTrainingStep:
        return self.scheduler.prepare_training_step(batch, objective, algorithm_state, scheduling, scenario_step)

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        return self.scheduler.execute_training_job(payload)

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        return self.scheduler.train_candidate(payload)

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        return self.scheduler.activate_candidate(candidate)

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        self.scheduler.reject_candidate(candidate, decision)

    @staticmethod
    def _model_candidate(candidate: UpdateCandidate) -> ModelCandidate:
        if not isinstance(candidate, ModelCandidate):
            raise TypeError(f"runtime training requires ModelCandidate, got {type(candidate).__name__}")
        return candidate

    @classmethod
    def _prepared_candidate(cls, prepared: PreparedStep) -> ModelCandidate:
        candidate = prepared.candidate
        if candidate is None:
            raise TypeError("runtime settlement requires a candidate step")
        return cls._model_candidate(candidate)


__all__ = ["RuntimeCandidateBackend"]
