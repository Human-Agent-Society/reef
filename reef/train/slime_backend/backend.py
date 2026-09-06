"""Slime adapter for Reef's common TrainingBackend lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.base import RuntimeContractError, TrainingRuntime
from reef.runtime.candidates import CandidateTrainingDeferred, ModelCandidate, StaleCandidate
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.evaluation.contracts import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.types import TrainingBatch, TrainStepResult


class SlimeTrainingBackend(TrainingBackend):
    """Turn Slime's runtime protocol into Reef's common backend lifecycle."""

    def __init__(
        self,
        runtime: TrainingRuntime,
        step_preparer: str,
        *,
        loss_family: str | None = None,
        scenario: str | None = None,
    ) -> None:
        if not step_preparer:
            raise ValueError("step_preparer must be non-empty")
        self._runtime = runtime
        self._step_preparer = step_preparer
        self._loss_family = loss_family
        self._scenario = scenario

    @property
    def runtime(self) -> TrainingRuntime:
        return self._runtime

    @property
    def step_preparer(self) -> str:
        return self._step_preparer

    @property
    def dispatched(self) -> bool:
        return True

    def experiment_config(self) -> Mapping[str, Any]:
        return {
            "runtime": "slime",
            "step_preparer": self._step_preparer,
            **({"loss_family": self._loss_family} if self._loss_family is not None else {}),
        }

    def initial_state(self) -> Mapping[str, Any]:
        return {}

    def recover_pending_step(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
    ) -> None:
        scenario = self._runtime_scenario()
        if scenario is None:
            self._runtime.reconcile_training_job(
                scenario_step,
                committed_training_job_id=committed_training_job_id,
                committed_training_without_job_id=committed_training_without_job_id,
            )
            return
        self._runtime.reconcile_training_job(
            scenario_step,
            committed_training_job_id=committed_training_job_id,
            committed_training_without_job_id=committed_training_without_job_id,
            scenario=scenario,
        )

    def acknowledge_commit(self, scenario_step: int, training_job_id: str) -> None:
        """Tell Slime that Reef durably committed the selected training job."""
        scenario = self._runtime_scenario()
        if scenario is None:
            self._runtime.reconcile_training_job(scenario_step, committed_training_job_id=training_job_id)
            return
        self._runtime.reconcile_training_job(
            scenario_step,
            committed_training_job_id=training_job_id,
            scenario=scenario,
        )

    def _runtime_scenario(self) -> str | None:
        """Name the scenario only to a runtime that trains several at once.

        Single-scenario runtimes (including user-written ones) keep the
        original ``reconcile_training_job`` signature.
        """
        if self._scenario is not None and self._runtime.concurrent_training_scenarios:
            return self._scenario
        return None

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        prepared = self._runtime.prepare_training_step(
            batch,
            self._step_preparer,
            state,
            scenario_step,
        )
        next_state = dict(prepared.next_algorithm_state)
        metrics = dict(prepared.metrics)
        if prepared.action == "skip":
            return PreparedStep.skipped(state=next_state, metrics=metrics)
        if prepared.payload is None:
            raise RuntimeContractError("training runtime prepared a train step without a payload")
        runtime = self._runtime
        payload = dict(prepared.payload)
        if self._scenario is not None and runtime.concurrent_training_scenarios:
            # A runtime that trains several scenarios' adapters needs to know
            # whose slot this job fills; the job identity then includes it.
            payload["scenario"] = self._scenario
        try:
            candidate = runtime.train_candidate(payload)
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
            self._runtime.reject_candidate(candidate, decision)
            return TrainStepResult(
                state=prepared.state,
                metrics=metrics,
                source_runtime_load_id=candidate.current_runtime_load_id,
            )
        activated = self._runtime.activate_candidate(candidate)
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
        self._runtime.reject_candidate(
            candidate,
            SelectionDecision(
                "reject",
                "reef_abort",
                "1",
                "candidate processing failed before settlement",
                evaluation,
            ),
        )

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


__all__ = ["SlimeTrainingBackend"]
