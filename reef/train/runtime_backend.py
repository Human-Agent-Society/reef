"""Coordinate separate training and inference through the existing training lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any

from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.runtime.base import (
    InferenceRuntime,
    PreparedTrainingStep,
    RuntimeContractError,
    TrainingJobResult,
    TrainingRuntime,
)
from reef.runtime.candidates import ActivatedModel, CandidateTrainingDeferred, ModelCandidate, StaleCandidate
from reef.runtime.training_group import TrainingRuntimeError
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.types import TrainingBatch, TrainStepResult


class RuntimeTrainingBackend(TrainingBackend):
    """Turn the training runtime protocol into Reef's common backend lifecycle."""

    def __init__(
        self,
        training_runtime: TrainingRuntime,
        step_preparer: str,
        *,
        inference_runtime: InferenceRuntime,
        loss_family: str | None = None,
        scenario: str | None = None,
    ) -> None:
        if not step_preparer:
            raise ValueError("step_preparer must be non-empty")
        self.training_runtime = training_runtime
        self.inference_runtime = inference_runtime
        self._step_preparer = step_preparer
        self._loss_family = loss_family
        self._scenario = scenario
        status = training_runtime.training_job_status()
        self._colocated = bool(status and status.get("colocate"))
        if status is None:
            inference_runtime.mark_published()
        else:
            self._sync_inference_admission(status)

    @property
    def step_preparer(self) -> str:
        return self._step_preparer

    @property
    def dispatched(self) -> bool:
        return True

    def experiment_config(self) -> Mapping[str, Any]:
        return {
            "runtime": type(self.training_runtime).__name__,
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
        if committed_training_job_id is not None and (
            not isinstance(committed_training_job_id, str) or not committed_training_job_id
        ):
            raise TrainingRuntimeError("committed_training_job_id must be a non-empty string or None")
        if not isinstance(committed_training_without_job_id, bool):
            raise TrainingRuntimeError("committed_training_without_job_id must be a boolean")
        scenario = self._scenario if self.training_runtime.concurrent_training_scenarios else None
        training_job = self.training_runtime.training_job_status()
        if training_job is None:
            if committed_training_job_id is not None:
                self._finish_committed_training_job(committed_training_job_id)
            return
        status = training_job["status"]
        if self._sync_inference_admission(training_job):
            return
        job_scenario = training_job.get("scenario")
        if scenario is not None and isinstance(job_scenario, str) and job_scenario != scenario:
            # The pending job belongs to another scenario sharing this
            # runtime; admission is engine-global and already synced above,
            # but its commit handshake is that scenario's to finish.
            return
        if status == "REJECTING":
            training_job_id = training_job.get("training_job_id")
            if not isinstance(training_job_id, str) or not training_job_id:
                raise TrainingRuntimeError("rejecting training job is missing its durable identity")
            self.training_runtime.reject_training_job(training_job_id)
            self.inference_runtime.resume_admission()
            return
        if status not in {"UPDATING_WEIGHTS", "READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            return
        rollout_id = training_job.get("rollout_id")
        training_job_id = training_job.get("training_job_id")
        if (
            not isinstance(rollout_id, int)
            or isinstance(rollout_id, bool)
            or not isinstance(training_job_id, str)
            or not training_job_id
        ):
            raise TrainingRuntimeError("training-job status is missing its durable identity")
        if status == "UPDATING_WEIGHTS":
            self.inference_runtime.resume_weight_update(training_job_id)
        if (
            status == "COMPLETE"
            and training_job.get("commit_acknowledged") is not True
            and scenario_step == rollout_id + 1
            and committed_training_job_id is None
            and committed_training_without_job_id
        ):
            # Older bridges resumed before Reef committed and could not write
            # their job identity into the old commit schema. The exact
            # next-step training record is the strongest durable migration
            # proof available; rollback/non-training commits are excluded.
            self._finish_committed_training_job(training_job_id)
            return
        if scenario_step > rollout_id and committed_training_job_id == training_job_id:
            self._finish_committed_training_job(training_job_id)

    def acknowledge_commit(self, scenario_step: int, training_job_id: str) -> None:
        self.recover_pending_step(scenario_step, committed_training_job_id=training_job_id)

    def _finish_committed_training_job(self, training_job_id: str) -> None:
        self.inference_runtime.acknowledge_publication(training_job_id)
        self.inference_runtime.mark_published()
        self.inference_runtime.resume_admission()

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        prepared = self.prepare_training_step(
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
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedTrainingStep:
        return self.training_runtime.prepare_training_step(
            batch,
            step_preparer,
            algorithm_state,
            scenario_step,
            serving_runtime_load_id=(
                self.inference_runtime.serving_runtime_load_id()
                if self.training_runtime.max_staleness > 0
                else self.inference_runtime.current_runtime_load_id()
            ),
        )

    def execute_training_job(
        self,
        payload: Mapping[str, Any],
    ) -> TrainingJobResult:
        if self._colocated:
            # New requests wait without occupying a service worker and bind
            # the head committed below. Already-admitted requests stay inside
            # SGLang: the colocated bridge retracts their KV before handing
            # the GPUs to Megatron, then SGLang re-prefills them after resume.
            self.inference_runtime.pause_admission()

        try:
            checkpoint = self._validated_result(self.training_runtime.execute_training_job(payload))
        except BaseException:
            # A colocated pause may have succeeded before the backend rejected
            # the job. Reopen only when the durable status proves that no
            # training or checkpoint work started.
            job_state = None
            if self._colocated:
                with suppress(Exception):
                    job_state = (self.training_runtime.training_job_status() or {}).get("status")
            if job_state == "IDLE":
                self.inference_runtime.resume_admission()
            raise
        if checkpoint.outcome in {"stale", "storage_blocked"}:
            if self._colocated:
                self.inference_runtime.resume_admission()
            return checkpoint
        if checkpoint.outcome == "complete":
            if (self.training_runtime.training_job_status() or {}).get("commit_acknowledged") is True:
                self.inference_runtime.resume_admission()
            else:
                self.inference_runtime.pause_admission()
            return checkpoint
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("deferred weight updates require a checkpoint with a training_job_id")

        if not self._colocated:
            # Training and checkpointing may overlap inference on disjoint
            # GPUs. Close admission only for the short serving-weight update.
            self.inference_runtime.pause_admission()
        updated = self.inference_runtime.resume_weight_update(checkpoint.training_job_id)
        return TrainingJobResult(
            "complete",
            updated.runtime_load_id,
            checkpoint.checkpoint_path,
            metrics=checkpoint.metrics,
            training_job_id=checkpoint.training_job_id,
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        current = self.inference_runtime.current_runtime_load_id()
        if self._colocated:
            self.inference_runtime.pause_admission()
        try:
            candidate = self.training_runtime.train_candidate(payload)
        except (CandidateTrainingDeferred, StaleCandidate):
            if self._colocated:
                self.inference_runtime.resume_admission()
            raise
        except BaseException:
            status = None
            if self._colocated:
                with suppress(Exception):
                    status = self.training_runtime.training_job_status()
            if status is not None and status.get("status") == "IDLE":
                self.inference_runtime.resume_admission()
            raise
        if not isinstance(candidate, ModelCandidate):
            raise RuntimeContractError("training runtime must return ModelCandidate")
        return replace(candidate, current_runtime_load_id=current)

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        self.inference_runtime.pause_admission()
        return self.inference_runtime.activate_candidate(candidate)

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        self.training_runtime.reject_candidate(candidate, decision)
        self.inference_runtime.resume_admission()

    def _sync_inference_admission(self, training_job: Mapping[str, Any]) -> bool:
        """Apply states that need no weight-update recovery; return if settled."""
        status = training_job["status"]
        if status in {"IDLE", "REJECTED"} or (
            status == "COMPLETE" and training_job.get("commit_acknowledged") is True
        ):
            self.inference_runtime.mark_published()
            self.inference_runtime.resume_admission()
            return True
        if status in {"RUNNING", "CHECKPOINT"}:
            if self._colocated:
                self.inference_runtime.pause_admission()
            else:
                if self.inference_runtime.current_runtime_load_id() is None:
                    self.inference_runtime.mark_published()
                self.inference_runtime.resume_admission()
            return True
        self.inference_runtime.pause_admission()
        return False

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result

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


__all__ = ["RuntimeTrainingBackend"]
