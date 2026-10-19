"""Generic training runtime over Reef's backend-neutral coordinator client.

Native integrations choose their connection and retain their own model code.
This runtime prepares scenario jobs and returns exported checkpoint candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.core.batches import StepScheduling, TrainingBatch, TrajectoryItem, trajectories
from reef.runtime.executor.connection import CoordinatorClient, training_job_status
from reef.runtime.interfaces import (
    CandidateTrainingDeferred,
    ModelCandidate,
    PreparedTrainingStep,
    StaleCandidate,
    TrainingJobResult,
    TrainingMethod,
    TrainingRuntime,
    TrainingRuntimeError,
)


class ExecutorTrainingRuntime(TrainingRuntime):
    """Train and export checkpoints through a supplied worker control connection."""

    def __init__(self, train_group_handle: CoordinatorClient) -> None:
        self._train_group_handle = train_group_handle

    @property
    def train_group_handle(self) -> CoordinatorClient:
        return self._train_group_handle

    @property
    def concurrent_training_scenarios(self) -> bool:
        return self.training_job_status().get("lora_mode") == "scenario"

    def training_job_status(self) -> Mapping[str, Any]:
        return training_job_status(self._train_group_handle)

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
        *,
        parent_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        prepared = self._train_group_handle.prepare_training_step(batch, method, algorithm_state, scheduling)
        if not isinstance(prepared, PreparedTrainingStep):
            raise TrainingRuntimeError(
                f"train group handle returned invalid prepared training step: {type(prepared).__name__}"
            )
        if prepared.action == "skip":
            return prepared
        if prepared.payload is None:
            raise TrainingRuntimeError("non-skip training preparation must carry a payload")
        payload = dict(prepared.payload)
        samples = _ordered_samples(batch, payload.pop("source_rows", None))
        payload.update(self.version_fields(samples, parent_runtime_load_id))
        # The backend numbers its own checkpoints; the job records the scenario step it trains.
        payload["scenario_step"] = scenario_step
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state=prepared.next_algorithm_state,
            metrics=prepared.metrics,
        )

    def version_fields(self, samples: Sequence[TrajectoryItem], parent_runtime_load_id: str | None) -> dict[str, Any]:
        """Return the versions of one job: the producing versions of each sample, and the parent version.

        The parent version is the version that the job trains from. These
        fields are training metadata, not a check. The record-to-record stage
        dropped stale samples before the batch formed.
        """
        spans = [
            [
                {"start": span.start, "end": span.end, "runtime_load_id": span.runtime_load_id}
                for span in parse_runtime_load_spans(sample.training.get("runtime_load_spans", []))
            ]
            for sample in samples
        ]
        fields: dict[str, Any] = {
            "producing_runtime_load_ids": [sample.training.get("runtime_load_id", None) for sample in samples]
        }
        if any(spans):
            fields["producing_runtime_load_spans"] = spans
        if parent_runtime_load_id is not None:
            fields["parent_runtime_load_id"] = parent_runtime_load_id
        return fields

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        return self._validated_result(self._train_group_handle.execute_training_job(payload))

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        checkpoint = self.execute_training_job(payload)
        if checkpoint.outcome == "storage_blocked":
            if not isinstance(checkpoint.storage, Mapping):
                raise TrainingRuntimeError("training runtime returned invalid checkpoint storage status")
            raise CandidateTrainingDeferred(checkpoint.storage)
        if checkpoint.outcome == "stale":
            raise StaleCandidate(checkpoint.metrics)
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("candidate training must stop after exporting a checkpoint")
        return ModelCandidate(
            candidate_id=checkpoint.training_job_id,
            training_job_id=checkpoint.training_job_id,
            checkpoint_path=checkpoint.checkpoint_path,
            current_runtime_load_id=None,
            training_metrics=dict(checkpoint.metrics or {}),
        )

    def reject_candidate(self, candidate: ModelCandidate) -> None:
        self.reject_training_job(candidate.training_job_id)

    def reject_training_job(self, training_job_id: str) -> None:
        self._train_group_handle.reject_training_candidate(training_job_id)

    def shutdown(self) -> None:
        self._train_group_handle.shutdown()

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result


def _ordered_samples(batch: TrainingBatch, source_rows: Any) -> tuple[TrajectoryItem, ...]:
    """The batch's policy samples in the order the prepared payload's rows use them."""
    samples = trajectories(batch)
    if source_rows is not None:
        # Wire rows follow the step schedule (epochs repeat rows, shuffle
        # reorders rollouts); producing versions and timestamps must follow
        # that same order.
        try:
            samples = tuple(samples[row] for row in source_rows)
        except (IndexError, TypeError) as exc:
            raise TrainingRuntimeError(f"prepared payload names invalid source rows: {exc}") from exc
    if not samples:
        raise TrainingRuntimeError("a training job requires at least one policy sample")
    return samples
