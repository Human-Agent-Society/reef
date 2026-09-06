"""Scenario aggregate and its durable recovery metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.artifact.repository import Repository
from reef.core.reports import ReportBase
from reef.records import RecordStore
from reef.runtime.base import InferenceRuntime
from reef.runtime.inference import InferenceBackend
from reef.scenario.binding import ScenarioBinding
from reef.scenario.checkpoint_strategy import CheckpointStrategy
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.scenario.commit_protocol import ScenarioCommitProtocol
from reef.scenario.snapshot import SCENARIO_SNAPSHOT_METADATA_KEY, snapshot_metadata_for
from reef.surface.base import Surface
from reef.train.backend import StepExecution
from reef.train.trainer import Trainer
from reef.train.types import TrainingBatch, TrainStepResult


class Scenario:
    """Scenario aggregate owning one runtime binding and release chain."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        repository: Repository,
        checkpoint_strategy: CheckpointStrategy,
        records: RecordStore,
        trainer: Trainer,
        scenario_step: int = 0,
        process_id: str | None = None,
        commit_log: CommitLog | None = None,
        recovered_head_record: CommitRecord | None = None,
    ) -> None:
        self._name = name
        self._binding = binding
        self._surface = binding.surface
        self._records = records
        self._trainer = trainer
        self._artifact_chain = ArtifactReleaseChain(repository, process_id=process_id)
        self._commit_protocol = ScenarioCommitProtocol(
            name=name,
            binding=binding,
            artifacts=self._artifact_chain,
            checkpoint_strategy=checkpoint_strategy,
            trainer=trainer,
            scenario_step=scenario_step,
            commit_log=commit_log,
            recovered_head_record=recovered_head_record,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def runtime(self) -> InferenceRuntime | None:
        """Inference or training runtime bound to this scenario."""
        return self._binding.runtime

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The recipe's declared report contract, enforced at ingress when set."""
        return self._binding.report_type

    @property
    def inference_backend(self) -> InferenceBackend | None:
        return self._binding.inference_backend

    @property
    def repository(self) -> Repository:
        """The release chain's scenario-scoped artifact repository.

        The public read path to artifact heads (base, current, checkpoint);
        mutation goes through Scenario methods so it stays serialized with
        rollback and commit.
        """
        return self._artifact_chain.repository

    @property
    def records(self) -> RecordStore:
        return self._records

    @property
    def trainer(self) -> Trainer:
        """Inspection-only view of the trainer.

        Every mutating path goes through a Scenario method so it is serialized
        against rollback and commit by the commit protocol lock. Reading state
        that is not part of a transaction (step-preparer identity, consumption
        watermarks, processor schema) is safe here; do not reserve batches,
        replace results, or compact through this handle.
        """
        return self._trainer

    @property
    def scenario_step(self) -> int:
        return self._commit_protocol.step

    @property
    def surface(self) -> Surface:
        """The serving surface built for this scenario."""
        return self._surface

    def prepare_training_step(self) -> TrainStepResult | None:
        """Prepare one local-backend step while excluding rollback and commit."""
        with self._commit_protocol.lock:
            return self._trainer.run_once(self.scenario_step)

    def reserve_training_batch(self) -> TrainingBatch | None:
        """Reserve one backend-training batch while excluding rollback and commit."""
        with self._commit_protocol.lock:
            return self._trainer.reserve_training_batch()

    def execute_reserved_training_step(self) -> StepExecution:
        """Run the bound dispatched backend for the reserved batch."""
        return self._trainer.execute_reserved_step(self.scenario_step)

    def reject_pending(self, metrics: Mapping[str, Any] | None = None) -> None:
        """Drop the reserved batch and durably compact whatever it released."""
        with self._commit_protocol.lock:
            self._trainer.reject_pending(metrics)

    def reingest(self, *, up_to_sequence: int, consumed_ids: frozenset[str]) -> None:
        """Rebuild processor memory from retained rows behind a recovered watermark."""
        with self._commit_protocol.lock:
            self._trainer.reingest(up_to_sequence=up_to_sequence, consumed_ids=consumed_ids)

    def restore_record_progress(self, *, after_sequence: int, offset: int) -> None:
        """Resume record consumption at a recovered commit's high-water mark."""
        with self._commit_protocol.lock:
            self._trainer.restore_record_progress(after_sequence=after_sequence, offset=offset)

    @property
    def commit_log(self) -> CommitLog | None:
        """The commit protocol's durable journal, for read-only inspection."""
        return self._commit_protocol.commit_log

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """The non-blocking step and durable training-outcome snapshot."""
        return self._commit_protocol.commit_status

    @property
    def committed_training_job_id(self) -> str | None:
        """Training-job identity proven by the current durable commit."""
        with self._commit_protocol.lock:
            commit_log = self._commit_protocol.commit_log
            records = () if commit_log is None else commit_log.records()
            if not records or records[-1].step != self.scenario_step:
                return None
            return records[-1].training_job_id

    @property
    def committed_training_without_job_id(self) -> bool:
        """Whether the current head is a pre-identity training commit."""
        with self._commit_protocol.lock:
            commit_log = self._commit_protocol.commit_log
            records = () if commit_log is None else commit_log.records()
            if not records or records[-1].step != self.scenario_step:
                return False
            record = records[-1]
            return record.operation == "training" and record.operation_verified and record.training_job_id is None

    def metrics_for_version(self, release_id: str) -> Mapping[str, Any] | None:
        """Metrics of the training step that published ``release_id``, if logged."""
        return self._commit_protocol.metrics_for_version(release_id)

    def releases(self) -> tuple[dict[str, Any], ...]:
        return self._commit_protocol.releases()

    def artifact_for_version(self, release_id: str) -> Artifact:
        """Materialize a catalog version for read-only serving; absence raises ArtifactNotFound."""
        return self._commit_protocol.artifact_for_version(release_id)

    def entries_for_version(self, release_id: str) -> tuple[Mapping[str, Any], ...] | None:
        """The composition entries behind a catalog version, if its training commit logged them."""
        return self._commit_protocol.entries_for_version(release_id)

    def artifact_snapshot(
        self,
        release_id: str | None = None,
    ) -> tuple[Artifact, Mapping[str, Any] | None]:
        """Freeze one artifact and its gate metrics without waiting for preparation."""
        return self._commit_protocol.artifact_snapshot(release_id)

    def current_artifact_ref(self) -> ArtifactRef:
        return self._artifact_chain.current

    def rollback(self, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        return self._commit_protocol.rollback(release_id, operation=operation)

    def commit(self, result: TrainStepResult) -> Any:
        return self._commit_protocol.commit(result)

    def close(self) -> None:
        """Tear down what this scenario instance owns: trainer, then records.

        The dispatcher calls this on shutdown and when a durable reload
        replaces the instance — the one guarantee processors with background
        workers rely on (see :meth:`DataProcessor.close`). Safe to call more
        than once; the trainer closes first so no processor thread can touch
        the record store after it closes.
        """
        self._trainer.close()
        self._records.close()

    def to_snapshot_metadata(self) -> dict[str, object]:
        return snapshot_metadata_for(
            name=self.name,
            base_artifact=self.repository.base_artifact,
            scenario_step=self.scenario_step,
        )


__all__ = [
    "SCENARIO_SNAPSHOT_METADATA_KEY",
    "ReleaseNotRestorable",
    "Scenario",
]
