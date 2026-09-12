"""Scenario committer coordinating training and artifact publication.

Artifact storage and head movement live in ``reef.artifact``; the durable
record values and checkpoint metadata formats live in ``commits``. This
module owns the scenario-specific ordering across trainer state, store
settlement, checkpoint policy, surfaces, and artifact
operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactNotFound,
    ArtifactPublicationError,
    ArtifactRef,
    LiveWeightArtifactRef,
)
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.core.errors import ReefError
from reef.recipe.checkpoint_strategy import CheckpointStrategy
from reef.scenario.binding import ScenarioBinding
from reef.scenario.releases import ScenarioReleases
from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord, RecordProgress, scenario_metadata_for
from reef.storage.scenario import ScenarioStore, ScenarioStoreConflict
from reef.surface.base import ArtifactActivator
from reef.train.trainer import Trainer
from reef.train.types import (
    DurableWeightsPublication,
    LiveWeightPublication,
    NoArtifactPublication,
    PreparedCommit,
    SavedArtifactPublication,
    TrainStepResult,
)


@dataclass(frozen=True)
class _ArtifactHeadSync:
    state: Literal["synchronized", "pending", "conflict"]
    release_id: str
    error: str | None = None


class ScenarioCommitter:
    """Order one scenario's commit, rollback, and retry effects; delegate release queries."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        artifacts: ArtifactReleaseChain,
        checkpoint_strategy: CheckpointStrategy,
        trainer: Trainer,
        scenario_step: int = 0,
        store: ScenarioStore,
        recovered_head_record: CommitRecord | None = None,
    ) -> None:
        if not isinstance(scenario_step, int) or scenario_step < 0:
            raise ValueError("scenario_step must be non-negative")
        if store.durable:
            artifacts.repository.require_staged_commit_support()
        self._name = name
        self._binding = binding
        self._artifacts = artifacts
        self._checkpoint_strategy = checkpoint_strategy
        self._trainer = trainer
        self._step = scenario_step
        self._store = store
        self._lock = RLock()
        # Preparation holds the operation lock across proposer calls and
        # evaluation episodes, neither of which changes committed releases.
        # Readers share only the publication lock with commit and rollback.
        # Writers must acquire the operation lock first; readers never take it.
        self._publication_lock = RLock()
        self._releases = ScenarioReleases(
            name=name,
            artifacts=artifacts,
            store=store,
            publication_lock=self._publication_lock,
            scenario_step=scenario_step,
        )
        records = (
            (() if recovered_head_record is None else (recovered_head_record,))
            if not store.durable
            else store.history()
        )
        self._latest_training_record = next(
            (record for record in reversed(records) if record.operation == "training"),
            None,
        )
        self._artifact_head_sync = _ArtifactHeadSync("synchronized", artifacts.checkpoint.release_id)
        self._commit_status = (scenario_step, self._latest_training_record, self._artifact_head_sync)

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def artifacts(self) -> ArtifactReleaseChain:
        return self._artifacts

    @property
    def checkpoint_strategy(self) -> CheckpointStrategy:
        return self._checkpoint_strategy

    @property
    def store(self) -> ScenarioStore:
        return self._store

    @property
    def step(self) -> int:
        return self._step

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """Current step and latest training outcome read without blocking."""
        step, record, head_sync = self._commit_status
        return {
            "scenario_step": step,
            "artifact_head_sync": asdict(head_sync),
            "last_committed_step": (
                None
                if record is None
                else {
                    "step": record.step,
                    "recorded_at": record.recorded_at,
                    "metrics": None if record.metrics is None else deepcopy(record.metrics),
                }
            ),
        }

    def advance_to(self, step: int) -> None:
        if step != self._step + 1:
            raise ValueError(f"scenario step must advance from {self._step} to {self._step + 1}")
        self._step = step
        self._commit_status = (step, self._latest_training_record, self._artifact_head_sync)

    def releases(self) -> tuple[dict[str, Any], ...]:
        with self._publication_lock:
            return self._releases.releases(self._step)

    def rollback(self, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        """Publish a durable copy of an older version as a new fenced commit; promote uses the same path."""
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        release_id = release_id.strip()
        with self._lock, self._publication_lock:
            current_ref = self._artifacts.current
            if current_ref.release_id == release_id:
                return current_ref
            target = self._releases.find_release(release_id)
            if target is None:
                raise ArtifactNotFound(f"scenario {self._name!r} has no release {release_id!r}")
            target_ref, target_checkpoint = target
            if not target_checkpoint or isinstance(target_ref, LiveWeightArtifactRef):
                raise ReleaseNotRestorable(
                    f"scenario {self._name!r} release {release_id!r} has no durable checkpoint bytes"
                )
            if self._trainer.pending_batch is not None:
                raise ReefError("cannot rollback while a training result is pending commit")

            artifacts = self._artifacts
            checkpoint = artifacts.checkpoint
            next_step = self._step + 1
            prepared = self._trainer.prepare_commit(None)
            recorded = self._recorded_operation_retry(prepared, operation, release_id, next_step)
            if recorded is not None:
                recorded = self._store.commit_step(expected_step=self._step, commit=recorded)
                self._reconcile_recorded_artifact(recorded)
                self._settle_trainer_commit(prepared, recorded, next_step)
                return recorded.artifact_ref
            if self._store.durable:
                self._synchronize_checkpoint()
            source = artifacts.resolve(target_ref)
            durable = self._store.durable
            surface = self._binding.surface
            self._binding.artifact_validator.validate(source)
            if surface.loader is not None:
                surface.loader.load(source, self._binding.runtime)
            staged = artifacts.stage(next_step, source, parent=checkpoint)
            try:
                commit_metadata = scenario_metadata_for(
                    name=self._name,
                    base_artifact=artifacts.base,
                    scenario_step=next_step,
                    algorithm_state=prepared.algorithm_state,
                    record_progress=RecordProgress(
                        high_water_sequence=prepared.high_water_sequence,
                        high_water_offset=prepared.high_water_offset,
                        compacted_ids=prepared.compacted_ids,
                        consumed_ids=prepared.consumed_ids,
                    ),
                    metrics=prepared.metrics,
                    training_job_id=prepared.training_job_id,
                    operation=operation,
                    rollback_target_release_id=release_id,
                )
                commit_metadata["rollback"] = {
                    "target_release_id": release_id,
                }
                published_ref = artifacts.publish(
                    staged,
                    expected_parent=checkpoint,
                    metadata={
                        **dict(source.metadata),
                        SCENARIO_METADATA_KEY: commit_metadata,
                    },
                    advance_heads=not durable,
                )
                if isinstance(surface.loader, ArtifactActivator):
                    surface.loader.activate(artifacts.resolve(published_ref), self._binding.runtime, source=source)
                record = self._append_commit_record(
                    step=next_step,
                    artifact_ref=published_ref,
                    checkpoint=True,
                    prepared=prepared,
                    operation=operation,
                    rollback_target_release_id=release_id,
                )
                if durable:
                    self._install_committed_checkpoint(
                        published_ref, expected=current_ref, expected_checkpoint=checkpoint
                    )
            except Exception:
                artifacts.discard(staged)
                raise
            self._settle_trainer_commit(prepared, record, next_step)
            return published_ref

    def commit(self, result: TrainStepResult) -> Any:
        """Commit a pending training result as one atomic version record."""
        with self._lock, self._publication_lock:
            next_step = self._step + 1
            prepared = self._trainer.prepare_commit(result)
            recorded = self._recorded_training_retry(prepared, result, next_step)
            if recorded is not None:
                recorded = self._store.commit_step(expected_step=self._step, commit=recorded)
                self._reconcile_recorded_artifact(recorded)
                self._settle_trainer_commit(prepared, recorded, next_step)
                return result.state

            if self._store.durable:
                self._synchronize_checkpoint()
            publication = result.publication
            if isinstance(publication, DurableWeightsPublication):
                # Checkpoint policy lives here, so a backend that exported
                # weights hands over both options and this is the only place
                # that picks one.
                if self._should_checkpoint(result):
                    publication = SavedArtifactPublication(
                        Artifact.local(
                            Path(publication.checkpoint_path),
                            metadata={"runtime_load_id": publication.runtime_load_id},
                        )
                    )
                else:
                    publication = LiveWeightPublication(publication.runtime_load_id)

            if isinstance(publication, LiveWeightPublication):
                return self._commit_live_weights(result, publication, prepared)
            if isinstance(publication, NoArtifactPublication):
                return self._commit_without_artifact(result, prepared)
            return self._commit_saved_artifact(result, publication, prepared)

    def _should_checkpoint(self, result: TrainStepResult) -> bool:
        return self._checkpoint_strategy.should_checkpoint(self._name, self._step + 1, result)

    def _commit_live_weights(
        self,
        result: TrainStepResult,
        publication: LiveWeightPublication,
        prepared: PreparedCommit,
    ) -> Any:
        artifacts = self._artifacts
        next_step = self._step + 1
        if self._should_checkpoint(result):
            raise ReefError(
                "checkpoint-selected training results must include the artifact returned by execute_training_job"
            )
        if result.pending:
            # Live weights have no durable bytes to promote later, so holding them back is not expressible.
            raise ReefError("a pending release requires a durable checkpoint; this step publishes live weights only")
        head, live_ref = artifacts.prepare_live(step=next_step, runtime_load_id=publication.runtime_load_id)
        if not self._store.durable:
            artifacts.advance(live_ref, expected=head)
        record = self._append_commit_record(
            step=next_step,
            artifact_ref=live_ref,
            checkpoint=False,
            prepared=prepared,
        )
        if self._store.durable:
            artifacts.advance(live_ref, expected=head)
        self._settle_trainer_commit(prepared, record, next_step)
        return result.state

    def _commit_without_artifact(self, result: TrainStepResult, prepared: PreparedCommit) -> Any:
        # No pending check: a pending step carries durable bytes by construction, so it never lands here.
        next_step = self._step + 1
        record = self._append_commit_record(
            step=next_step,
            artifact_ref=self._artifacts.current,
            checkpoint=False,
            prepared=prepared,
        )
        self._settle_trainer_commit(prepared, record, next_step)
        return result.state

    def _commit_saved_artifact(
        self,
        result: TrainStepResult,
        publication: SavedArtifactPublication,
        prepared: PreparedCommit,
    ) -> Any:
        artifacts = self._artifacts
        next_step = self._step + 1
        checkpoint = artifacts.checkpoint
        head = artifacts.current
        self._binding.artifact_validator.validate(publication.artifact)
        pending = result.pending
        durable = self._store.durable
        checkpointed = pending or self._should_checkpoint(result)
        local_artifact = artifacts.stage(next_step, publication.artifact, parent=checkpoint)
        try:
            # A pending release is recorded but never activated and moves no head.
            # The engine must confirm the new revision before anything moves
            # the served head: the staged bytes load first, and the version
            # minted by publication then aliases them.
            if not pending:
                self._activate(local_artifact)
            if checkpointed:
                commit_metadata = scenario_metadata_for(
                    name=self._name,
                    base_artifact=artifacts.base,
                    scenario_step=next_step,
                    algorithm_state=prepared.algorithm_state,
                    record_progress=RecordProgress(
                        high_water_sequence=prepared.high_water_sequence,
                        high_water_offset=prepared.high_water_offset,
                        compacted_ids=prepared.compacted_ids,
                        consumed_ids=prepared.consumed_ids,
                    ),
                    metrics=prepared.metrics,
                    training_job_id=prepared.training_job_id,
                )
                published_ref = artifacts.publish(
                    local_artifact,
                    expected_parent=checkpoint,
                    metadata={
                        **dict(publication.artifact.metadata),
                        SCENARIO_METADATA_KEY: commit_metadata,
                    },
                    advance_heads=not pending and not durable,
                )
                if not pending:
                    self._activate(artifacts.resolve(published_ref), source=local_artifact)
                record = self._append_commit_record(
                    step=next_step,
                    artifact_ref=published_ref,
                    checkpoint=True,
                    prepared=prepared,
                    pending=pending,
                )
            else:
                if not durable:
                    # Without a durable commit point, reject a changed serving
                    # head before recording or compacting the prepared step.
                    artifacts.advance(local_artifact.ref, expected=head)
                record = self._append_commit_record(
                    step=next_step,
                    artifact_ref=local_artifact.ref,
                    checkpoint=False,
                    prepared=prepared,
                )
            if not pending:
                if checkpointed and durable:
                    self._install_committed_checkpoint(published_ref, expected=head, expected_checkpoint=checkpoint)
                elif not checkpointed and durable:
                    artifacts.advance(local_artifact.ref, expected=head)
        except Exception:
            # A lost append acknowledgment can leave a committed local release.
            # Its bytes must survive so retry can settle that exact record.
            if not any(row.artifact_ref == local_artifact.ref for row in self._store.history()):
                artifacts.discard(local_artifact)
            raise

        self._settle_trainer_commit(prepared, record, next_step)
        return result.state

    def _install_committed_checkpoint(
        self, ref: ArtifactRef, *, expected: ArtifactRef, expected_checkpoint: ArtifactRef
    ) -> None:
        """Install the refs already committed in the store, then synchronize artifact storage."""
        self._artifacts.install_checkpoint(ref, expected=expected, expected_checkpoint=expected_checkpoint)
        self._refresh_committed_checkpoint()

    def _refresh_committed_checkpoint(self) -> None:
        # The step is already committed. Expose the pointer failure in
        # commit_status without making the caller repeat a successful step.
        # A new commit must synchronize first and propagates any failure.
        with suppress(ArtifactPublicationError, ArtifactConflict):
            self._synchronize_checkpoint()

    def _synchronize_checkpoint(self) -> None:
        checkpoint = self._artifacts.checkpoint
        try:
            self._artifacts.repository.synchronize_checkpoint()
        except ArtifactConflict as exc:
            self._artifact_head_sync = _ArtifactHeadSync("conflict", checkpoint.release_id, str(exc))
            raise
        except ArtifactPublicationError as exc:
            self._artifact_head_sync = _ArtifactHeadSync("pending", checkpoint.release_id, str(exc))
            raise
        else:
            self._artifact_head_sync = _ArtifactHeadSync("synchronized", checkpoint.release_id)
        finally:
            self._commit_status = (self._step, self._latest_training_record, self._artifact_head_sync)

    def _activate(self, artifact: Artifact, *, source: Artifact | None = None) -> None:
        loader = self._binding.surface.loader
        if isinstance(loader, ArtifactActivator):
            loader.activate(artifact, self._binding.runtime, source=source)

    def _settle_trainer_commit(self, prepared: PreparedCommit, record: CommitRecord, next_step: int) -> None:
        """Finish recoverable effects before exposing the prepared state."""
        self._trainer.commit_applied(prepared.algorithm_state)
        self._trainer.compaction_applied(prepared.compacted_ids)
        self._trainer.commit(prepared)
        if not self._store.durable:
            self._artifact_head_sync = _ArtifactHeadSync("synchronized", self._artifacts.checkpoint.release_id)
        if record.operation == "training":
            self._latest_training_record = record
        self.advance_to(next_step)

    def _recorded_training_retry(
        self,
        prepared: PreparedCommit,
        result: TrainStepResult,
        next_step: int,
    ) -> CommitRecord | None:
        records = self._store.history()
        if not records or records[-1].step != next_step:
            return None
        record = records[-1]
        matches = (
            record.operation == "training"
            and record.pending == result.pending
            and self._record_matches_prepared(record, prepared)
        )
        if not matches:
            raise ScenarioStoreConflict(f"commit step {next_step} conflicts with the pending training result")
        return record

    def _recorded_operation_retry(
        self,
        prepared: PreparedCommit,
        operation: str,
        target_release_id: str,
        next_step: int,
    ) -> CommitRecord | None:
        records = self._store.history()
        if not records or records[-1].step != next_step:
            return None
        record = records[-1]
        if (
            record.operation != operation
            or record.rollback_target_release_id != target_release_id
            or not self._record_matches_prepared(record, prepared)
        ):
            raise ScenarioStoreConflict(f"commit step {next_step} conflicts with the pending {operation}")
        return record

    @staticmethod
    def _record_matches_prepared(record: CommitRecord, prepared: PreparedCommit) -> bool:
        return (
            record.algorithm_state == prepared.algorithm_state
            and record.high_water_sequence == prepared.high_water_sequence
            and record.high_water_offset == prepared.high_water_offset
            and record.compacted_ids == prepared.compacted_ids
            and record.consumed_ids == prepared.consumed_ids
            and record.metrics == prepared.metrics
            and record.training_job_id == prepared.training_job_id
        )

    def _reconcile_recorded_artifact(self, record: CommitRecord) -> None:
        current = self._artifacts.current
        if record.pending:
            return
        if current == record.artifact_ref:
            if record.checkpoint:
                self._refresh_committed_checkpoint()
            return
        records = self._store.history()
        previous = next(
            (prior.artifact_ref for prior in reversed(records[:-1]) if not prior.pending),
            self._releases.creation_artifact,
        )
        if record.checkpoint:
            self._install_committed_checkpoint(
                record.artifact_ref, expected=previous, expected_checkpoint=self._artifacts.checkpoint
            )
        else:
            self._artifacts.advance(record.artifact_ref, expected=previous)

    def _append_commit_record(
        self,
        *,
        step: int,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        prepared: PreparedCommit,
        operation: str = "training",
        rollback_target_release_id: str | None = None,
        pending: bool = False,
    ) -> CommitRecord:
        record = CommitRecord(
            scenario=self._name,
            step=step,
            artifact_ref=artifact_ref,
            checkpoint=checkpoint,
            pending=pending,
            algorithm_state=prepared.algorithm_state,
            high_water_sequence=prepared.high_water_sequence,
            high_water_offset=prepared.high_water_offset,
            compacted_ids=prepared.compacted_ids,
            consumed_ids=prepared.consumed_ids,
            operation=operation,
            rollback_target_release_id=rollback_target_release_id,
            metrics=prepared.metrics,
            training_job_id=prepared.training_job_id,
        )
        return self._store.commit_step(expected_step=self._step, commit=record)

    def metrics_for_version(self, release_id: str) -> Mapping[str, Any] | None:
        return self._releases.metrics_for_version(release_id)

    def entries_for_version(self, release_id: str) -> tuple[Mapping[str, Any], ...] | None:
        return self._releases.entries_for_version(release_id)

    def artifact_for_version(self, release_id: str) -> Artifact:
        return self._releases.artifact_for_version(release_id)

    def artifact_with_metrics(
        self,
        release_id: str | None = None,
    ) -> tuple[Artifact, Mapping[str, Any] | None]:
        return self._releases.artifact_with_metrics(release_id)


__all__ = [
    "ScenarioCommitter",
]
