"""Scenario commit protocol coordinating training and artifact publication.

Artifact storage and head movement live in ``reef.artifact``; the durable
record and snapshot formats live in ``commit_log`` and ``snapshot``. This
module owns the scenario-specific ordering across trainer state, commit-log
durability, record compaction, checkpoint policy, surfaces, and artifact
operations.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

from reef.artifact.artifact import (
    Artifact,
    ArtifactError,
    ArtifactNotFound,
    ArtifactRef,
    LiveWeightArtifactRef,
    is_local_release,
)
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.core.errors import ReefError
from reef.scenario.binding import ScenarioBinding
from reef.scenario.checkpoint_strategy import CheckpointStrategy
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.scenario.snapshot import SCENARIO_SNAPSHOT_METADATA_KEY, RecordProgress, snapshot_metadata_for
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


class ScenarioCommitProtocol:
    """Own one scenario's atomic commit, rollback, catalog, and recovery rules."""

    def __init__(
        self,
        *,
        name: str,
        binding: ScenarioBinding,
        artifacts: ArtifactReleaseChain,
        checkpoint_strategy: CheckpointStrategy,
        trainer: Trainer,
        scenario_step: int = 0,
        commit_log: CommitLog | None = None,
        recovered_head_record: CommitRecord | None = None,
    ) -> None:
        if not isinstance(scenario_step, int) or scenario_step < 0:
            raise ValueError("scenario_step must be non-negative")
        self._name = name
        self._binding = binding
        self._artifacts = artifacts
        self._checkpoint_strategy = checkpoint_strategy
        self._trainer = trainer
        self._step = scenario_step
        self._commit_log = commit_log
        self._creation_artifact = self._resolve_creation_artifact()
        self._lock = RLock()
        records = (
            (() if recovered_head_record is None else (recovered_head_record,))
            if commit_log is None
            else commit_log.records()
        )
        self._latest_training_record = next(
            (record for record in reversed(records) if record.operation == "training"),
            None,
        )
        self._commit_status_snapshot = (scenario_step, self._latest_training_record)

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
    def commit_log(self) -> CommitLog | None:
        return self._commit_log

    @property
    def step(self) -> int:
        return self._step

    @property
    def commit_status(self) -> Mapping[str, Any]:
        """Current step and latest training outcome from one non-blocking snapshot."""
        step, record = self._commit_status_snapshot
        return {
            "scenario_step": step,
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
        self._commit_status_snapshot = (step, self._latest_training_record)

    def releases(self) -> tuple[dict[str, Any], ...]:
        """List committed releases newest first."""
        with self._lock:
            records = () if self._commit_log is None else self._commit_log.records()
            rows = [
                self._release_row(
                    artifact_ref=self._creation_artifact,
                    checkpoint=True,
                    recorded_at=None,
                    operation="creation",
                    current=self._step == 0,
                )
            ]
            rows.extend(
                self._release_row(
                    artifact_ref=record.artifact_ref,
                    checkpoint=record.checkpoint,
                    recorded_at=record.recorded_at,
                    operation=record.operation,
                    current=record.step == self._step,
                    rollback_target_release_id=record.rollback_target_release_id,
                    high_water_sequence=record.high_water_sequence,
                    high_water_offset=record.high_water_offset,
                    metrics=record.metrics,
                )
                for record in records
            )
            if not records and self._step > 0:
                rows.append(
                    self._release_row(
                        artifact_ref=self._artifacts.current,
                        checkpoint=True,
                        recorded_at=None,
                        operation="recovery",
                        current=True,
                    )
                )
            return tuple(reversed(rows))

    def rollback(self, release_id: str) -> ArtifactRef:
        """Publish a durable copy of an older version as a new fenced commit."""
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        release_id = release_id.strip()
        with self._lock:
            current_ref = self._artifacts.current
            if current_ref.release_id == release_id:
                return current_ref
            target = self._find_release_id(release_id)
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
            source = artifacts.resolve(target_ref)
            surface = self._binding.surface
            self._binding.artifact_validator.validate(source)
            if surface.loader is not None:
                surface.loader.load(source, self._binding.runtime)
            prepared = self._trainer.commit()
            staged = artifacts.stage(next_step, source, parent=checkpoint)
            try:
                snapshot_metadata = snapshot_metadata_for(
                    name=self._name,
                    base_artifact=artifacts.base,
                    scenario_step=next_step,
                    algorithm_state=prepared.algorithm_state,
                    prepared=prepared,
                    operation="rollback",
                    rollback_target_release_id=release_id,
                )
                snapshot_metadata["rollback"] = {
                    "target_release_id": release_id,
                }
                published_ref = artifacts.publish(
                    staged,
                    expected_parent=checkpoint,
                    metadata={
                        **dict(source.metadata),
                        SCENARIO_SNAPSHOT_METADATA_KEY: snapshot_metadata,
                    },
                )
                if isinstance(surface.loader, ArtifactActivator):
                    surface.loader.activate(artifacts.resolve(published_ref), self._binding.runtime, source=source)
                self._append_commit_record(
                    step=next_step,
                    artifact_ref=published_ref,
                    checkpoint=True,
                    prepared=prepared,
                    operation="rollback",
                    rollback_target_release_id=release_id,
                )
                self._finish_trainer_commit(prepared)
            except Exception:
                artifacts.discard(staged)
                raise
            self.advance_to(next_step)
            return published_ref

    def commit(self, result: TrainStepResult) -> Any:
        """Commit a pending training result as one atomic version record."""
        with self._lock:
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
                return self._commit_live_weights(result, publication)
            if isinstance(publication, NoArtifactPublication):
                return self._commit_without_artifact(result)
            return self._commit_saved_artifact(result, publication)

    def _should_checkpoint(self, result: TrainStepResult) -> bool:
        return self._checkpoint_strategy.should_checkpoint(self._name, self._step + 1, result)

    def _commit_live_weights(self, result: TrainStepResult, publication: LiveWeightPublication) -> Any:
        artifacts = self._artifacts
        next_step = self._step + 1
        if self._should_checkpoint(result):
            raise ReefError(
                "checkpoint-selected training results must include the artifact returned by execute_training_job"
            )
        head, live_ref = artifacts.prepare_live(step=next_step, runtime_load_id=publication.runtime_load_id)
        prepared = self._trainer.commit()
        self._append_commit_record(
            step=next_step,
            artifact_ref=live_ref,
            checkpoint=False,
            prepared=prepared,
        )
        self._finish_trainer_commit(prepared)
        artifacts.advance(live_ref, expected=head)
        self.advance_to(next_step)
        return result.state

    def _commit_without_artifact(self, result: TrainStepResult) -> Any:
        next_step = self._step + 1
        prepared = self._trainer.commit()
        self._append_commit_record(
            step=next_step,
            artifact_ref=self._artifacts.current,
            checkpoint=False,
            prepared=prepared,
        )
        self._finish_trainer_commit(prepared)
        self.advance_to(next_step)
        return result.state

    def _commit_saved_artifact(self, result: TrainStepResult, publication: SavedArtifactPublication) -> Any:
        artifacts = self._artifacts
        next_step = self._step + 1
        checkpoint = artifacts.checkpoint
        head = artifacts.current
        self._binding.artifact_validator.validate(publication.artifact)
        prepared = self._trainer.commit()
        local_artifact = artifacts.stage(next_step, publication.artifact, parent=checkpoint)
        try:
            # The engine must confirm the new revision before anything moves
            # the served head: the staged bytes load first, and the version
            # minted by publication then aliases them.
            self._activate(local_artifact)
            if self._should_checkpoint(result):
                snapshot_metadata = snapshot_metadata_for(
                    name=self._name,
                    base_artifact=artifacts.base,
                    scenario_step=next_step,
                    algorithm_state=prepared.algorithm_state,
                    prepared=prepared,
                )
                published_ref = artifacts.publish(
                    local_artifact,
                    expected_parent=checkpoint,
                    metadata={
                        **dict(publication.artifact.metadata),
                        SCENARIO_SNAPSHOT_METADATA_KEY: snapshot_metadata,
                    },
                )
                self._activate(artifacts.resolve(published_ref), source=local_artifact)
                self._append_commit_record(
                    step=next_step,
                    artifact_ref=published_ref,
                    checkpoint=True,
                    prepared=prepared,
                )
            else:
                artifacts.advance(local_artifact.ref, expected=head)
                self._append_commit_record(
                    step=next_step,
                    artifact_ref=local_artifact.ref,
                    checkpoint=False,
                    prepared=prepared,
                )
            self._finish_trainer_commit(prepared)
        except Exception:
            artifacts.discard(local_artifact)
            raise

        self.advance_to(next_step)
        return result.state

    def _activate(self, artifact: Artifact, *, source: Artifact | None = None) -> None:
        loader = self._binding.surface.loader
        if isinstance(loader, ArtifactActivator):
            loader.activate(artifact, self._binding.runtime, source=source)

    def _finish_trainer_commit(self, prepared: PreparedCommit) -> None:
        """Run post-commit effects only after the version record is durable."""
        self._trainer.commit_applied(prepared.algorithm_state)
        self._trainer.apply_compaction(prepared.compacted_ids)

    def _append_commit_record(
        self,
        *,
        step: int,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        prepared: PreparedCommit,
        operation: str = "training",
        rollback_target_release_id: str | None = None,
    ) -> CommitRecord:
        record = CommitRecord(
            scenario=self._name,
            step=step,
            artifact_ref=artifact_ref,
            checkpoint=checkpoint,
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
        if self._commit_log is not None:
            self._commit_log.append(record)
        if record.operation == "training":
            self._latest_training_record = record
        return record

    def _find_release_id(self, release_id: str) -> tuple[ArtifactRef, bool] | None:
        records = () if self._commit_log is None else self._commit_log.records()
        for record in reversed(records):
            if record.artifact_ref.release_id == release_id:
                return record.artifact_ref, record.checkpoint
        if self._creation_artifact.release_id == release_id:
            return self._creation_artifact, True
        return None

    def _resolve_creation_artifact(self) -> ArtifactRef:
        """The artifact the scenario was forked from, for the release catalog.

        A fresh scenario is still on it, so the chain head is the creation
        artifact. After a recovery the fork point is reconstructed from the
        first commit record: normally the durable parent of step 1's artifact
        (a checkpoint, live, or local ref still knows its parent version); if
        step 1 never recorded a durable parent (a plain non-checkpoint ref),
        that ref itself is the earliest version the catalog can show. When
        the parent has since disappeared from the backend, fall back to the
        repository base artifact.
        """
        if self._step == 0:
            return self._artifacts.current
        records = () if self._commit_log is None else self._commit_log.records()
        if records and records[0].step == 1:
            first = records[0]
            ref = first.artifact_ref
            if (
                first.checkpoint or isinstance(ref, LiveWeightArtifactRef) or is_local_release(ref.release_id)
            ) and ref.parent_release_id is not None:
                try:
                    return self._artifacts.repository.backend.resolve_release(ref.parent_release_id)
                except ArtifactNotFound:
                    pass
            elif not first.checkpoint:
                return ref
        return self._artifacts.base

    def metrics_for_version(self, release_id: str) -> Mapping[str, Any] | None:
        """Metrics of the training step that published ``release_id``, if logged."""
        if self._commit_log is None:
            return None
        for record in self._commit_log.records():
            if record.artifact_ref.release_id == release_id and record.operation == "training":
                return record.metrics
        return None

    def artifact_for_version(self, release_id: str) -> Artifact:
        """Materialize a catalog version for a read-only content serve.

        The read-side counterpart of ``rollback``: the same catalog lookup,
        but no head moves and no commit is written; the caller only wants the
        version's file tree. Every failure is ``ArtifactNotFound`` naming the
        version (not ``ReleaseNotRestorable``, which answers a rejected
        write): to a reader, a version whose bytes are gone and a version
        that was recorded but not kept are the same absent content.
        """
        if not isinstance(release_id, str) or not release_id.strip():
            raise ValueError("release_id must be a non-empty string")
        release_id = release_id.strip()
        with self._lock:
            found = self._find_release_id(release_id)
        if found is None:
            raise ArtifactNotFound(f"scenario {self._name!r} has no release {release_id!r}")
        ref, _ = found
        if isinstance(ref, LiveWeightArtifactRef):
            raise ArtifactNotFound(
                f"scenario {self._name!r} release {release_id!r} is live weights and has no file tree"
            )
        try:
            return self._artifacts.resolve(ref)
        except ArtifactError as exc:
            raise ArtifactNotFound(f"scenario {self._name!r} cannot restore release {release_id!r}: {exc}") from exc

    @staticmethod
    def _release_row(
        *,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        recorded_at: float | None,
        operation: str,
        current: bool,
        rollback_target_release_id: str | None = None,
        high_water_sequence: int = 0,
        high_water_offset: int = 0,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "release_id": artifact_ref.release_id,
            "parent_release_id": artifact_ref.parent_release_id,
            "content_id": artifact_ref.content_id,
            "content_kind": "live_weights" if isinstance(artifact_ref, LiveWeightArtifactRef) else "saved_artifact",
            "checkpoint": checkpoint,
            "restorable": checkpoint and not isinstance(artifact_ref, LiveWeightArtifactRef),
            "recorded_at": recorded_at,
            "operation": operation,
            "current": current,
            "record_progress": {
                "high_water_sequence": high_water_sequence,
                "high_water_offset": high_water_offset,
            },
        }
        if rollback_target_release_id is not None:
            row["rollback_target_release_id"] = rollback_target_release_id
        if isinstance(artifact_ref, LiveWeightArtifactRef):
            row["runtime_load_id"] = artifact_ref.runtime_load_id
        if metrics is not None:
            row["metrics"] = dict(metrics)
        return row

    @staticmethod
    def recover_head(
        scenario: str,
        commit_log: CommitLog | None,
        *,
        snapshot_step: int,
        snapshot_state: Mapping[str, Any] | None,
        snapshot_record_progress: RecordProgress | None,
        checkpoint_head: ArtifactRef,
        snapshot_training_job_id: str | None = None,
        snapshot_metrics: Mapping[str, Any] | None = None,
        snapshot_operation: str | None = None,
        snapshot_rollback_target_release_id: str | None = None,
    ) -> CommitRecord | None:
        """Pick the committed head record to resume from, healing the log."""
        records: tuple[CommitRecord, ...] = ()
        log_label = "<no commit log>"
        if commit_log is not None:
            records = commit_log.records()
            log_label = str(commit_log.path)
        for record in records:
            if record.scenario != scenario:
                raise ReefError(f"commit log {log_label} holds records for {record.scenario!r}, not {scenario!r}")

        applicable = [record for record in records if record.step > snapshot_step]
        for previous, current in itertools.pairwise(applicable):
            if current.step != previous.step + 1:
                raise ReefError(
                    f"commit log {log_label} jumps from step {previous.step} to {current.step}; the log is corrupt"
                )
        if applicable and applicable[0].step != snapshot_step + 1:
            raise ReefError(
                f"commit log {log_label} resumes at step {applicable[0].step}, "
                f"expected {snapshot_step + 1} after the checkpointed step {snapshot_step}; the log is corrupt"
            )

        last_step = records[-1].step if records else 0
        if snapshot_step > last_step:
            adopted = CommitRecord(
                scenario=scenario,
                step=snapshot_step,
                artifact_ref=checkpoint_head,
                checkpoint=True,
                algorithm_state=snapshot_state,
                high_water_sequence=(
                    snapshot_record_progress.high_water_sequence
                    if snapshot_record_progress is not None
                    else (records[-1].high_water_sequence if records else 0)
                ),
                high_water_offset=(
                    snapshot_record_progress.high_water_offset
                    if snapshot_record_progress is not None
                    else (records[-1].high_water_offset if records else 0)
                ),
                compacted_ids=(
                    snapshot_record_progress.compacted_ids if snapshot_record_progress is not None else frozenset()
                ),
                consumed_ids=(
                    snapshot_record_progress.consumed_ids if snapshot_record_progress is not None else frozenset()
                ),
                operation=snapshot_operation or "training",
                operation_verified=snapshot_operation is not None,
                rollback_target_release_id=snapshot_rollback_target_release_id,
                metrics=snapshot_metrics,
                training_job_id=snapshot_training_job_id,
            )
            if commit_log is not None:
                commit_log.append(adopted)
            return adopted

        if applicable:
            return applicable[-1]
        if records and records[-1].step == snapshot_step:
            return records[-1]
        return None


__all__ = [
    "ScenarioCommitProtocol",
]
