"""Per-scenario commit log: the single durable commit point for training steps.

Every step-advancing commit appends exactly one record here, and every other
store is derived from it: record-store compaction is replayed from the
records' high-water marks, the checkpoint head is restored from the last
record's ref, and the trainer's algorithm state and read progress resume from
the same record. The log is append-only JSONL; the append (+fsync) is the
commit point, ordered so a crash anywhere else is recoverable by replay:

- live step:       commit trainer -> append record -> compact -> advance head
- checkpoint step: commit trainer -> publish version -> append record -> compact
- local step:      commit trainer -> stage -> advance head -> append record -> compact
- no-artifact:     commit trainer -> append record -> compact
- rollback:        restore surface -> commit trainer -> publish copy -> append record -> compact

A checkpoint publish therefore never loses its record (the repository write comes
first), and a crash between append and compaction is healed at recovery by
re-applying the recorded deletions. A local step may advance the head before
appending because a staged ref is process-local and never a recovery source:
a crash before the append loses the head move and the record together, so the
step simply never committed. If the checkpoint head is ahead of the log
(crash between publish and append), recovery adopts the checkpoint head — its
snapshot metadata carries the same record fields — and appends it to heal the
log. Rollback follows the checkpoint order and appends a record whose
operation points at the restored step while the fencing step keeps increasing.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact.artifact import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError
from reef.scenario.snapshot import parse_record_progress

RECORD_KIND = "reef-commit/4"


class CommitLogError(ReefError):
    """The commit log is corrupt or a record does not match the schema."""


class CommitRecord:
    """One committed training step: the atomic version record.

    ``step``, ``artifact_ref`` and ``algorithm_state`` advance together or not
    at all; ``record_progress`` pins the record high-water mark the step
    consumed, the rows its batch consumed, and the rows its compaction
    deleted, so the record store and processor memory can be re-derived after
    a crash. ``consumed_ids`` is ``None`` on records written before the field
    existed; recovery treats an unknown consumed set as a reason not to
    reingest, never as an empty one.
    """

    def __init__(
        self,
        *,
        scenario: str,
        step: int,
        artifact_ref: ArtifactRef,
        checkpoint: bool,
        algorithm_state: Mapping[str, Any] | None,
        high_water_sequence: int,
        high_water_offset: int,
        compacted_ids: frozenset[str] = frozenset(),
        consumed_ids: frozenset[str] | None = None,
        recorded_at: float | None = None,
        operation: str = "training",
        operation_verified: bool = True,
        rollback_target_artifact_version: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        training_job_id: str | None = None,
    ) -> None:
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise CommitLogError("commit record step must be a positive integer")
        for name, value in (("high_water_sequence", high_water_sequence), ("high_water_offset", high_water_offset)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CommitLogError(f"commit record {name} must be a non-negative integer")
        self.scenario = scenario
        self.step = step
        self.artifact_ref = artifact_ref
        self.checkpoint = checkpoint
        self.algorithm_state = None if algorithm_state is None else dict(algorithm_state)
        self.high_water_sequence = high_water_sequence
        self.high_water_offset = high_water_offset
        self.compacted_ids = frozenset(compacted_ids)
        self.consumed_ids = None if consumed_ids is None else frozenset(consumed_ids)
        self.recorded_at = time.time() if recorded_at is None else recorded_at
        if operation not in ("training", "rollback"):
            raise CommitLogError("commit record operation must be 'training' or 'rollback'")
        if not isinstance(operation_verified, bool):
            raise CommitLogError("commit record operation_verified must be a boolean")
        if operation == "rollback":
            if not isinstance(rollback_target_artifact_version, str) or not rollback_target_artifact_version:
                raise CommitLogError("rollback commit requires rollback_target_artifact_version")
        elif rollback_target_artifact_version is not None:
            raise CommitLogError("training commit must not carry rollback_target_artifact_version")
        self.operation = operation
        self.operation_verified = operation_verified
        self.rollback_target_artifact_version = rollback_target_artifact_version
        if metrics is not None and not isinstance(metrics, Mapping):
            raise CommitLogError("commit record metrics must be an object or null")
        self.metrics = None if metrics is None else dict(metrics)
        if training_job_id is not None and (not isinstance(training_job_id, str) or not training_job_id):
            raise CommitLogError("commit record training_job_id must be a non-empty string or null")
        if operation != "training" and training_job_id is not None:
            raise CommitLogError("only training commits may carry training_job_id")
        self.training_job_id = training_job_id

    def to_dict(self) -> dict[str, Any]:
        record_progress: dict[str, Any] = {
            "high_water_sequence": self.high_water_sequence,
            "high_water_offset": self.high_water_offset,
            "compacted_ids": sorted(self.compacted_ids),
        }
        if self.consumed_ids is not None:
            record_progress["consumed_ids"] = sorted(self.consumed_ids)
        value = {
            "record": RECORD_KIND,
            "scenario": self.scenario,
            "step": self.step,
            "artifact_ref": encode_artifact_ref(self.artifact_ref),
            "checkpoint": self.checkpoint,
            "algorithm_state": self.algorithm_state,
            "record_progress": record_progress,
            "recorded_at": self.recorded_at,
        }
        if self.operation != "training":
            value["operation"] = self.operation
            value["rollback_target_artifact_version"] = self.rollback_target_artifact_version
        if not self.operation_verified:
            value["operation_verified"] = False
        if self.metrics is not None:
            value["metrics"] = self.metrics
        if self.training_job_id is not None:
            value["training_job_id"] = self.training_job_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommitRecord:
        if value.get("record") != RECORD_KIND:
            raise CommitLogError(f"unknown commit record kind: {value.get('record')!r}")
        scenario = value.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise CommitLogError("commit record requires scenario")
        step = value.get("step")
        if not isinstance(step, int):
            raise CommitLogError("commit record step must be an integer")
        raw_ref = value.get("artifact_ref")
        if not isinstance(raw_ref, Mapping):
            raise CommitLogError("commit record requires artifact_ref")
        checkpoint = value.get("checkpoint")
        if not isinstance(checkpoint, bool):
            raise CommitLogError("commit record checkpoint must be a boolean")
        algorithm_state = value.get("algorithm_state")
        if algorithm_state is not None and not isinstance(algorithm_state, Mapping):
            raise CommitLogError("commit record algorithm_state must be an object or null")
        raw_progress = value.get("record_progress")
        if not isinstance(raw_progress, Mapping):
            raise CommitLogError("commit record requires record_progress")
        try:
            artifact_ref = decode_artifact_ref(raw_ref)
        except ValueError as exc:
            raise CommitLogError(f"commit record {exc}") from exc
        try:
            record_progress = parse_record_progress(raw_progress, context="commit record")
        except ValueError as exc:
            raise CommitLogError(str(exc)) from exc
        recorded_at = value.get("recorded_at")
        if not isinstance(recorded_at, int | float) or isinstance(recorded_at, bool):
            raise CommitLogError("commit record recorded_at must be a number")
        operation = value.get("operation", "training")
        operation_verified = value.get("operation_verified", True)
        rollback_target_artifact_version = value.get("rollback_target_artifact_version")
        return cls(
            scenario=scenario,
            step=step,
            artifact_ref=artifact_ref,
            checkpoint=checkpoint,
            algorithm_state=algorithm_state,
            high_water_sequence=record_progress.high_water_sequence,
            high_water_offset=record_progress.high_water_offset,
            compacted_ids=record_progress.compacted_ids,
            consumed_ids=record_progress.consumed_ids,
            recorded_at=float(recorded_at),
            operation=operation,
            operation_verified=operation_verified,
            rollback_target_artifact_version=rollback_target_artifact_version,
            metrics=value.get("metrics"),
            training_job_id=value.get("training_job_id"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommitRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"CommitRecord(scenario={self.scenario!r}, step={self.step}, version={self.artifact_ref.version!r})"


class CommitLog:
    """Append-only JSONL commit log for one scenario.

    Appends serialize the record to a single line and fsync before returning;
    that is the commit point. Reads tolerate exactly one torn tail (a crash
    mid-append) and reject corruption anywhere else.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: CommitRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> tuple[CommitRecord, ...]:
        if not self._path.exists():
            return ()
        records: list[CommitRecord] = []
        with open(self._path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(lines) - 1:
                    break  # torn tail from a crash mid-append; the commit never happened
                raise CommitLogError(f"commit log {self._path} has a corrupt record at line {index + 1}") from exc
            if not isinstance(value, dict):
                raise CommitLogError(f"commit log {self._path} line {index + 1} is not a record object")
            try:
                records.append(CommitRecord.from_dict(value))
            except CommitLogError as exc:
                raise CommitLogError(f"commit log {self._path} line {index + 1}: {exc}") from exc
        return tuple(records)
