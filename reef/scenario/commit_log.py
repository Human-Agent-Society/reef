"""Per-scenario commit log: the single durable commit point for training steps.

Every step-advancing commit appends exactly one record here, and every other
store is derived from it: record-store compaction is replayed from the
records' high-water marks, the checkpoint head is restored from the last
record's ref, and the trainer's algorithm state and read progress resume from
the same record. The log is append-only JSONL; the idempotent append (+fsync)
is the commit point, ordered so a crash anywhere else is recoverable by replay:

- live step:       prepare trainer -> append record -> advance head -> apply trainer
- checkpoint step: prepare trainer -> publish version -> append record -> apply trainer
- local step:      prepare trainer -> stage -> advance head -> append record -> apply trainer
- no-artifact:     prepare trainer -> append record -> apply trainer
- rollback:        restore surface -> prepare trainer -> publish copy -> append record -> apply trainer

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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from reef.artifact.artifact import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError
from reef.scenario.snapshot import parse_record_progress

RECORD_KIND = "reef-commit/5"


class CommitLogError(ReefError):
    """The commit log is corrupt or a record does not match the schema."""


@dataclass(frozen=True, kw_only=True, eq=False)
class CommitRecord:
    """One committed training step: the atomic release record.

    ``step``, ``artifact_ref`` and ``algorithm_state`` advance together or not
    at all; ``record_progress`` pins the record high-water mark the step
    consumed, the rows its batch consumed, and the rows its compaction
    deleted, so the record store and processor memory can be re-derived after
    a crash.
    """

    scenario: str
    step: int
    artifact_ref: ArtifactRef
    checkpoint: bool
    algorithm_state: Mapping[str, Any] | None
    high_water_sequence: int
    high_water_offset: int
    compacted_ids: frozenset[str] = frozenset()
    consumed_ids: frozenset[str] = frozenset()
    recorded_at: float = field(default_factory=time.time)
    operation: str = "training"
    operation_verified: bool = True
    pending: bool = False
    rollback_target_release_id: str | None = None
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 1:
            raise CommitLogError("commit record step must be a positive integer")
        for name, value in (
            ("high_water_sequence", self.high_water_sequence),
            ("high_water_offset", self.high_water_offset),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CommitLogError(f"commit record {name} must be a non-negative integer")
        if self.operation not in ("training", "rollback", "promote"):
            raise CommitLogError("commit record operation must be 'training', 'rollback', or 'promote'")
        if not isinstance(self.operation_verified, bool):
            raise CommitLogError("commit record operation_verified must be a boolean")
        if self.operation in ("rollback", "promote"):
            if not isinstance(self.rollback_target_release_id, str) or not self.rollback_target_release_id:
                raise CommitLogError(f"{self.operation} commit requires rollback_target_release_id")
        elif self.rollback_target_release_id is not None:
            raise CommitLogError("training commit must not carry rollback_target_release_id")
        if not isinstance(self.pending, bool) or (self.pending and self.operation != "training"):
            raise CommitLogError("only a training commit may be pending")
        if self.metrics is not None and not isinstance(self.metrics, Mapping):
            raise CommitLogError("commit record metrics must be an object or null")
        if self.training_job_id is not None and (
            not isinstance(self.training_job_id, str) or not self.training_job_id
        ):
            raise CommitLogError("commit record training_job_id must be a non-empty string or null")
        if self.operation != "training" and self.training_job_id is not None:
            raise CommitLogError("only training commits may carry training_job_id")
        # Own every mutable value the record was handed: the log is the durable
        # commit point, so a caller mutating its state or metrics afterwards
        # must not change what was recorded.
        object.__setattr__(
            self, "algorithm_state", None if self.algorithm_state is None else dict(self.algorithm_state)
        )
        object.__setattr__(self, "metrics", None if self.metrics is None else deepcopy(dict(self.metrics)))
        object.__setattr__(self, "compacted_ids", frozenset(self.compacted_ids))
        object.__setattr__(self, "consumed_ids", frozenset(self.consumed_ids))

    def to_dict(self) -> dict[str, Any]:
        record_progress: dict[str, Any] = {
            "high_water_sequence": self.high_water_sequence,
            "high_water_offset": self.high_water_offset,
            "compacted_ids": sorted(self.compacted_ids),
        }
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
            value["rollback_target_release_id"] = self.rollback_target_release_id
        if not self.operation_verified:
            value["operation_verified"] = False
        if self.pending:
            value["pending"] = True
        if self.metrics is not None:
            value["metrics"] = deepcopy(self.metrics)
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
        rollback_target_release_id = value.get("rollback_target_release_id")
        pending = value.get("pending", False)
        return cls(
            pending=pending,
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
            rollback_target_release_id=rollback_target_release_id,
            metrics=value.get("metrics"),
            training_job_id=value.get("training_job_id"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommitRecord):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"CommitRecord(scenario={self.scenario!r}, step={self.step}, release={self.artifact_ref.release_id!r})"


class CommitLog:
    """Append-only JSONL commit log for one scenario.

    Appends serialize the record to a single line and fsync before returning;
    that is the commit point. Retrying the same scenario step is a no-op, while
    different content for an existing step is a conflict. Reads tolerate
    exactly one torn tail (a crash mid-append) and reject corruption elsewhere.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cached_records: list[CommitRecord] | None = None
        self._cached_steps: dict[int, CommitRecord] | None = None
        self._cached_snapshot: tuple[CommitRecord, ...] | None = None
        self._cached_signature: tuple[int, int, int, int] | None = None
        self._run_segment = 0
        self._run_step = 0

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: CommitRecord) -> None:
        encoded = (
            json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        with self._lock:
            self._truncate_partial_tail()
            previous_signature = self._file_signature()
            cached_records = self._cached_records
            records: Sequence[CommitRecord]
            if cached_records is None or previous_signature != self._cached_signature:
                records = self._reload_cache(previous_signature)
            else:
                records = cached_records
            cached_steps = self._cached_steps
            existing = (
                cached_steps.get(record.step)
                if cached_steps is not None
                else next((item for item in records if item.step == record.step), None)
            )
            if existing is not None:
                if self._same_commit(existing, record):
                    return
                raise CommitLogError(f"commit step {record.step} conflicts with its existing record")
            if records and record.step < records[-1].step:
                raise CommitLogError(f"commit step {record.step} precedes existing step {records[-1].step}")
            with open(self._path, "ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            current_signature = self._file_signature()
            cached_records = self._cached_records
            if (
                cached_records is not None
                and previous_signature == self._cached_signature
                and self._is_exclusive_append(previous_signature, current_signature, len(encoded))
            ):
                cached_records.append(record)
                if cached_steps is not None:
                    cached_steps[record.step] = record
                self._cached_snapshot = None
                self._advance_run_position(record)
                self._cached_signature = current_signature
            else:
                self._invalidate_cache()

    @staticmethod
    def _same_commit(left: CommitRecord, right: CommitRecord) -> bool:
        return {key: value for key, value in left.to_dict().items() if key != "recorded_at"} == {
            key: value for key, value in right.to_dict().items() if key != "recorded_at"
        }

    def records(self) -> tuple[CommitRecord, ...]:
        with self._lock:
            signature = self._file_signature()
            cached_records = self._cached_records
            if cached_records is not None and signature == self._cached_signature:
                if self._cached_snapshot is None:
                    self._cached_snapshot = tuple(cached_records)
                return self._cached_snapshot
            return self._reload_cache(signature)

    def training_run_position(self) -> tuple[int, int]:
        """Return the latest rollback step and training steps after it."""
        with self._lock:
            signature = self._file_signature()
            if self._cached_records is not None and signature == self._cached_signature:
                return self._run_segment, self._run_step
            records = self._reload_cache(signature)
            if self._cached_records is not None and signature == self._cached_signature:
                return self._run_segment, self._run_step
            return self._position(records)

    def _file_signature(self) -> tuple[int, int, int, int] | None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _truncate_partial_tail(self) -> None:
        try:
            with open(self._path, "r+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    return
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) == b"\n":
                    return
                handle.seek(0)
                content = handle.read()
                handle.truncate(content.rfind(b"\n") + 1)
                handle.flush()
                os.fsync(handle.fileno())
        except FileNotFoundError:
            return

    def _cache(self, records: tuple[CommitRecord, ...], signature: tuple[int, int, int, int] | None) -> None:
        self._cached_records = list(records)
        self._cached_steps = {record.step: record for record in records}
        self._cached_snapshot = records
        self._cached_signature = signature
        self._run_segment, self._run_step = self._position(records)

    @staticmethod
    def _is_exclusive_append(
        previous: tuple[int, int, int, int] | None,
        current: tuple[int, int, int, int] | None,
        appended_bytes: int,
    ) -> bool:
        if current is None:
            return False
        if previous is None:
            return current[2] == appended_bytes
        return current[:2] == previous[:2] and current[2] == previous[2] + appended_bytes

    def _reload_cache(self, signature: tuple[int, int, int, int] | None) -> tuple[CommitRecord, ...]:
        self._invalidate_cache()
        records, complete = self._read_records()
        if complete and signature == self._file_signature():
            self._cache(records, signature)
        return records

    def _invalidate_cache(self) -> None:
        self._cached_records = None
        self._cached_steps = None
        self._cached_snapshot = None
        self._cached_signature = None

    @staticmethod
    def _position(records: tuple[CommitRecord, ...]) -> tuple[int, int]:
        run_segment = 0
        run_step = 0
        for record in records:
            if record.operation == "rollback":
                run_segment = record.step
                run_step = 0
            elif record.operation == "training" and record.step > run_segment:
                run_step += 1
        return run_segment, run_step

    def _advance_run_position(self, record: CommitRecord) -> None:
        if record.operation == "rollback":
            self._run_segment = record.step
            self._run_step = 0
        elif record.operation == "training" and record.step > self._run_segment:
            self._run_step += 1

    def _read_records(self) -> tuple[tuple[CommitRecord, ...], bool]:
        if not self._path.exists():
            return (), True
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
                    return tuple(records), False  # torn tail from a crash mid-append
                raise CommitLogError(f"commit log {self._path} has a corrupt record at line {index + 1}") from exc
            if not isinstance(value, dict):
                raise CommitLogError(f"commit log {self._path} line {index + 1} is not a record object")
            try:
                records.append(CommitRecord.from_dict(value))
            except CommitLogError as exc:
                raise CommitLogError(f"commit log {self._path} line {index + 1}: {exc}") from exc
        return tuple(records), True
