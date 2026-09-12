"""Persisted scenario values shared by storage and artifact metadata adapters.

This module owns commit records, checkpoint snapshots, and record progress.
It defines their existing validation and commit encoding without importing
storage implementations, artifact operations, or the training lifecycle.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from reef.core.artifact_ref import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError


@dataclass(frozen=True)
class RecordProgress:
    """Record-consumption watermark pinned by a snapshot or commit record.

    ``consumed_ids`` names the rows the step's batch consumed.
    """

    high_water_sequence: int
    high_water_offset: int
    compacted_ids: frozenset[str] = frozenset()
    consumed_ids: frozenset[str] = frozenset()


def parse_record_progress(value: object, *, context: str) -> RecordProgress:
    """Validate one record_progress mapping; shared by snapshot and commit-log parsing.

    ``context`` prefixes every error message (e.g. ``"scenario snapshot"`` or
    ``"commit record"``). Raises ``ValueError``; callers with their own error
    types translate it.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} record_progress must be an object")
    for name in ("high_water_sequence", "high_water_offset"):
        field = value.get(name)
        if not isinstance(field, int) or isinstance(field, bool) or field < 0:
            raise ValueError(f"{context} record_progress.{name} must be a non-negative integer")
    compacted_ids = value.get("compacted_ids")
    if not isinstance(compacted_ids, list) or any(not isinstance(item, str) for item in compacted_ids):
        raise ValueError(f"{context} record_progress.compacted_ids must be a list of strings")
    consumed_ids = value.get("consumed_ids")
    if not isinstance(consumed_ids, list) or any(not isinstance(item, str) for item in consumed_ids):
        raise ValueError(f"{context} record_progress.consumed_ids must be a list of strings")
    return RecordProgress(
        high_water_sequence=value["high_water_sequence"],
        high_water_offset=value["high_water_offset"],
        compacted_ids=frozenset(compacted_ids),
        consumed_ids=frozenset(consumed_ids),
    )


@dataclass(frozen=True)
class ScenarioSnapshot:
    """Parsed, validated scenario registration metadata."""

    scenario: str
    base_artifact: ArtifactRef
    scenario_step: int
    algorithm_state: Mapping[str, Any] | None
    record_progress: RecordProgress | None
    training_job_id: str | None = None
    operation: str | None = None
    rollback_target_release_id: str | None = None
    metrics: Mapping[str, Any] | None = None


RECORD_KIND = "reef-commit/5"


class CommitLogError(ReefError):
    """The commit log is corrupt or a record does not match the schema."""


@dataclass(frozen=True, kw_only=True, eq=False)
class CommitRecord:
    """One committed training step: the atomic release record.

    ``step``, ``artifact_ref`` and ``algorithm_state`` advance together or not
    at all; ``record_progress`` pins the record high-water mark the step
    consumed, the rows its batch consumed, and the rows its compaction
    retired, so the record store and processor memory can be re-derived after
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


__all__ = [
    "RECORD_KIND",
    "CommitLogError",
    "CommitRecord",
    "RecordProgress",
    "ScenarioSnapshot",
    "parse_record_progress",
]
