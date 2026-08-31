"""Durable scenario snapshot metadata: schema, serialization, and parsing.

A scenario registration lives in the artifact backend's metadata under
``SCENARIO_SNAPSHOT_METADATA_KEY``. The snapshot pins the durable binding
(scenario name, recipe, base artifact) plus enough recovery state
(scenario step, algorithm state, record-consumption progress) for a
checkpoint to resume training after a crash. This module is the format
counterpart of ``commit_log.py``: the log journals every commit, the
snapshot summarizes the last checkpointed one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from reef.artifact.artifact import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.train.types import PreparedCommit

SCENARIO_SNAPSHOT_METADATA_KEY = "scenario_snapshot"
SCENARIO_SNAPSHOT_KIND = "reef-scenario/3"


@dataclass(frozen=True)
class RecordProgress:
    """Record-consumption watermark pinned by a snapshot or commit record.

    ``consumed_ids`` names the rows the step's batch consumed; ``None`` marks
    a record written before the field existed, when the consumed set was not
    recorded and recovery cannot know it.
    """

    high_water_sequence: int
    high_water_offset: int
    compacted_ids: frozenset[str] = frozenset()
    consumed_ids: frozenset[str] | None = None


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
    compacted_ids = value.get("compacted_ids", [])
    if not isinstance(compacted_ids, list) or any(not isinstance(item, str) for item in compacted_ids):
        raise ValueError(f"{context} record_progress.compacted_ids must be a list of strings")
    # Absent on records written before the field existed; None keeps that
    # distinguishable from a step that consumed nothing.
    consumed_ids = value.get("consumed_ids")
    if consumed_ids is not None and (
        not isinstance(consumed_ids, list) or any(not isinstance(item, str) for item in consumed_ids)
    ):
        raise ValueError(f"{context} record_progress.consumed_ids must be a list of strings")
    return RecordProgress(
        high_water_sequence=value["high_water_sequence"],
        high_water_offset=value["high_water_offset"],
        compacted_ids=frozenset(compacted_ids),
        consumed_ids=None if consumed_ids is None else frozenset(consumed_ids),
    )


@dataclass(frozen=True)
class ScenarioSnapshot:
    """Parsed, validated scenario registration metadata."""

    scenario: str
    recipe: str
    base_artifact: ArtifactRef
    scenario_step: int
    algorithm_state: Mapping[str, Any] | None
    record_progress: RecordProgress | None
    training_job_id: str | None = None
    operation: str | None = None
    rollback_target_artifact_version: str | None = None


def snapshot_metadata_for(
    *,
    name: str,
    recipe: str,
    base_artifact: ArtifactRef,
    scenario_step: int = 0,
    algorithm_state: Mapping[str, Any] | None = None,
    prepared: PreparedCommit | None = None,
    operation: str = "training",
    rollback_target_artifact_version: str | None = None,
) -> dict[str, object]:
    if not isinstance(scenario_step, int) or scenario_step < 0:
        raise ValueError("scenario_step must be non-negative")
    metadata: dict[str, object] = {
        "format": SCENARIO_SNAPSHOT_KIND,
        "scenario": name,
        "recipe": recipe,
        "scenario_step": scenario_step,
        "base_artifact": encode_artifact_ref(base_artifact),
        "operation": operation,
    }
    if operation not in ("training", "rollback"):
        raise ValueError("scenario snapshot operation must be 'training' or 'rollback'")
    if operation == "rollback":
        if not isinstance(rollback_target_artifact_version, str) or not rollback_target_artifact_version:
            raise ValueError("rollback scenario snapshot requires rollback_target_artifact_version")
        metadata["rollback_target_artifact_version"] = rollback_target_artifact_version
    elif rollback_target_artifact_version is not None:
        raise ValueError("training scenario snapshot must not carry rollback_target_artifact_version")
    if algorithm_state is not None:
        metadata["algorithm_state"] = dict(algorithm_state)
    if prepared is not None:
        metadata["record_progress"] = {
            "high_water_sequence": prepared.high_water_sequence,
            "high_water_offset": prepared.high_water_offset,
            "compacted_ids": sorted(prepared.compacted_ids),
            "consumed_ids": sorted(prepared.consumed_ids),
        }
        if prepared.training_job_id is not None:
            metadata["training_job_id"] = prepared.training_job_id
    return metadata


def parse_snapshot_metadata(value: Mapping[str, Any]) -> ScenarioSnapshot:
    if value.get("format") != SCENARIO_SNAPSHOT_KIND:
        raise ValueError(f"unsupported scenario snapshot format: {value.get('format')!r}")
    scenario = value.get("scenario")
    recipe = value.get("recipe")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario snapshot requires scenario")
    if not isinstance(recipe, str) or not recipe:
        raise ValueError("scenario snapshot requires recipe")
    raw_base = value.get("base_artifact")
    if not isinstance(raw_base, Mapping):
        raise ValueError("scenario snapshot requires base_artifact")
    try:
        base_artifact = decode_artifact_ref(raw_base)
    except ValueError as exc:
        raise ValueError(f"invalid scenario snapshot base_artifact: {exc}") from exc
    scenario_step = value.get("scenario_step", 0)
    if not isinstance(scenario_step, int) or scenario_step < 0:
        raise ValueError("scenario snapshot scenario_step must be non-negative")
    algorithm_state = value.get("algorithm_state")
    if algorithm_state is not None:
        if not isinstance(algorithm_state, Mapping):
            raise ValueError("scenario snapshot algorithm_state must be an object")
        algorithm_state = MappingProxyType(dict(algorithm_state))
    raw_progress = value.get("record_progress")
    record_progress: RecordProgress | None = None
    if raw_progress is not None:
        record_progress = parse_record_progress(raw_progress, context="scenario snapshot")
    training_job_id = value.get("training_job_id")
    if training_job_id is not None and (not isinstance(training_job_id, str) or not training_job_id):
        raise ValueError("scenario snapshot training_job_id must be a non-empty string or null")
    operation = value.get("operation")
    rollback_target_artifact_version = value.get("rollback_target_artifact_version")
    if operation is None and "rollback" in value:
        legacy_rollback = value.get("rollback")
        if not isinstance(legacy_rollback, Mapping):
            raise ValueError("scenario snapshot rollback must be an object")
        operation = "rollback"
        rollback_target_artifact_version = legacy_rollback.get("target_artifact_version")
    if operation is not None and operation not in ("training", "rollback"):
        raise ValueError("scenario snapshot operation must be 'training' or 'rollback'")
    if operation == "rollback":
        if not isinstance(rollback_target_artifact_version, str) or not rollback_target_artifact_version:
            raise ValueError("rollback scenario snapshot requires rollback_target_artifact_version")
        if training_job_id is not None:
            raise ValueError("rollback scenario snapshot cannot carry training_job_id")
    elif rollback_target_artifact_version is not None:
        raise ValueError("non-rollback scenario snapshot cannot carry rollback_target_artifact_version")
    return ScenarioSnapshot(
        scenario=scenario,
        recipe=recipe,
        base_artifact=base_artifact,
        scenario_step=scenario_step,
        algorithm_state=algorithm_state,
        record_progress=record_progress,
        training_job_id=training_job_id,
        operation=operation,
        rollback_target_artifact_version=rollback_target_artifact_version,
    )


__all__ = [
    "SCENARIO_SNAPSHOT_KIND",
    "SCENARIO_SNAPSHOT_METADATA_KEY",
    "RecordProgress",
    "ScenarioSnapshot",
    "parse_record_progress",
    "parse_snapshot_metadata",
    "snapshot_metadata_for",
]
