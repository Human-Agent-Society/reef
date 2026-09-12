"""Assemble and parse artifact snapshot metadata for a scenario.

A scenario registration lives in the artifact backend's metadata under
``SCENARIO_SNAPSHOT_METADATA_KEY``. The snapshot pins the durable binding
(scenario name and base artifact) plus enough recovery state
(scenario step, algorithm state, record-consumption progress) for a
checkpoint to resume training after a crash. ``state`` owns the shared snapshot
and record-progress values. This adapter builds their metadata envelope from a
prepared training commit and parses the envelope read from the artifact backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

from reef.artifact.artifact import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.scenario.state import RecordProgress, ScenarioSnapshot, parse_record_progress
from reef.train.types import PreparedCommit

SCENARIO_SNAPSHOT_METADATA_KEY = "scenario_snapshot"
SCENARIO_SNAPSHOT_KIND = "reef-scenario/4"


def snapshot_metadata_for(
    *,
    name: str,
    base_artifact: ArtifactRef,
    scenario_step: int = 0,
    algorithm_state: Mapping[str, Any] | None = None,
    prepared: PreparedCommit | None = None,
    operation: str = "training",
    rollback_target_release_id: str | None = None,
) -> dict[str, object]:
    if not isinstance(scenario_step, int) or scenario_step < 0:
        raise ValueError("scenario_step must be non-negative")
    metadata: dict[str, object] = {
        "format": SCENARIO_SNAPSHOT_KIND,
        "scenario": name,
        "scenario_step": scenario_step,
        "base_artifact": encode_artifact_ref(base_artifact),
        "operation": operation,
    }
    if operation not in ("training", "rollback", "promote"):
        raise ValueError("scenario snapshot operation must be 'training' or 'rollback'")
    if operation in ("rollback", "promote"):
        if not isinstance(rollback_target_release_id, str) or not rollback_target_release_id:
            raise ValueError("rollback scenario snapshot requires rollback_target_release_id")
        metadata["rollback_target_release_id"] = rollback_target_release_id
    elif rollback_target_release_id is not None:
        raise ValueError("training scenario snapshot must not carry rollback_target_release_id")
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
        if prepared.metrics is not None:
            metadata["metrics"] = deepcopy(dict(prepared.metrics))
    return metadata


def parse_snapshot_metadata(value: Mapping[str, Any]) -> ScenarioSnapshot:
    if value.get("format") != SCENARIO_SNAPSHOT_KIND:
        raise ValueError(f"unsupported scenario snapshot format: {value.get('format')!r}")
    scenario = value.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario snapshot requires scenario")
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
    if scenario_step > 0 and record_progress is None:
        raise ValueError("scenario snapshot requires record_progress after step zero")
    training_job_id = value.get("training_job_id")
    if training_job_id is not None and (not isinstance(training_job_id, str) or not training_job_id):
        raise ValueError("scenario snapshot training_job_id must be a non-empty string or null")
    metrics = value.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, Mapping):
            raise ValueError("scenario snapshot metrics must be an object or null")
        metrics = MappingProxyType(deepcopy(dict(metrics)))
    operation = value.get("operation")
    rollback_target_release_id = value.get("rollback_target_release_id")
    if operation is not None and operation not in ("training", "rollback", "promote"):
        raise ValueError("scenario snapshot operation must be 'training' or 'rollback'")
    if operation in ("rollback", "promote"):
        if not isinstance(rollback_target_release_id, str) or not rollback_target_release_id:
            raise ValueError("rollback scenario snapshot requires rollback_target_release_id")
        if training_job_id is not None:
            raise ValueError("rollback scenario snapshot cannot carry training_job_id")
    elif rollback_target_release_id is not None:
        raise ValueError("non-rollback scenario snapshot cannot carry rollback_target_release_id")
    return ScenarioSnapshot(
        scenario=scenario,
        base_artifact=base_artifact,
        scenario_step=scenario_step,
        algorithm_state=algorithm_state,
        record_progress=record_progress,
        training_job_id=training_job_id,
        metrics=metrics,
        operation=operation,
        rollback_target_release_id=rollback_target_release_id,
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
