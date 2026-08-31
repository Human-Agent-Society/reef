"""Durable training-job marker: the bridge's crash-recovery state machine.

One marker file (:data:`reef.runtime.names.LATEST_JOB_MARKER_FILENAME`) lives
next to the HF checkpoints and records the latest training job's identity and
stage:

- ``RUNNING`` — training started; a marker found in this state on recovery is
  ambiguous (the optimizer may or may not have stepped) and requires operator
  intervention.
- ``CHECKPOINT`` — training and checkpoint save completed.
- ``UPDATING_WEIGHTS`` — serving publication started but is not proven complete.
- ``READY_TO_COMMIT`` — all engines serve one version and remain paused.
- ``HEAD_COMMITTED`` — Reef acknowledged its durable commit; resume may retry.
- ``COMPLETE`` — engines resumed and the whole transaction is replayable.
- ``REJECTING`` — rejection is durable while incumbent serving recovery runs.
- ``REJECTED`` — Reef declined the checkpoint without changing serving weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from reef.runtime.base import TrainingJobResult
from reef.runtime.names import LATEST_JOB_MARKER_FILENAME
from reef.train.slime_backend.reef_adapters.training_job.durable_io import read_json
from reef.train.slime_backend.reef_adapters.training_job.durable_io import write_json as write_marker

MarkerDisposition = Literal["replay", "resume", "conflict", "fresh"]
MarkerStatus = Literal[
    "RUNNING",
    "CHECKPOINT",
    "UPDATING_WEIGHTS",
    "READY_TO_COMMIT",
    "HEAD_COMMITTED",
    "COMPLETE",
    "REJECTING",
    "REJECTED",
]

_MARKER_TRANSITIONS: dict[MarkerStatus, frozenset[MarkerStatus]] = {
    "RUNNING": frozenset({"CHECKPOINT"}),
    "CHECKPOINT": frozenset({"UPDATING_WEIGHTS", "REJECTING"}),
    "UPDATING_WEIGHTS": frozenset({"READY_TO_COMMIT"}),
    "READY_TO_COMMIT": frozenset({"HEAD_COMMITTED"}),
    "HEAD_COMMITTED": frozenset({"COMPLETE"}),
    "COMPLETE": frozenset(),
    "REJECTING": frozenset({"REJECTED"}),
    "REJECTED": frozenset(),
}

__all__ = [
    "MarkerDisposition",
    "MarkerStatus",
    "marker_checkpoint_result",
    "marker_disposition",
    "marker_path",
    "marker_result",
    "marker_rollouts",
    "read_marker",
    "transition_marker",
    "write_marker",
]


def marker_path(hf_template: str) -> Path:
    """The single marker location derived from the HF checkpoint template."""
    return Path(hf_template.format(rollout_id=0)).expanduser().parent / LATEST_JOB_MARKER_FILENAME


def read_marker(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except ValueError as exc:
        raise RuntimeError(f"invalid training marker: {path}") from exc
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("status") not in {
        "RUNNING",
        "CHECKPOINT",
        "UPDATING_WEIGHTS",
        "READY_TO_COMMIT",
        "HEAD_COMMITTED",
        "COMPLETE",
        "REJECTING",
        "REJECTED",
    }:
        raise RuntimeError(f"invalid training marker: {path}")
    if (
        not isinstance(value.get("job_id"), str)
        or not value["job_id"]
        or not isinstance(value.get("rollout_id"), int)
        or isinstance(value["rollout_id"], bool)
        or value["rollout_id"] < 0
    ):
        raise RuntimeError(f"invalid training marker: {path}")
    commit_acknowledged = value.get("commit_acknowledged")
    if commit_acknowledged is not None and not isinstance(commit_acknowledged, bool):
        raise RuntimeError(f"invalid training marker commit acknowledgement: {path}")
    if value["status"] == "HEAD_COMMITTED" and commit_acknowledged is not True:
        raise RuntimeError(f"training marker state requires a commit acknowledgement: {path}")
    if value["status"] != "RUNNING" and (
        not isinstance(value.get("checkpoint_path"), str)
        or not value["checkpoint_path"]
        or Path(value["checkpoint_path"]).is_symlink()
        or not Path(value["checkpoint_path"]).is_dir()
    ):
        raise RuntimeError(f"training marker has no checkpoint: {path}")
    if value["status"] in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"} and (
        not isinstance(value.get("runtime_load_id"), str) or not value["runtime_load_id"]
    ):
        raise RuntimeError(f"invalid training marker: {path}")
    return value


def transition_marker(
    path: Path,
    marker: dict[str, Any],
    status: MarkerStatus,
    **updates: Any,
) -> dict[str, Any]:
    """Durably advance one edge of the training-job state machine."""
    current = marker.get("status")
    if current not in _MARKER_TRANSITIONS or status not in _MARKER_TRANSITIONS[current]:
        raise RuntimeError(f"invalid training marker transition {current!r} -> {status!r}")
    marker.update(updates, status=status)
    write_marker(path, marker)
    return marker


def marker_disposition(marker: Mapping[str, Any] | None, job_id: str) -> MarkerDisposition:
    """Classify a recovered marker against an incoming job identity.

    - ``replay``: the same job already completed; return its recorded result.
    - ``resume``: the same job trained and checkpointed; only the serving
      publication remains.
    - ``conflict``: a different job is mid-flight; operator recovery required.
    - ``fresh``: nothing blocks running this job from the start.
    """
    status = None if marker is None else marker["status"]
    match (marker is not None and marker["job_id"] == job_id, status):
        case (True, "READY_TO_COMMIT" | "HEAD_COMMITTED" | "COMPLETE"):
            return "replay"
        case (True, "CHECKPOINT" | "REJECTED"):
            return "resume"
        case (
            _,
            "RUNNING" | "CHECKPOINT" | "UPDATING_WEIGHTS" | "READY_TO_COMMIT" | "HEAD_COMMITTED" | "REJECTING",
        ):
            return "conflict"
        case _:
            return "fresh"


def marker_result(marker: Mapping[str, Any], *, metrics: Mapping[str, Any] | None = None) -> TrainingJobResult:
    merged_metrics = _marker_metrics(marker)
    if metrics:
        merged_metrics.update(metrics)
    return TrainingJobResult(
        outcome="complete",
        runtime_load_id=str(marker["runtime_load_id"]),
        checkpoint_path=str(marker["checkpoint_path"]),
        metrics=merged_metrics or None,
        training_job_id=str(marker["job_id"]),
    )


def marker_checkpoint_result(marker: Mapping[str, Any]) -> TrainingJobResult:
    """Return a typed result for a job waiting on its serving-weight update."""
    return TrainingJobResult(
        outcome="checkpoint",
        runtime_load_id=str(marker.get("runtime_load_id", "pending")),
        checkpoint_path=str(marker["checkpoint_path"]),
        metrics=_marker_metrics(marker) or None,
        training_job_id=str(marker["job_id"]),
    )


def marker_rollouts(marker: Mapping[str, Any] | None) -> set[int]:
    rollout_id = None if marker is None else marker.get("rollout_id")
    return {rollout_id} if isinstance(rollout_id, int) else set()


def _marker_metrics(marker: Mapping[str, Any]) -> dict[str, Any]:
    """Merge durable method telemetry with Slime worker/loss metrics."""
    durable = marker.get("metrics")
    worker = marker.get("train_metrics")
    return {
        **(dict(durable) if isinstance(durable, Mapping) else {}),
        **(dict(worker) if isinstance(worker, Mapping) else {}),
    }
