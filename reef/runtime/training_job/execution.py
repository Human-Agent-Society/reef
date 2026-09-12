"""Idempotent training-step coordination without concrete model dependencies."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from reef.runtime.base import TrainingJobResult
from reef.runtime.training_job.marker import (
    marker_checkpoint_result,
    marker_disposition,
    marker_result,
    read_marker,
    transition_marker,
    write_marker,
)
from reef.runtime.training_job.state import TrainingJobState


def max_staleness(payload: Mapping[str, Any]) -> int:
    value = payload.get("max_staleness", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("training job max_staleness must be a non-negative integer")
    return value


def uses_staleness_admission(payload: Mapping[str, Any]) -> bool:
    """Whether the serving version is an admission fence, not job identity."""
    return max_staleness(payload) > 0 or "producing_runtime_load_ids" in payload


def training_job_id(payload: Mapping[str, Any]) -> str:
    """Preserve the retry-stable identity of the shared training payload."""
    identity = dict(payload)
    identity.pop("max_staleness", None)
    if uses_staleness_admission(payload):
        # A newer admission fence on retry must not repeat an optimizer step.
        identity.pop("expected_runtime_load_id", None)
    encoded = json.dumps(identity, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Backend-selected checkpoint index and path, optionally scoped to a scenario."""

    rollout_id: int
    path: Path
    scenario: str | None = None
    scenario_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, int) or isinstance(self.rollout_id, bool) or self.rollout_id < 0:
            raise ValueError("checkpoint rollout_id must be non-negative")
        if self.scenario is not None:
            if not isinstance(self.scenario, str) or not self.scenario:
                raise ValueError("checkpoint scenario must be non-empty")
            if (
                not isinstance(self.scenario_step, int)
                or isinstance(self.scenario_step, bool)
                or self.scenario_step < 0
            ):
                raise ValueError("checkpoint scenario_step must be non-negative")
        elif self.scenario_step is not None:
            raise ValueError("checkpoint scenario_step requires a scenario")


@dataclass(frozen=True)
class TrainingMetrics:
    """Worker/loss metrics and durable method telemetry from one optimizer step."""

    training: Mapping[str, Any] = field(default_factory=dict)
    durable: Mapping[str, Any] = field(default_factory=dict)


class PreparedTrainingJob(Protocol):
    """A prepared job whose reservation stays held through checkpoint recording.

    ``train`` may change model/optimizer state and returns all training metrics.
    ``save_checkpoint`` synchronously persists every required model/optimizer
    checkpoint and backend recovery metadata. Neither method may publish serving
    weights or advance Reef's job marker.
    """

    @property
    def checkpoint(self) -> TrainingCheckpoint: ...

    def train(self) -> TrainingMetrics: ...

    def save_checkpoint(self) -> None: ...


class TrainingJobBackend(Protocol):
    """Prepare/admit jobs without changing model or optimizer state.

    Validation, scoring, data packing and storage admission finish before the
    prepared job is yielded. The context retains resource reservations until
    Reef records the checkpoint (including on failure); it must not suppress
    exceptions. An early result may only be stale or storage-blocked.
    """

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        rollout_id: int,
        prior_marker: Mapping[str, Any] | None,
    ) -> AbstractContextManager[PreparedTrainingJob | TrainingJobResult]: ...


class TrainingExecution:
    """Own retry classification, RUNNING/CHECKPOINT transitions and train ordering.

    The caller serializes execution with publication and shutdown. Replay never
    prepares a backend job. RUNNING is intentionally ambiguous after a failure:
    Reef cannot infer whether an optimizer stepped, so automatic retry refuses.
    """

    def __init__(self, path: Path | None, backend: TrainingJobBackend, state: TrainingJobState) -> None:
        self._path = path
        self._backend = backend
        self._state = state

    def recover(self) -> dict[str, Any] | None:
        """Read restart state without guessing whether an optimizer step completed."""
        marker = read_marker(self._path) if self._path is not None else None
        if marker is not None and marker["status"] == "RUNNING":
            raise RuntimeError(f"ambiguous training job {marker['job_id']}")
        return marker

    def execute(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        job_id = training_job_id(payload)
        rollout_id = payload.get("rollout_id")
        if not isinstance(rollout_id, int) or isinstance(rollout_id, bool) or rollout_id < 0:
            raise ValueError("training job rollout_id must be non-negative")
        if self._path is None:
            raise RuntimeError("training job checkpoint path is not configured")
        marker = read_marker(self._path)
        disposition = marker_disposition(marker, job_id)
        if disposition == "conflict":
            if marker is None:
                raise RuntimeError("conflicting training disposition has no marker")
            raise RuntimeError(f"training marker is {marker['status']}; operator recovery required")
        if disposition != "fresh":
            if marker is None:
                raise RuntimeError("replayed training disposition has no marker")
            if marker["status"] == "COMPLETE":
                return marker_result(marker)
            return marker_checkpoint_result(marker)
        with self._backend.prepare(payload, job_id=job_id, rollout_id=rollout_id, prior_marker=marker) as prepared:
            if isinstance(prepared, TrainingJobResult):
                if prepared.outcome not in {"stale", "storage_blocked"}:
                    raise RuntimeError("training preparation may only return stale or storage_blocked")
                return prepared
            checkpoint = prepared.checkpoint
            running: dict[str, Any] = {"status": "RUNNING", "job_id": job_id, "rollout_id": checkpoint.rollout_id}
            if checkpoint.scenario is not None:
                running.update(scenario=checkpoint.scenario, scenario_step=checkpoint.scenario_step)
            write_marker(self._path, running)
            self._state.phase = "training"
            try:
                metrics = prepared.train()
                self._state.phase = "checkpointing"
                prepared.save_checkpoint()
                if checkpoint.path.is_symlink() or not checkpoint.path.is_dir():
                    raise RuntimeError(f"checkpoint is missing or unsafe: {checkpoint.path}")
                # Replay must see all telemetry with the checkpoint, even if
                # the process dies immediately after this transition.
                updates: dict[str, Any] = {"checkpoint_path": str(checkpoint.path)}
                if metrics.durable:
                    updates["metrics"] = dict(metrics.durable)
                if metrics.training:
                    updates["train_metrics"] = dict(metrics.training)
                transition_marker(self._path, running, "CHECKPOINT", **updates)
            except BaseException:
                self._state.phase = "training_failed" if self._state.phase == "training" else "checkpoint_failed"
                # Later retries replace the RPC error; retain the original
                # worker/checkpoint failure in the coordinator's process log.
                traceback.print_exc(file=sys.stderr)
                raise
            return marker_checkpoint_result(running)
        raise RuntimeError("training preparation suppressed an execution failure")
