"""Backend-neutral ordering for checkpoint publication and commit acknowledgement."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from reef.runtime.adapter_residency import AdapterCapacityExhausted, AdapterEvictionFailed
from reef.runtime.training_job.marker import read_marker, transition_marker, write_marker
from reef.runtime.training_job.state import TrainingJobState


class WeightPublisher(Protocol):
    """Model operations needed by Reef's durable publication transaction.

    Transport stays in the backend: ``publish`` must verify that all engines
    received the returned runtime load ID, without resuming generation.
    ``recover`` replaces uncertain engines and restores other committed adapters;
    its marker is ``None`` when no durable training job exists.
    ``pause`` and ``resume`` are idempotent barriers across every serving engine.
    ``abort`` prevents inference after an uncertain or partial update.
    """

    def recover(self, marker: Mapping[str, Any] | None) -> None: ...

    def pause(self) -> None: ...

    def publish(self, marker: Mapping[str, Any], *, force_full: bool) -> str: ...

    def republish(self, runtime_load_id: str, marker: Mapping[str, Any] | None) -> str:
        """Resend unchanged trainer weights with a full transfer and the same identity."""

    def resume(self) -> None: ...

    def restore_incumbent(self) -> None: ...

    def abort(self) -> None: ...


@dataclass(frozen=True)
class PublicationResult:
    """A durable result and whether this call performed its publication."""

    marker: Mapping[str, Any]
    published: bool


class TrainingPublication:
    """Own marker transitions and the barrier that gates serving on Reef's commit.

    The enclosing training coordinator must serialize these operations with
    training and shutdown. A single shared ``phase`` also lets its training and
    checkpoint operations report progress through the same health endpoint.
    No tensor data passes through this object. A missing marker path supports
    legacy coordinators that never execute durable training jobs.
    """

    def __init__(self, path: Path | None, publisher: WeightPublisher) -> None:
        self._path = path
        self._publisher = publisher
        self.state = TrainingJobState()

    @property
    def phase(self) -> str:
        return self.state.phase

    @phase.setter
    def phase(self, phase: str) -> None:
        self.state.phase = phase

    def _require_path(self) -> Path:
        if self._path is None:
            raise RuntimeError("training job checkpoint path is not configured")
        return self._path

    def _job_marker(self, job_id: str) -> dict[str, Any]:
        if not job_id:
            raise ValueError("training_job_id must be non-empty")
        marker = read_marker(self._require_path())
        if marker is None or marker["job_id"] != job_id:
            raise RuntimeError(f"unknown training job {job_id!r}")
        return marker

    def _abort(self) -> None:
        self.phase = "weight_sync_failed"
        with suppress(Exception):
            self._publisher.abort()

    def publish(self, job_id: str) -> PublicationResult:
        """Publish one checkpoint, leaving every engine paused."""
        marker = self._job_marker(job_id)
        status = marker["status"]
        if status in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            return PublicationResult(marker, published=False)
        if status not in {"CHECKPOINT", "UPDATING_WEIGHTS"}:
            raise RuntimeError(f"training job is {status}; operator recovery required")
        recovering = status == "UPDATING_WEIGHTS"
        try:
            if recovering:
                self._publisher.recover(marker)
            self._publisher.pause()
            # A failed pause has changed no weights. Keep CHECKPOINT retryable.
            if not recovering:
                transition_marker(self._require_path(), marker, "UPDATING_WEIGHTS")
            self.phase = "publishing"
            version = self._publisher.publish(marker, force_full=recovering)
            if not isinstance(version, str) or not version:
                raise RuntimeError("weight publisher returned an empty runtime load ID")
            transition_marker(self._require_path(), marker, "READY_TO_COMMIT", runtime_load_id=version)
            self.phase = "awaiting_commit"
        except AdapterEvictionFailed:
            self._abort()
            raise
        except AdapterCapacityExhausted:
            # No weights left the trainer; terminating unrelated adapters would
            # not resolve capacity pressure. The pending marker still gates commit.
            self.phase = "serving"
            raise
        except BaseException:
            self._abort()
            raise
        return PublicationResult(marker, published=True)

    def republish(self, runtime_load_id: str) -> str:
        """Restore replaced engines without bypassing the durable commit gate.

        A trainer with an unfinished or rejected candidate cannot represent the
        incumbent. Such jobs must use their normal publication/recovery path.
        The owner must retain the last verified identity across failed attempts.
        """
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise ValueError("republication requires a non-empty runtime load ID")
        if self.phase == "stopped":
            raise RuntimeError("training coordinator is stopped")
        marker = read_marker(self._path) if self._path is not None else None
        if marker is not None:
            if marker["status"] not in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
                raise RuntimeError(f"cannot republish serving from {marker['status']}; use training job recovery")
            if marker["runtime_load_id"] != runtime_load_id:
                raise RuntimeError("serving runtime load ID does not match the training marker")
        try:
            # Fence before recovery: replacement engines must inherit pause
            # intent, and monitoring must not restart before verified transfer.
            self._publisher.pause()
            self.phase = "publishing"
            self._publisher.recover(marker)
            published = self._publisher.republish(runtime_load_id, marker)
            if published != runtime_load_id:
                raise RuntimeError(
                    f"serving republication changed runtime load ID {runtime_load_id!r} to {published!r}"
                )
        except BaseException:
            self._abort()
            raise
        self.finish_recovery(marker, published)
        return published

    def reject(self, job_id: str) -> Mapping[str, Any]:
        """Durably reject before restoring the incumbent's released resources."""
        marker = self._job_marker(job_id)
        status = marker["status"]
        if status == "REJECTED":
            return marker
        if status == "CHECKPOINT":
            transition_marker(self._require_path(), marker, "REJECTING")
        elif status != "REJECTING":
            raise RuntimeError(f"cannot reject training job {job_id!r} from {status}")
        self._publisher.restore_incumbent()
        transition_marker(self._require_path(), marker, "REJECTED")
        self.phase = "serving"
        return marker

    def acknowledge(self, job_id: str) -> None:
        """Persist Reef's commit acknowledgement before allowing requests again."""
        marker = self._job_marker(job_id)
        if marker["status"] == "COMPLETE":
            if marker.get("commit_acknowledged") is not True:
                write_marker(self._require_path(), {**marker, "commit_acknowledged": True})
            return
        if marker["status"] == "READY_TO_COMMIT":
            transition_marker(self._require_path(), marker, "HEAD_COMMITTED", commit_acknowledged=True)
        if marker["status"] != "HEAD_COMMITTED":
            raise RuntimeError(f"cannot acknowledge training job {job_id!r} from {marker['status']}")
        self._publisher.resume()
        transition_marker(self._require_path(), marker, "COMPLETE", commit_acknowledged=True)
        self.phase = "serving"

    def prepare_recovery(self, marker: dict[str, Any] | None) -> None:
        """Fence a recovered candidate before the backend republishes its checkpoint."""
        if marker is None or marker["status"] not in {
            "CHECKPOINT",
            "UPDATING_WEIGHTS",
            "READY_TO_COMMIT",
            "HEAD_COMMITTED",
        }:
            return
        try:
            self._publisher.pause()
            if marker["status"] == "CHECKPOINT":
                transition_marker(self._require_path(), marker, "UPDATING_WEIGHTS")
        except BaseException:
            self._abort()
            raise

    def finish_recovery(self, marker: dict[str, Any] | None, runtime_load_id: str) -> None:
        """Record recovered publication while preserving the durable commit gate.

        The backend has already restored checkpoint tensors and verified every
        engine. Previously published jobs must retain their serving identity.
        """
        status = None if marker is None else marker["status"]
        try:
            if not isinstance(runtime_load_id, str) or not runtime_load_id:
                raise RuntimeError("weight publisher returned an empty runtime load ID")
            if (
                marker is not None
                and status in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}
                and runtime_load_id != marker["runtime_load_id"]
            ):
                raise RuntimeError(
                    "checkpoint republication changed runtime load ID "
                    f"{marker['runtime_load_id']!r} to {runtime_load_id!r}"
                )
            if marker is not None and status == "UPDATING_WEIGHTS":
                transition_marker(self._require_path(), marker, "READY_TO_COMMIT", runtime_load_id=runtime_load_id)
                self.phase = "awaiting_commit"
            elif status == "READY_TO_COMMIT":
                self.phase = "awaiting_commit"
            elif marker is not None and status == "HEAD_COMMITTED":
                self.acknowledge(str(marker["job_id"]))
            elif status in {None, "COMPLETE", "REJECTED"}:
                self._publisher.resume()
                self.phase = "serving"
            else:
                raise RuntimeError(f"training marker is {status}; operator recovery required")
        except BaseException:
            self._abort()
            raise
