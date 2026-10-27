"""Durable step orchestration for TTT-Discover harness runs.

One search step is one Reef training job. Reef commits a job at its
submission, at each saved checkpoint and at its end, so the controller counts
finished jobs (``finished_jobs`` in ``GET /reef/status``), not scenario steps.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATE_FORMAT_VERSION = 1


class TTTDRunStateError(RuntimeError):
    """The search archive and Reef's durable training state do not align."""


class TTTDTrainingTimeoutError(RuntimeError):
    """Reef did not commit a TTTD training step before its deadline."""


@dataclass(frozen=True)
class TTTDRunIdentity:
    scenario: str
    model: str
    recipe: str
    inference_path: str
    instruction_sha256: str
    groups_per_step: int
    rollouts_per_group: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    enable_thinking: bool
    exploration: float
    invalid_reward: float


@dataclass(frozen=True)
class ScenarioTrainingFailure:
    step: int
    reason: str
    release_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioTrainingStatus:
    """One scenario's training as ``GET /reef/status`` reports it.

    ``finished_jobs`` is the number of training jobs that ended, a
    non-negative int; one job trains one search step. ``running_job_id`` is
    the running job's ID, or ``None`` while no job runs. ``runtime_load_id``
    is the serving load ID, or ``None``. ``failed_steps`` are the TTTD steps
    the processor discarded. ``failed_job_error`` is the error of the last
    commit's job when that job failed, or ``None``. ``ReefTrainingStatusClient``
    makes the value and ``TTTDRunController`` reads it.
    """

    finished_jobs: int
    running_job_id: str | None
    runtime_load_id: str | None
    job_data_ready: bool
    failed_steps: tuple[ScenarioTrainingFailure, ...] = ()
    failed_job_error: str | None = None


class SearchHarness(ABC):
    archive: Any

    @abstractmethod
    def run_step(self, step: int) -> Sequence[Any]: ...


class TrainingStatusReader(ABC):
    @abstractmethod
    def scenario_status(self, scenario: str) -> ScenarioTrainingStatus | None: ...


class ReefTrainingStatusClient(TrainingStatusReader):
    """Read Reef's public training status without coupling the harness to Ray."""

    def __init__(
        self,
        service_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 30.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("status timeout must be positive")
        self._url = service_url.rstrip("/") + "/reef/status"
        self._token = token
        self._timeout_s = float(timeout_s)
        self._opener = opener

    def scenario_status(self, scenario: str) -> ScenarioTrainingStatus | None:
        request = urllib.request.Request(self._url)
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        try:
            with self._opener(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read())
        except Exception as exc:
            raise RuntimeError(f"cannot read Reef training status from {self._url}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("Reef training status is not an object")
        if payload.get("error"):
            raise RuntimeError(f"Reef training worker failed: {payload['error']}")
        preload_errors = payload.get("preload_errors")
        if isinstance(preload_errors, Mapping) and scenario in preload_errors:
            raise RuntimeError(f"Reef could not restore scenario {scenario!r}: {preload_errors[scenario]}")
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, Mapping) or scenario not in scenarios:
            return None
        value = scenarios[scenario]
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Reef status for scenario {scenario!r} is not an object")
        finished_jobs = value.get("finished_jobs")
        if not isinstance(finished_jobs, int) or isinstance(finished_jobs, bool) or finished_jobs < 0:
            raise RuntimeError(f"Reef status for scenario {scenario!r} has invalid finished_jobs {finished_jobs!r}")
        running_job = value.get("training_job")
        running_job_id = running_job.get("job_id") if isinstance(running_job, Mapping) else None
        if running_job is not None and (not isinstance(running_job_id, str) or not running_job_id):
            raise RuntimeError(f"Reef status for scenario {scenario!r} has invalid training_job {running_job!r}")
        runtime_load_id = value.get("current_runtime_load_id")
        if runtime_load_id is not None and (not isinstance(runtime_load_id, str) or not runtime_load_id):
            raise RuntimeError(
                f"Reef status for scenario {scenario!r} has invalid runtime load ID {runtime_load_id!r}"
            )
        processor = value.get("processor")
        failed_values = processor.get("failed_steps", []) if isinstance(processor, Mapping) else []
        if not isinstance(failed_values, list):
            raise RuntimeError(f"Reef status for scenario {scenario!r} has invalid processor failures")
        failed_steps: list[ScenarioTrainingFailure] = []
        for failure in failed_values:
            if not isinstance(failure, Mapping):
                raise RuntimeError(f"Reef status for scenario {scenario!r} has an invalid processor failure")
            failed_step = failure.get("step")
            reason = failure.get("reason")
            versions = failure.get("release_ids")
            if (
                not isinstance(failed_step, int)
                or isinstance(failed_step, bool)
                or failed_step < 0
                or not isinstance(reason, str)
                or not reason
                or not isinstance(versions, list)
                or not all(isinstance(version, str) and version for version in versions)
            ):
                raise RuntimeError(f"Reef status for scenario {scenario!r} has an invalid processor failure")
            failed_steps.append(
                ScenarioTrainingFailure(
                    step=failed_step,
                    reason=reason,
                    release_ids=tuple(versions),
                )
            )
        # The last commit's job outcome: a failed job ends with event "failed" and its error.
        last_commit = value.get("last_committed_step")
        metrics = last_commit.get("metrics") if isinstance(last_commit, Mapping) else None
        last_job = metrics.get("training_job") if isinstance(metrics, Mapping) else None
        failed_job_error = (
            str(last_job.get("error") or "the training job failed")
            if isinstance(last_job, Mapping) and last_job.get("event") == "failed"
            else None
        )
        return ScenarioTrainingStatus(
            finished_jobs=finished_jobs,
            running_job_id=running_job_id,
            runtime_load_id=runtime_load_id,
            job_data_ready=value.get("job_data_ready") is True,
            failed_steps=tuple(failed_steps),
            failed_job_error=failed_job_error,
        )


class TTTDRunStateStore:
    """Atomically pair the post-step PUCT archive with Reef's committed step."""

    def __init__(self, path: str | Path, identity: TTTDRunIdentity) -> None:
        self.path = Path(path)
        self.identity = identity

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TTTDRunStateError(f"cannot read TTTD state {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format_version") != STATE_FORMAT_VERSION:
            raise TTTDRunStateError(f"unsupported TTTD state format in {self.path}")
        if payload.get("identity") != asdict(self.identity):
            raise TTTDRunStateError(
                f"TTTD state identity does not match this run: {payload.get('identity')!r} != {asdict(self.identity)!r}"
            )
        phase = payload.get("phase")
        next_step = payload.get("next_step")
        if phase not in {"pending", "committed"}:
            raise TTTDRunStateError(f"TTTD state has invalid phase {phase!r}")
        if not isinstance(next_step, int) or isinstance(next_step, bool) or next_step < 1:
            raise TTTDRunStateError(f"TTTD state has invalid next_step {next_step!r}")
        if not isinstance(payload.get("archive"), Mapping):
            raise TTTDRunStateError("TTTD state has no PUCT archive")
        return payload

    def save_pending(
        self,
        *,
        next_step: int,
        previous_runtime_load_id: str | None,
        archive: Mapping[str, Any],
    ) -> None:
        self._write(
            {
                "format_version": STATE_FORMAT_VERSION,
                "identity": asdict(self.identity),
                "phase": "pending",
                "next_step": next_step,
                "previous_runtime_load_id": previous_runtime_load_id,
                "archive": dict(archive),
            }
        )

    def save_committed(
        self,
        *,
        next_step: int,
        runtime_load_id: str,
        archive: Mapping[str, Any],
    ) -> None:
        if not runtime_load_id:
            raise ValueError("committed TTTD state requires a runtime load ID")
        self._write(
            {
                "format_version": STATE_FORMAT_VERSION,
                "identity": asdict(self.identity),
                "phase": "committed",
                "next_step": next_step,
                "runtime_load_id": runtime_load_id,
                "archive": dict(archive),
            }
        )

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TTTDRunOutcome:
    results: tuple[Any, ...]
    start_step: int
    next_step: int
    runtime_load_id: str | None


class TTTDRunController:
    """Run search steps only after the preceding step's Reef training job ends."""

    def __init__(
        self,
        harness: SearchHarness,
        status_reader: TrainingStatusReader,
        state_store: TTTDRunStateStore,
        *,
        train_timeout_s: float = 14_400.0,
        poll_interval_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        emit: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if train_timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("training timeout and poll interval must be positive")
        self.harness = harness
        self.status_reader = status_reader
        self.state_store = state_store
        self.train_timeout_s = float(train_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self._sleep = sleep
        self._emit = emit or (lambda _event: None)

    def run(self, total_steps: int) -> TTTDRunOutcome:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        next_step, runtime_load_id = self._restore()
        start_step = next_step
        if next_step > total_steps:
            raise TTTDRunStateError(
                f"saved TTTD state is already at step {next_step}, beyond requested total {total_steps}"
            )

        all_results: list[Any] = []
        for step in range(next_step, total_steps):
            before = self.status_reader.scenario_status(self.state_store.identity.scenario)
            if before is None:
                if step != 0:
                    raise TTTDRunStateError(f"Reef has no status for resumed TTTD step {step}")
                previous_runtime_load_id = None
            else:
                if before.running_job_id is not None or before.finished_jobs != step:
                    raise TTTDRunStateError(
                        f"Reef finished {before.finished_jobs} training jobs (running: {before.running_job_id}), "
                        f"but the PUCT archive expects {step}"
                    )
                previous_runtime_load_id = before.runtime_load_id

            self._emit({"event": "tttd_step_started", "step": step, "runtime_load_id": previous_runtime_load_id})
            results = tuple(self.harness.run_step(step))
            expected_rollouts = (
                self.state_store.identity.groups_per_step * self.state_store.identity.rollouts_per_group
            )
            if len(results) != expected_rollouts:
                raise RuntimeError(f"TTTD step {step} returned {len(results)} rollouts, expected {expected_rollouts}")
            archive = self.harness.archive.state_dict()
            self.state_store.save_pending(
                next_step=step + 1,
                previous_runtime_load_id=previous_runtime_load_id,
                archive=archive,
            )
            committed = self._wait_for_step(step + 1, previous_runtime_load_id)
            self.state_store.save_committed(
                next_step=step + 1,
                runtime_load_id=committed.runtime_load_id,
                archive=archive,
            )
            runtime_load_id = committed.runtime_load_id
            all_results.extend(results)
            candidates = self.harness.archive.candidates
            self._emit(
                {
                    "event": "tttd_step_committed",
                    "step": step,
                    "next_step": step + 1,
                    "runtime_load_id": runtime_load_id,
                    "archive_size": len(candidates),
                    "archive_best_reward": max(candidate.reward for candidate in candidates),
                }
            )

        return TTTDRunOutcome(
            results=tuple(all_results),
            start_step=start_step,
            next_step=total_steps,
            runtime_load_id=runtime_load_id,
        )

    def _restore(self) -> tuple[int, str | None]:
        saved = self.state_store.load()
        if saved is None:
            current = self.status_reader.scenario_status(self.state_store.identity.scenario)
            if current is not None and (current.finished_jobs != 0 or current.running_job_id is not None):
                raise TTTDRunStateError(
                    f"Reef already finished {current.finished_jobs} training jobs "
                    f"(running: {current.running_job_id}), but no PUCT state exists"
                )
            return 0, None if current is None else current.runtime_load_id

        self.harness.archive.load_state_dict(saved["archive"])
        next_step = int(saved["next_step"])
        if saved["phase"] == "pending":
            previous = saved.get("previous_runtime_load_id")
            if previous is not None and not isinstance(previous, str):
                raise TTTDRunStateError("pending TTTD state has an invalid previous runtime load ID")
            committed = self._wait_for_step(next_step, previous)
            self.state_store.save_committed(
                next_step=next_step,
                runtime_load_id=committed.runtime_load_id,
                archive=saved["archive"],
            )
            return next_step, committed.runtime_load_id

        expected_version = saved.get("runtime_load_id")
        if not isinstance(expected_version, str) or not expected_version:
            raise TTTDRunStateError("committed TTTD state has no runtime load ID")
        current = self._wait_for_current_step(next_step)
        if not isinstance(current.runtime_load_id, str) or not current.runtime_load_id:
            raise TTTDRunStateError(f"Reef restored step {next_step} without a serving runtime load ID")
        if current.runtime_load_id != expected_version:
            # Serving versions are engine-session scoped. A restarted runtime
            # republishes the checkpoint under a new version while preserving
            # Reef's durable scenario step, which is the cross-session fence.
            self.state_store.save_committed(
                next_step=next_step,
                runtime_load_id=current.runtime_load_id,
                archive=saved["archive"],
            )
            self._emit(
                {
                    "event": "tttd_runtime_load_id_rebound",
                    "step": next_step,
                    "previous_runtime_load_id": expected_version,
                    "runtime_load_id": current.runtime_load_id,
                }
            )
        return next_step, current.runtime_load_id

    def _wait_for_current_step(
        self,
        expected_step: int,
        *,
        previous_runtime_load_id: str | None = None,
    ) -> ScenarioTrainingStatus:
        deadline = time.monotonic() + self.train_timeout_s
        last: ScenarioTrainingStatus | None = None
        while time.monotonic() < deadline:
            last = self.status_reader.scenario_status(self.state_store.identity.scenario)
            if last is not None:
                failed_rollout = expected_step - 1
                failure = next((item for item in last.failed_steps if item.step == failed_rollout), None)
                if failure is not None:
                    raise TTTDRunStateError(
                        f"TTTD step {failure.step} failed ({failure.reason}); releases={list(failure.release_ids)!r}"
                    )
                if last.finished_jobs > expected_step:
                    raise TTTDRunStateError(
                        f"Reef finished {last.finished_jobs} training jobs, beyond PUCT step {expected_step}"
                    )
                if last.finished_jobs == expected_step and last.running_job_id is None:
                    if last.failed_job_error is not None:
                        raise TTTDRunStateError(
                            f"the training job of TTTD step {expected_step - 1} failed: {last.failed_job_error}"
                        )
                    if previous_runtime_load_id is None or last.runtime_load_id != previous_runtime_load_id:
                        # Reef commits the served checkpoint before the runtime
                        # acknowledges it and refreshes its public serving
                        # version. Keep polling through that short
                        # reconciliation window so a successful update is not
                        # mistaken for stale weights after checkpoint recovery.
                        return last
            self._sleep(self.poll_interval_s)
        raise TTTDTrainingTimeoutError(
            f"Reef scenario did not restore step {expected_step} after {self.train_timeout_s:g}s: {last}"
        )

    def _wait_for_step(
        self,
        expected_step: int,
        previous_runtime_load_id: str | None,
    ) -> ScenarioTrainingStatus:
        current = self._wait_for_current_step(
            expected_step,
            previous_runtime_load_id=previous_runtime_load_id,
        )
        if not isinstance(current.runtime_load_id, str) or not current.runtime_load_id:
            raise TTTDRunStateError(f"Reef committed step {expected_step} without a serving runtime load ID")
        return current


__all__ = [
    "ReefTrainingStatusClient",
    "ScenarioTrainingFailure",
    "ScenarioTrainingStatus",
    "TTTDRunController",
    "TTTDRunIdentity",
    "TTTDRunOutcome",
    "TTTDRunStateError",
    "TTTDRunStateStore",
    "TTTDTrainingTimeoutError",
]
