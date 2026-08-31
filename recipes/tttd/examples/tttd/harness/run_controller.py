"""Durable step orchestration for TTT-Discover harness runs."""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


STATE_FORMAT_VERSION = 1


class TTTDRunStateError(RuntimeError):
    """The search archive and Reef's durable training state do not align."""


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
    artifact_versions: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioTrainingStatus:
    scenario_step: int
    weight_version: str | None
    batch_ready: bool
    failed_steps: tuple[ScenarioTrainingFailure, ...] = ()


class SearchHarness(Protocol):
    archive: Any

    def run_step(self, step: int) -> Sequence[Any]: ...


class TrainingStatusReader(Protocol):
    def scenario_status(self, scenario: str) -> ScenarioTrainingStatus | None: ...


class ReefTrainingStatusClient:
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
        step = value.get("scenario_step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise RuntimeError(f"Reef status for scenario {scenario!r} has invalid scenario_step {step!r}")
        weight_version = value.get("current_weight_version")
        if weight_version is not None and (not isinstance(weight_version, str) or not weight_version):
            raise RuntimeError(f"Reef status for scenario {scenario!r} has invalid weight version {weight_version!r}")
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
            versions = failure.get("artifact_versions")
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
                    artifact_versions=tuple(versions),
                )
            )
        return ScenarioTrainingStatus(
            scenario_step=step,
            weight_version=weight_version,
            batch_ready=value.get("batch_ready") is True,
            failed_steps=tuple(failed_steps),
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
        previous_weight_version: str | None,
        archive: Mapping[str, Any],
    ) -> None:
        self._write(
            {
                "format_version": STATE_FORMAT_VERSION,
                "identity": asdict(self.identity),
                "phase": "pending",
                "next_step": next_step,
                "previous_weight_version": previous_weight_version,
                "archive": dict(archive),
            }
        )

    def save_committed(
        self,
        *,
        next_step: int,
        weight_version: str,
        archive: Mapping[str, Any],
    ) -> None:
        if not weight_version:
            raise ValueError("committed TTTD state requires a weight version")
        self._write(
            {
                "format_version": STATE_FORMAT_VERSION,
                "identity": asdict(self.identity),
                "phase": "committed",
                "next_step": next_step,
                "weight_version": weight_version,
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
    weight_version: str | None


class TTTDRunController:
    """Run search steps only after the preceding Reef transaction commits."""

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
        next_step, weight_version = self._restore()
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
                previous_weight_version = None
            else:
                if before.scenario_step != step:
                    raise TTTDRunStateError(
                        f"Reef is at scenario step {before.scenario_step}, but the PUCT archive expects {step}"
                    )
                previous_weight_version = before.weight_version

            self._emit({"event": "tttd_step_started", "step": step, "weight_version": previous_weight_version})
            results = tuple(self.harness.run_step(step))
            expected_rollouts = (
                self.state_store.identity.groups_per_step * self.state_store.identity.rollouts_per_group
            )
            if len(results) != expected_rollouts:
                raise RuntimeError(f"TTTD step {step} returned {len(results)} rollouts, expected {expected_rollouts}")
            archive = self.harness.archive.state_dict()
            self.state_store.save_pending(
                next_step=step + 1,
                previous_weight_version=previous_weight_version,
                archive=archive,
            )
            committed = self._wait_for_step(step + 1, previous_weight_version)
            self.state_store.save_committed(
                next_step=step + 1,
                weight_version=committed.weight_version,
                archive=archive,
            )
            weight_version = committed.weight_version
            all_results.extend(results)
            candidates = self.harness.archive.candidates
            self._emit(
                {
                    "event": "tttd_step_committed",
                    "step": step,
                    "next_step": step + 1,
                    "weight_version": weight_version,
                    "archive_size": len(candidates),
                    "archive_best_reward": max(candidate.reward for candidate in candidates),
                }
            )

        return TTTDRunOutcome(
            results=tuple(all_results),
            start_step=start_step,
            next_step=total_steps,
            weight_version=weight_version,
        )

    def _restore(self) -> tuple[int, str | None]:
        saved = self.state_store.load()
        if saved is None:
            current = self.status_reader.scenario_status(self.state_store.identity.scenario)
            if current is not None and current.scenario_step != 0:
                raise TTTDRunStateError(
                    f"Reef scenario is already at step {current.scenario_step}, but no PUCT state exists"
                )
            return 0, None if current is None else current.weight_version

        self.harness.archive.load_state_dict(saved["archive"])
        next_step = int(saved["next_step"])
        if saved["phase"] == "pending":
            previous = saved.get("previous_weight_version")
            if previous is not None and not isinstance(previous, str):
                raise TTTDRunStateError("pending TTTD state has an invalid previous weight version")
            committed = self._wait_for_step(next_step, previous)
            self.state_store.save_committed(
                next_step=next_step,
                weight_version=committed.weight_version,
                archive=saved["archive"],
            )
            return next_step, committed.weight_version

        expected_version = saved.get("weight_version")
        if not isinstance(expected_version, str) or not expected_version:
            raise TTTDRunStateError("committed TTTD state has no weight version")
        current = self._wait_for_current_step(next_step)
        if not isinstance(current.weight_version, str) or not current.weight_version:
            raise TTTDRunStateError(f"Reef restored step {next_step} without a serving weight version")
        if current.weight_version != expected_version:
            # Serving versions are engine-session scoped. A restarted runtime
            # republishes the checkpoint under a new version while preserving
            # Reef's durable scenario step, which is the cross-session fence.
            self.state_store.save_committed(
                next_step=next_step,
                weight_version=current.weight_version,
                archive=saved["archive"],
            )
            self._emit(
                {
                    "event": "tttd_weight_version_rebound",
                    "step": next_step,
                    "previous_weight_version": expected_version,
                    "weight_version": current.weight_version,
                }
            )
        return next_step, current.weight_version

    def _wait_for_current_step(
        self,
        expected_step: int,
        *,
        previous_weight_version: str | None = None,
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
                        f"TTTD step {failure.step} failed ({failure.reason}); "
                        f"artifact versions={list(failure.artifact_versions)!r}"
                    )
                if last.scenario_step > expected_step:
                    raise TTTDRunStateError(
                        f"Reef advanced to step {last.scenario_step}, beyond PUCT step {expected_step}"
                    )
                if last.scenario_step == expected_step and (
                    previous_weight_version is None or last.weight_version != previous_weight_version
                ):
                    # Reef commits the durable scenario head before the runtime
                    # acknowledges that commit and refreshes its public serving
                    # version. Keep polling through that short reconciliation
                    # window so a successful update is not mistaken for stale
                    # weights after checkpoint recovery.
                    return last
            self._sleep(self.poll_interval_s)
        raise TimeoutError(
            f"Reef scenario did not restore step {expected_step} after {self.train_timeout_s:g}s: {last}"
        )

    def _wait_for_step(
        self,
        expected_step: int,
        previous_weight_version: str | None,
    ) -> ScenarioTrainingStatus:
        current = self._wait_for_current_step(
            expected_step,
            previous_weight_version=previous_weight_version,
        )
        if not isinstance(current.weight_version, str) or not current.weight_version:
            raise TTTDRunStateError(f"Reef committed step {expected_step} without a serving weight version")
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
]
