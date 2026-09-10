"""Durable step orchestration for Guidance-TTT runs.

One search step is one Reef training transaction. The controller runs a step
only after the previous transaction has committed: it samples ``G x R``
guidance rollouts, waits for the durable train/checkpoint/publish job, checks
that the LoRA optimizer really consumed the reserved batch, and only then
pairs the post-step archive with that committed step.

Nothing here is task specific. The Harbor agent supplies the harness, the
archive, and the run settings; ``run()`` returns the rollouts and the per-step
summaries that the agent hands back to Harbor.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

RESUME_FORMAT_VERSION = 1


class GuidanceRunStateError(RuntimeError):
    """The Guidance archive and Reef's durable training state do not align."""


@dataclass(frozen=True)
class GuidanceRunIdentity:
    """Settings a resumed run must reproduce exactly."""

    model: str
    executor: str
    gpt_oss_reasoning_effort: str
    groups_per_step: int
    rollouts_per_group: int
    guidance_max_tokens: int
    sequence_length: int
    lora_rank: int
    tensor_parallel_size: int


@dataclass(frozen=True)
class GuidanceRunOutcome:
    results: tuple[Any, ...]
    start_step: int
    next_step: int
    step_summaries: tuple[dict[str, Any], ...]
    runtime_load_id: str | None


class SearchHarness(Protocol):
    def run_step(self, step: int) -> Sequence[Any]: ...


class TrainingBridge(Protocol):
    """Reef's durable training transaction, seen from the harness."""

    def start_step(self) -> int: ...

    def wait_for_step(self, *, expected_completed_steps: int, expected_rollout_id: int) -> Mapping[str, Any]: ...


class RayTrainingBridge:
    """Read the durable training job through Reef's Ray train bridge.

    The bridge exposes the optimizer transaction (grad norm, LoRA parameter
    counts, consumed batch) that ``/reef/status`` deliberately does not, so a
    step that trained on nothing cannot be mistaken for a healthy one.
    """

    def __init__(
        self,
        service_url: str,
        scenario: str,
        *,
        token: str | None = None,
        ray_address: str,
        ray_namespace: str = "reef",
        ray_actor_name: str = "reef-train-bridge",
        timeout_s: float = 14_400.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        if timeout_s <= 0 or poll_interval_s < 0:
            raise ValueError("training timeout must be positive and the poll interval non-negative")
        import ray

        if not ray.is_initialized():
            import logging

            ray.init(address=ray_address, namespace=ray_namespace, logging_level=logging.ERROR)
        self._ray = ray
        self._actor = ray.get_actor(ray_actor_name, namespace=ray_namespace)
        self._status_url = service_url.rstrip("/") + "/reef/status"
        self._token = token
        self._scenario = scenario
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)

    def start_step(self) -> int:
        return int(self._ray.get(self._actor.start_rollout_id.remote()))

    def wait_for_step(self, *, expected_completed_steps: int, expected_rollout_id: int) -> Mapping[str, Any]:
        return wait_for_training_step(
            health=lambda: dict(self._ray.get(self._actor.health.remote(), timeout=30)),
            status=self._scenario_status,
            expected_completed_steps=expected_completed_steps,
            expected_rollout_id=expected_rollout_id,
            timeout_s=self._timeout_s,
            poll_interval_s=self._poll_interval_s,
        )

    def _scenario_status(self) -> dict[str, Any]:
        request = urllib.request.Request(self._status_url)
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
        return scenario_status(body, self._scenario)


def wait_for_training_step(
    *,
    health: Callable[[], Mapping[str, Any]],
    status: Callable[[], Mapping[str, Any]],
    expected_completed_steps: int,
    expected_rollout_id: int,
    timeout_s: float,
    poll_interval_s: float,
) -> Mapping[str, Any]:
    """Wait for Reef's asynchronous durable train/checkpoint/publish job."""
    deadline = time.monotonic() + timeout_s
    last_health: Mapping[str, Any] = {}
    last_status: Mapping[str, Any] = {}
    while time.monotonic() < deadline:
        last_health = health()
        if last_health.get("ok") is not True:
            raise RuntimeError(f"durable training bridge failed: {last_health}")
        last_status = status()
        failure = failed_training_step(last_status, expected_rollout_id)
        if failure is not None:
            raise RuntimeError(
                f"TTTD step {expected_rollout_id} failed ({failure['reason']}) because its reports span "
                f"releases {failure['release_ids']!r}"
            )
        if last_health.get("completed_train_steps") == expected_completed_steps:
            if last_health.get("last_train_rollout_id") != expected_rollout_id:
                raise RuntimeError(f"durable training completed the wrong rollout: {last_health}")
            if last_health.get("phase") == "serving":
                return last_health
        if poll_interval_s:
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"training rollout {expected_rollout_id} did not complete after {timeout_s:g}s: "
        f"bridge={last_health}; reef={last_status}"
    )


def scenario_status(body: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    """Read one scenario's generic asynchronous training status."""
    training_error = body.get("error")
    if training_error is not None:
        if not isinstance(training_error, str) or not training_error:
            raise RuntimeError(f"Reef returned an invalid training error: {body!r}")
        raise RuntimeError(f"Reef training failed: {training_error}")
    scenarios = body.get("scenarios")
    status = scenarios.get(scenario) if isinstance(scenarios, Mapping) else None
    if not isinstance(status, Mapping):
        raise RuntimeError(f"Reef returned no training status for {scenario!r}: {body!r}")
    return dict(status)


def failed_training_step(status: Mapping[str, Any], step: int) -> dict[str, Any] | None:
    """Return this step's terminal processor failure, if present."""
    processor = status.get("processor")
    failed_steps = processor.get("failed_steps") if isinstance(processor, Mapping) else None
    if not isinstance(failed_steps, list):
        raise RuntimeError(f"Reef returned invalid processor status: {status!r}")
    for failure in failed_steps:
        if (
            not isinstance(failure, Mapping)
            or not isinstance(failure.get("reason"), str)
            or not failure["reason"]
            or not isinstance(failure.get("release_ids"), list)
        ):
            raise RuntimeError(f"Reef returned invalid processor status: {status!r}")
        if failure.get("step") == step:
            return dict(failure)
    return None


def require_step_success(
    *,
    expected_rollouts: int,
    actual_rollouts: int,
    retained_trajectories: int,
    bridge_health: Mapping[str, Any],
    expected_completed_train_steps: int,
    expected_rollout_id: int,
    grad_norm: Any,
    lora_rank: int,
) -> float:
    """Fail closed unless the LoRA optimizer really consumed this step."""
    if actual_rollouts != expected_rollouts:
        raise RuntimeError(f"sampled {actual_rollouts} rollouts, expected {expected_rollouts}")
    if bridge_health.get("ok") is not True or bridge_health.get("phase") != "serving":
        raise RuntimeError(f"bridge did not return to serving: {bridge_health}")
    if bridge_health.get("completed_train_steps") != expected_completed_train_steps:
        raise RuntimeError(f"expected {expected_completed_train_steps} completed train steps: {bridge_health}")
    if bridge_health.get("last_train_rollout_id") != expected_rollout_id:
        raise RuntimeError(f"expected last rollout id {expected_rollout_id}: {bridge_health}")
    if not isinstance(grad_norm, (int, float)) or not math.isfinite(float(grad_norm)) or grad_norm <= 0:
        raise RuntimeError(f"optimizer did not report a positive finite grad norm: {grad_norm!r}")
    metrics = bridge_health.get("last_train_metrics") or {}
    reported_batch = metrics.get("train/global_batch_size")
    if reported_batch != retained_trajectories:
        raise RuntimeError(f"trainer consumed global batch {reported_batch!r}, expected {retained_trajectories}")
    trainable = metrics.get("train/lora_trainable_parameters")
    base_trainable = metrics.get("train/lora_base_trainable_parameters")
    lora_b_nonzero = metrics.get("train/lora_b_nonzero")
    lora_b_l1 = metrics.get("train/lora_b_l1")
    if not isinstance(trainable, (int, float)) or trainable <= 0:
        raise RuntimeError(f"rank-{lora_rank} LoRA exposed no trainable adapter parameters: {metrics}")
    if base_trainable != 0:
        raise RuntimeError(f"LoRA step left {base_trainable!r} base parameters trainable")
    if not isinstance(lora_b_nonzero, (int, float)) or lora_b_nonzero <= 0:
        raise RuntimeError(f"LoRA-B remained zero after the optimizer step: {metrics}")
    if not isinstance(lora_b_l1, (int, float)) or not math.isfinite(lora_b_l1) or lora_b_l1 <= 0:
        raise RuntimeError(f"LoRA-B did not receive a finite nonzero update: {metrics}")
    return float(grad_norm)


class GuidanceRunStateStore:
    """Pair the post-step Guidance archive with Reef's committed step.

    ``library.json`` is the working archive the harness mutates during a step.
    It is copied to ``committed-library.json`` only after the matching training
    transaction is durable, so a resumed run restarts from an archive that the
    served checkpoint has actually seen.
    """

    def __init__(self, state_dir: str | Path, identity: GuidanceRunIdentity) -> None:
        self.state_dir = Path(state_dir)
        self.identity = identity
        self.working_library_path = self.state_dir / "library.json"
        self.committed_library_path = self.state_dir / "committed-library.json"
        self.resume_path = self.state_dir / "resume-state.json"
        self.result_path = self.state_dir / "result.json"

    def load(self) -> dict[str, Any] | None:
        if not self.resume_path.is_file():
            return None
        state = read_json(self.resume_path)
        if state.get("format_version") != RESUME_FORMAT_VERSION:
            raise GuidanceRunStateError(f"unsupported resume state format in {self.resume_path}")
        mismatches = [
            f"{key}={state.get(key)!r}, requested={value!r}"
            for key, value in asdict(self.identity).items()
            if state.get(key) != value
        ]
        if mismatches:
            raise GuidanceRunStateError("resume settings mismatch: " + "; ".join(mismatches))
        next_step = state.get("next_step")
        if not isinstance(next_step, int) or isinstance(next_step, bool) or next_step < 1:
            raise GuidanceRunStateError(f"resume state has invalid next_step {next_step!r}")
        return state

    def restore_working_library(self) -> None:
        if not self.committed_library_path.is_file():
            raise GuidanceRunStateError(f"committed Guidance archive is missing: {self.committed_library_path}")
        shutil.copy2(self.committed_library_path, self.working_library_path)

    def commit_library(self) -> None:
        atomic_copy(self.working_library_path, self.committed_library_path)

    def save(self, *, next_step: int, step_summaries: Sequence[Mapping[str, Any]], extra: Mapping[str, Any]) -> None:
        write_json(
            self.resume_path,
            {
                "format_version": RESUME_FORMAT_VERSION,
                **asdict(self.identity),
                **dict(extra),
                "next_step": next_step,
                "step_summaries": [dict(summary) for summary in step_summaries],
            },
        )


@dataclass
class GuidanceRunController:
    """Run Guidance-TTT steps against one durable Reef training scenario."""

    harness: SearchHarness
    library: Any
    bridge: TrainingBridge
    state_store: GuidanceRunStateStore
    checkpoint_root: Path
    resume_extra: Mapping[str, Any]
    emit: Callable[[Mapping[str, Any]], None]

    def run(self, total_steps: int) -> GuidanceRunOutcome:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        identity = self.state_store.identity
        groups = identity.groups_per_step
        rollouts = identity.rollouts_per_group
        expected_rollouts = groups * rollouts

        saved = self.state_store.load()
        start_step = int(saved["next_step"]) if saved else 0
        if start_step >= total_steps:
            raise GuidanceRunStateError(f"run already completed {start_step} steps; requested total is {total_steps}")
        bridge_start_step = self.bridge.start_step()
        if bridge_start_step != start_step:
            raise GuidanceRunStateError(
                f"checkpoint starts at rollout {bridge_start_step}, but the Guidance archive starts at {start_step}"
            )

        step_summaries = [dict(summary) for summary in (saved or {}).get("step_summaries", [])]
        all_results: list[Any] = []
        runtime_load_id: str | None = None
        for step in range(start_step, total_steps):
            started_at = time.monotonic()
            self.emit({"event": "guidance_step_started", "step": step})
            results = tuple(self.harness.run_step(step))
            bridge_health = self.bridge.wait_for_step(
                expected_completed_steps=step - start_step + 1,
                expected_rollout_id=step,
            )
            grouped_rewards = [
                [result.reward for result in results[offset : offset + rollouts]]
                for offset in range(0, len(results), rollouts)
            ]
            nonconstant_groups = sum(
                any(reward != rewards[0] for reward in rewards[1:]) for rewards in grouped_rewards
            )
            valid_rollouts = sum(result.verification.valid for result in results)
            # The TTTD processor keeps one group on an otherwise fully
            # constant step so the frozen-base KL transaction remains
            # well-defined. Mirror that cardinality in the qualification
            # checks instead of claiming that the trainer consumed no data.
            retained_trajectories = (nonconstant_groups or 1) * rollouts
            metrics = dict(bridge_health.get("last_train_metrics") or {})
            grad_norm = require_step_success(
                expected_rollouts=expected_rollouts,
                actual_rollouts=len(results),
                retained_trajectories=retained_trajectories,
                bridge_health=bridge_health,
                expected_completed_train_steps=step - start_step + 1,
                expected_rollout_id=step,
                grad_norm=metrics.get("train/grad_norm"),
                lora_rank=identity.lora_rank,
            )
            if not (self.checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
                raise RuntimeError(f"durable Megatron checkpoint is missing under {self.checkpoint_root}")

            snapshot = self.library.snapshot()
            best_node = snapshot["nodes"].get(snapshot.get("best_node_id")) or {}
            summary = {
                "step": step,
                "elapsed_s": time.monotonic() - started_at,
                "sampled_rollouts": len(results),
                "valid_rollouts": valid_rollouts,
                "well_formed_guidance": sum(result.guidance_format_ok for result in results),
                "nonconstant_groups": nonconstant_groups,
                "degenerate_step": valid_rollouts == 0 or nonconstant_groups == 0,
                "retained_trajectories": retained_trajectories,
                "grad_norm": grad_norm,
                "best_raw_score": best_node.get("raw_score"),
                "train_metrics": metrics,
            }
            step_summaries.append(summary)
            self.state_store.commit_library()
            self.state_store.save(
                next_step=step + 1,
                step_summaries=step_summaries,
                extra=self.resume_extra,
            )
            runtime_load_id = bridge_health.get("runtime_load_id") or runtime_load_id
            all_results.extend(results)
            self.emit({"event": "guidance_step_committed", **summary})

        return GuidanceRunOutcome(
            results=tuple(all_results),
            start_step=start_step,
            next_step=total_steps,
            step_summaries=tuple(step_summaries),
            runtime_load_id=runtime_load_id,
        )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


__all__ = [
    "GuidanceRunController",
    "GuidanceRunIdentity",
    "GuidanceRunOutcome",
    "GuidanceRunStateError",
    "GuidanceRunStateStore",
    "RayTrainingBridge",
    "atomic_copy",
    "failed_training_step",
    "read_json",
    "require_step_success",
    "scenario_status",
    "wait_for_training_step",
    "write_json",
]
