"""Durable step orchestration for Guidance-TTT runs.

One search step is one Reef training job. The controller runs a step only
after the previous job has ended: it samples ``G x R`` guidance rollouts,
waits on Reef's HTTP status until the job that trains them ends, checks that
the LoRA optimizer really consumed the reserved batch, and only then pairs the
post-step archive with that committed step.

Nothing here is task specific. The Harbor agent supplies the harness, the
archive, and the run settings; ``run()`` returns the rollouts and the per-step
summaries that the agent hands back to Harbor.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RESUME_FORMAT_VERSION = 1


class GuidanceRunStateError(RuntimeError):
    """The Guidance archive and Reef's durable training state do not align."""


class GuidanceTrainingTimeoutError(RuntimeError):
    """Reef did not commit a Guidance-TTT training step before its deadline."""


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


class SearchHarness(ABC):
    @abstractmethod
    def run_step(self, step: int) -> Sequence[Any]: ...


class TrainingProgress(ABC):
    """Reef's training of one scenario, seen from the harness."""

    @abstractmethod
    def position(self) -> TrainingPosition:
        """Where the scenario's training stands now."""

    @abstractmethod
    def wait_for_step(self, *, after: TrainingPosition, expected_rollout_id: int) -> TrainingJobOutcome:
        """Wait until the scenario finished the one training job that follows ``after``."""


@dataclass(frozen=True)
class TrainingPosition:
    """A scenario's training as ``GET /reef/status`` reports it.

    ``scenario_step`` is Reef's committed scenario step and ``finished_jobs``
    the number of training jobs that ended, both non-negative ints.
    ``running_job_id`` is the running job's ID, a non-empty string, or ``None``
    while no job runs. A scenario that the status does not list yet has not
    trained: ``TrainingPosition(0, 0, None)``. The controller reads it before
    a step and waits for the job after it.
    """

    scenario_step: int
    finished_jobs: int
    running_job_id: str | None

    def __post_init__(self) -> None:
        for name, value in (("scenario_step", self.scenario_step), ("finished_jobs", self.finished_jobs)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"Reef returned an invalid {name}: {value!r}")
        if self.running_job_id is not None and (not isinstance(self.running_job_id, str) or not self.running_job_id):
            raise RuntimeError(f"Reef returned an invalid running training job: {self.running_job_id!r}")

    @classmethod
    def from_status(cls, status: Mapping[str, Any]) -> TrainingPosition:
        """Read one scenario's status block; a block without job fields is not a job-loop trainer."""
        if "training_job" not in status or "finished_jobs" not in status:
            raise RuntimeError(f"Reef reports no training job state for the scenario: {status!r}")
        job = status["training_job"]
        if job is not None and not isinstance(job, Mapping):
            raise RuntimeError(f"Reef returned an invalid training job: {job!r}")
        return cls(
            scenario_step=status.get("scenario_step"),
            finished_jobs=status["finished_jobs"],
            running_job_id=None if job is None else job.get("job_id"),
        )


@dataclass(frozen=True)
class TrainingJobOutcome:
    """One training job of a Guidance step that ended and saved its checkpoint.

    ``job_id`` is the job's non-empty ID. ``rollout_id`` is the Slime rollout
    the job saved, a non-negative int. ``checkpoint_metrics`` are the metrics
    Reef committed for the checkpoint: Slime's training metrics (for example
    ``train/grad_norm``) and the evaluation and selection result.
    ``position`` is the scenario's training after the job, with no job
    running. ``runtime_load_id`` is the serving load ID the status reports
    then, or ``None``. ``wait_for_training_step`` makes the value and the
    controller reads it.
    """

    job_id: str
    rollout_id: int
    checkpoint_metrics: Mapping[str, Any]
    position: TrainingPosition
    runtime_load_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise RuntimeError(f"Reef returned an invalid training job ID: {self.job_id!r}")
        if not isinstance(self.rollout_id, int) or isinstance(self.rollout_id, bool) or self.rollout_id < 0:
            raise RuntimeError(f"training job {self.job_id} reported an invalid rollout ID: {self.rollout_id!r}")
        if self.position.running_job_id is not None:
            raise RuntimeError(f"training job {self.job_id} ended while job {self.position.running_job_id} runs")


class StatusTrainingProgress(TrainingProgress):
    """Read a scenario's training through Reef's HTTP API.

    ``GET /reef/status`` reports the scenario step, the running job and the
    number of finished jobs. ``GET /reef/scenarios/<scenario>/commits`` holds
    each job's committed checkpoint metrics and its end. No Ray connection is
    needed.
    """

    def __init__(
        self,
        service_url: str,
        scenario: str,
        *,
        token: str | None = None,
        timeout_s: float = 14_400.0,
        poll_interval_s: float = 2.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        if timeout_s <= 0 or poll_interval_s < 0 or request_timeout_s <= 0:
            raise ValueError("training timeout must be positive and the poll interval non-negative")
        self.service_url = service_url.rstrip("/")
        self.token = token
        self.scenario = scenario
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.request_timeout_s = float(request_timeout_s)

    def position(self) -> TrainingPosition:
        status = self.scenario_status()
        return TrainingPosition(0, 0, None) if status is None else TrainingPosition.from_status(status)

    def wait_for_step(self, *, after: TrainingPosition, expected_rollout_id: int) -> TrainingJobOutcome:
        return wait_for_training_step(
            status=self.scenario_status,
            commits=self.commits_after,
            after=after,
            expected_rollout_id=expected_rollout_id,
            timeout_s=self.timeout_s,
            poll_interval_s=self.poll_interval_s,
        )

    def scenario_status(self) -> dict[str, Any] | None:
        return scenario_status(self.get_json("/reef/status"), self.scenario)

    def commits_after(self, scenario_step: int) -> list[dict[str, Any]]:
        """Every commit of the scenario after ``scenario_step``, in step order."""
        path = f"/reef/scenarios/{urllib.parse.quote(self.scenario, safe='')}/commits"
        commits: list[dict[str, Any]] = []
        after_step: int | None = scenario_step
        while after_step is not None:
            page = self.get_json(f"{path}?{urllib.parse.urlencode({'after_step': after_step, 'limit': 100})}")
            entries = page.get("commits")
            if not isinstance(entries, list) or not all(isinstance(entry, Mapping) for entry in entries):
                raise RuntimeError(f"Reef returned invalid commits: {page!r}")
            commits.extend(dict(entry) for entry in entries)
            after_step = page.get("next_after_step")
        return commits

    def get_json(self, path: str) -> Mapping[str, Any]:
        request = urllib.request.Request(self.service_url + path)
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
            body = json.loads(response.read())
        if not isinstance(body, Mapping):
            raise RuntimeError(f"Reef returned a non-object for {path}: {body!r}")
        return body


def wait_for_training_step(
    *,
    status: Callable[[], Mapping[str, Any] | None],
    commits: Callable[[int], Sequence[Mapping[str, Any]]],
    after: TrainingPosition,
    expected_rollout_id: int,
    timeout_s: float,
    poll_interval_s: float,
) -> TrainingJobOutcome:
    """Wait for the one training job that follows ``after`` to end, and read its checkpoint.

    The step is done when the scenario step advanced past ``after``, one more
    job finished, and no job runs. The scenario's training error, a TTTD step
    failure of ``expected_rollout_id``, and a failed job raise ``RuntimeError``.
    """
    expected_jobs = after.finished_jobs + 1
    deadline = time.monotonic() + timeout_s
    last_status: Mapping[str, Any] | None = None
    while time.monotonic() < deadline:
        last_status = status()
        if last_status is not None:
            failure = failed_training_step(last_status, expected_rollout_id)
            if failure is not None:
                raise RuntimeError(
                    f"TTTD step {expected_rollout_id} failed ({failure['reason']}) because its reports span "
                    f"releases {failure['release_ids']!r}"
                )
            position = TrainingPosition.from_status(last_status)
            if position.finished_jobs > expected_jobs:
                raise RuntimeError(f"Reef finished {position.finished_jobs} training jobs, expected {expected_jobs}")
            if (
                position.running_job_id is None
                and position.finished_jobs == expected_jobs
                and position.scenario_step > after.scenario_step
            ):
                return finished_job_outcome(
                    commits(after.scenario_step),
                    position,
                    last_status.get("current_runtime_load_id"),
                    expected_rollout_id,
                )
        if poll_interval_s:
            time.sleep(poll_interval_s)
    raise GuidanceTrainingTimeoutError(
        f"training rollout {expected_rollout_id} did not complete after {timeout_s:g}s: reef={last_status}"
    )


def finished_job_outcome(
    commits: Sequence[Mapping[str, Any]],
    position: TrainingPosition,
    runtime_load_id: str | None,
    expected_rollout_id: int,
) -> TrainingJobOutcome:
    """The outcome of the one job that ended in ``commits``; a failed job or a missing checkpoint raises."""
    events: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for commit in commits:
        metrics = commit.get("metrics")
        job = metrics.get("training_job") if isinstance(metrics, Mapping) else None
        if isinstance(job, Mapping):
            events.append((job, metrics))
    ended = [(job, metrics) for job, metrics in events if job.get("event") in ("finished", "failed")]
    if len(ended) != 1:
        raise RuntimeError(f"expected one training job to end, found {[job for job, _ in ended]!r}")
    job, end_metrics = ended[0]
    if job["event"] == "failed":
        raise RuntimeError(f"Reef training job {job.get('job_id')} failed: {job.get('error')}")
    checkpoints = [
        metrics
        for event, metrics in events
        if event.get("event") == "checkpoint" and event.get("job_id") == job["job_id"]
    ]
    if len(checkpoints) != 1:
        raise RuntimeError(f"training job {job['job_id']} saved {len(checkpoints)} checkpoints, expected one")
    outcome = TrainingJobOutcome(
        job_id=job["job_id"],
        rollout_id=end_metrics.get("rollout_id"),
        checkpoint_metrics=dict(checkpoints[0]),
        position=position,
        runtime_load_id=runtime_load_id,
    )
    if outcome.rollout_id != expected_rollout_id:
        raise RuntimeError(
            f"training job {outcome.job_id} saved rollout {outcome.rollout_id}, expected {expected_rollout_id}"
        )
    return outcome


def scenario_status(body: Mapping[str, Any], scenario: str) -> dict[str, Any] | None:
    """Read one scenario's training status; ``None`` while the status does not list the scenario."""
    training_error = body.get("error")
    if training_error is not None:
        if not isinstance(training_error, str) or not training_error:
            raise RuntimeError(f"Reef returned an invalid training error: {body!r}")
        raise RuntimeError(f"Reef training failed: {training_error}")
    preload_errors = body.get("preload_errors")
    if isinstance(preload_errors, Mapping) and scenario in preload_errors:
        raise RuntimeError(f"Reef could not restore scenario {scenario!r}: {preload_errors[scenario]}")
    scenarios = body.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise RuntimeError(f"Reef returned no scenario statuses: {body!r}")
    status = scenarios.get(scenario)
    if status is None:
        return None
    if not isinstance(status, Mapping):
        raise RuntimeError(f"Reef returned an invalid status for {scenario!r}: {status!r}")
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
    outcome: TrainingJobOutcome,
    lora_rank: int,
    allow_zero_signal: bool = False,
) -> float:
    """Fail closed unless the LoRA optimizer really consumed this step; return its grad norm."""
    if actual_rollouts != expected_rollouts:
        raise RuntimeError(f"sampled {actual_rollouts} rollouts, expected {expected_rollouts}")
    metrics = outcome.checkpoint_metrics
    grad_norm = metrics.get("train/grad_norm")
    if (
        not isinstance(grad_norm, (int, float))
        or not math.isfinite(float(grad_norm))
        or grad_norm < 0
        or (grad_norm == 0 and not allow_zero_signal)
    ):
        raise RuntimeError(f"optimizer did not report the expected finite grad norm: {grad_norm!r}")
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
    if (
        not isinstance(lora_b_nonzero, (int, float))
        or lora_b_nonzero < 0
        or (lora_b_nonzero == 0 and not allow_zero_signal)
    ):
        raise RuntimeError(f"LoRA-B remained zero after the optimizer step: {metrics}")
    if (
        not isinstance(lora_b_l1, (int, float))
        or not math.isfinite(lora_b_l1)
        or lora_b_l1 < 0
        or (lora_b_l1 == 0 and not allow_zero_signal)
    ):
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
    progress: TrainingProgress
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
        position = self.progress.position()
        if position.running_job_id is not None:
            raise GuidanceRunStateError(
                f"Reef is still running training job {position.running_job_id}; resume after it ends"
            )
        if position.finished_jobs != start_step:
            raise GuidanceRunStateError(
                f"Reef finished {position.finished_jobs} training jobs, but the Guidance archive starts at {start_step}"
            )

        step_summaries = [dict(summary) for summary in (saved or {}).get("step_summaries", [])]
        all_results: list[Any] = []
        runtime_load_id: str | None = None
        for step in range(start_step, total_steps):
            started_at = time.monotonic()
            self.emit({"event": "guidance_step_started", "step": step})
            results = tuple(self.harness.run_step(step))
            outcome = self.progress.wait_for_step(after=position, expected_rollout_id=step)
            position = outcome.position
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
            metrics = dict(outcome.checkpoint_metrics)
            grad_norm = require_step_success(
                expected_rollouts=expected_rollouts,
                actual_rollouts=len(results),
                retained_trajectories=retained_trajectories,
                outcome=outcome,
                lora_rank=identity.lora_rank,
                allow_zero_signal=nonconstant_groups == 0,
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
            runtime_load_id = outcome.runtime_load_id or runtime_load_id
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
    "GuidanceTrainingTimeoutError",
    "StatusTrainingProgress",
    "TrainingJobOutcome",
    "TrainingPosition",
    "TrainingProgress",
    "atomic_copy",
    "failed_training_step",
    "finished_job_outcome",
    "read_json",
    "require_step_success",
    "scenario_status",
    "wait_for_training_step",
    "write_json",
]
