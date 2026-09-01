"""Terminal-Bench evaluation owned by Harbor's task verifier."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .composition import CompositionCandidate
from .config import HARD_TASKS, TERMINAL_BENCH_DATASET_COMMIT, TERMINAL_BENCH_DATASET_URL
from .evaluation import ALL_SPLITS, EvaluationResult, TrialEvidence

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class HarborTrialOutcome:
    reward: float
    trajectory: tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    status: str = "completed"
    error: str | None = None
    verifier: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.reward) <= 1.0:
            raise ValueError("Harbor trial reward must be between zero and one")
        if self.estimated_cost_usd < 0 or self.wall_time_s < 0:
            raise ValueError("Harbor trial cost and wall time must be non-negative")
        if not self.status:
            raise ValueError("Harbor trial outcome requires a status")


class TrialExecutor(Protocol):
    def __call__(
        self,
        candidate: CompositionCandidate,
        candidate_path: Path,
        task_id: str,
        trial_index: int,
        round_index: int,
        trial_dir: Path,
    ) -> HarborTrialOutcome: ...


class TrialHook(Protocol):
    def __call__(self, trial_id: str) -> None: ...


class TrialCostHook(Protocol):
    def __call__(self, trial_id: str, cost_usd: float) -> None: ...


class HarborInfrastructureError(RuntimeError):
    """Harbor did not return complete, verifier-owned benchmark evidence."""


async def verify_dataset_registry_pin(
    task_ids: Sequence[str] = HARD_TASKS,
    *,
    dataset_config_factory: Callable[..., Any] | None = None,
) -> None:
    """Fail if Harbor's named dataset no longer resolves to the recorded Git tree."""
    if dataset_config_factory is None:
        from harbor.models.job.config import DatasetConfig

        dataset_config_factory = DatasetConfig
    requested = tuple(str(task_id) for task_id in task_ids)
    if not requested or len(set(requested)) != len(requested):
        raise RuntimeError("dataset registry verification requires unique task ids")
    task_configs = await dataset_config_factory(
        name="terminal-bench",
        version="2.0",
        task_names=list(requested),
    ).get_task_configs()
    resolved: dict[str, Any] = {}
    for task_config in task_configs:
        path = getattr(task_config, "path", None)
        name = Path(path).name if path is not None else ""
        if not name or name in resolved:
            raise RuntimeError("Terminal-Bench registry returned an invalid or duplicate task path")
        resolved[name] = task_config
    if set(resolved) != set(requested):
        missing = sorted(set(requested) - set(resolved))
        extra = sorted(set(resolved) - set(requested))
        raise RuntimeError(f"Terminal-Bench registry task mismatch: missing={missing}, extra={extra}")
    for task_id in requested:
        task_config = resolved[task_id]
        if Path(task_config.path) != Path(task_id):
            raise RuntimeError(f"Terminal-Bench task {task_id!r} resolved to {task_config.path!s}")
        if task_config.git_url != TERMINAL_BENCH_DATASET_URL:
            raise RuntimeError(f"Terminal-Bench task {task_id!r} resolved to unexpected URL {task_config.git_url!r}")
        if task_config.git_commit_id != TERMINAL_BENCH_DATASET_COMMIT:
            raise RuntimeError(
                f"Terminal-Bench task {task_id!r} resolved to unexpected commit {task_config.git_commit_id!r}"
            )


def _pinned_task_config(task_id: str) -> Any:
    """Build a Harbor task config from immutable inputs, not mutable registry state."""
    from harbor.models.trial.config import TaskConfig

    if task_id not in HARD_TASKS:
        raise RuntimeError(f"Terminal-Bench task {task_id!r} is outside the pinned hard subset")
    return TaskConfig(
        path=Path(task_id),
        git_url=TERMINAL_BENCH_DATASET_URL,
        git_commit_id=TERMINAL_BENCH_DATASET_COMMIT,
        source="terminal-bench",
    )


class HarborEvaluator:
    """Evaluate fixed task splits without exposing verifier ownership to search."""

    def __init__(
        self,
        *,
        splits: Mapping[str, Sequence[str]],
        trials_per_task: int,
        output_dir: Path,
        target_model: str,
        target_base_url: str,
        target_api_key_env: str,
        pi_binary: Path,
        pi_timeout_s: float = 1800.0,
        trial_executor: TrialExecutor | None = None,
        before_trial: TrialHook | None = None,
        after_trial: TrialCostHook | None = None,
        budget_namespace: str = "default",
    ) -> None:
        if set(splits) != set(ALL_SPLITS):
            raise ValueError(f"Harbor evaluator requires exactly the splits {ALL_SPLITS}")
        normalized = {name: tuple(str(task) for task in splits[name]) for name in ALL_SPLITS}
        if any(not tasks or any(not task for task in tasks) for tasks in normalized.values()):
            raise ValueError("each Harbor task split must be non-empty")
        all_tasks = sum(normalized.values(), ())
        if len(set(all_tasks)) != len(all_tasks):
            raise ValueError("Harbor task splits must be disjoint")
        if len({_safe_segment(task) for task in all_tasks}) != len(all_tasks):
            raise ValueError("Harbor task ids collide after path normalization")
        if trials_per_task < 1:
            raise ValueError("Harbor trials per task must be positive")
        if not target_model or not target_base_url or not target_api_key_env:
            raise ValueError("Harbor target model, base URL, and credential environment are required")
        if pi_timeout_s <= 0:
            raise ValueError("Pi timeout must be positive")
        if not budget_namespace or ":" in budget_namespace:
            raise ValueError("budget namespace must be non-empty and contain no colon")
        self.splits = normalized
        self.trials_per_task = trials_per_task
        self.output_dir = Path(output_dir).resolve()
        self.target_model = target_model
        self.target_base_url = target_base_url.rstrip("/")
        self.target_api_key_env = target_api_key_env
        self.pi_binary = Path(pi_binary).resolve()
        self.pi_timeout_s = float(pi_timeout_s)
        self._trial_executor = trial_executor or self._execute_harbor_trial
        self._before_trial = before_trial
        self._after_trial = after_trial
        self._budget_namespace = budget_namespace

    def evaluate(
        self,
        candidate: CompositionCandidate,
        *,
        split: str,
        round_index: int,
    ) -> EvaluationResult:
        if split not in self.splits:
            raise ValueError(f"unknown evaluation split {split!r}")
        candidate_path = self._persist_candidate(candidate)
        trials: list[TrialEvidence] = []
        for task_id in self.splits[split]:
            for trial_index in range(self.trials_per_task):
                trial_dir = (
                    self.output_dir
                    / "trials"
                    / candidate.content_hash
                    / f"round-{round_index:04d}"
                    / split
                    / _safe_segment(task_id)
                    / f"trial-{trial_index:02d}"
                )
                trial_dir.mkdir(parents=True, exist_ok=True)
                evidence_path = trial_dir / "evidence.json"
                identity = (
                    f"{self._budget_namespace}:{candidate.content_hash}:round-{round_index:04d}:"
                    f"{split}:{task_id}:trial-{trial_index:02d}"
                )
                if evidence_path.is_file():
                    evidence = _read_evidence(evidence_path, task_id=task_id, trial_index=trial_index)
                    _validate_completed_evidence(evidence)
                    if self._after_trial is not None:
                        self._after_trial(identity, evidence.estimated_cost_usd)
                    trials.append(evidence)
                    continue
                if self._before_trial is not None:
                    self._before_trial(identity)
                try:
                    outcome = self._trial_executor(
                        candidate,
                        candidate_path,
                        task_id,
                        trial_index,
                        round_index,
                        trial_dir,
                    )
                except Exception as exc:
                    attempt = len(tuple(trial_dir.glob("executor-error-*.json"))) + 1
                    self._write_once(
                        trial_dir / f"executor-error-{attempt:02d}.json",
                        {"error_type": type(exc).__name__, "error": str(exc)},
                    )
                    raise HarborInfrastructureError(f"Harbor trial execution failed: {exc}") from exc
                evidence = TrialEvidence(
                    task_id=task_id,
                    trial=trial_index,
                    reward=outcome.reward,
                    trajectory=outcome.trajectory,
                    usage=outcome.usage,
                    estimated_cost_usd=outcome.estimated_cost_usd,
                    wall_time_s=outcome.wall_time_s,
                    status=outcome.status,
                    error=outcome.error,
                    verifier=outcome.verifier,
                )
                try:
                    _validate_completed_evidence(evidence)
                except HarborInfrastructureError:
                    attempt = len(tuple(trial_dir.glob("infrastructure-attempt-*.json"))) + 1
                    failure_path = trial_dir / f"infrastructure-attempt-{attempt:02d}.json"
                    self._write_once(failure_path, evidence.to_jsonable())
                    if self._after_trial is not None:
                        self._after_trial(f"{identity}:infrastructure-{attempt:02d}", evidence.estimated_cost_usd)
                    raise
                self._write_once(evidence_path, evidence.to_jsonable())
                if self._after_trial is not None:
                    self._after_trial(identity, evidence.estimated_cost_usd)
                trials.append(evidence)
        return EvaluationResult(split=split, trials=tuple(trials))

    def completed_evaluations(self, split: str) -> int:
        """Count complete candidate/round split batches for restart and budget fill."""
        if split not in self.splits:
            raise ValueError(f"unknown evaluation split {split!r}")
        trial_root = self.output_dir / "trials"
        expected = len(self.splits[split]) * self.trials_per_task
        groups: dict[tuple[str, str], int] = {}
        for path in trial_root.glob(f"*/round-*/{split}/*/trial-*/evidence.json"):
            relative = path.relative_to(trial_root)
            key = (relative.parts[0], relative.parts[1])
            groups[key] = groups.get(key, 0) + 1
        return sum(count == expected for count in groups.values())

    def _persist_candidate(self, candidate: CompositionCandidate) -> Path:
        path = self.output_dir / "candidates" / candidate.content_hash / "composition.json"
        self._write_text_once(path, candidate.canonical_json + "\n")
        return path

    def _execute_harbor_trial(
        self,
        candidate: CompositionCandidate,
        candidate_path: Path,
        task_id: str,
        trial_index: int,
        round_index: int,
        trial_dir: Path,
    ) -> HarborTrialOutcome:
        return asyncio.run(
            self._execute_harbor_trial_async(
                candidate,
                candidate_path,
                task_id,
                trial_index,
                round_index,
                trial_dir,
            )
        )

    async def _execute_harbor_trial_async(
        self,
        _candidate: CompositionCandidate,
        candidate_path: Path,
        task_id: str,
        _trial_index: int,
        _round_index: int,
        trial_dir: Path,
    ) -> HarborTrialOutcome:
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import AgentConfig
        from harbor.models.trial.config import EnvironmentConfig, TrialConfig
        from harbor.trial.trial import Trial

        config = TrialConfig(
            task=_pinned_task_config(task_id),
            trial_name=trial_dir.name,
            trials_dir=trial_dir.parent,
            agent=AgentConfig(
                import_path="harness.harbor_agent:HarborAgent",
                model_name=self.target_model,
                kwargs={
                    "composition_path": str(candidate_path),
                    "target_base_url": self.target_base_url,
                    "target_api_key_env": self.target_api_key_env,
                    "pi_binary": str(self.pi_binary),
                    "pi_timeout_s": self.pi_timeout_s,
                },
            ),
            environment=EnvironmentConfig(type=EnvironmentType.DOCKER, delete=True),
        )
        result = await (await Trial.create(config)).run()
        return _outcome_from_trial_result(result)

    @staticmethod
    def _write_once(path: Path, payload: Any) -> None:
        HarborEvaluator._write_text_once(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _write_text_once(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"evaluation evidence changed across restart: {path}")
            return
        path.write_text(text, encoding="utf-8")


def _outcome_from_trial_result(result: Any) -> HarborTrialOutcome:
    verifier_result = getattr(result, "verifier_result", None)
    rewards = dict(getattr(verifier_result, "rewards", None) or {})
    reward = float(rewards.get("reward", 0.0))
    context = getattr(result, "agent_result", None)
    metadata = getattr(context, "metadata", None) if context is not None else None
    reef_metadata = metadata.get("reef_meta_harness", {}) if isinstance(metadata, Mapping) else {}
    trajectory: tuple[Mapping[str, Any], ...] = ()
    trajectory_path = reef_metadata.get("trajectory_path") if isinstance(reef_metadata, Mapping) else None
    if trajectory_path:
        raw_trajectory = json.loads(Path(str(trajectory_path)).read_text(encoding="utf-8"))
        if not isinstance(raw_trajectory, list) or not all(isinstance(event, Mapping) for event in raw_trajectory):
            raise RuntimeError("Pi trajectory artifact is not a list of events")
        trajectory = tuple(raw_trajectory)

    exception = getattr(result, "exception_info", None)
    pi_exit_code = reef_metadata.get("pi_exit_code") if isinstance(reef_metadata, Mapping) else None
    status = "completed"
    error = None
    if exception is not None:
        status = f"trial_failed:{getattr(exception, 'exception_type', 'unknown')}"
        error = str(getattr(exception, "exception_message", ""))
    elif pi_exit_code not in (None, 0):
        status = f"pi_failed:{pi_exit_code}"
        error = str(reef_metadata.get("pi_stderr") or "")
    elif context is None:
        status = "agent_context_missing"
        error = "Harbor returned no agent context"
    elif not trajectory_path:
        status = "trajectory_missing"
        error = "Harbor agent returned no Pi trajectory artifact"
    elif verifier_result is None or "reward" not in rewards:
        status = "verifier_missing"
        error = "Harbor returned no verifier reward"

    started = getattr(result, "started_at", None)
    finished = getattr(result, "finished_at", None)
    wall_time_s = max((finished - started).total_seconds(), 0.0) if started and finished else 0.0
    usage = {
        "input_tokens": int(getattr(context, "n_input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(context, "n_cache_tokens", 0) or 0),
        "output_tokens": int(getattr(context, "n_output_tokens", 0) or 0),
    }
    return HarborTrialOutcome(
        reward=reward,
        trajectory=trajectory,
        usage=usage,
        estimated_cost_usd=float(getattr(context, "cost_usd", 0.0) or 0.0),
        wall_time_s=wall_time_s,
        status=status,
        error=error,
        verifier={
            "rewards": rewards,
            "task_checksum": str(getattr(result, "task_checksum", "")),
            "source": getattr(result, "source", None),
        },
    )


def _safe_segment(value: str) -> str:
    safe = _SAFE_SEGMENT.sub("-", value).strip("-.")
    return safe[:96] or "task"


def _validate_completed_evidence(evidence: TrialEvidence) -> None:
    verifier = evidence.verifier
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if evidence.status != "completed":
        raise HarborInfrastructureError(f"Harbor trial returned infrastructure status {evidence.status!r}")
    if not evidence.trajectory:
        raise HarborInfrastructureError("Harbor trial returned no Pi trajectory")
    if not isinstance(rewards, Mapping) or "reward" not in rewards:
        raise HarborInfrastructureError("Harbor trial returned no verifier reward")
    if not verifier.get("task_checksum"):
        raise HarborInfrastructureError("Harbor trial returned no task checksum")


def _read_evidence(path: Path, *, task_id: str, trial_index: int) -> TrialEvidence:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"trial evidence is not an object: {path}")
    if value.get("task_id") != task_id or value.get("trial") != trial_index:
        raise RuntimeError(f"trial evidence identity does not match its path: {path}")
    trajectory = value.get("trajectory")
    usage = value.get("usage")
    verifier = value.get("verifier")
    if not isinstance(trajectory, list) or not all(isinstance(event, Mapping) for event in trajectory):
        raise RuntimeError(f"trial evidence trajectory is invalid: {path}")
    return TrialEvidence(
        task_id=task_id,
        trial=trial_index,
        reward=float(value["reward"]),
        trajectory=tuple(trajectory),
        usage=dict(usage) if isinstance(usage, Mapping) else {},
        estimated_cost_usd=float(value.get("estimated_cost_usd", 0.0)),
        wall_time_s=float(value.get("wall_time_s", 0.0)),
        status=str(value.get("status") or "completed"),
        error=str(value["error"]) if value.get("error") is not None else None,
        verifier=dict(verifier) if isinstance(verifier, Mapping) else {},
    )
