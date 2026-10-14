"""Harness candidate evaluation: run the evaluation tasks as episodes, then decide by the scores.

A :class:`HarnessCandidate` is the candidate a harness backend proposes.
:class:`HarnessEpisodeEvaluator` implements :class:`CandidateEvaluator` for it.
It runs each evaluation task on the candidate tree and, for a comparison, on
the current tree, and returns one :class:`EvaluationResult` with the scores of
each side. The method's :class:`EpisodeScorer` (``evolution.evaluate``) scores
each episode. Two selection policies read the result:
:class:`ScoreComparisonPlugin` selects a candidate with more task wins than
losses, and :class:`FloorPlugin` runs the candidate alone and selects it when
every task scores at least ``floor_score``.

The evaluator's internal parts are in this module too:

- :class:`EpisodeEvaluationWorker` runs and scores one episode in a worker
  process and returns a :class:`ScoredEpisode`.
- :class:`EvaluationWorkerPool` runs those workers on the executor that
  :func:`evaluation_selection` chooses from ``execution.evolution``.
"""

from __future__ import annotations

import json
import logging
import math
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Self

from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.core.plugins import ConfiguredPlugin, PluginConfigError
from reef.core.tasks import primary_reward
from reef.evaluation.evaluators import BackendEvaluateMixin, SelectionPlugin
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import EPISODE_OWNER_LEASE, EpisodeExecutor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import (
    TrajectoryError,
    agent_work,
    failed_trial_error,
    final_assistant_text,
    root_header,
    stage_path,
    sum_agents,
)
from reef.harness.tree.mutations import Mutation, nodes_from_entries
from reef.harness.tree.nodes import clip_record_text, redact_secret_shaped
from reef.harness.tree.render import render_composition, tree_files
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import (
    ExecutorSelection,
    ExecutorSettings,
    local_gpu_assignments,
    select_executor,
    worker_requirements,
)
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.requirements import ExecutionRequirements


#: The record file beside one evaluation episode's kept trajectory.
RECORD_EPISODE_FILE = "episode.json"


class EpisodeScorer(ConfiguredPlugin, ABC):
    """Base class for scoring one harness-evolution episode.

    Implement:
        ``score`` — given a task name and its episode result, return a
        float score. Higher is better; non-finite values raise.
    """

    @abstractmethod
    def score(self, task: str, result: EpisodeResult) -> float:
        """Score one episode result for a task."""

    def execution_requirements(self) -> ExecutionRequirements:
        """Override when scoring loads a local GPU model or needs cluster placement.

        API calls do not request local GPUs. GPU scorers must initialize their
        model lazily in the allocated worker, not while the recipe is built.
        """
        return ExecutionRequirements()

    def score_with_models(self, task: str, result: EpisodeResult, models: ModelBindings) -> float:
        """Model-based judges override this method and use the supplied bindings."""
        return self.score(task, result)


def verifier_reward(task: str, result: EpisodeResult) -> float:
    """The Harbor verifier's reward for a task directory episode; :class:`VerifierRewardScorer` scores with it.

    The terminus runner writes one ``verifier`` row per episode with the task it played and the rewards
    its verifier wrote; Harbor's primary reward is the ``reward`` entry, else the sole entry. A failed
    episode or a verifier that wrote nothing scores 0 (the gate has already set aside an episode whose
    trial never ran); an episode that exited without a row scores 0; a row for another task, several
    rewards without a ``reward`` entry, or a reward that is not a finite number is an error.
    """
    rows = [event for event in result.trajectory if event.get("type") == "verifier"]
    if not rows and result.exit_code:
        return 0.0
    if len(rows) != 1:
        raise ValueError(f"expected one verifier record for {task!r}, found {len(rows)}")
    row = rows[0]
    if row.get("task") != task:
        raise ValueError(f"the verifier record names {row.get('task')!r}, not {task!r}")
    if row.get("failed"):
        return 0.0
    rewards = row.get("rewards")
    reward = primary_reward(rewards) if isinstance(rewards, Mapping) else row.get("reward")
    if reward is None:
        if isinstance(rewards, Mapping) and rewards:
            raise ValueError(f"the verifier for {task!r} wrote {sorted(rewards)} and no 'reward' entry")
        return 0.0
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"the verifier reward for {task!r} must be a finite number, not {reward!r}")
    return float(reward)


class VerifierRewardScorer(EpisodeScorer):
    """Score an episode by its Harbor verifier's reward (:func:`verifier_reward`); for a task manifest evaluation."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return verifier_reward(task, result)


@dataclass(frozen=True)
class FailureObservation:
    """One failed episode as the evaluation saw it, before any diffing."""

    task: str
    stage: str
    cause: str

    def to_dict(self) -> dict[str, str]:
        return {"task": self.task, "stage": self.stage, "cause": self.cause}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureObservation:
        return cls(task=str(value["task"]), stage=str(value["stage"]), cause=str(value["cause"]))


@dataclass(frozen=True)
class ScoredEpisode:
    """One evaluation episode as the evaluation keeps it: the score, or why it has none, and what the trajectory showed."""

    score: float | None
    failure: FailureObservation | None
    residue: int = 0
    agents: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The root's stage path and end reason; ``None`` when no trajectory was read.
    path: dict[str, Any] | None = None
    #: The final assistant text the trajectory holds, the reply a text grader reads; ``None`` when it holds none.
    reply: str | None = None
    #: Whether the reader found a session log; a scored episode without one left a text grader nothing to read.
    transcript_read: bool = True
    #: Remote workers return the kept trajectory; the driver owns its durable path.
    record_archive: bytes | None = field(default=None, repr=False)


@dataclass(repr=False)
class EpisodeEvaluationWorker:
    """Only episode/scorer state crosses the process boundary, never the recipe tree."""

    descriptor: AdapterDescriptor
    scorer: EpisodeScorer
    binary: str | None
    timeout: float
    executor: EpisodeExecutor
    forbid_residue: bool
    owner_lease: bool = False
    transfer_records: bool = False

    def __post_init__(self) -> None:
        self.executor.preflight()

    def run(
        self, files: Mapping[str, str], task: str, keep_dir: Path | None = None, models: ModelBindings | None = None
    ) -> ScoredEpisode:
        if keep_dir is None or not self.transfer_records:
            return self._run_and_score(files, task, keep_dir, models)
        # Remote workers must not interpret the driver's path as a local path.
        # Keep the trajectory on the worker, then return it with the scored result.
        with tempfile.TemporaryDirectory(prefix="reef-worker-record-") as temporary:
            record_dir = Path(temporary) / "episode"
            scored = self._run_and_score(files, task, record_dir, models)
            buffer = BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(record_dir, arcname=".")
            return replace(scored, record_archive=buffer.getvalue())

    def _run_and_score(
        self, files: Mapping[str, str], task: str, keep_dir: Path | None, models: ModelBindings | None
    ) -> ScoredEpisode:
        """Score one side's episode; a ``None`` score marks an episode that
        could not run. The observation keeps what the exception handling
        would otherwise discard: the failure's stage and cause. A native turn
        that ended on an error could not run either; any other nonzero exit
        still scores, as before, and is observed alongside the score.
        The residue counts files the episode left outside the cleanup
        whitelist; with ``forbid_residue`` a littering episode scores as one
        that could not run. ``keep_dir`` receives the episode's trajectory
        files before its root is removed; a copy that fails is not an
        episode failure and propagates, so the step aborts instead of
        scoring a result shaped by a disk error."""
        token = EPISODE_OWNER_LEASE.set(self.owner_lease)
        try:
            result = run_episode(
                self.descriptor,
                files,
                task,
                binary=self.binary,
                timeout=self.timeout,
                executor=self.executor,
                keep_dir=keep_dir,
            )
        except EpisodeError as error:
            scored = ScoredEpisode(None, FailureObservation(task=task, stage="launch", cause=str(error)))
            write_episode_record(keep_dir, task, None, scored)
            return scored
        except TrajectoryError as error:
            scored = ScoredEpisode(None, FailureObservation(task=task, stage="trajectory", cause=str(error)))
            write_episode_record(keep_dir, task, None, scored)
            return scored
        finally:
            EPISODE_OWNER_LEASE.reset(token)
        scored = self._score_result(result, task, models)
        write_episode_record(keep_dir, task, result, scored)
        return scored

    def _score_result(self, result: EpisodeResult, task: str, models: ModelBindings | None = None) -> ScoredEpisode:
        """The score and the observations of an episode that ran."""
        residue = len(result.residue)
        agents = agent_work(result.trajectory)
        path = stage_path(result.trajectory)
        if residue and self.forbid_residue:
            cause = f"{residue} file(s) outside the cleanup whitelist: {result.residue[0]}"
            return ScoredEpisode(
                None, FailureObservation(task=task, stage="residue", cause=cause), residue, agents, path
            )
        trial_error = failed_trial_error(result.trajectory)
        if trial_error:
            # The terminus runner recorded a trial that never ran (the image did not build, the agent could not
            # start): no answer was given, so it ranks below every real score instead of tying a zero.
            return ScoredEpisode(
                None, FailureObservation(task=task, stage="trial", cause=trial_error), residue, agents, path
            )
        if path.get("error") is not None:
            # The native loop ended its turn on an error (a tree that cannot load, a graph that cannot run, an
            # agent whose failure ended the run): nothing it wrote is an answer, so it ranks below every real
            # score instead of tying a zero.
            error = path["error"]
            cause = f"{error.get('code', 'error')}: {error.get('message', '')}".strip(": ")
            if path.get("errored_agent"):
                cause = f"agent {path['errored_agent']}: {cause}"
            # A loop turn walks no graph: the failure names the loop when the root's header does.
            stage = "loop" if root_header(result.trajectory).get("loop") else "graph"
            return ScoredEpisode(None, FailureObservation(task=task, stage=stage, cause=cause), residue, agents, path)
        score = float(
            self.scorer.score(task, result) if models is None else self.scorer.score_with_models(task, result, models)
        )
        if not math.isfinite(score):
            raise ValueError(f"episode scorer returned a non-finite score {score!r} for task {task!r}")
        reply = final_assistant_text(result.trajectory)
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return ScoredEpisode(
                score, FailureObservation(task=task, stage="exit", cause=cause), residue, agents, path, reply
            )
        # An empty trajectory is no failure: a grader that reads files scored the run as it stands. The summary
        # still says no transcript was read, so a low score a text grader gave is not blamed on the request.
        return ScoredEpisode(score, None, residue, agents, path, reply, transcript_read=bool(result.trajectory))


def write_episode_record(
    keep_dir: Path | None, task: str, result: EpisodeResult | None, scored: ScoredEpisode
) -> None:
    """``episode.json`` beside the kept trajectory: what the scorer saw, so a result can be re-derived from the record."""
    if keep_dir is None:
        return
    record = {
        "task": clip_record_text(task),
        "score": scored.score,
        "failure": None if scored.failure is None else scored.failure.to_dict(),
        "path": scored.path,
        "exit_code": None if result is None else result.exit_code,
        "stdout": None if result is None else clip_record_text(result.stdout),
        "stderr": None if result is None else clip_record_text(result.stderr),
        "residue": None if result is None else [clip_record_text(str(path)) for path in result.residue],
    }
    # An episode that never wrote an event has no trajectory copy, so the directory may not exist yet.
    keep_dir.mkdir(parents=True, exist_ok=True)
    with open(keep_dir / RECORD_EPISODE_FILE, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, default=str) + "\n")


class EvaluationWorkerPool:
    """One lazy, fixed-size worker group per backend; never replay failed work."""

    def __init__(self, selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec):
        self._selection = selection
        self._requirements = requirements
        self._worker = worker
        self._executor: Executor | None = None
        self._lock = Lock()
        self._closed = False

    def evaluate(self, pairings, *, models: ModelBindings | None = None):
        # Drain a batch before admitting the next, including ordinary scorer errors.
        with self._lock:
            if self._closed:
                if self._executor is not None and self._executor.failure is not None:
                    self._executor.check_health()
                raise RuntimeError("evolution worker pool is closed")
            if not pairings:
                return []
            if self._executor is None:
                config = evaluation_executor_config(self._selection, self._requirements, self._worker)
                try:
                    self._executor = Executor.create(config)
                except BaseException:
                    self._closed = True
                    raise
            pending = []
            try:
                for index, pairing in enumerate(pairings):
                    pending.append(
                        self._executor.rpc(
                            index % self._requirements.workers,
                            "run",
                            args=pairing,
                            kwargs={} if models is None else {"models": models},
                            non_block=True,
                        )
                    )
            except BaseException:
                # Partial submission cannot be replayed safely. Retire the pool.
                self._closed = True
                self._executor.shutdown()
                raise
            results = []
            first_error: Exception | None = None
            try:
                for future in pending:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                if first_error is not None:
                    raise first_error
                return results
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    self._closed = True
                    self._executor.shutdown()
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown()


def legacy_worker_settings(
    settings: ExecutorSettings, workers: int | None = None, gpus: float | None = None
) -> ExecutorSettings:
    """Compatibility boundary for the old recipe/Python resource arguments."""
    if workers is not None:
        if settings.workers is not None and settings.workers != workers:
            raise ValueError("episode_workers conflicts with execution.evolution.workers")
        settings = replace(settings, workers=workers)
    if gpus is not None:
        if settings.resources.gpus_per_worker is not None and settings.resources.gpus_per_worker != gpus:
            raise ValueError("worker_resources.num_gpus conflicts with execution.evolution.resources.gpus_per_worker")
        settings = replace(settings, resources=replace(settings.resources, gpus_per_worker=gpus))
    return settings


def evaluation_selection(
    scorer: EpisodeScorer,
    workers: int | None,
    settings: ExecutorSettings,
    gpus_per_worker: float | None = None,
) -> tuple[ExecutorSelection, ExecutionRequirements]:
    declared = scorer.execution_requirements()
    if not isinstance(declared, ExecutionRequirements):
        raise TypeError("EpisodeScorer.execution_requirements must return ExecutionRequirements")
    settings = legacy_worker_settings(settings, workers, gpus_per_worker)
    requirements = worker_requirements(settings, declared)
    gpus = requirements.gpus_per_worker
    configured_gpus = settings.options.get("num_gpus", gpus)
    if configured_gpus != gpus:
        raise ValueError(
            "declare GPU needs through execution.evolution.resources.gpus_per_worker, not executor.options.num_gpus"
        )
    configured_cpus = settings.options.get("num_cpus", requirements.cpus_per_worker)
    if settings.resources.cpus_per_worker is not None and configured_cpus != requirements.cpus_per_worker:
        raise ValueError("executor.options.num_cpus conflicts with execution.evolution.resources.cpus_per_worker")
    # Preserve old Ray profiles which request CPUs through options.num_cpus.
    requirements = replace(requirements, cpus_per_worker=configured_cpus)
    runtime_env = settings.options.get("runtime_env", {})
    if not isinstance(runtime_env, dict):
        raise ValueError("evolution executor runtime_env must be an object")
    if "CUDA_VISIBLE_DEVICES" in runtime_env.get("env_vars", {}):
        raise ValueError(
            "Ray owns evolution worker CUDA visibility; declare execution.evolution.resources.gpus_per_worker"
        )
    if settings.options.get("max_restarts", 0) != 0 or settings.options.get("max_task_retries", 0) != 0:
        raise ValueError("evolution executors must not replay evaluations")
    return select_executor(settings, role="evolution", requirements=requirements), requirements


def evaluation_executor_config(
    selection: ExecutorSelection, requirements: ExecutionRequirements, worker: WorkerSpec
) -> ExecutorConfig:
    backend = Executor.get_class(selection.settings.backend)
    options = dict(selection.settings.options)
    if issubclass(backend, RayExecutor) or (
        selection.settings.backend not in ("uni", "mp")
        and (requirements.gpus_per_worker or selection.settings.resources.cpus_per_worker is not None)
    ):
        options = {**options, "num_cpus": requirements.cpus_per_worker, "num_gpus": requirements.gpus_per_worker}
    logging.getLogger(__name__).info(
        "evolution: executor=%s workers=%s gpus_per_worker=%s (%s)",
        selection.settings.backend,
        requirements.workers,
        requirements.gpus_per_worker,
        selection.reason,
    )
    workers = (worker,) * requirements.workers
    if selection.settings.backend == "mp" and requirements.gpus_per_worker:
        workers = tuple(
            replace(worker, options={"cuda_visible_devices": ",".join(devices)})
            for devices in local_gpu_assignments(requirements)
        )
    return ExecutorConfig(backend=backend, options=options, workers=workers)


#: The trees an evaluation step runs, in pairing order; a policy that evaluates the candidate alone names the first only.
EVALUATION_SIDES: tuple[str, ...] = ("candidate", "current")
#: One evaluation episode: the rendered files, the task, and the directory that keeps its record (``None``: no record).
EpisodePairing = tuple[Mapping[str, str], str, Path | None]
#: The directory under a candidate's step record that holds one directory per evaluation episode.
RECORD_EPISODES_DIR = "episodes"
#: How many candidate episodes a step's evaluation summarizes, and how much of each text it keeps: a summary for
#: the pages, never the traffic.
EPISODE_SUMMARIES = 8
EPISODE_SUMMARY_CHARS = 240


@dataclass(frozen=True, kw_only=True)
class HarnessCandidate(UpdateCandidate):
    """One rendered harness mutation compared with the current composition."""

    candidate_files: Mapping[str, str]
    current_files: Mapping[str, str]
    candidate_entries: tuple[Mapping[str, Any], ...]
    current_entries: tuple[Mapping[str, Any], ...]
    mutations: tuple[Mutation, ...]
    #: Seed tasks, then promoted traffic prompts; the seed set when promotion is off.
    evaluation_tasks: tuple[str, ...] = ()
    #: Legacy constructor keyword; new candidates use evaluation_tasks.
    gate_tasks: tuple[str, ...] = ()
    #: The candidate is the rollback target, so selecting it rolls back.
    recheck: bool = False
    #: The inbox proposal these mutations came from, settled with the result; None for the method's own.
    proposal_id: str | None = None
    #: The step record directory claimed for it at prepare time; ``None`` with the record off.
    record_dir: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()


def clip_redacted(text: str, limit: int) -> str:
    """``text`` redacted as the record is, cut at ``limit`` characters with an ellipsis marker."""
    text = redact_secret_shaped(text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def episode_record_name(side: str, task_index: int, repeat: int) -> str:
    """The record directory of one evaluation episode: ``<side>-<task index>``, a repeat adding ``-<repeat>``."""
    return f"{side}-{task_index}" if repeat == 0 else f"{side}-{task_index}-{repeat}"


class HarnessEpisodeEvaluator(CandidateEvaluator):
    """Measure a :class:`HarnessCandidate` by running its evaluation tasks as episodes.

    Each task runs on the candidate tree and, for a comparison, on the current
    tree, in a worker pool on the selected executor. The episodes render each
    tree with the served model binding, which never enters the candidate's
    files. This object owns the worker pool, so the caller must :meth:`close` it.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        scorer: EpisodeScorer,
        tasks: tuple[str, ...],
        models: ModelBindings,
        binary: str | None,
        episode_timeout_s: float,
        episode_repeats: int,
        executor: EpisodeExecutor,
        forbid_residue: bool,
        worker_selection: ExecutorSelection,
        worker_requirements: ExecutionRequirements,
    ) -> None:
        self.episode_repeats = episode_repeats
        self.worker_selection = worker_selection
        self.worker_requirements = worker_requirements
        self._descriptor = descriptor
        self._tasks = tasks
        self.use_models(models)
        self._worker_pool = EvaluationWorkerPool(
            worker_selection,
            worker_requirements,
            WorkerSpec(
                EpisodeEvaluationWorker,
                args=(
                    descriptor,
                    scorer,
                    binary,
                    episode_timeout_s,
                    executor,
                    forbid_residue,
                    worker_selection.settings.backend != "uni",
                    worker_selection.settings.backend not in ("uni", "mp"),
                ),
            ),
        )

    def close(self) -> None:
        self._worker_pool.close()

    def use_models(self, models: ModelBindings) -> None:
        """Run later episodes with ``models``: the served binding renders into each tree, and scorers receive all."""
        # Resolving the binding here refuses an adapter without a matching model_binding before any episode runs.
        self._binding_nodes = models.served.compose_nodes(self._descriptor)
        self._models = models

    def render(self, entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        """The tree plus the model binding, and the entries list beside them where the adapter carries one."""
        files = render_composition((*nodes_from_entries(entries), *self._binding_nodes), self._descriptor)
        return {**files, **tree_files(self._descriptor, entries)}

    def evaluation_tasks(self, candidate: HarnessCandidate) -> tuple[str, ...]:
        """The tasks the candidate carries; an older candidate carries none and falls back to the seed set."""
        return candidate.evaluation_tasks or candidate.gate_tasks or self._tasks

    def episode_count(self, candidate: HarnessCandidate, side_count: int) -> int:
        """How many episodes :meth:`evaluate` runs for ``candidate`` on ``side_count`` sides."""
        return len(self.evaluation_tasks(candidate)) * self.episode_repeats * side_count

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        """Run the evaluation episodes of the named ``sides`` and return their scores.

        The default runs the candidate and the current tree as pairs. A policy
        that evaluates the candidate alone (a floor) passes ``("candidate",)``: no
        current episode runs, ``current_scores`` is empty and the other
        ``current_*`` keys are absent, and ``evaluation_sides`` records the choice.
        """
        if not isinstance(candidate, HarnessCandidate):
            raise TypeError(f"harness evaluation requires HarnessCandidate, got {type(candidate).__name__}")
        sides = tuple(sides)
        if not sides or any(side not in EVALUATION_SIDES for side in sides):
            raise ValueError(f"sides must name one or both of {EVALUATION_SIDES}, got {sides!r}")
        entries = {"candidate": candidate.candidate_entries, "current": candidate.current_entries}
        files = {side: self.render(entries[side]) for side in sides}
        episodes_dir = None if candidate.record_dir is None else candidate.record_dir / RECORD_EPISODES_DIR
        # Episodes interleave candidate and current inside each pairing, so
        # anything that drifts during the run (upstream load, rate limits)
        # lands on both sides of a pair instead of one whole side. A repeat
        # is one more pairing of the same task; the selector compares the
        # vectors positionally, so every pairing tallies on its own. Each
        # episode is independent work in its own root, so with more than one
        # worker the pairings run in one pool - a large task set costs one
        # wave instead of a long turn-taking pass - and the results are read
        # back in submission order either way.
        pairings: list[EpisodePairing] = [
            (
                files[side],
                task,
                None if episodes_dir is None else episodes_dir / episode_record_name(side, index, repeat),
            )
            for index, task in enumerate(self.evaluation_tasks(candidate))
            for repeat in range(self.episode_repeats)
            for side in sides
        ]
        scored = self.run_episodes(pairings)
        runs = {side: scored[offset :: len(sides)] for offset, side in enumerate(sides)}
        tasks = {side: [pairing[1] for pairing in pairings][offset :: len(sides)] for offset, side in enumerate(sides)}
        scores = {side: tuple(run.score for run in runs[side]) for side in sides}
        metrics: dict[str, Any] = {
            "candidate_scores": scores.get("candidate", ()),
            "current_scores": scores.get("current", ()),
            "episode_failures": sum(score is None for side in sides for score in scores[side]),
            "episode_repeats": self.episode_repeats,
        }
        for side in sides:
            # Failure observations ride the evaluation so settlement can
            # build the committed side's manifest from the decision alone.
            metrics[f"{side}_failures"] = tuple(run.failure.to_dict() for run in runs[side] if run.failure is not None)
            metrics[f"{side}_residue"] = sum(run.residue for run in runs[side])
            metrics[f"{side}_score"] = float(sum(score for score in scores[side] if score is not None))
            # Per agent sums over the side's episodes, so a result says which agent did the work.
            metrics[f"{side}_agents"] = sum_agents(run.agents for run in runs[side])
            # Per episode, in pairing order: the root's stage path and how its turn ended.
            metrics[f"{side}_paths"] = tuple(run.path for run in runs[side])
        if "candidate" in sides:
            # What the first candidate episodes were graded on, so a rejected step names the task, the reply and why.
            metrics["candidate_episodes"] = [
                {
                    "task": clip_redacted(task, EPISODE_SUMMARY_CHARS),
                    "score": run.score,
                    "failure": (
                        None if run.failure is None else clip_redacted(run.failure.cause, EPISODE_SUMMARY_CHARS)
                    ),
                    "reply": None if run.reply is None else clip_redacted(run.reply, EPISODE_SUMMARY_CHARS),
                    # Only when missing, so a summary of an ordinary episode keeps its shape.
                    **({} if run.transcript_read else {"transcript_read": False}),
                }
                for task, run in list(zip(tasks["candidate"], runs["candidate"], strict=True))[:EPISODE_SUMMARIES]
            ]
        if sides != EVALUATION_SIDES:
            metrics["evaluation_sides"] = list(sides)
        return EvaluationResult(evaluator="harness_episode_pairs", evaluator_version="1", metrics=metrics)

    def run_episodes(self, pairings: Sequence[EpisodePairing]) -> list[ScoredEpisode]:
        """Run and score each pairing in the worker pool, in submission order.

        A remote worker returns the kept trajectory as an archive; it is
        extracted here into the pairing's record directory.
        """
        scored = self._worker_pool.evaluate(pairings, models=self._models)
        for (_, _, keep_dir), result in zip(pairings, scored, strict=True):
            # A worker returns an archive only for a pairing that keeps its record.
            if keep_dir is None or result.record_archive is None:
                continue
            try:
                # Never merge into an earlier episode's record. Extraction
                # rejects path traversal, escaping links and special files.
                keep_dir.mkdir(parents=True)
                with tarfile.open(fileobj=BytesIO(result.record_archive), mode="r:gz") as archive:
                    archive.extractall(keep_dir, filter="data")
            except (OSError, tarfile.TarError) as exc:
                raise TrajectoryKeepError(f"cannot keep remote episode trajectory under {keep_dir}: {exc}") from exc
        return scored


class ScoreComparisonMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that selects when wins exceed losses by ``min_win_margin`` (0: plain majority)."""

    def __init__(self, *, min_win_margin: int = 0) -> None:
        if isinstance(min_win_margin, bool) or not isinstance(min_win_margin, int) or min_win_margin < 0:
            raise ValueError("min_win_margin must be an integer of at least 0")
        super().__init__()
        self._min_win_margin = min_win_margin

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        candidate_scores, current_scores = score_vectors(evaluation)
        wins, losses = score_comparison_tally(candidate_scores, current_scores)
        ties = len(candidate_scores) - wins - losses
        selected = wins - losses > self._min_win_margin
        metrics: dict[str, Any] = {"wins": wins, "losses": losses, "ties": ties}
        if self._min_win_margin:
            metrics["min_win_margin"] = self._min_win_margin
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="score_comparison",
            policy_version="1",
            reason=(
                f"candidate won {wins} task pairings and lost {losses}"
                if selected
                else f"candidate did not exceed its {losses} losses with {wins} wins"
            ),
            evaluation=evaluation,
            metrics=metrics,
        )


class ScoreComparisonPlugin(ScoreComparisonMixin, BackendEvaluateMixin, SelectionPlugin):
    """Cordis's default evaluation: measure through the candidate backend, decide by score comparison."""

    def __init__(self, candidate_backend: Any, *, min_win_margin: int = 0) -> None:
        super().__init__(min_win_margin=min_win_margin)
        self._candidate_backend = candidate_backend

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        unknown = sorted(str(key) for key in config if key != "min_win_margin")
        if unknown:
            raise PluginConfigError(f"the score_comparison selection takes min_win_margin, not {', '.join(unknown)}")
        return cls(candidate_backend, min_win_margin=config.get("min_win_margin", 0))


class FloorMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that selects when every evaluation task scores at least ``floor_score``.

    A floor is absolute, not a comparison: only the candidate's scores are
    read, and an episode that could not run (score ``None``) missed it.
    """

    def __init__(self, *, floor_score: float = 1.0) -> None:
        if (
            isinstance(floor_score, bool)
            or not isinstance(floor_score, (int, float))
            or not math.isfinite(floor_score)
            or floor_score <= 0
        ):
            raise ValueError("floor_score must be a positive number")
        super().__init__()
        self._floor_score = float(floor_score)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        passed = sum(1 for score in scores if score is not None and score >= self._floor_score)
        failed = len(scores) - passed
        if not scores:
            reason = "no evaluation task was scored"
        elif failed:
            reason = f"candidate missed the floor on {failed} of {len(scores)} tasks"
        else:
            reason = f"candidate met the floor on all {len(scores)} tasks"
        return SelectionDecision(
            outcome="select" if scores and not failed else "reject",
            policy="floor",
            policy_version="1",
            reason=reason,
            evaluation=evaluation,
            metrics={"passed": passed, "failed": failed, "floor_score": self._floor_score},
        )


class FloorPlugin(FloorMixin, BackendEvaluateMixin, SelectionPlugin):
    """Evaluate the candidate alone through the candidate backend, decide by the floor; the current release is not run."""

    def __init__(self, candidate_backend: Any, *, floor_score: float = 1.0) -> None:
        super().__init__(floor_score=floor_score)
        self._candidate_backend = candidate_backend

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self._candidate_backend.evaluate(candidate, sides=("candidate",))

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        unknown = sorted(str(key) for key in config if key != "floor_score")
        if unknown:
            raise PluginConfigError(f"the floor selection takes floor_score, not {', '.join(unknown)}")
        return cls(candidate_backend, floor_score=config.get("floor_score", 1.0))


def score_vectors(
    evaluation: EvaluationResult,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    candidate = tuple(evaluation.metrics.get("candidate_scores", ()))
    current = tuple(evaluation.metrics.get("current_scores", ()))
    return candidate, current


def score_comparison_tally(candidate: tuple[float | None, ...], current: tuple[float | None, ...]) -> tuple[int, int]:
    wins = losses = 0
    for candidate_score, current_score in zip(candidate, current, strict=True):
        candidate_rank = candidate_score if candidate_score is not None else float("-inf")
        current_rank = current_score if current_score is not None else float("-inf")
        if candidate_rank > current_rank:
            wins += 1
        elif candidate_rank < current_rank:
            losses += 1
    return wins, losses
