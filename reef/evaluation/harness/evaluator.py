"""Evaluate harness trees by running runs, with the history as the score cache.

:class:`HarnessTaskEvaluator` renders a tree with the served model
binding, runs each task as a run in a worker pool, and scores it with
the deployment's :class:`TaskScorer` (``evolution.evaluation.scorer``).
It is the harness tree runner of ``RolloutRunner``: :meth:`HarnessTaskEvaluator.run_rollouts`
runs ``RolloutRequest`` values without the cache.
:meth:`HarnessTaskEvaluator.runs` scores any tree on any tasks, for a
method's ``tools.evaluate``.
:meth:`HarnessTaskEvaluator.evaluate` scores a :class:`HarnessCandidate`
and the head tree for a publication policy. Both run only the runs that
the history does not hold: a run of the same tree, task, repeat and
evaluation context that did not fail is reused.
"""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from reef.core.evaluation import (
    CandidateEvaluator,
    EvaluationResult,
    PolicyHarness,
    PolicyModel,
    RolloutRequest,
    RolloutRunner,
    RolloutScore,
    RolloutScores,
    UpdateCandidate,
)
from reef.evaluation.harness.history import History, RunRecord
from reef.evaluation.harness.runner import EvaluationWorkerPool, TaskEvaluationWorker, TaskRun
from reef.evaluation.harness.scoring import ScoredRun, TaskScorer
from reef.evaluation.harness.tasks import EvaluationTask
from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tasks.harbor_task import can_run_in_task_container
from reef.harness.tasks.model_binding import ModelBinding, ModelBindings
from reef.harness.tasks.run import TrajectoryKeepError
from reef.harness.tasks.trajectory import sum_agents
from reef.harness.tree import HarnessTree, TreePatch, render_tree
from reef.harness.tree.checks import clip_record_text
from reef.runtime.executor import WorkerSpec
from reef.runtime.executor.config import ExecutorSelection
from reef.runtime.executor.requirements import ExecutionRequirements

#: The trees a candidate evaluation runs, in pairing order; a policy that evaluates the candidate alone names the
#: first.
EVALUATION_SIDES: tuple[str, ...] = ("candidate", "current")
#: How many candidate runs an evaluation summarizes, and how much of each text it keeps: a summary for the
#: pages, never the traffic.
RUN_SUMMARIES = 8
RUN_SUMMARY_CHARS = 240


@dataclass(frozen=True)
class EvaluationContext:
    """Everything a run's score depends on besides the tree and the task.

    Values: ``model`` is the served model. ``named_models`` pairs each name of
    ``evolution.models`` with its model, sorted by name, each name once: a
    judge scorer reads these models. ``adapter`` and ``adapter_version`` name
    the harness adapter and its pinned version (``None`` for an adapter
    without one); ``binary`` is the deployment's own harness binary
    (``evolution.binary``), ``None`` when the pinned one runs. ``executor`` is
    the run executor's configuration (``evolution.executor`` and
    ``evolution.sandbox``) as one canonical JSON text, which only ``key``
    reads. ``scorer`` names the scorer class and its configuration.
    ``task_timeout_s``, a finite number above 0, and ``forbid_residue``
    are the run settings.
    Purpose: the evaluator makes one for the current models; ``key`` is the
    context part of every cache key, so a score is never reused across
    contexts.
    """

    model: str
    named_models: tuple[tuple[str, str], ...]
    adapter: str
    adapter_version: str | None
    binary: str | None
    executor: str
    scorer: str
    task_timeout_s: float
    forbid_residue: bool

    def __post_init__(self) -> None:
        names = [name for name, _ in self.named_models]
        if names != sorted(set(names)):
            raise ValueError("EvaluationContext.named_models must be sorted by name, each name once")
        timeout = self.task_timeout_s
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
            raise ValueError("EvaluationContext.task_timeout_s must be a finite number")
        if timeout <= 0:
            raise ValueError("EvaluationContext.task_timeout_s must be above 0")

    def key(self) -> str:
        """The first 16 hexadecimal digits of the SHA-256 digest of the context's JSON form."""
        encoded = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, kw_only=True)
class HarnessCandidate(UpdateCandidate):
    """One candidate tree, evaluated against the head tree.

    Values: ``candidate_tree`` is the tree under evaluation, and
    ``current_tree`` is the head tree the step started from.
    ``evaluation_tasks`` are the tasks both sides run, the seed tasks first.
    ``step`` ≥ 1 is the step that made it.
    Purpose: ``HarnessBackend.prepare_step`` makes it; the evaluator scores
    both trees and the publication policy decides on it.
    """

    candidate_tree: HarnessTree
    current_tree: HarnessTree
    evaluation_tasks: tuple[EvaluationTask, ...]
    step: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("HarnessCandidate.step must be a positive integer")

    @property
    def patch(self) -> TreePatch:
        """``current_tree.diff(candidate_tree)``: the smallest patch that makes the candidate from the head tree."""
        return self.current_tree.diff(self.candidate_tree)


class HarnessCandidateEvaluator(CandidateEvaluator, ABC):
    """Score a harness candidate on one or both sides.

    Purpose: a publication policy measures through it; the harness backend
    implements it over its evaluator, and the evaluator implements it itself.
    """

    @abstractmethod
    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        """The scores of ``candidate`` on the named ``sides``, from ``EVALUATION_SIDES``."""


#: The hosts that name the Reef host only from the Reef host itself.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def container_binding(binding: ModelBinding, agent_host: str) -> ModelBinding:
    """``binding`` as a task's container reaches it: ``agent_host`` in place of a loopback host of its base URL,
    the port and path kept. A base URL with any other host is reachable as it is and stays."""
    parts = urllib.parse.urlsplit(binding.base_url)
    if parts.hostname not in LOOPBACK_HOSTS:
        return binding
    netloc = agent_host if parts.port is None else f"{agent_host}:{parts.port}"
    return replace(binding, base_url=urllib.parse.urlunsplit(parts._replace(netloc=netloc)))


class HarnessTaskEvaluator(HarnessCandidateEvaluator, RolloutRunner[ScoredRun]):
    """Score harness trees on tasks by running runs, reusing the scores that the history holds.

    This is the harness tree runner of ``RolloutRunner``: a request's harness
    is this adapter with a rendered tree as its plugin, and its model is the
    served model. The runs render each tree with the served model binding,
    which never enters a tree's files. ``binary`` is the installed binary the runs
    run; ``binary_setting`` is the deployment's ``evolution.binary``, which
    the evaluation context names, and ``executor_setting`` is the run
    executor's configuration as canonical JSON text, which the context
    digests. ``task_directories`` says that every task is a Harbor task
    directory. An adapter whose command takes the task as text then plays
    each task in the task's own container (``task_containers``), and the
    served model's base URL there names ``agent_host``, the address at which
    the container reaches Reef. The evaluator owns the worker pool, so the
    caller must :meth:`close` it.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        scorer: TaskScorer,
        scorer_name: str,
        models: ModelBindings,
        binary: str | None,
        binary_setting: str | None,
        executor_setting: str,
        task_timeout_s: float,
        task_repeats: int,
        executor: ProgramExecutor,
        forbid_residue: bool,
        worker_selection: ExecutorSelection,
        worker_requirements: ExecutionRequirements,
        history: History,
        task_directories: bool,
        agent_host: str = "host.docker.internal",
    ) -> None:
        if task_repeats < 1:
            raise ValueError("task_repeats must be at least 1")
        self.task_directories = task_directories
        self.task_containers = task_directories and not descriptor.takes_task_directory
        if self.task_containers and not can_run_in_task_container(descriptor):
            raise ValueError(
                f"adapter {descriptor.name!r} cannot play a Harbor task directory: it needs an npm install"
            )
        if not agent_host:
            raise ValueError("agent_host must name the address at which a task container reaches Reef")
        self.agent_host = agent_host
        self.descriptor = descriptor
        self.scorer_name = scorer_name
        self.binary_setting = binary_setting
        self.executor_setting = executor_setting
        self.task_timeout_s = task_timeout_s
        self.task_repeats = task_repeats
        self.forbid_residue = forbid_residue
        self.history = history
        self.task_keys: dict[str, str] = {}
        self.use_models(models)
        self.worker_pool = EvaluationWorkerPool(
            worker_selection,
            worker_requirements,
            WorkerSpec(
                TaskEvaluationWorker,
                args=(
                    descriptor,
                    scorer,
                    binary,
                    task_timeout_s,
                    executor,
                    forbid_residue,
                    worker_selection.settings.backend != "uni",
                    worker_selection.settings.backend not in ("uni", "mp"),
                    self.task_containers,
                ),
            ),
        )

    def close(self) -> None:
        self.worker_pool.close()

    def use_models(self, models: ModelBindings) -> None:
        """Run later runs with ``models``: the served binding renders into each tree, and scorers receive all."""
        # Resolving the binding here refuses an adapter without a matching model_binding before any task runs.
        models.served.config_overlays(self.descriptor)
        self.models = models

    def context(self) -> EvaluationContext:
        """The evaluation context of the current models."""
        return EvaluationContext(
            model=self.models.served.model,
            named_models=tuple(sorted((name, binding.model) for name, binding in self.models.named.items())),
            adapter=self.descriptor.name,
            adapter_version=None if self.descriptor.install is None else self.descriptor.install.version,
            binary=self.binary_setting,
            executor=self.executor_setting,
            scorer=self.scorer_name,
            task_timeout_s=self.task_timeout_s,
            forbid_residue=self.forbid_residue,
        )

    @property
    def served_model(self) -> PolicyModel:
        """The model that every request must name: the served model, which the runs call."""
        return PolicyModel(self.models.served.model)

    def render(self, tree: HarnessTree) -> dict[str, str]:
        """``tree``'s files for one run, pointed at the served model: through ``agent_host`` in a task's
        container, which cannot reach the Reef host's loopback by its own name."""
        served = self.models.served
        if self.task_containers:
            served = container_binding(served, self.agent_host)
        return render_tree(tree, self.descriptor, served.config_overlays(self.descriptor))

    def policy_harness(self, tree: HarnessTree) -> PolicyHarness:
        """``tree`` as a run request's harness: this adapter, with ``tree``'s rendered files as its plugin."""
        return PolicyHarness(self.descriptor.name, self.render(tree))

    def run_rollouts(self, requests: Sequence[RolloutRequest]) -> list[ScoredRun]:
        """Run and score one run of each request in the worker pool, in request order, without the cache.

        Raise ``ValueError`` before any task runs when a request names
        another adapter than this one, or another model than the served one. A
        remote worker returns the kept record as an archive; it is extracted
        here into the request's ``record_dir``.
        """
        served = self.served_model
        for request in requests:
            if request.harness.base != self.descriptor.name:
                raise ValueError(
                    f"this runner runs the {self.descriptor.name!r} adapter, not {request.harness.base!r}"
                )
            if request.model != served:
                raise ValueError(f"this runner serves {served}, not {request.model}")
        # A request names the task by what it plays; the shared contract carries no answer.
        task_of = EvaluationTask.of_directory if self.task_directories else EvaluationTask.of_instruction
        return self.run_in_pool(
            [TaskRun(dict(request.harness.plugin), task_of(request.task), request.record_dir) for request in requests]
        )

    def run_in_pool(self, runs: list[TaskRun]) -> list[ScoredRun]:
        """Run and score ``runs`` in the worker pool, in order; a remote worker's kept record is extracted here."""
        scored = self.worker_pool.run(runs, self.models)
        for run, result in zip(runs, scored, strict=True):
            # A worker returns an archive only for a run that keeps its record.
            if result.record_archive is not None and run.record_dir is not None:
                keep_remote_record(result.record_archive, run.record_dir)
        return scored

    def rollout_score(self, task_run: ScoredRun) -> RolloutScore:
        return task_run.score

    def task_key(self, task: EvaluationTask) -> str:
        """``task.key()``; a task directory's digest is computed once per directory."""
        if task.directory is None:
            return task.key()
        key = self.task_keys.get(task.text)
        if key is None:
            key = self.task_keys[task.text] = task.key()
        return key

    def run(self, requests: Sequence[tuple[HarnessTree, EvaluationTask, int]], *, step: int) -> list[RunRecord]:
        """One run record per ``(tree, task, repeat)`` request, in request order.

        Every tree must be a candidate of the history. A request that the
        cache holds takes the cached record. The other requests run as one
        wave in the worker pool, in request order, and their records join the
        history.
        """
        context = self.context().key()
        checksums = [tree.checksum() for tree, _, _ in requests]
        keys = [self.task_key(task) for _, task, _ in requests]
        records: dict[int, RunRecord] = {}
        for index, (checksum, key, (_, _, repeat)) in enumerate(zip(checksums, keys, requests, strict=True)):
            if checksum not in self.history.candidates:
                raise ValueError(f"tree {checksum} is not a candidate of the history")
            cached = self.history.cached(checksum, key, repeat, context)
            if cached is not None:
                records[index] = cached
        missing = [index for index in range(len(requests)) if index not in records]
        rendered: dict[str, dict[str, str]] = {}
        run_ids: list[str] = []
        to_run: list[TaskRun] = []
        for index in missing:
            tree, task, _ = requests[index]
            if checksums[index] not in rendered:
                rendered[checksums[index]] = dict(self.render(tree))
            run_id, record_dir = self.history.new_run()
            run_ids.append(run_id)
            to_run.append(TaskRun(rendered[checksums[index]], task, record_dir))
        scored = self.run_in_pool(to_run)
        for index, run_id, task_run, result in zip(missing, run_ids, to_run, scored, strict=True):
            record = RunRecord.from_scored(
                result,
                run_id=run_id,
                tree=checksums[index],
                task=task_run.task.text,
                task_id=task_run.task.id,
                task_key=keys[index],
                repeat=requests[index][2],
                context=context,
                step=step,
            )
            self.history.add_run(record)
            records[index] = record
        return [records[index] for index in range(len(requests))]

    def runs(
        self, tree: HarnessTree, tasks: Sequence[EvaluationTask], *, repeats: int, step: int
    ) -> tuple[RunRecord, ...]:
        """``repeats`` run records of ``tree`` on each task, task by task; cached ones are reused."""
        return tuple(self.run(run_requests(tree, tasks, repeats), step=step))

    def runs_to_run(self, tree: HarnessTree, tasks: Sequence[EvaluationTask], *, repeats: int) -> int:
        """How many of the runs that :meth:`runs` asks for the cache does not hold, so they would run."""
        context = self.context().key()
        checksum = tree.checksum()
        return sum(
            self.history.cached(checksum, self.task_key(task), repeat, context) is None
            for _, task, repeat in run_requests(tree, tasks, repeats)
        )

    def run_count(self, candidate: HarnessCandidate, side_count: int) -> int:
        """How many runs :meth:`evaluate` asks for on ``side_count`` sides, cached ones included."""
        return len(candidate.evaluation_tasks) * self.task_repeats * side_count

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        """Score the named ``sides`` of ``candidate`` on its evaluation tasks.

        The default scores the candidate and the current tree as pairs.
        ``candidate_scores`` and ``current_scores`` are each side's
        ``RolloutScores`` JSON, by task; a side without tasks has an empty list.
        A policy that evaluates the candidate alone (a floor) passes
        ``("candidate",)``: no current task runs, ``current_scores`` is
        empty, the other ``current_*`` keys are absent, and
        ``evaluation_sides`` records the choice. The runs that must run
        interleave the sides inside each pairing, so a drift during the run
        (upstream load, rate limits) lands on both sides of a pair.
        """
        if not isinstance(candidate, HarnessCandidate):
            raise TypeError(f"harness evaluation requires HarnessCandidate, got {type(candidate).__name__}")
        sides = tuple(sides)
        if not sides or any(side not in EVALUATION_SIDES for side in sides):
            raise ValueError(f"sides must name one or both of {EVALUATION_SIDES}, got {sides!r}")
        trees = {"candidate": candidate.candidate_tree, "current": candidate.current_tree}
        requests = [
            (trees[side], task, repeat)
            for task in candidate.evaluation_tasks
            for repeat in range(self.task_repeats)
            for side in sides
        ]
        known = len(self.history.runs)
        records = self.run(requests, step=candidate.step)
        ran = len(self.history.runs) - known
        runs = {side: records[offset :: len(sides)] for offset, side in enumerate(sides)}
        scores = {side: [record.score for record in runs[side]] for side in sides}
        tasks = candidate.evaluation_tasks
        metrics: dict[str, Any] = {
            # The run scores of each side, by task.
            **{
                f"{side}_scores": (
                    RolloutScores.by_task([task.id for task in tasks], scores[side]).to_json()
                    if side in scores and tasks
                    else []
                )
                for side in EVALUATION_SIDES
            },
            # Runs that failed or gave no answer: every failure except a scored run that exited nonzero.
            "run_failures": sum(
                record.failure is not None and record.failure.stage != "exit"
                for side in sides
                for record in runs[side]
            ),
            "task_repeats": self.task_repeats,
            "runs_run": ran,
            "runs_cached": len(records) - ran,
        }
        for side in sides:
            metrics[f"{side}_residue"] = sum(record.residue for record in runs[side])
            metrics[f"{side}_score"] = math.fsum(score.value for score in scores[side] if score.value is not None)
            # Per agent sums over the side's runs, so a result says which agent did the work.
            metrics[f"{side}_agents"] = sum_agents(record.agents for record in runs[side])
            # Per run, in pairing order: the root's stage path and how its turn ended.
            metrics[f"{side}_paths"] = tuple(record.path for record in runs[side])
        if "candidate" in sides:
            # What the first candidate runs were graded on, so a rejected step names the task, the reply and why.
            metrics["candidate_runs"] = [
                {
                    "task": clip_record_text(record.task, RUN_SUMMARY_CHARS),
                    "score": record.score.to_json(),
                    "failure": (
                        None if record.failure is None else clip_record_text(record.failure.cause, RUN_SUMMARY_CHARS)
                    ),
                    "reply": None if record.reply is None else clip_record_text(record.reply, RUN_SUMMARY_CHARS),
                    "run": record.run_id,
                    # Only when missing, so a summary of an ordinary run keeps its shape.
                    **({} if record.transcript_read else {"transcript_read": False}),
                }
                for record in runs["candidate"][:RUN_SUMMARIES]
            ]
        if sides != EVALUATION_SIDES:
            metrics["evaluation_sides"] = list(sides)
        return EvaluationResult(evaluator="harness_runs", evaluator_version="2", metrics=metrics)


def run_requests(
    tree: HarnessTree, tasks: Sequence[EvaluationTask], repeats: int
) -> list[tuple[HarnessTree, EvaluationTask, int]]:
    """The ``(tree, task, repeat)`` requests of ``repeats`` runs of ``tree`` on each task, task by task.

    ``ValueError`` when ``repeats`` is below 1.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return [(tree, task, repeat) for task in tasks for repeat in range(repeats)]


def keep_remote_record(archive: bytes, record_dir: Path) -> None:
    """Extract a remote worker's record archive into ``record_dir``, which must not exist yet.

    Extraction refuses path traversal, links that escape the directory and
    special files, so a remote record never merges into another one.
    """
    try:
        record_dir.mkdir(parents=True)
        with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as opened:
            opened.extractall(record_dir, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise TrajectoryKeepError(f"cannot keep remote run trajectory under {record_dir}: {exc}") from exc


__all__ = [
    "EVALUATION_SIDES",
    "RUN_SUMMARIES",
    "RUN_SUMMARY_CHARS",
    "EvaluationContext",
    "HarnessCandidate",
    "HarnessCandidateEvaluator",
    "HarnessTaskEvaluator",
    "keep_remote_record",
]
