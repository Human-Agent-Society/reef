"""Evaluate harness trees by running episodes, with the history as the score cache.

:class:`HarnessEpisodeEvaluator` renders a tree with the served model
binding, runs each task as an episode in a worker pool, and scores it with
the deployment's :class:`EpisodeScorer` (``evolution.evaluation.scorer``).
:meth:`HarnessEpisodeEvaluator.episodes` scores any tree on any tasks, for a
method's ``tools.evaluate``.
:meth:`HarnessEpisodeEvaluator.evaluate` scores a :class:`HarnessCandidate`
and the head tree for a publication policy. Both run only the episodes that
the history does not hold: a scored episode of the same tree, task, repeat
and evaluation context is reused.
"""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from reef.core.evaluation import CandidateEvaluator, EvaluationResult, UpdateCandidate
from reef.evaluation.harness.history import EpisodeRecord, History
from reef.evaluation.harness.runner import EpisodeEvaluationWorker, EpisodeRun, EvaluationWorkerPool
from reef.evaluation.harness.scoring import EpisodeScorer
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import TrajectoryKeepError
from reef.harness.episodes.trajectory import sum_agents
from reef.harness.tree import HarnessTree, TreePatch, render_tree
from reef.harness.tree.checks import redact_secret_shaped
from reef.runtime.executor import WorkerSpec
from reef.runtime.executor.config import ExecutorSelection
from reef.runtime.executor.requirements import ExecutionRequirements
from reef.sandbox.process import EpisodeExecutor
from reef.storage.tasks import files_digest

#: The trees a candidate evaluation runs, in pairing order; a policy that evaluates the candidate alone names the
#: first.
EVALUATION_SIDES: tuple[str, ...] = ("candidate", "current")
#: How many candidate episodes an evaluation summarizes, and how much of each text it keeps: a summary for the
#: pages, never the traffic.
EPISODE_SUMMARIES = 8
EPISODE_SUMMARY_CHARS = 240


@dataclass(frozen=True)
class EvaluationContext:
    """Everything an episode's score depends on besides the tree and the task.

    Values: ``model`` is the served model. ``named_models`` pairs each name of
    ``evolution.models`` with its model, sorted by name, each name once: a
    judge scorer reads these models. ``adapter`` and ``adapter_version`` name
    the harness adapter and its pinned version (``None`` for an adapter
    without one); ``binary`` is the deployment's own harness binary
    (``evolution.binary``), ``None`` when the pinned one runs. ``executor`` is
    the episode executor's configuration (``evolution.executor`` and
    ``evolution.sandbox``) as one canonical JSON text, which only ``key``
    reads. ``scorer`` names the scorer class and its configuration.
    ``episode_timeout_s``, a finite number above 0, and ``forbid_residue``
    are the episode settings.
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
    episode_timeout_s: float
    forbid_residue: bool

    def __post_init__(self) -> None:
        names = [name for name, _ in self.named_models]
        if names != sorted(set(names)):
            raise ValueError("EvaluationContext.named_models must be sorted by name, each name once")
        timeout = self.episode_timeout_s
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
            raise ValueError("EvaluationContext.episode_timeout_s must be a finite number")
        if timeout <= 0:
            raise ValueError("EvaluationContext.episode_timeout_s must be above 0")

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
    evaluation_tasks: tuple[str, ...]
    step: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("HarnessCandidate.step must be a positive integer")

    @property
    def patch(self) -> TreePatch:
        """``current_tree.diff(candidate_tree)``: the smallest patch that makes the candidate from the head tree."""
        return self.current_tree.diff(self.candidate_tree)


def clip_redacted(text: str, limit: int) -> str:
    """``text`` redacted as the record is, cut at ``limit`` characters with an ellipsis marker."""
    text = redact_secret_shaped(text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class HarnessCandidateEvaluator(CandidateEvaluator, ABC):
    """Score a harness candidate on one or both sides.

    Purpose: a publication policy measures through it; the harness backend
    implements it over its evaluator, and the evaluator implements it itself.
    """

    @abstractmethod
    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        """The scores of ``candidate`` on the named ``sides``, from ``EVALUATION_SIDES``."""


class HarnessEpisodeEvaluator(HarnessCandidateEvaluator):
    """Score harness trees on tasks by running episodes, reusing the scores that the history holds.

    The episodes render each tree with the served model binding, which never
    enters a tree's files. ``binary`` is the installed binary the episodes
    run; ``binary_setting`` is the deployment's ``evolution.binary``, which
    the evaluation context names, and ``executor_setting`` is the episode
    executor's configuration as canonical JSON text, which the context
    digests. The evaluator owns the worker pool, so the caller must
    :meth:`close` it.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        scorer: EpisodeScorer,
        scorer_name: str,
        models: ModelBindings,
        binary: str | None,
        binary_setting: str | None,
        executor_setting: str,
        episode_timeout_s: float,
        episode_repeats: int,
        executor: EpisodeExecutor,
        forbid_residue: bool,
        worker_selection: ExecutorSelection,
        worker_requirements: ExecutionRequirements,
        history: History,
    ) -> None:
        if episode_repeats < 1:
            raise ValueError("episode_repeats must be at least 1")
        self.descriptor = descriptor
        self.scorer_name = scorer_name
        self.binary_setting = binary_setting
        self.executor_setting = executor_setting
        self.episode_timeout_s = episode_timeout_s
        self.episode_repeats = episode_repeats
        self.forbid_residue = forbid_residue
        self.history = history
        self.task_keys: dict[str, str] = {}
        self.use_models(models)
        self.worker_pool = EvaluationWorkerPool(
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
        self.worker_pool.close()

    def use_models(self, models: ModelBindings) -> None:
        """Run later episodes with ``models``: the served binding renders into each tree, and scorers receive all."""
        # Resolving the binding here refuses an adapter without a matching model_binding before any episode runs.
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
            episode_timeout_s=self.episode_timeout_s,
            forbid_residue=self.forbid_residue,
        )

    def render(self, tree: HarnessTree) -> dict[str, str]:
        """``tree``'s files for one episode, pointed at the served model."""
        return render_tree(tree, self.descriptor, self.models.served.config_overlays(self.descriptor))

    def task_key(self, task: str) -> str:
        """What identifies a task's content: the digest of a task directory's files, or of a prompt's text."""
        key = self.task_keys.get(task)
        if key is None:
            if self.descriptor.is_prompt_task_directory:
                key = files_digest(Path(task))
            else:
                key = hashlib.sha256(task.encode("utf-8")).hexdigest()
            self.task_keys[task] = key
        return key

    def run(self, requests: Sequence[tuple[HarnessTree, str, int]], *, step: int) -> list[EpisodeRecord]:
        """One episode record per ``(tree, task, repeat)`` request, in request order.

        Every tree must be a candidate of the history. A request that the
        cache holds takes the cached record. The other requests run as one
        wave in the worker pool, in request order, and their records join the
        history.
        """
        context = self.context().key()
        checksums = [tree.checksum() for tree, _, _ in requests]
        keys = [self.task_key(task) for _, task, _ in requests]
        records: dict[int, EpisodeRecord] = {}
        for index, (checksum, key, (_, _, repeat)) in enumerate(zip(checksums, keys, requests, strict=True)):
            if checksum not in self.history.candidates:
                raise ValueError(f"tree {checksum} is not a candidate of the history")
            cached = self.history.cached(checksum, key, repeat, context)
            if cached is not None:
                records[index] = cached
        missing = [index for index in range(len(requests)) if index not in records]
        rendered: dict[str, dict[str, str]] = {}
        episodes: list[str] = []
        runs: list[EpisodeRun] = []
        for index in missing:
            tree, task, _ = requests[index]
            if checksums[index] not in rendered:
                rendered[checksums[index]] = self.render(tree)
            episode, record_dir = self.history.new_episode()
            episodes.append(episode)
            runs.append(EpisodeRun(rendered[checksums[index]], task, record_dir))
        scored = self.worker_pool.run(runs, self.models)
        for index, episode, run, result in zip(missing, episodes, runs, scored, strict=True):
            if result.record_archive is not None:
                keep_remote_record(result.record_archive, run.record_dir)
            record = EpisodeRecord.from_scored(
                result,
                episode=episode,
                tree=checksums[index],
                task=run.task,
                task_key=keys[index],
                repeat=requests[index][2],
                context=context,
                step=step,
            )
            self.history.add_episode(record)
            records[index] = record
        return [records[index] for index in range(len(requests))]

    def episodes(
        self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int, step: int
    ) -> tuple[EpisodeRecord, ...]:
        """``repeats`` episode records of ``tree`` on each task, task by task; cached ones are reused."""
        return tuple(self.run(episode_requests(tree, tasks, repeats), step=step))

    def episodes_to_run(self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int) -> int:
        """How many of the episodes that :meth:`episodes` asks for the cache does not hold, so they would run."""
        context = self.context().key()
        checksum = tree.checksum()
        return sum(
            self.history.cached(checksum, self.task_key(task), repeat, context) is None
            for _, task, repeat in episode_requests(tree, tasks, repeats)
        )

    def episode_count(self, candidate: HarnessCandidate, side_count: int) -> int:
        """How many episodes :meth:`evaluate` asks for on ``side_count`` sides, cached ones included."""
        return len(candidate.evaluation_tasks) * self.episode_repeats * side_count

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        """Score the named ``sides`` of ``candidate`` on its evaluation tasks.

        The default scores the candidate and the current tree as pairs. A
        policy that evaluates the candidate alone (a floor) passes
        ``("candidate",)``: no current episode runs, ``current_scores`` is
        empty, the other ``current_*`` keys are absent, and
        ``evaluation_sides`` records the choice. The episodes that must run
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
            for repeat in range(self.episode_repeats)
            for side in sides
        ]
        known = len(self.history.episodes)
        records = self.run(requests, step=candidate.step)
        ran = len(self.history.episodes) - known
        runs = {side: records[offset :: len(sides)] for offset, side in enumerate(sides)}
        scores = {side: tuple(record.score for record in runs[side]) for side in sides}
        metrics: dict[str, Any] = {
            "candidate_scores": scores.get("candidate", ()),
            "current_scores": scores.get("current", ()),
            "episode_failures": sum(score is None for side in sides for score in scores[side]),
            "episode_repeats": self.episode_repeats,
            "episodes_run": ran,
            "episodes_cached": len(records) - ran,
        }
        for side in sides:
            metrics[f"{side}_residue"] = sum(record.residue for record in runs[side])
            metrics[f"{side}_score"] = float(sum(score for score in scores[side] if score is not None))
            # Per agent sums over the side's episodes, so a result says which agent did the work.
            metrics[f"{side}_agents"] = sum_agents(record.agents for record in runs[side])
            # Per episode, in pairing order: the root's stage path and how its turn ended.
            metrics[f"{side}_paths"] = tuple(record.path for record in runs[side])
        if "candidate" in sides:
            # What the first candidate episodes were graded on, so a rejected step names the task, the reply and why.
            metrics["candidate_episodes"] = [
                {
                    "task": clip_redacted(record.task, EPISODE_SUMMARY_CHARS),
                    "score": record.score,
                    "failure": (
                        None if record.failure is None else clip_redacted(record.failure.cause, EPISODE_SUMMARY_CHARS)
                    ),
                    "reply": None if record.reply is None else clip_redacted(record.reply, EPISODE_SUMMARY_CHARS),
                    "episode": record.episode,
                    # Only when missing, so a summary of an ordinary episode keeps its shape.
                    **({} if record.transcript_read else {"transcript_read": False}),
                }
                for record in runs["candidate"][:EPISODE_SUMMARIES]
            ]
        if sides != EVALUATION_SIDES:
            metrics["evaluation_sides"] = list(sides)
        return EvaluationResult(evaluator="harness_episodes", evaluator_version="2", metrics=metrics)


def episode_requests(tree: HarnessTree, tasks: Sequence[str], repeats: int) -> list[tuple[HarnessTree, str, int]]:
    """The ``(tree, task, repeat)`` requests of ``repeats`` episodes of ``tree`` on each task, task by task.

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
        raise TrajectoryKeepError(f"cannot keep remote episode trajectory under {record_dir}: {exc}") from exc


__all__ = [
    "EPISODE_SUMMARIES",
    "EPISODE_SUMMARY_CHARS",
    "EVALUATION_SIDES",
    "EvaluationContext",
    "HarnessCandidate",
    "HarnessCandidateEvaluator",
    "HarnessEpisodeEvaluator",
    "keep_remote_record",
]
