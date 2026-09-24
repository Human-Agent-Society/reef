"""The harness evolution backend itself: propose, evaluate, select, render.

The package docstring in :mod:`reef.train.cordis_backend` describes the step
this implements. This module holds the backend and the candidate types it
owns. ``reef.recipe.cordis`` assembles it; shared tree admission lives in
``reef.harness.tree.mutations``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import tarfile
import tempfile
import threading
import time
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.core.requirements import MAX_REQUIRES, merge_requires, parse_requires
from reef.core.trajectories import recorded_payload, source_record_id
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.compose import Context
from reef.harness.compose.loader import EntryOptions, Loader
from reef.harness.episodes.e2b import E2BExecutor
from reef.harness.episodes.executor import EPISODE_OWNER_LEASE, EpisodeExecutor, LocalExecutor, SandboxExecutor
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings, ModelBindingsResolver, usage_of
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError
from reef.harness.episodes.vendor_install import install_prefix, resolve_binary
from reef.harness.tree.mutations import (
    Mutation,
    MutationError,
    _load_error,
    _loader_entries,
    _native_refusal,
    _nodes_from,
    admit_mutations,
)
from reef.harness.tree.nodes import (
    ALWAYS_REVIEWED_KINDS,
    NODE_KINDS,
    RESERVED_ENTRY_IDS,
    directive_shaped,
    redact_secret_shaped,
    secret_shaped,
)
from reef.harness.tree.render import render_composition
from reef.runtime.executor import Executor, WorkerSpec
from reef.runtime.executor.config import ExecutorSettings
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.cordis_backend.contracts import ProposalValidator, StepProgress, StepProgressReader, StepRecords
from reef.train.cordis_backend.execution import EvaluationWorkerPool, evaluation_selection
from reef.train.cordis_backend.manifest import FailureManifest, FailureObservation
from reef.train.cordis_backend.manifest import FailureRecord as FailureRecord  # re-export: manifest entry type
from reef.train.cordis_backend.manifest import advance
from reef.train.cordis_backend.proposals import Proposal, ProposalInbox
from reef.train.cordis_backend.strategies import (
    AgentHost,
    EpisodeScorer,
    Promoter,
    Proposer,
    ProposerCalls,
    StepProposal,
    accepts_keyword,
    accepts_manifest,
)
from reef.train.evaluation.evaluators import BackendEvaluateMixin, CandidatePluginFactory
from reef.train.types import TrainingBatch, TrainStepResult, TrajectoryItem, trajectories


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
    ) -> _ScoredEpisode:
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
    ) -> _ScoredEpisode:
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
            scored = _ScoredEpisode(None, FailureObservation(task=task, stage="launch", cause=str(error)))
            _write_episode_record(keep_dir, task, None, scored)
            return scored
        except TrajectoryError as error:
            scored = _ScoredEpisode(None, FailureObservation(task=task, stage="trajectory", cause=str(error)))
            _write_episode_record(keep_dir, task, None, scored)
            return scored
        finally:
            EPISODE_OWNER_LEASE.reset(token)
        scored = self._score_result(result, task, models)
        _write_episode_record(keep_dir, task, result, scored)
        return scored

    def _score_result(self, result: EpisodeResult, task: str, models: ModelBindings | None = None) -> _ScoredEpisode:
        """The score and the observations of an episode that ran."""
        residue = len(result.residue)
        agents = _agent_work(result.trajectory)
        path = _stage_path(result.trajectory)
        if residue and self.forbid_residue:
            cause = f"{residue} file(s) outside the cleanup whitelist: {result.residue[0]}"
            return _ScoredEpisode(
                None, FailureObservation(task=task, stage="residue", cause=cause), residue, agents, path
            )
        trial_error = _failed_trial_error(result.trajectory)
        if trial_error:
            # The terminus runner recorded a trial that never ran (the image did not build, the agent could not
            # start): no answer was given, so it ranks below every real score instead of tying a zero.
            return _ScoredEpisode(
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
            stage = "loop" if _root_header(result.trajectory).get("loop") else "graph"
            return _ScoredEpisode(None, FailureObservation(task=task, stage=stage, cause=cause), residue, agents, path)
        score = float(
            self.scorer(task, result) if models is None else self.scorer.score_with_models(task, result, models)
        )
        if not math.isfinite(score):
            raise ValueError(f"episode scorer returned a non-finite score {score!r} for task {task!r}")
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return _ScoredEpisode(
                score, FailureObservation(task=task, stage="exit", cause=cause), residue, agents, path
            )
        return _ScoredEpisode(score, None, residue, agents, path)


def _failed_trial_error(trajectory: Sequence[Mapping[str, Any]]) -> str:
    """The error of a terminus trial that never ran (a failed ``verifier`` row with an error), else empty."""
    for event in trajectory:
        if event.get("type") == "verifier" and event.get("failed") and event.get("error"):
            return str(event["error"])
    return ""


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


#: Characters kept per text in the step record; a longer text ends in a clip marker.
RECORD_TEXT_CAP = 20_000
#: The record files one step writes under its claimed directory (``<step>``, a retried step ``<step>-<attempt>``).
RECORD_PROPOSER_FILE = "proposer.json"
RECORD_MUTATIONS_FILE = "mutations.json"
RECORD_EPISODES_DIR = "episodes"
RECORD_EPISODE_FILE = "episode.json"
#: The trees an evaluation step runs, in pairing order; a policy that evaluates the candidate alone names the first only.
EVALUATION_SIDES: tuple[str, ...] = ("candidate", "current")


def _clip(text: str) -> str:
    """``text`` with credential-shaped literals redacted, cut at the record cap with a marker naming how much was dropped.

    The record holds model traffic and proposals before the tree boundary
    saw them, so the boundary's credential tripwire runs here too."""
    text = redact_secret_shaped(text)
    if len(text) <= RECORD_TEXT_CAP:
        return text
    return f"{text[:RECORD_TEXT_CAP]}... [clipped {len(text) - RECORD_TEXT_CAP} chars]"


def _bounded(value: Any) -> Any:
    """A copy of a request value that JSON can write, with every string redacted and cut at the record cap."""
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, Mapping):
        return {str(key): _bounded(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [_bounded(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _clip(str(value))


def _mutation_record(mutation: Mutation) -> dict[str, Any]:
    """A mutation as the commit record, the rejected history and the step record persist it: op, id and full options."""
    options = None if mutation.options is None else dict(mutation.options)
    return {"op": mutation.op, "id": mutation.id, "options": options}


def _episode_name(side: str, task_index: int, repeat: int) -> str:
    """The record directory of one evaluation episode: ``<side>-<task index>``, a repeat adding ``-<repeat>``."""
    return f"{side}-{task_index}" if repeat == 0 else f"{side}-{task_index}-{repeat}"


def _prompt_of(sample: TrajectoryItem) -> str | None:
    """The trace's last user message, the prompt ``evaluate`` scores; ``None`` for a tool-only turn."""
    messages = recorded_payload(sample).get("messages")
    if not isinstance(messages, Sequence):
        return None
    for message in reversed(list(messages)):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, Sequence) and not isinstance(content, str):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"
            )
            if text.strip():
                return text
    return None


def _default_promote(samples: Sequence[TrajectoryItem]) -> list[str]:
    """The default policy: every trace's user prompt is a candidate task."""
    prompts = (_prompt_of(sample) for sample in samples)
    return [prompt for prompt in prompts if prompt is not None]


def _client_of(sample: TrajectoryItem) -> str:
    """The client that sent a trace: its x-reef-tag-client, else its session tag, else ``untagged``."""
    metadata = recorded_payload(sample).get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    for key in ("client", "session"):
        value = tags.get(key) if isinstance(tags, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value
    return "untagged"


def _source_of(sample: TrajectoryItem) -> dict[str, Any]:
    """What a proposer may know about where a sample came from; the text itself stays untrusted."""
    return {"record": source_record_id(sample), "client": _client_of(sample), "untrusted": True}


def _screened(prompt: str) -> bool:
    """Whether a trace prompt contains a credential or instruction override barred from task records."""
    return secret_shaped(prompt) or directive_shaped(prompt)


def _admit_promoted(
    existing: Sequence[str],
    candidates: Sequence[str],
    seed: frozenset[str],
    cap: int,
    *,
    per_client: int = 0,
    clients: Mapping[str, str] | None = None,
    counts: dict[str, int] | None = None,
) -> list[str]:
    """Append new candidate prompts under the caps; a screened or malformed prompt is skipped, never failed."""
    promoted = list(existing)
    seen = set(seed) | set(promoted)
    counts = {} if counts is None else counts
    for prompt in candidates:
        if len(promoted) >= cap:
            break
        if not isinstance(prompt, str) or not prompt.strip() or prompt in seen or _screened(prompt):
            continue
        # Untagged traffic has no identity to count under, so only a tagged client meets the per-client cap.
        client = (clients or {}).get(prompt, "untagged")
        if per_client and client != "untagged":
            if counts.get(client, 0) >= per_client:
                continue
            counts[client] = counts.get(client, 0) + 1
        promoted.append(prompt)
        seen.add(prompt)
    return promoted


#: The most activity lines a step keeps; the oldest go first.
MAX_ACTIVITY = 300


class _StepCalls(ProposerCalls):
    """One step's model-call budget, record and live activity, shared by the budgeted bindings and an agent's gateway.

    A cap of 0 is no budget; the record is the list the step writes to ``proposer.json``; the activity is what the
    request page shows while the step runs, read from another thread through :meth:`activity`.
    """

    def __init__(self, cap: int, record: list[dict[str, Any]]) -> None:
        self._cap = cap
        self._spent = 0
        self._record = record
        self._activity: deque[dict[str, Any]] = deque(maxlen=MAX_ACTIVITY)
        self._lock = threading.Lock()

    def spend(self) -> None:
        if self._cap and self._spent >= self._cap:
            raise RuntimeError(f"model call budget of {self._cap} per evolve step exhausted")
        self._spent += 1

    def record(self, entry: Mapping[str, Any]) -> None:
        self._record.append(dict(entry))

    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        line: dict[str, Any] = {"at": time.time(), "kind": kind, "text": " ".join(text.split())[:300]}
        if failed:
            line["failed"] = True
        with self._lock:
            self._activity.append(line)

    def activity(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(line) for line in self._activity)


class _BudgetedBinding(ModelBinding):
    """A ModelBinding that delegates ``chat`` to a wrapped binding under a
    shared per-step call budget and records every call in the step record.

    A subclass so the proposer still receives ModelBinding values, but it
    holds the real binding and forwards to its ``chat`` (never the base
    implementation), so a method's own binding behavior is preserved. The
    shared counter is a mutable one-element list so every binding in the set
    decrements the same budget; a cap of 0 is no budget. The record is the
    step's list: one entry per call with the model, the request, the reply
    or the error, the provider response when available, the seconds it took and the
    ``usage`` (input and output tokens), every text cut at the record cap.
    """

    _inner: ModelBinding
    _calls: _StepCalls

    def __init__(self, inner: ModelBinding, calls: _StepCalls) -> None:
        super().__init__(
            base_url=inner.base_url,
            model=inner.model,
            api_key=inner.api_key,
            api=inner.api,
            timeout_s=inner.timeout_s,
        )
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_calls", calls)

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str:
        self._calls.spend()
        self._calls.note("model", f"asking {self.model}")
        kwargs: dict[str, Any] = dict(params)
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        entry: dict[str, Any] = {"model": self.model, "messages": _bounded(messages), "params": _bounded(kwargs)}
        # ``chat`` returns text only; keep the provider response too, including any reasoning it exposed.
        if isinstance(self._inner, ModelBinding):
            object.__setattr__(self._inner, "_last_usage", None)
            object.__setattr__(self._inner, "_last_response", None)
        started = time.monotonic()
        try:
            reply = self._inner.chat(messages, **kwargs)
        except BaseException as exc:
            # The failed call is the step's decision too: the record keeps it before the error propagates.
            entry["error"] = _clip(f"{type(exc).__name__}: {exc}")
            raise
        else:
            entry["reply"] = _clip(reply) if isinstance(reply, str) else _bounded(reply)
            return reply
        finally:
            # A provider may spend its budget on reasoning and return no final text; retain that response on error too.
            response = self._inner.last_response() if isinstance(self._inner, ModelBinding) else None
            if response is not None:
                entry["response"] = _bounded(response)
            entry["seconds"] = round(time.monotonic() - started, 3)
            usage = self._inner.last_usage() if isinstance(self._inner, ModelBinding) else None
            if usage is not None:
                entry["usage"] = usage
            self._calls.record(entry)
            self._note_answer(entry)

    def _note_answer(self, entry: Mapping[str, Any]) -> None:
        """One activity line for a finished call: how long it took and its tokens, or its error."""
        seconds = entry.get("seconds", 0)
        if "error" in entry:
            self._calls.note("model", f"{self.model} failed after {seconds:g} s: {entry['error']}", failed=True)
            return
        usage = entry.get("usage")
        tokens = (
            f", {int(usage.get('input_tokens', 0) or 0):,} → {int(usage.get('output_tokens', 0) or 0):,} tokens"
            if isinstance(usage, Mapping)
            else ""
        )
        self._calls.note("model", f"{self.model} answered in {seconds:g} s{tokens}")

    def complete(self, body: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """A method's raw request goes through the same budget and record as ``chat``: ``body`` in, ``response`` out."""
        self._calls.spend()
        self._calls.note("model", f"asking {self.model}")
        kwargs: dict[str, Any] = {} if timeout_s is None else {"timeout_s": timeout_s}
        entry: dict[str, Any] = {"model": self.model, "body": _bounded(body), "params": _bounded(kwargs)}
        started = time.monotonic()
        try:
            response = self._inner.complete(body, **kwargs)
        except BaseException as exc:
            entry["error"] = _clip(f"{type(exc).__name__}: {exc}")
            entry["seconds"] = round(time.monotonic() - started, 3)
            self._calls.record(entry)
            self._note_answer(entry)
            raise
        entry["response"] = _bounded(response)
        entry["seconds"] = round(time.monotonic() - started, 3)
        usage = usage_of(response)
        if usage is not None:
            entry["usage"] = usage
        self._calls.record(entry)
        self._note_answer(entry)
        return response


def _proposer_tokens(record: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Input and output tokens summed over the record's calls that reported usage."""
    usages = [entry["usage"] for entry in record if isinstance(entry.get("usage"), Mapping)]
    return (
        sum(int(usage.get("input_tokens", 0) or 0) for usage in usages),
        sum(int(usage.get("output_tokens", 0) or 0) for usage in usages),
    )


def _budgeted_bindings(models: ModelBindings, calls: _StepCalls) -> ModelBindings:
    """The proposer's view: every ``chat`` shares the step's budget and lands in its record.

    The bindings are wrapped whatever the cap, so the record sees every call;
    with a cap of 0 nothing is refused. ``calls`` is per prepare_step call,
    so a cap bounds one step's model bill, never the campaign's.
    """

    def wrap(binding: ModelBinding) -> ModelBinding:
        return _BudgetedBinding(binding, calls)

    return ModelBindings(
        served=wrap(models.served),
        named={name: wrap(models[name]) for name in models if name != "served"},
    )


def tree_files(descriptor: AdapterDescriptor, entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The entries list as the file ``files.tree`` names, verbatim; empty for an adapter that declares none."""
    if descriptor.tree_path is None:
        return {}
    return {descriptor.tree_path: json.dumps([dict(entry) for entry in entries], indent=2, sort_keys=True) + "\n"}


class ScoreComparisonMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that selects when wins exceed losses by ``min_win_margin`` (0: plain majority)."""

    def __init__(self, *, min_win_margin: int = 0) -> None:
        if isinstance(min_win_margin, bool) or not isinstance(min_win_margin, int) or min_win_margin < 0:
            raise ValueError("min_win_margin must be an integer of at least 0")
        super().__init__()
        self._min_win_margin = min_win_margin

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        candidate_scores, current_scores = _score_vectors(evaluation)
        wins, losses = _score_comparison_tally(candidate_scores, current_scores)
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


class ScoreComparisonPlugin(ScoreComparisonMixin, BackendEvaluateMixin):
    """Cordis's default evaluation: measure through the candidate backend, decide by score comparison."""

    def __init__(self, candidate_backend: Any, *, min_win_margin: int = 0) -> None:
        super().__init__(min_win_margin=min_win_margin)
        self._candidate_backend = candidate_backend


@dataclass(frozen=True)
class ScoreComparisonPluginFactory(CandidatePluginFactory):
    """Bind a scenario's score comparison policy with its configured margin."""

    min_win_margin: int = 0

    def build(self, candidate_backend: CandidateEvaluator) -> CandidateEvaluationPlugin:
        return ScoreComparisonPlugin(candidate_backend, min_win_margin=self.min_win_margin)


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


class FloorPlugin(FloorMixin, BackendEvaluateMixin):
    """Evaluate the candidate alone through the candidate backend, decide by the floor; the current release is not run."""

    def __init__(self, candidate_backend: Any, *, floor_score: float = 1.0) -> None:
        super().__init__(floor_score=floor_score)
        self._candidate_backend = candidate_backend

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self._candidate_backend.evaluate(candidate, sides=("candidate",))


@dataclass(frozen=True)
class FloorPluginFactory(CandidatePluginFactory):
    """Bind a scenario's floor policy with its configured floor score."""

    floor_score: float = 1.0

    def build(self, candidate_backend: CandidateEvaluator) -> CandidateEvaluationPlugin:
        return FloorPlugin(candidate_backend, floor_score=self.floor_score)


def _score_vectors(
    evaluation: EvaluationResult,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    candidate = tuple(evaluation.metrics.get("candidate_scores", ()))
    current = tuple(evaluation.metrics.get("current_scores", ()))
    return candidate, current


def _score_comparison_tally(candidate: tuple[float | None, ...], current: tuple[float | None, ...]) -> tuple[int, int]:
    wins = losses = 0
    for candidate_score, current_score in zip(candidate, current, strict=True):
        candidate_rank = candidate_score if candidate_score is not None else float("-inf")
        current_rank = current_score if current_score is not None else float("-inf")
        if candidate_rank > current_rank:
            wins += 1
        elif candidate_rank < current_rank:
            losses += 1
    return wins, losses


class CordisBackend(CandidateBackend, ProposalValidator, StepRecords, StepProgressReader):
    """Settle one proposal per step through episode pairs.

    A proposal is one ``Mutation`` or a sequence of them. A sequence applies
    under one snapshot and settles under one selection decision: the whole set
    publishes together or reverts together.

    Owns a compose Entry tree (``Loader``) of harness
    nodes. The tree is in-memory; its state is serialized through the
    algorithm state (``"entries"`` key) so the commit log and snapshot
    metadata persist and recover it. A selected mutation renders to files and
    returns a ``TrainStepResult`` carrying the artifact; a rejected mutation
    restores the prior tree and returns a no-artifact ``TrainStepResult``.

    ``seed`` is the first-boot composition: entry options exactly like the
    state's ``entries``, carried by ``initial_state`` and loaded once at
    construction so an invalid seed refuses boot. A recovered state brings
    its own entries and therefore always wins over the seed.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        propose: Proposer,
        score_episode: EpisodeScorer,
        tasks: tuple[str, ...],
        models: ModelBindings | ModelBinding,
        model_resolver: ModelBindingsResolver | None = None,
        binary: str | None = None,
        episode_timeout_s: float = 600.0,
        episode_repeats: int = 1,
        forbid_residue: bool = False,
        max_steps: int = 0,
        max_failure_streak: int = 0,
        max_model_calls_per_step: int = 0,
        executor: EpisodeExecutor | None = None,
        promote_failures: bool = False,
        max_promoted_tasks: int = 50,
        max_promoted_per_client: int = 5,
        promote: Promoter | None = None,
        recheck_every: int = 0,
        max_rejected_history: int = 25,
        publish: str = "auto",
        review_kinds: tuple[str, ...] = (),
        seed: tuple[Mapping[str, Any], ...] = (),
        episode_workers: int | None = None,
        proposals_dir: str | Path | None = None,
        max_pending_proposals: int = 8,
        step_record_dir: str | Path | None = None,
        worker_executor: ExecutorSettings | None = None,
        worker_gpus: float | None = None,
        agent_executor: EpisodeExecutor | None = None,
        agent_timeout_s: float = 1800.0,
        agent_trial_timeout_s: float = 300.0,
    ) -> None:
        if not tasks:
            raise ValueError("harness evolution requires a non-empty task set")
        if step_record_dir is not None and not str(step_record_dir):
            raise ValueError("step_record_dir must be a non-empty path when set")
        if isinstance(models, ModelBinding):
            models = ModelBindings(served=models)
        if not isinstance(models, ModelBindings):
            raise TypeError(f"harness evolution requires ModelBindings, got {type(models).__name__}")
        self._ctx = Context()
        self._loader = Loader(self._ctx, NODE_KINDS.get)
        self._descriptor = descriptor
        self._propose = propose
        self._score_episode = score_episode
        self._tasks = tasks
        self._models = models
        self._model_resolver = model_resolver
        # The served binding renders into episodes only. It is resolved once
        # here so an adapter without a matching model_binding refuses boot,
        # not the first step.
        self._binding_nodes = models.served.compose_nodes(descriptor)
        # Checked once at boot: an out-of-tree Proposer subclass whose
        # ``__call__`` predates the manifest keyword is called without it.
        self._propose_accepts_manifest = accepts_manifest(propose.__call__)
        self._propose_accepts_rejected = accepts_keyword(propose.__call__, "rejected")
        self._propose_accepts_sources = accepts_keyword(propose.__call__, "sources")
        self._propose_accepts_entries = accepts_keyword(propose.__call__, "entries")
        if (
            isinstance(episode_timeout_s, bool)
            or not isinstance(episode_timeout_s, (int, float))
            or episode_timeout_s <= 0
        ):
            raise ValueError("episode_timeout_s must be a positive number")
        if isinstance(episode_repeats, bool) or not isinstance(episode_repeats, int) or episode_repeats < 1:
            raise ValueError("episode_repeats must be an integer of at least 1")
        if not isinstance(forbid_residue, bool):
            raise ValueError("forbid_residue must be a boolean")
        for label, value in (
            ("max_steps", max_steps),
            ("max_failure_streak", max_failure_streak),
            ("max_model_calls_per_step", max_model_calls_per_step),
            ("recheck_every", recheck_every),
            ("max_rejected_history", max_rejected_history),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 disables the limit)")
        self._worker_selection, self._worker_requirements = evaluation_selection(
            score_episode, episode_workers, worker_executor or ExecutorSettings(), worker_gpus
        )
        Executor.get_class(self._worker_selection.settings.backend)
        self._episode_timeout_s = float(episode_timeout_s)
        self._episode_repeats = episode_repeats
        self._forbid_residue = forbid_residue
        self._max_steps = max_steps
        self._max_failure_streak = max_failure_streak
        self._max_model_calls_per_step = max_model_calls_per_step
        # Preflighted at build (recipe.build), so a hosted deployment that
        # requires the sandbox fails to start, not at the first episode.
        self._executor = executor or LocalExecutor()
        if max_promoted_tasks < 0:
            raise ValueError("max_promoted_tasks must be at least 0")
        if max_promoted_per_client < 0:
            raise ValueError("max_promoted_per_client must be at least 0")
        self._promote_failures = promote_failures
        self._max_promoted_tasks = max_promoted_tasks
        self._max_promoted_per_client = max_promoted_per_client
        self._promote_task = promote
        self._promote_accepts_manifest = promote is not None and accepts_manifest(promote.__call__)
        self._recheck_every = recheck_every
        self._max_rejected_history = max_rejected_history
        if publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")
        self._publish = publish
        self._review_kinds = frozenset(review_kinds)
        self._seed = tuple(dict(entry) for entry in seed)
        # Selected compositions render into temporary source trees before the
        # scenario committer copies them into repository-owned storage.
        # Track only trees created by this backend so a durable commit can
        # remove its source without touching caller-owned Artifact.local paths.
        self._rendered_publications: dict[int, Artifact] = {}
        # Agent proposals wait here between the route that admitted them and the step that takes them.
        self._proposals = None if proposals_dir is None else ProposalInbox(Path(proposals_dir), max_pending_proposals)
        # Created at boot so an unwritable record path refuses to start, not the first step.
        self._step_record_dir = None if step_record_dir is None else Path(step_record_dir)
        self._current_step_record: Path | None = None
        # Written by the training thread, read by the service's request page from another: each write is one
        # assignment of a frozen value, which is all the synchronization a reader that tolerates a stale phase needs.
        self._step_progress: StepProgress | None = None
        # The running step's calls, whose activity the progress reports; replaced by the next step's.
        self._step_calls: _StepCalls | None = None
        if self._step_record_dir is not None:
            try:
                self._step_record_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"step_record_dir {self._step_record_dir} cannot be created: {exc}") from exc
        self._validate_seed()
        # Evolution needs the binary at boot, just like the model binding.
        # Install failures propagate with the missing tool or vendor error.
        prefix = install_prefix(descriptor)
        if isinstance(self._executor, E2BExecutor):
            self._executor = self._executor.for_adapter(descriptor, binary=binary)
            self._executor.preflight()
            self._binary = self._executor.binary
        else:
            self._binary = binary if binary is not None else resolve_binary(descriptor, prefix=prefix)
        if binary is None and descriptor.install is not None and isinstance(self._executor, SandboxExecutor):
            # Bind the whole install: npm launchers are symlinks into packages,
            # and git installs need both their editable source and their venv.
            self._executor = replace(self._executor, base_paths=(*self._executor.base_paths, str(prefix)))
        for label, timeout in (("agent_timeout_s", agent_timeout_s), ("agent_trial_timeout_s", agent_trial_timeout_s)):
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError(f"{label} must be a positive number")
        # The agent proposer runs the same installed binary as the episodes, under its own isolation.
        if binary is None and descriptor.install is not None and isinstance(agent_executor, SandboxExecutor):
            agent_executor = replace(agent_executor, base_paths=(*agent_executor.base_paths, str(prefix)))
        if isinstance(agent_executor, E2BExecutor):
            agent_executor = agent_executor.for_adapter(descriptor)
            agent_executor.preflight()
        self._agent_executor = agent_executor
        self._agent_timeout_s = float(agent_timeout_s)
        self._agent_trial_timeout_s = float(agent_trial_timeout_s)
        self._evaluation_pool = EvaluationWorkerPool(
            self._worker_selection,
            self._worker_requirements,
            WorkerSpec(
                EpisodeEvaluationWorker,
                args=(
                    descriptor,
                    score_episode,
                    self._binary,
                    self._episode_timeout_s,
                    self._executor,
                    forbid_residue,
                    self._worker_selection.settings.backend != "uni",
                    self._worker_selection.settings.backend not in ("uni", "mp"),
                ),
            ),
        )
        # Direct Python users should close explicitly; GC is a fallback only.
        self._pool_finalizer = weakref.finalize(self, self._evaluation_pool.close)

    def close(self) -> None:
        self._pool_finalizer()

    @property
    def proposals(self) -> ProposalInbox | None:
        return self._proposals

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def admit(
        self, entries: Sequence[Mapping[str, Any]], mutations: Sequence[Mutation]
    ) -> tuple[list[EntryOptions], str | None]:
        """The route's admission: ``admit_mutations`` under this backend's adapter, touching no live state."""
        return admit_mutations(entries, mutations, self._descriptor)

    def _validate_seed(self) -> None:
        """Load the seed into the tree once so a bad seed refuses construction.

        This is the same load path ``prepare_step`` runs on the
        state entries, so a node validation failure surfaces here - at
        recipe build time, naming the failing entry - instead of an empty
        or broken composition tying every comparison at run time.
        """
        seen: set[str] = set()
        for options in self._seed:
            for key in ("id", "name"):
                value = options.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"seed entry {options!r} requires a non-empty string {key!r}")
            if options["id"] in seen:
                raise ValueError(f"seed names entry {options['id']!r} twice")
            seen.add(str(options["id"]))
        self._loader.root.update([dict(options) for options in self._seed])
        for options in self._seed:
            error = self._load_error(str(options["id"]))
            if error is not None:
                raise ValueError(f"seed entry {options['id']!r} rejected: {error}")
        refusal = _native_refusal(self._seed, self._descriptor)
        if refusal is not None:
            raise ValueError(f"seed {refusal}")

    def initial_state(self) -> Mapping[str, Any]:
        return {"steps": 0, "entries": [dict(entry) for entry in self._seed]}

    def shipped_content_update(self, state: Mapping[str, Any], published_tree: Path) -> TrainStepResult | None:
        """Republish the reef-owned entries this Reef seeds when the served tree renders them differently.

        Reef-owned entries (``RESERVED_ENTRY_IDS``) are copied from Reef's own assets
        into a scenario's seed, and no proposal may change them, so a scenario keeps
        the copy it was created with. The seed here is read from the running Reef's
        assets; an entry whose rendered files differ from ``published_tree``, or that
        the served tree lacks, is replaced in (or appended to) the state's entries,
        and the whole tree is rendered again from them, the way a step publishes it.
        """
        shipped = {str(entry["id"]): dict(entry) for entry in self._seed if entry["id"] in RESERVED_ENTRY_IDS}
        if not shipped:
            return None
        # A kind renders shared config files for any node; an entry owns the files beyond those.
        shared = render_composition((), self._descriptor)
        stale = []
        for entry_id, entry in shipped.items():
            owned = render_composition(((str(entry["name"]), entry.get("config")),), self._descriptor)
            for relative, text in owned.items():
                if relative in shared:
                    continue
                path = published_tree / relative
                if not path.is_file() or path.read_text(encoding="utf-8") != text:
                    stale.append(entry_id)
                    break
        if not stale:
            return None
        entries = [dict(entry) for entry in state["entries"]]
        present = {str(entry.get("id")) for entry in entries}
        entries = [dict(shipped[str(entry.get("id"))]) if entry.get("id") in stale else entry for entry in entries]
        entries.extend(dict(shipped[entry_id]) for entry_id in stale if entry_id not in present)
        published = {
            **render_composition(_nodes_from(entries), self._descriptor),
            **tree_files(self._descriptor, entries),
        }
        artifact = Artifact.local(_write_rendered_files(published))
        # commit_applied discards the rendered directory once the commit is durable, as it does a step's.
        step = int(state["steps"])
        replaced = self._rendered_publications.get(step)
        if replaced is not None:
            replaced.discard()
        self._rendered_publications[step] = artifact
        # The loader takes the entries from the committed state when the next step prepares.
        return TrainStepResult(
            {**state, "entries": entries}, {"shipped_content_update": {"entries": stale}}, artifact=artifact
        )

    @property
    def step_progress(self) -> StepProgress | None:
        """The running step's phase, start and activity so far, ``None`` between steps (see ``StepProgress``)."""
        progress, calls = self._step_progress, self._step_calls
        if progress is None or calls is None:
            return progress
        return replace(progress, activity=calls.activity())

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        self._current_step_record = None
        self._step_progress = None
        try:
            prepared = self._prepare_step(batch, state, scenario_step)
        except BaseException:
            self._step_progress = None
            raise
        progress = self._step_progress
        if prepared.outcome != "candidate" or progress is None:
            # A skip ends the step here: neither settle_step nor abort_step follows it.
            self._step_progress = None
        else:
            self._step_progress = replace(progress, phase="evaluating")
        return prepared

    def _prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        samples = trajectories(batch)
        if self._model_resolver is not None:
            self._models = self._model_resolver.resolve()
            self._binding_nodes = self._models.served.compose_nodes(self._descriptor)
        steps = int(state.get("steps", 0)) + 1
        entries = state.get("entries")
        if entries is not None:
            self._loader.root.update([dict(options) for options in entries])
            # Recovered state meets the same admission check as a seed (#476).
            # A workdir written before the admission check may hold entries the plugins
            # now refuse, and stepping on would republish them into this
            # step's commit record, snapshot metadata, and artifact tree.
            # The raise lands before any render or commit, so the failure
            # writes no record at all.
            for options in entries:
                error = self._load_error(str(options.get("id")))
                if error is not None:
                    message = f"recovered state entry {options.get('id')!r} rejected: {error}"
                    if "inline credential" in error:
                        message += (
                            "; this state predates the credential admission check and the existing "
                            "commit log and snapshot metadata already hold the credential: rotate "
                            "the credential, then edit the entry before resuming"
                        )
                    raise ValueError(message)

        # The previous step's manifest threads through every state this step
        # can return; the key is written only when present, so its absence
        # stays "unknown", never an empty manifest (the consumed_ids rule).
        previous_manifest = state.get("failure_manifest")
        carried: dict[str, Any] = {} if previous_manifest is None else {"failure_manifest": previous_manifest}
        # Carried through skips so a budget or streak skip keeps the grown suite.
        promoted = list(state.get("promoted_tasks", ()))
        if promoted:
            carried["promoted_tasks"] = promoted
        promoted_clients = dict(state.get("promoted_clients", {}))
        if promoted_clients:
            carried["promoted_clients"] = promoted_clients
        # Carried through skips so a no-commit step keeps the rollback target.
        rollback_entries = state.get("rollback_entries")
        rollback_evaluation_context = state.get("rollback_evaluation_context", state.get("rollback_gated_against"))
        if rollback_entries is not None:
            carried["rollback_entries"] = rollback_entries
            carried["rollback_evaluation_context"] = rollback_evaluation_context
        rejected = list(state.get("rejected_proposals", ()))
        if rejected:
            carried["rejected_proposals"] = rejected

        metrics: dict[str, Any] = {"steps": steps, "traces": len(samples)}
        # The budgets stop a runaway automatic loop; a skip consumes its batch, so an instruction runs instead.
        if batch.request is None and self._max_steps and steps > self._max_steps:
            return PreparedStep.skipped(
                state={"steps": steps, "entries": self._entries(), **carried},
                metrics={**metrics, "skipped": f"step budget of {self._max_steps} exhausted"},
            )
        streak = int(state.get("failure_streak", 0))
        if batch.request is None and self._max_failure_streak and streak >= self._max_failure_streak:
            return PreparedStep.skipped(
                state={"steps": steps, "entries": self._entries(), **carried},
                metrics={
                    **metrics,
                    "skipped": f"failure streak breaker open after {streak} consecutive rejections",
                },
            )
        manifest = None if previous_manifest is None else FailureManifest.from_state(previous_manifest)
        # Failing traces become permanent evaluation tasks; off by default. The method picks which, Reef screens them.
        # An instruction step consumes the failures it carries, so it promotes them too or they are lost.
        if self._promote_failures:
            if self._promote_task is None:
                candidates: Sequence[str] = _default_promote(samples)
            elif self._promote_accepts_manifest:
                candidates = self._promote_task(samples, manifest=manifest)
            else:
                candidates = self._promote_task(samples)
            clients = {prompt: _client_of(sample) for sample in samples if (prompt := _prompt_of(sample))}
            promoted = _admit_promoted(
                promoted,
                candidates,
                frozenset(self._tasks),
                self._max_promoted_tasks,
                per_client=self._max_promoted_per_client,
                clients=clients,
                counts=promoted_clients,
            )
            if promoted:
                carried["promoted_tasks"] = promoted
            if promoted_clients:
                carried["promoted_clients"] = promoted_clients
            metrics["screened_tasks"] = sum(
                1 for prompt in candidates if isinstance(prompt, str) and _screened(prompt)
            )
        evaluation_tasks = (*self._tasks, *(task for task in promoted if task not in frozenset(self._tasks)))
        if self._promote_failures:
            metrics["evaluation_task_count"] = len(evaluation_tasks)
            metrics["promoted_tasks"] = len(evaluation_tasks) - len(self._tasks)
        step_dir = self._claim_step_dir(steps)
        self._current_step_record = step_dir
        if step_dir is not None:
            metrics["step_record"] = str(step_dir)
        # From here the step is under way for the request page; the proposer runs next.
        self._step_calls = None
        self._step_progress = StepProgress(
            request_id=None if batch.request is None else batch.request.id,
            phase="proposing",
            started_at=time.time(),
            step_record=None if step_dir is None else str(step_dir),
        )
        # Re-evaluate the last-good tree against the published one on cadence or when the served model changed.
        drifted = rollback_evaluation_context is not None and rollback_evaluation_context != self.evaluation_context()
        due = bool(self._recheck_every) and steps % self._recheck_every == 0
        if batch.request is None and self._recheck_every and rollback_entries is not None and (due or drifted):
            target = [dict(entry) for entry in rollback_entries]
            published = [dict(entry) for entry in self._entries()]
            metrics["recheck"] = True
            metrics["recheck_reason"] = "drift" if drifted else "cadence"
            # A recheck asks the proposer nothing, so its record holds episodes only.
            metrics["proposer_calls"] = 0
            metrics["proposer_seconds"] = 0.0
            metrics["proposer_input_tokens"] = metrics["proposer_output_tokens"] = 0
            return PreparedStep.with_candidate(
                HarnessCandidate(
                    candidate_id=f"{batch.batch_id}:recheck",
                    candidate_files=render_composition(self._nodes_from(target), self._descriptor),
                    current_files=render_composition(self._nodes_from(published), self._descriptor),
                    candidate_entries=tuple(target),
                    current_entries=tuple(published),
                    mutations=(),
                    evaluation_tasks=evaluation_tasks,
                    recheck=True,
                    record_dir=step_dir,
                ),
                state={"steps": steps, **carried},
                metrics=metrics,
            )
        snapshot = tuple(dict(entry) for entry in self._entries())
        skipped_state = {"steps": steps, "entries": [dict(entry) for entry in snapshot], **carried}
        record: list[dict[str, Any]] = []
        # An agent's pending proposal goes first; the method proposes only when none waits.
        inbox = self.proposals
        # A manual request owns this step; an unrelated inbox proposal must
        # not consume its batch without the proposer reading the instruction.
        claimed = None if inbox is None or batch.request is not None else inbox.claim()
        if inbox is not None and claimed is not None:
            metrics["proposal"] = {
                "id": claimed.id,
                "session": claimed.session,
                "release_id": claimed.release_id,
                "reason": claimed.reason,
            }
            # An agent's proposal asks the method nothing, so the step's proposer record is empty.
            self._write_record(step_dir, RECORD_PROPOSER_FILE, record)
            metrics["proposer_calls"] = 0
            metrics["proposer_seconds"] = 0.0
            metrics["proposer_input_tokens"] = metrics["proposer_output_tokens"] = 0
            try:
                mutations = _proposal_mutations(claimed)
            except MutationError as error:
                inbox.refuse(claimed.id, str(error))
                self._write_record(step_dir, RECORD_MUTATIONS_FILE, [])
                return PreparedStep.skipped(state=skipped_state, metrics={**metrics, "skipped": str(error)})
        else:
            calls = _StepCalls(self._max_model_calls_per_step, record)
            self._step_calls = calls
            models = _budgeted_bindings(self._models, calls)
            extra: dict[str, Any] = {}
            if self._propose.runs_agent:
                # No agent executor configured: the proposer gets None and answers without an agent.
                extra["agent_host"] = self._agent_host(step_dir, calls)
                if extra["agent_host"] is None and batch.request is not None:
                    calls.note("proposer", "the agent proposer is off on this host; the text proposer answers")
            if self._propose_accepts_manifest:
                extra["manifest"] = manifest
            if self._propose_accepts_rejected:
                extra["rejected"] = tuple(rejected)
            if self._propose_accepts_sources:
                extra["sources"] = tuple(_source_of(sample) for sample in samples)
            if self._propose_accepts_entries:
                # The tree as entry options, so a method can name the entry an update or remove targets.
                extra["entries"] = tuple(dict(entry) for entry in snapshot)
            handed: dict[str, Any] | None = None
            if batch.request is not None:
                if not self._propose.reads_requests:
                    raise ValueError("an instruction step requires a proposer that accepts 'requests'")
                # A fresh dict the method may extend: the requires items it adds join the commit's after the screens.
                handed = {"id": batch.request.id, **batch.request.to_dict(), "untrusted": True}
                extra["requests"] = (handed,)
            try:
                proposal = self._propose(self._nodes(), samples, models, **extra)
            finally:
                # Written even when propose raised: the calls before the failure are the decision's record.
                self._write_record(step_dir, RECORD_PROPOSER_FILE, record)
            metrics["proposer_calls"] = len(record)
            metrics["proposer_seconds"] = round(sum(float(entry.get("seconds", 0.0)) for entry in record), 3)
            # Recorded, not charged: the platform meters served traffic, the evolve step only counts its own.
            metrics["proposer_input_tokens"], metrics["proposer_output_tokens"] = _proposer_tokens(record)
            if batch.request is not None and handed is not None:
                requires, refused = _merged_requires(batch.request.requires, handed.get("requires"))
                metrics["training_request"] = {"id": batch.request.id, **batch.request.to_dict(), "requires": requires}
                if refused:
                    metrics["training_request"]["refused_requires"] = refused
            if isinstance(proposal, StepProposal):
                # The notes are the method's own record of the step; the backend writes them and never reads them.
                notes = _bounded(proposal.notes)
                if notes:
                    metrics["proposal_notes"] = notes
                proposal = proposal.mutations
            mutations = (proposal,) if isinstance(proposal, Mutation) else tuple(proposal or ())
        # The parsed proposal lands before admission, so a refused one is on file too, redacted and clipped
        # like the proposer's traffic: the tree boundary has not seen it yet.
        self._write_record(step_dir, RECORD_MUTATIONS_FILE, [_bounded(_mutation_record(m)) for m in mutations])
        if not mutations:
            return PreparedStep.skipped(state=skipped_state, metrics={**metrics, "skipped": "no proposal"})

        current_files = render_composition(self._nodes(), self._descriptor)
        # Admission runs again here even for a proposal the route admitted: the head may have moved since.
        admitted, refusal = admit_mutations(snapshot, mutations, self._descriptor)
        if refusal is not None:
            if inbox is not None and claimed is not None:
                inbox.refuse(claimed.id, refusal)
            return PreparedStep.skipped(state=skipped_state, metrics={**metrics, "skipped": refusal})
        self._loader.root.update([dict(entry) for entry in admitted])
        return PreparedStep.with_candidate(
            HarnessCandidate(
                candidate_id=f"{batch.batch_id}:candidate",
                candidate_files=render_composition(self._nodes(), self._descriptor),
                current_files=current_files,
                candidate_entries=tuple(dict(entry) for entry in self._entries()),
                current_entries=snapshot,
                mutations=mutations,
                evaluation_tasks=evaluation_tasks,
                proposal_id=None if claimed is None else claimed.id,
                record_dir=step_dir,
            ),
            state={"steps": steps, **carried},
            metrics=metrics,
        )

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        """Run the evaluation episodes of the named ``sides`` and return their scores.

        The default runs the candidate and the current tree as pairs. A policy
        that evaluates the candidate alone (a floor) passes ``("candidate",)``: no
        current episode runs, ``current_scores`` is empty and the other
        ``current_*`` keys are absent, and ``evaluation_sides`` records the choice.
        """
        candidate = self._require_harness_candidate(candidate)
        sides = tuple(sides)
        if not sides or any(side not in EVALUATION_SIDES for side in sides):
            raise ValueError(f"sides must name one or both of {EVALUATION_SIDES}, got {sides!r}")
        # Episodes run against the tree plus the model binding. The binding
        # is appended at render time and never enters the candidate's files,
        # so the published artifact carries no endpoint or credential.
        entries = {"candidate": candidate.candidate_entries, "current": candidate.current_entries}
        files = {side: self._render_for_episode(entries[side]) for side in sides}
        # Episodes interleave candidate and current inside each pairing, so
        # anything that drifts during the run (upstream load, rate limits)
        # lands on both sides of a pair instead of one whole side. A repeat
        # is one more pairing of the same task; the selector compares the
        # vectors positionally, so every pairing tallies on its own. Each
        # episode is independent work in its own root, so with more than one
        # worker the pairings run in one pool - a large task set costs one
        # wave instead of a long turn-taking pass - and the results are read
        # back in submission order either way.
        # An older candidate carries no evaluation_tasks and falls back to the seed set.
        evaluation_tasks = candidate.evaluation_tasks or candidate.gate_tasks or self._tasks
        episodes_dir = None if candidate.record_dir is None else candidate.record_dir / RECORD_EPISODES_DIR
        pairings = [
            (files[side], task, None if episodes_dir is None else episodes_dir / _episode_name(side, index, repeat))
            for index, task in enumerate(evaluation_tasks)
            for repeat in range(self._episode_repeats)
            for side in sides
        ]
        progress = self._step_progress
        if progress is not None:
            # The evaluation's size for the page; the pool answers all at once, so no per-episode count is kept.
            self._step_progress = replace(progress, phase="evaluating", episodes_total=len(pairings))
        scored = self._evaluate_pairings(pairings)
        runs = {side: scored[offset :: len(sides)] for offset, side in enumerate(sides)}
        scores = {side: tuple(run.score for run in runs[side]) for side in sides}
        metrics: dict[str, Any] = {
            "candidate_scores": scores.get("candidate", ()),
            "current_scores": scores.get("current", ()),
            "episode_failures": sum(score is None for side in sides for score in scores[side]),
            "episode_repeats": self._episode_repeats,
        }
        for side in sides:
            # Failure observations ride the evaluation so settlement can
            # build the committed side's manifest from the decision alone.
            metrics[f"{side}_failures"] = tuple(run.failure.to_dict() for run in runs[side] if run.failure is not None)
            metrics[f"{side}_residue"] = sum(run.residue for run in runs[side])
            metrics[f"{side}_score"] = float(sum(score for score in scores[side] if score is not None))
            # Per agent sums over the side's episodes, so a result says which agent did the work.
            metrics[f"{side}_agents"] = _sum_agents(run.agents for run in runs[side])
            # Per episode, in pairing order: the root's stage path and how its turn ended.
            metrics[f"{side}_paths"] = tuple(run.path for run in runs[side])
        if sides != EVALUATION_SIDES:
            metrics["evaluation_sides"] = list(sides)
        return EvaluationResult(evaluator="harness_episode_pairs", evaluator_version="1", metrics=metrics)

    def settle_step(
        self,
        prepared: PreparedStep,
        decision: SelectionDecision,
    ) -> TrainStepResult:
        # The result is in: the page reads the committed row from here on, the trainer's reserved batch covering
        # the commit window.
        self._step_progress = None
        candidate = self._candidate_from(prepared)
        metrics = dict(prepared.metrics)

        # Raw score vectors and failure observations stay inside the
        # structured evaluation record. The policy owns any derived
        # comparison metrics, while this backend keeps the stable episode
        # totals at the top level of the commit metrics.
        evaluation_metrics = dict(decision.evaluation.metrics)
        evaluation_metrics.pop("candidate_scores", None)
        evaluation_metrics.pop("current_scores", None)
        candidate_failures = evaluation_metrics.pop("candidate_failures", ())
        current_failures = evaluation_metrics.pop("current_failures", ())

        # The manifest describes the composition this step commits: the
        # candidate when selected, otherwise the retained current tree.
        committed_side = "candidate" if decision.selected else "current"
        observed = candidate_failures if decision.selected else current_failures
        previous_state = prepared.state.get("failure_manifest")
        previous = None if previous_state is None else FailureManifest.from_state(previous_state)
        # A side the evaluation did not run showed nothing, so its manifest carries over untouched, as through a skip.
        manifest = None
        if committed_side in evaluation_metrics.get(
            "evaluation_sides", evaluation_metrics.get("gate_sides", EVALUATION_SIDES)
        ):
            manifest = advance(
                previous,
                int(prepared.state["steps"]),
                tuple(FailureObservation.from_dict(value) for value in observed),
            )
            metrics["failures"] = {
                "new": len(manifest.new),
                "persisting": len(manifest.persisting),
                "fixed": len(manifest.fixed),
            }
        # A recheck is not a proposal, so it leaves the failure streak alone.
        if candidate.recheck:
            streak = int(prepared.state.get("failure_streak", 0))
        else:
            streak = 0 if decision.selected else int(prepared.state.get("failure_streak", 0)) + 1
        metrics["evaluation_context"] = self.evaluation_context()
        # A publish stores the replaced tree and its evaluation stamp as the rollback target; a rollback consumes it.
        rollback_entries = prepared.state.get("rollback_entries")
        rollback_evaluation_context = prepared.state.get(
            "rollback_evaluation_context", prepared.state.get("rollback_gated_against")
        )
        if candidate.recheck and decision.selected:
            metrics["rolled_back"] = True
            rollback_entries = None
            rollback_evaluation_context = None
        elif candidate.recheck:
            metrics["rolled_back"] = False
        elif decision.selected and self._recheck_every:
            rollback_entries = list(candidate.current_entries)
            rollback_evaluation_context = metrics["evaluation_context"]
        state = {**prepared.state, "failure_streak": streak}
        if manifest is not None:
            state["failure_manifest"] = manifest.to_state()
        state.pop("rollback_entries", None)
        state.pop("rollback_evaluation_context", None)
        state.pop("rollback_gated_against", None)
        if rollback_entries is not None:
            state["rollback_entries"] = rollback_entries
            state["rollback_evaluation_context"] = rollback_evaluation_context
        # A real rejection joins a bounded record the proposer can read back, options included.
        if not candidate.recheck and not decision.selected and self._max_rejected_history:
            rejected = list(prepared.state.get("rejected_proposals", ()))
            rejected.append(
                {
                    "step": int(prepared.state["steps"]),
                    "mutations": [_mutation_record(mutation) for mutation in candidate.mutations],
                    "reason": decision.reason,
                }
            )
            state["rejected_proposals"] = rejected[-self._max_rejected_history :]

        metrics.update(
            {
                **evaluation_metrics,
                **decision.metrics,
                "selected": decision.selected,
                # Compatibility for existing catalogs and clients. New code
                # should inspect the structured selection record.
                "published": decision.selected,
                "selection": {"candidate_id": candidate.candidate_id, **decision.to_dict()},
            }
        )
        if len(candidate.mutations) == 1:
            metrics["mutation"] = _mutation_record(candidate.mutations[0])
        else:
            metrics["mutations"] = [_mutation_record(mutation) for mutation in candidate.mutations]

        if candidate.proposal_id is not None and self.proposals is not None:
            selection_result = {"step": int(state["steps"]), "selected": decision.selected, "reason": decision.reason}
            self.proposals.settle(candidate.proposal_id, selection_result)

        if decision.selected:
            entries = [dict(entry) for entry in candidate.candidate_entries]
            # The loader writes live changes into the rows it holds, config included; the returned state shares none.
            self._loader.root.update([copy.deepcopy(entry) for entry in entries])
            # The published tree carries its entries list too, so a resident process can mount it entry by entry.
            published = {**candidate.candidate_files, **tree_files(self._descriptor, entries)}
            artifact = Artifact.local(_write_rendered_files(published))
            step = int(state["steps"])
            replaced = self._rendered_publications.get(step)
            if replaced is not None:
                replaced.discard()
            self._rendered_publications[step] = artifact
            # Under review, or when a reviewed kind is touched, the release waits for a promote; loop code is
            # reviewed whether or not the deployment listed it.
            pending = self._publish == "review" or bool(
                (self._review_kinds | ALWAYS_REVIEWED_KINDS) & self._mutation_kinds(candidate)
            )
            return TrainStepResult({**state, "entries": entries}, metrics, artifact=artifact, pending=pending)

        entries = [dict(entry) for entry in candidate.current_entries]
        self._loader.root.update([copy.deepcopy(entry) for entry in entries])
        return TrainStepResult({**state, "entries": entries}, metrics)

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Discard this backend's render source after its commit is durable."""
        artifact = self._rendered_publications.pop(int(state["steps"]), None)
        if artifact is not None:
            artifact.discard()
        # The tree follows the durable state, whatever settlement or a subclass left in it.
        self._loader.root.update([copy.deepcopy(entry) for entry in state["entries"]])

    def abort_step(self, prepared: PreparedStep) -> None:
        self._step_progress = None
        candidate = self._candidate_from(prepared)
        self._loader.root.update([dict(entry) for entry in candidate.current_entries])
        if candidate.proposal_id is not None and self.proposals is not None:
            # Filed, not left in claimed/ forever: the inbox never returns to a claimed file on its own.
            self.proposals.refuse(candidate.proposal_id, "step aborted before a result")

    @classmethod
    def _candidate_from(cls, prepared: PreparedStep) -> HarnessCandidate:
        candidate = prepared.candidate
        if candidate is None:
            raise TypeError("harness settlement requires a candidate step")
        return cls._require_harness_candidate(candidate)

    @staticmethod
    def _require_harness_candidate(candidate: UpdateCandidate) -> HarnessCandidate:
        if not isinstance(candidate, HarnessCandidate):
            raise TypeError(f"harness evaluation requires HarnessCandidate, got {type(candidate).__name__}")
        return candidate

    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]:
        """Read only this scenario's retained step files."""
        from reef.train.cordis_backend.record_history import read_step_records

        return read_step_records(self._step_record_dir, directory, relative)

    def failed_step_metrics(self) -> Mapping[str, Any]:
        """Keep the failed attempt's exact directory when the trainer consumes its instruction after reload."""
        return {} if self._current_step_record is None else {"step_record": str(self._current_step_record)}

    def _agent_host(self, step_dir: Path | None, calls: ProposerCalls) -> AgentHost | None:
        """What an agent proposer runs in this step, or ``None`` when the deployment configured no agent."""
        if self._agent_executor is None:
            return None
        return AgentHost(
            descriptor=self._descriptor,
            binary=self._binary,
            executor=self._agent_executor,
            step_dir=step_dir,
            calls=calls,
            timeout_s=self._agent_timeout_s,
            trial_timeout_s=self._agent_trial_timeout_s,
        )

    def _claim_step_dir(self, step: int) -> Path | None:
        """Create and return a fresh record directory for ``step``; ``None`` with the record off."""
        if self._step_record_dir is None:
            return None
        # A retried step keeps the earlier attempt on file: its directory is never reused.
        attempt = 1
        step_dir = self._step_record_dir / str(step)
        while True:
            try:
                step_dir.mkdir(parents=True)
            except FileExistsError:
                attempt += 1
                step_dir = self._step_record_dir / f"{step}-{attempt}"
                continue
            return step_dir

    @staticmethod
    def _write_record(step_dir: Path | None, name: str, payload: Any) -> None:
        """One record file of the step as JSON; nothing is written with the record off."""
        if step_dir is None:
            return
        # Exclusive create: a record file on disk is a record and is never replaced.
        with open(step_dir / name, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, default=str) + "\n")

    @staticmethod
    def _agent_work(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
        return _agent_work(trajectory)

    def _evaluate_pairings(self, pairings):
        scored = self._evaluation_pool.evaluate(pairings, models=self._models)
        for pairing, result in zip(pairings, scored, strict=True):
            if result.record_archive is not None:
                keep_dir = pairing[2]
                try:
                    # Never merge into an earlier episode's record. Extraction
                    # rejects path traversal, escaping links and special files.
                    keep_dir.mkdir(parents=True)
                    with tarfile.open(fileobj=BytesIO(result.record_archive), mode="r:gz") as archive:
                        archive.extractall(keep_dir, filter="data")
                except (OSError, tarfile.TarError) as exc:
                    raise TrajectoryKeepError(
                        f"cannot keep remote episode trajectory under {keep_dir}: {exc}"
                    ) from exc
        return scored

    @staticmethod
    def _mutation_kinds(candidate: HarnessCandidate) -> frozenset[str]:
        """Node kinds the mutations touch; update and remove read the kind off the pre-mutation tree."""
        # The kind after the mutation: an update cannot change it (_apply refuses), and a remove reads it
        # off the tree it left, so the pre-mutation entries answer for every op that is not a create.
        by_id = {str(entry.get("id")): str(entry.get("name")) for entry in candidate.current_entries}
        kinds: set[str] = set()
        for mutation in candidate.mutations:
            name = (mutation.options or {}).get("name") if mutation.op == "create" else by_id.get(mutation.id)
            if name:
                kinds.add(str(name))
        return frozenset(kinds)

    def evaluation_context(self) -> dict[str, Any]:
        """The served model and adapter version this step's evaluation runs against."""
        return {
            "model": self._models.served.model,
            "adapter": self._descriptor.name,
            "adapter_version": self._descriptor.install.version if self._descriptor.install else None,
        }

    def _render_for_episode(self, entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        """The tree plus the model binding, and the entries list beside them where the adapter carries one."""
        files = render_composition((*_nodes_from(entries), *self._binding_nodes), self._descriptor)
        return {**files, **tree_files(self._descriptor, entries)}

    def _nodes(self) -> tuple[tuple[str, Any], ...]:
        """The enabled composition in tree order, as (kind, config) pairs."""
        return _nodes_from(self._entries())

    @staticmethod
    def _nodes_from(entries: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
        return _nodes_from(entries)

    def _entries(self) -> list[EntryOptions]:
        """Live entry options in tree order."""
        return _loader_entries(self._loader)

    def _load_error(self, id_: str) -> str | None:
        return _load_error(self._loader, id_, self._descriptor)


def _proposer_requires(
    base: Sequence[Mapping[str, Any]], handed: object
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The items a proposer added to its request mapping, each under the shape and text screens admission runs.

    An item is the proposer's when its name is not among the person's
    ``base`` items, wherever the proposer put it; one that is malformed, or
    whose name, check or prompt is credential or directive shaped, is
    dropped alone and named once in the log, the rest stand and the
    mutations stand.
    Returns the kept items and the refused ones, each refused as the bounded
    ``item`` with the ``reason`` it was dropped, so the step can record them."""
    log = logging.getLogger(__name__)
    added: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    if handed is None:
        return added, refused
    if not isinstance(handed, Sequence) or isinstance(handed, (str, bytes)):
        log.warning("propose: the requires it added are dropped: not a list")
        refused.append({"item": _bounded(handed), "reason": "not a list"})
        return added, refused
    names = {str(item.get("name")) for item in base}
    for item in handed:
        # The person's items passed admission and the person's copy wins: an edit of one is not the proposer's.
        if isinstance(item, Mapping) and str(item.get("name")) in names:
            continue
        try:
            (parsed,) = parse_requires([item])
        except ValueError as error:
            log.warning("propose: a requires item it added is dropped: %s", error)
            refused.append({"item": _bounded(item), "reason": str(error)})
            continue
        texts = (parsed["name"], str(parsed.get("check") or ""), str(parsed.get("prompt") or ""))
        if any(secret_shaped(text) for text in texts):
            log.warning("propose: a requires item it added carries a credential shaped literal; dropped")
            refused.append({"item": _bounded(item), "reason": "carries a credential shaped literal"})
            continue
        if any(directive_shaped(text) for text in texts):
            log.warning("propose: a requires item it added carries an instruction override phrasing; dropped")
            refused.append({"item": _bounded(item), "reason": "carries an instruction override phrasing"})
            continue
        added.append(parsed)
    return added, refused


def _merged_requires(
    base: Sequence[Mapping[str, Any]], handed: object
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The person's items, then what the proposer added by name, capped at ``MAX_REQUIRES`` naming the dropped.

    Returned with what :func:`_proposer_requires` refused, so the commit
    records the items the person will not see in the list."""
    added, refused = _proposer_requires(base, handed)
    merged = merge_requires(base, added)
    if len(merged) > MAX_REQUIRES:
        logging.getLogger(__name__).warning(
            "propose: requires capped at %d items; dropped: %s",
            MAX_REQUIRES,
            ", ".join(str(item["name"]) for item in merged[MAX_REQUIRES:]),
        )
        merged = merged[:MAX_REQUIRES]
    return merged, refused


def _proposal_mutations(proposal: Proposal) -> tuple[Mutation, ...]:
    """The mutations an inbox proposal carries, in the shape the backend applies; a bad shape is a MutationError."""
    mutations = []
    for record in proposal.mutations:
        op, id_ = record.get("op"), record.get("id")
        if not isinstance(op, str) or not isinstance(id_, str):
            raise MutationError(f"proposal {proposal.id} carries a mutation without a string op and id")
        options = record.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise MutationError(f"proposal {proposal.id} mutation {op} {id_!r} options must be an object")
        mutations.append(Mutation(op, id_, None if options is None else dict(options)))
    if not mutations:
        raise MutationError(f"proposal {proposal.id} carries no mutations")
    return tuple(mutations)


@dataclass(frozen=True)
class _ScoredEpisode:
    """One evaluation episode as the evaluation keeps it: the score, or why it has none, and what the trajectory showed."""

    score: float | None
    failure: FailureObservation | None
    residue: int = 0
    agents: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The root's stage path and end reason; ``None`` when no trajectory was read.
    path: dict[str, Any] | None = None
    #: Remote workers return the kept trajectory; the driver owns its durable path.
    record_archive: bytes | None = field(default=None, repr=False)


def _write_episode_record(
    keep_dir: Path | None, task: str, result: EpisodeResult | None, scored: _ScoredEpisode
) -> None:
    """``episode.json`` beside the kept trajectory: what the scorer saw, so a result can be re-derived from the record."""
    if keep_dir is None:
        return
    record = {
        "task": _clip(task),
        "score": scored.score,
        "failure": None if scored.failure is None else scored.failure.to_dict(),
        "path": scored.path,
        "exit_code": None if result is None else result.exit_code,
        "stdout": None if result is None else _clip(result.stdout),
        "stderr": None if result is None else _clip(result.stderr),
        "residue": None if result is None else [_clip(str(path)) for path in result.residue],
    }
    # An episode that never wrote an event has no trajectory copy, so the directory may not exist yet.
    keep_dir.mkdir(parents=True, exist_ok=True)
    with open(keep_dir / RECORD_EPISODE_FILE, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, default=str) + "\n")


def _root_header(trajectory: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The root's ``session`` line of a native-jsonl trajectory (the agents' files sort first); empty for other formats."""
    for event in trajectory:
        data = event.get("data") or {}
        if event.get("type") == "session" and str(data.get("agent") or "root") == "root":
            return data
    return {}


def _stage_path(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The root's ``stage/exit`` stage names in order and its ``turn/end`` reason kind; empty for other formats."""
    stages: list[str] = []
    reason: str | None = None
    errors: dict[str, Mapping[str, Any]] = {}
    agent: str | None = None
    for event in trajectory:
        type_, data = event.get("type"), event.get("data") or {}
        if type_ == "session":
            agent = str(data.get("agent") or "root")
        elif type_ == "turn/end" and agent is not None:
            details = data.get("reason") or {}
            kind = details.get("kind")
            if kind == "error" and isinstance(details.get("error"), Mapping):
                errors[agent] = details["error"]
            if agent == "root":
                reason = None if kind is None else str(kind)
        elif agent != "root":
            continue
        elif type_ == "stage/exit":
            stages.append(str(data.get("stage")))
    path: dict[str, Any] = {"stages": stages, "reason": reason}
    # The root's own error, else the error of an agent whose failure ended the run before the root wrote its end:
    # a subagent's model error aborts the whole run with no root turn/end.
    errored = "root" if "root" in errors else next((name for name in errors if reason is None), None)
    if errored is not None:
        path["error"] = dict(errors[errored])
        if errored != "root":
            path["errored_agent"] = errored
    return path


#: The counters a result carries per agent; the token pair is what the endpoint reported, zero when it reported none.
AGENT_COUNTERS = ("turns", "steps", "tool_calls", "tool_errors", "input_tokens", "output_tokens")


def _agent_work(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Turns, steps, tool calls, tool errors and reported tokens per agent of a
    native-jsonl trajectory; empty for other formats."""
    work: dict[str, dict[str, int]] = {}
    agent: str | None = None
    for event in trajectory:
        type_, data = event.get("type"), event.get("data") or {}
        if type_ == "session":
            agent = str(data.get("agent") or "root")
            work.setdefault(agent, dict.fromkeys(AGENT_COUNTERS, 0))["turns"] += 1
        elif agent is None:
            continue
        elif type_ in ("assistant/message", "context/compacted") and isinstance(data.get("usage"), Mapping):
            work[agent]["input_tokens"] += int(data["usage"].get("input_tokens", 0) or 0)
            work[agent]["output_tokens"] += int(data["usage"].get("output_tokens", 0) or 0)
        elif type_ == "step/start":
            work[agent]["steps"] += 1
        elif type_ == "tool/call":
            work[agent]["tool_calls"] += 1
        elif type_ == "tool/result" and data.get("is_error"):
            work[agent]["tool_errors"] += 1
    return work


def _sum_agents(runs: Any) -> dict[str, dict[str, int]]:
    total: dict[str, dict[str, int]] = {}
    for work in runs:
        for agent, counts in work.items():
            sums = total.setdefault(agent, dict.fromkeys(AGENT_COUNTERS, 0))
            for key, value in counts.items():
                sums[key] = sums.get(key, 0) + value
    return {agent: total[agent] for agent in sorted(total)}


def _write_rendered_files(files: Mapping[str, str]) -> Path:
    """Write a rendered file mapping to a temporary directory."""
    directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return directory
