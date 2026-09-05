"""The harness evolution backend itself: propose, evaluate, select, render.

The package docstring in :mod:`reef.train.cordis_backend` describes the step
this implements. This module holds the backend and the candidate types it
owns, so the package ``__init__`` stays a re-export surface and ``recipe.py``
can import the backend without an import cycle.
"""

from __future__ import annotations

import json
import math
import tarfile
import tempfile
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import EPISODE_OWNER_LEASE, EpisodeExecutor, LocalExecutor, SandboxExecutor
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError
from reef.harness.episodes.vendor_install import install_prefix, resolve_binary
from reef.harness.tree.nodes import NODE_KINDS, directive_shaped, redact_secret_shaped, secret_shaped
from reef.harness.tree.render import RenderError, render_composition
from reef.runtime.executor import Executor, WorkerSpec
from reef.runtime.executor.config import ExecutorSettings
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.cordis_backend.execution import EvaluationWorkerPool, evaluation_selection
from reef.train.cordis_backend.manifest import FailureManifest, FailureObservation
from reef.train.cordis_backend.manifest import FailureRecord as FailureRecord  # re-export: manifest entry type
from reef.train.cordis_backend.manifest import advance
from reef.train.cordis_backend.proposals import Proposal, ProposalInbox
from reef.train.cordis_backend.strategies import (
    EpisodeScorer,
    Mutation,
    MutationError,
    Promoter,
    Proposer,
    accepts_keyword,
    accepts_manifest,
)
from reef.train.evaluation.contracts import CandidateSelector, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.types import TraceBatch, TraceSample, TrainingBatch, TrainStepResult

from .compose import Context, FiberState
from .compose.loader import EntryOptions, Loader


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

    def run(self, files: Mapping[str, str], task: str, keep_dir: Path | None = None) -> _ScoredEpisode:
        if keep_dir is None or not self.transfer_records:
            return self._run_and_score(files, task, keep_dir)
        # Remote workers must not interpret the driver's path as a local path.
        # Keep the trajectory on the worker, then return it with the scored result.
        with tempfile.TemporaryDirectory(prefix="reef-worker-record-") as temporary:
            record_dir = Path(temporary) / "episode"
            scored = self._run_and_score(files, task, record_dir)
            buffer = BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(record_dir, arcname=".")
            return replace(scored, record_archive=buffer.getvalue())

    def _run_and_score(self, files: Mapping[str, str], task: str, keep_dir: Path | None) -> _ScoredEpisode:
        """Score one side's episode; a ``None`` score marks an episode that
        could not run. The observation keeps what the exception handling
        would otherwise discard: the failure's stage and cause. A nonzero
        exit still scores, as before, and is observed alongside the score.
        The residue counts files the episode left outside the cleanup
        whitelist; with ``forbid_residue`` a littering episode scores as one
        that could not run. ``keep_dir`` receives the episode's trajectory
        files before its root is removed; a copy that fails is not an
        episode failure and propagates, so the step aborts instead of
        scoring a verdict shaped by a disk error."""
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
        scored = self._score_result(result, task)
        _write_episode_record(keep_dir, task, result, scored)
        return scored

    def _score_result(self, result: EpisodeResult, task: str) -> _ScoredEpisode:
        """The score and the observations of an episode that ran."""
        residue = len(result.residue)
        agents = _agent_work(result.trajectory)
        path = _stage_path(result.trajectory)
        if residue and self.forbid_residue:
            cause = f"{residue} file(s) outside the cleanup whitelist: {result.residue[0]}"
            return _ScoredEpisode(
                None, FailureObservation(task=task, stage="residue", cause=cause), residue, agents, path
            )
        score = float(self.scorer(task, result))
        if not math.isfinite(score):
            raise ValueError(f"episode scorer returned a non-finite score {score!r} for task {task!r}")
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return _ScoredEpisode(
                score, FailureObservation(task=task, stage="exit", cause=cause), residue, agents, path
            )
        return _ScoredEpisode(score, None, residue, agents, path)


@dataclass(frozen=True, kw_only=True)
class HarnessCandidate(UpdateCandidate):
    """One rendered harness mutation compared with the current composition."""

    candidate_files: Mapping[str, str]
    current_files: Mapping[str, str]
    candidate_entries: tuple[Mapping[str, Any], ...]
    current_entries: tuple[Mapping[str, Any], ...]
    mutations: tuple[Mutation, ...]
    #: Seed tasks, then promoted traffic prompts; the seed set when promotion is off.
    gate_tasks: tuple[str, ...] = ()
    #: The candidate is the rollback target, so selecting it rolls back.
    recheck: bool = False
    #: The inbox proposal these mutations came from, settled with the verdict; None for the method's own.
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
    """The record directory of one gate episode: ``<side>-<task index>``, a repeat adding ``-<repeat>``."""
    return f"{side}-{task_index}" if repeat == 0 else f"{side}-{task_index}-{repeat}"


def _prompt_of(sample: TraceSample) -> str | None:
    """The trace's last user message, the prompt ``evaluate`` scores; ``None`` for a tool-only turn."""
    messages = sample.payload.get("messages") if isinstance(sample.payload, Mapping) else None
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


def _default_promote(samples: Sequence[TraceSample]) -> list[str]:
    """The default policy: every trace's user prompt is a candidate task."""
    prompts = (_prompt_of(sample) for sample in samples)
    return [prompt for prompt in prompts if prompt is not None]


def _client_of(sample: TraceSample) -> str:
    """The client that sent a trace: its x-reef-tag-client, else its session tag, else ``untagged``."""
    metadata = sample.payload.get("metadata") if isinstance(sample.payload, Mapping) else None
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    for key in ("client", "session"):
        value = tags.get(key) if isinstance(tags, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value
    return "untagged"


def _source_of(sample: TraceSample) -> dict[str, Any]:
    """What a proposer may know about where a sample came from; the text itself stays untrusted."""
    return {"record": sample.source_agent_record_id, "client": _client_of(sample), "untrusted": True}


def _screened(prompt: str) -> bool:
    """Whether a trace prompt fails the ledger's tripwires: a credential, or an instruction override."""
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


class _BudgetedBinding(ModelBinding):
    """A ModelBinding that delegates ``chat`` to a wrapped binding under a
    shared per-step call budget and records every call in the step record.

    A subclass so the proposer still receives ModelBinding values, but it
    holds the real binding and forwards to its ``chat`` (never the base
    implementation), so a method's own binding behavior is preserved. The
    shared counter is a mutable one-element list so every binding in the set
    decrements the same budget; a cap of 0 is no budget. The record is the
    step's list: one entry per call with the model, the request, the reply
    or the error, and the seconds it took, every text cut at the record cap.
    """

    _inner: ModelBinding
    _spent: list[int]
    _cap: int
    _record: list[dict[str, Any]]

    def __init__(self, inner: ModelBinding, spent: list[int], cap: int, record: list[dict[str, Any]]) -> None:
        super().__init__(
            base_url=inner.base_url,
            model=inner.model,
            api_key=inner.api_key,
            api=inner.api,
            timeout_s=inner.timeout_s,
        )
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_spent", spent)
        object.__setattr__(self, "_cap", cap)
        object.__setattr__(self, "_record", record)

    def _spend(self) -> None:
        if self._cap and self._spent[0] >= self._cap:
            raise RuntimeError(f"model call budget of {self._cap} per evolve step exhausted")
        self._spent[0] += 1

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str:
        self._spend()
        kwargs: dict[str, Any] = dict(params)
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        entry: dict[str, Any] = {"model": self.model, "messages": _bounded(messages), "params": _bounded(kwargs)}
        started = time.monotonic()
        try:
            reply = self._inner.chat(messages, **kwargs)
        except BaseException as exc:
            # The failed call is the step's decision too: the record keeps it before the error propagates.
            entry["error"] = _clip(f"{type(exc).__name__}: {exc}")
            entry["seconds"] = round(time.monotonic() - started, 3)
            self._record.append(entry)
            raise
        entry["reply"] = _clip(reply) if isinstance(reply, str) else _bounded(reply)
        entry["seconds"] = round(time.monotonic() - started, 3)
        self._record.append(entry)
        return reply

    def complete(self, body: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """A method's raw request goes through the same budget and record as ``chat``: ``body`` in, ``response`` out."""
        self._spend()
        kwargs: dict[str, Any] = {} if timeout_s is None else {"timeout_s": timeout_s}
        entry: dict[str, Any] = {"model": self.model, "body": _bounded(body), "params": _bounded(kwargs)}
        started = time.monotonic()
        try:
            response = self._inner.complete(body, **kwargs)
        except BaseException as exc:
            entry["error"] = _clip(f"{type(exc).__name__}: {exc}")
            entry["seconds"] = round(time.monotonic() - started, 3)
            self._record.append(entry)
            raise
        entry["response"] = _bounded(response)
        entry["seconds"] = round(time.monotonic() - started, 3)
        self._record.append(entry)
        return response


def _budgeted_bindings(models: ModelBindings, cap: int, record: list[dict[str, Any]]) -> ModelBindings:
    """The proposer's view: every ``chat`` shares one per-step budget and lands in ``record``.

    The bindings are wrapped whatever the cap, so the record sees every call;
    with ``cap`` 0 nothing is refused. The counter is per prepare_step call,
    so a cap bounds one step's model bill, never the campaign's.
    """
    spent: list[int] = [0]

    def wrap(binding: ModelBinding) -> ModelBinding:
        return _BudgetedBinding(binding, spent, cap, record)

    return ModelBindings(
        served=wrap(models.served), named={name: wrap(models[name]) for name in models if name != "served"}
    )


def tree_files(descriptor: AdapterDescriptor, entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The entries list as the file ``files.tree`` names, verbatim; empty for an adapter that declares none."""
    if descriptor.tree_path is None:
        return {}
    return {descriptor.tree_path: json.dumps([dict(entry) for entry in entries], indent=2, sort_keys=True) + "\n"}


def _nodes_from(entries: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
    return tuple((str(options["name"]), options.get("config")) for options in entries if not options.get("disabled"))


def _loader_entries(loader: Loader) -> list[EntryOptions]:
    """Live entry options in tree order."""
    result = []
    for options in loader.root.data:
        entry = loader.store.get(str(options.get("id")))
        if entry is not None:
            result.append(dict(entry.options))
    return result


def _unrenderable(descriptor: AdapterDescriptor, kind: str) -> str | None:
    """Why the adapter cannot render an entry of ``kind``; config nodes render through the config targets."""
    if kind == "config" or kind in descriptor.node_paths:
        return None
    return f"adapter {descriptor.name!r} does not render {kind} nodes"


def _load_error(loader: Loader, id_: str, descriptor: AdapterDescriptor) -> str | None:
    """Why the entry cannot serve: its kind's admission, its fiber's state, then whether the adapter renders it."""
    entry = loader.resolve(id_)
    kind = str(entry.options.get("name"))
    if entry.disabled:
        # Disabled is a serving state, not a validation bypass (#476): a
        # disabled entry builds no fiber, but its options persist in the
        # state verbatim, so its kind's admission gate runs directly here.
        plugin = NODE_KINDS.get(kind)
        if plugin is None:
            return f"unknown node kind {kind!r}"
        try:
            plugin(None, entry.options.get("config"))
        except ValueError as error:
            return str(error)
        return _unrenderable(descriptor, kind)
    fiber = entry.fiber
    if fiber is None:
        return f"unknown node kind {kind!r}"
    if fiber.error is not None:
        return str(fiber.error)
    if fiber.state is not FiberState.ACTIVE:
        return f"node fiber is {fiber.state.name}"
    return _unrenderable(descriptor, kind)


def _resolve(loader: Loader, id_: str) -> Any:
    try:
        return loader.resolve(id_)
    except LookupError as exc:
        raise MutationError(str(exc)) from exc


def _apply_mutation(loader: Loader, mutation: Mutation) -> None:
    """One mutation on the loader: create refuses an existing id, update a missing id or a changed kind, remove a missing id."""
    if mutation.op == "create":
        try:
            loader.resolve(mutation.id)
        except LookupError:
            pass
        else:
            raise MutationError(f"entry {mutation.id!r} already exists")
        if mutation.options is None:
            raise MutationError("create mutation must carry options")
        loader.create({**mutation.options, "id": mutation.id})
    elif mutation.op == "update":
        entry = _resolve(loader, mutation.id)
        if mutation.options is None:
            raise MutationError("update mutation must carry options")
        # The kind's plugin is the entry's admission gate and stays bound to the live fiber, so an
        # update that renamed the kind would validate under the old one; a kind change is remove + create.
        kind = mutation.options.get("name")
        if kind is not None and str(kind) != str(entry.options.get("name")):
            raise MutationError(
                f"update {mutation.id!r} cannot change the entry's kind from {entry.options.get('name')!r} "
                f"to {kind!r}; remove the entry and create it under the new kind"
            )
        loader.update(mutation.id, dict(mutation.options))
    else:
        _resolve(loader, mutation.id)
        loader.remove(mutation.id)
        loader.root.data[:] = [options for options in loader.root.data if str(options.get("id")) != mutation.id]


def admit_mutations(
    entries: Sequence[Mapping[str, Any]], mutations: Sequence[Mutation], descriptor: AdapterDescriptor
) -> tuple[list[EntryOptions], str | None]:
    """Apply ``mutations`` over a fresh admission loader and try the render: the new entries and None, or the old entries and the refusal.

    The one admission every proposal meets, the method's in ``prepare_step``
    and an agent's at the proposals route: each mutation under the rules of
    ``_apply_mutation``, a FAILED fiber refused with its error, a kind the
    adapter renders no path for refused naming the kind, and a render error
    refused with its message."""
    previous = [dict(entry) for entry in entries]
    loader = Loader(Context(), NODE_KINDS.get)
    loader.root.update([dict(entry) for entry in entries])
    try:
        for mutation in mutations:
            _apply_mutation(loader, mutation)
            if mutation.op == "remove":
                continue
            error = _load_error(loader, mutation.id, descriptor)
            if error is not None:
                raise MutationError(f"mutation {mutation.op} {mutation.id!r} rejected: {error}")
        admitted = _loader_entries(loader)
        render_composition(_nodes_from(admitted), descriptor)
        refusal = _native_refusal(admitted, descriptor)
        if refusal is not None:
            raise MutationError(refusal)
    except (MutationError, RenderError) as error:
        return previous, str(error)
    return admitted, None


def _native_refusal(entries: Sequence[Mapping[str, Any]], descriptor: AdapterDescriptor) -> str | None:
    """The first config entry a native host would refuse at boot, None when the tree is not native or all pass.

    The boot rules of the config plugin (target ``models`` only, no pinned
    binding field, a positive window) are checked here too, so a tree the
    serve process would roll back never wins a gate."""
    if descriptor.tree_path is None:
        return None
    from reef.harness.runners.native.plugins import check_native_config

    for entry in entries:
        if entry.get("name") != "config" or entry.get("disabled"):
            continue
        try:
            check_native_config(entry.get("config") or {})
        except ValueError as error:
            return f"entry {entry.get('id')!r} rejected: {error}"
    return None


class ScoreComparisonSelector(CandidateSelector):
    """Select a candidate when its wins exceed its losses by more than ``min_win_margin`` (0: plain majority)."""

    def __init__(self, min_win_margin: int = 0) -> None:
        if isinstance(min_win_margin, bool) or not isinstance(min_win_margin, int) or min_win_margin < 0:
            raise ValueError("min_win_margin must be an integer of at least 0")
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


class CordisBackend(TrainingBackend):
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
        # The served binding renders into episodes only. It is resolved once
        # here so an adapter without a matching model_binding refuses boot,
        # not the first step.
        self._binding_nodes = models.served.compose_nodes(descriptor)
        # Checked once at boot: an out-of-tree Proposer subclass whose
        # ``__call__`` predates the manifest keyword is called without it.
        self._propose_accepts_manifest = accepts_manifest(propose.__call__)
        self._propose_accepts_rejected = accepts_keyword(propose.__call__, "rejected")
        self._propose_accepts_sources = accepts_keyword(propose.__call__, "sources")
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
        # scenario commit protocol copies them into repository-owned storage.
        # Track only trees created by this backend so a durable commit can
        # remove its source without touching caller-owned Artifact.local paths.
        self._rendered_publications: dict[int, Artifact] = {}
        # Agent proposals wait here between the route that admitted them and the step that takes them.
        self.proposals = None if proposals_dir is None else ProposalInbox(Path(proposals_dir), max_pending_proposals)
        # Created at boot so an unwritable record path refuses to start, not the first step.
        self._step_record_dir = None if step_record_dir is None else Path(step_record_dir)
        if self._step_record_dir is not None:
            try:
                self._step_record_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"step_record_dir {self._step_record_dir} cannot be created: {exc}") from exc
        self._validate_seed()
        # Evolution needs the binary at boot, just like the model binding.
        # Install failures propagate with the missing tool or vendor error.
        prefix = install_prefix(descriptor)
        self._binary = binary if binary is not None else resolve_binary(descriptor, prefix=prefix)
        if binary is None and descriptor.install is not None and isinstance(self._executor, SandboxExecutor):
            # Bind the whole install: npm launchers are symlinks into packages,
            # and git installs need both their editable source and their venv.
            self._executor = replace(self._executor, base_paths=(*self._executor.base_paths, str(prefix)))
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
                    self._worker_selection.settings.backend not in ("uni", "local"),
                    self._worker_selection.settings.backend not in ("uni", "local", "mp"),
                ),
            ),
        )
        # Direct Python users should close explicitly; GC is a fallback only.
        self._pool_finalizer = weakref.finalize(self, self._evaluation_pool.close)

    def close(self) -> None:
        self._pool_finalizer()

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

    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep:
        if not isinstance(batch, TraceBatch):
            raise TypeError(f"harness evolution requires TraceBatch, got {type(batch).__name__}")
        steps = int(state.get("steps", 0)) + 1
        entries = state.get("entries")
        if entries is not None:
            self._loader.root.update([dict(options) for options in entries])
            # Recovered state meets the same admission gate as a seed (#476).
            # A workdir written before the gate may hold entries the plugins
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
                            "; this state predates the credential admission gate and the existing "
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
        rollback_gated_against = state.get("rollback_gated_against")
        if rollback_entries is not None:
            carried["rollback_entries"] = rollback_entries
            carried["rollback_gated_against"] = rollback_gated_against
        rejected = list(state.get("rejected_proposals", ()))
        if rejected:
            carried["rejected_proposals"] = rejected

        metrics: dict[str, Any] = {"steps": steps, "traces": len(batch.samples)}
        # The budgets are circuit breakers, not schedulers: a skipped step
        # commits its reason and consumes the batch, so a runaway loop stops
        # burning episodes and model calls instead of queueing forever.
        if self._max_steps and steps > self._max_steps:
            return PreparedStep.skipped(
                state={"steps": steps, "entries": self._entries(), **carried},
                metrics={**metrics, "skipped": f"step budget of {self._max_steps} exhausted"},
            )
        streak = int(state.get("failure_streak", 0))
        if self._max_failure_streak and streak >= self._max_failure_streak:
            return PreparedStep.skipped(
                state={"steps": steps, "entries": self._entries(), **carried},
                metrics={
                    **metrics,
                    "skipped": f"failure streak breaker open after {streak} consecutive rejections",
                },
            )
        manifest = None if previous_manifest is None else FailureManifest.from_state(previous_manifest)
        # Failing traces become permanent gate tasks; off by default. The method picks which, Reef screens them.
        if self._promote_failures:
            if self._promote_task is None:
                candidates: Sequence[str] = _default_promote(batch.samples)
            elif self._promote_accepts_manifest:
                candidates = self._promote_task(batch.samples, manifest=manifest)
            else:
                candidates = self._promote_task(batch.samples)
            clients = {prompt: _client_of(sample) for sample in batch.samples if (prompt := _prompt_of(sample))}
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
        gate_tasks = (*self._tasks, *(task for task in promoted if task not in frozenset(self._tasks)))
        if self._promote_failures:
            metrics["gate_tasks"] = len(gate_tasks)
            metrics["promoted_tasks"] = len(gate_tasks) - len(self._tasks)
        step_dir = self._claim_step_dir(steps)
        if step_dir is not None:
            metrics["step_record"] = str(step_dir)
        # Re-gate the last-good tree against the published one on cadence or when the served model changed.
        drifted = rollback_gated_against is not None and rollback_gated_against != self._gated_against()
        due = bool(self._recheck_every) and steps % self._recheck_every == 0
        if self._recheck_every and rollback_entries is not None and (due or drifted):
            target = [dict(entry) for entry in rollback_entries]
            published = [dict(entry) for entry in self._entries()]
            metrics["recheck"] = True
            metrics["recheck_reason"] = "drift" if drifted else "cadence"
            # A recheck asks the proposer nothing, so its record holds episodes only.
            metrics["proposer_calls"] = 0
            metrics["proposer_seconds"] = 0.0
            return PreparedStep.with_candidate(
                HarnessCandidate(
                    candidate_id=f"{batch.batch_id}:recheck",
                    candidate_files=render_composition(self._nodes_from(target), self._descriptor),
                    current_files=render_composition(self._nodes_from(published), self._descriptor),
                    candidate_entries=tuple(target),
                    current_entries=tuple(published),
                    mutations=(),
                    gate_tasks=gate_tasks,
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
        claimed = None if inbox is None else inbox.claim()
        if inbox is not None and claimed is not None:
            metrics["proposal"] = {"id": claimed.id, "session": claimed.session, "release_id": claimed.release_id}
            # An agent's proposal asks the method nothing, so the step's proposer record is empty.
            self._write_record(step_dir, RECORD_PROPOSER_FILE, record)
            metrics["proposer_calls"] = 0
            metrics["proposer_seconds"] = 0.0
            try:
                mutations = _proposal_mutations(claimed)
            except MutationError as error:
                inbox.refuse(claimed.id, str(error))
                self._write_record(step_dir, RECORD_MUTATIONS_FILE, [])
                return PreparedStep.skipped(state=skipped_state, metrics={**metrics, "skipped": str(error)})
        else:
            models = _budgeted_bindings(self._models, self._max_model_calls_per_step, record)
            extra: dict[str, Any] = {}
            if self._propose_accepts_manifest:
                extra["manifest"] = manifest
            if self._propose_accepts_rejected:
                extra["rejected"] = tuple(rejected)
            if self._propose_accepts_sources:
                extra["sources"] = tuple(_source_of(sample) for sample in batch.samples)
            try:
                proposal = self._propose(self._nodes(), batch.samples, models, **extra)
            finally:
                # Written even when propose raised: the calls before the failure are the decision's record.
                self._write_record(step_dir, RECORD_PROPOSER_FILE, record)
            metrics["proposer_calls"] = len(record)
            metrics["proposer_seconds"] = round(sum(float(entry.get("seconds", 0.0)) for entry in record), 3)
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
                gate_tasks=gate_tasks,
                proposal_id=None if claimed is None else claimed.id,
                record_dir=step_dir,
            ),
            state={"steps": steps, **carried},
            metrics=metrics,
        )

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        candidate = self._require_harness_candidate(candidate)
        # Episodes run against the tree plus the model binding. The binding
        # is appended at render time and never enters the candidate's files,
        # so the published artifact carries no endpoint or credential.
        candidate_files = self._render_for_episode(candidate.candidate_entries)
        current_files = self._render_for_episode(candidate.current_entries)
        # Episodes interleave candidate and current inside each pairing, so
        # anything that drifts during the run (upstream load, rate limits)
        # lands on both sides of a pair instead of one whole side. A repeat
        # is one more pairing of the same task; the selector compares the
        # vectors positionally, so every pairing tallies on its own. Each
        # episode is independent work in its own root, so with more than one
        # worker the pairings run in one pool - a large task set costs one
        # wave instead of a long turn-taking pass - and the results are read
        # back in submission order either way.
        # An older candidate carries no gate_tasks and falls back to the seed set.
        gate_tasks = candidate.gate_tasks or self._tasks
        episodes_dir = None if candidate.record_dir is None else candidate.record_dir / RECORD_EPISODES_DIR
        pairings = [
            (files, task, None if episodes_dir is None else episodes_dir / _episode_name(side, index, repeat))
            for index, task in enumerate(gate_tasks)
            for repeat in range(self._episode_repeats)
            for side, files in (("candidate", candidate_files), ("current", current_files))
        ]
        scored = self._evaluate_pairings(pairings)
        candidate_runs = scored[0::2]
        current_runs = scored[1::2]
        candidate_scores = tuple(run.score for run in candidate_runs)
        current_scores = tuple(run.score for run in current_runs)
        return EvaluationResult(
            evaluator="harness_episode_pairs",
            evaluator_version="1",
            metrics={
                "candidate_scores": candidate_scores,
                "current_scores": current_scores,
                # Failure observations ride the evaluation so settlement can
                # build the committed side's manifest from the decision alone.
                "candidate_failures": tuple(
                    run.failure.to_dict() for run in candidate_runs if run.failure is not None
                ),
                "current_failures": tuple(run.failure.to_dict() for run in current_runs if run.failure is not None),
                "episode_failures": sum(score is None for score in candidate_scores + current_scores),
                "episode_repeats": self._episode_repeats,
                "candidate_residue": sum(run.residue for run in candidate_runs),
                "current_residue": sum(run.residue for run in current_runs),
                "candidate_score": float(sum(score for score in candidate_scores if score is not None)),
                "current_score": float(sum(score for score in current_scores if score is not None)),
                # Per agent sums over the side's episodes, so a verdict says which agent did the work.
                "candidate_agents": _sum_agents(run.agents for run in candidate_runs),
                "current_agents": _sum_agents(run.agents for run in current_runs),
                # Per episode, in pairing order: the root's stage path and how its turn ended.
                "candidate_paths": tuple(run.path for run in candidate_runs),
                "current_paths": tuple(run.path for run in current_runs),
            },
        )

    def settle_step(
        self,
        prepared: PreparedStep,
        decision: SelectionDecision,
    ) -> TrainStepResult:
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
        observed = candidate_failures if decision.selected else current_failures
        previous_state = prepared.state.get("failure_manifest")
        previous = None if previous_state is None else FailureManifest.from_state(previous_state)
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
        metrics["gated_against"] = self._gated_against()
        # A publish stores the replaced tree and its gate stamp as the rollback target; a rollback consumes it.
        rollback_entries = prepared.state.get("rollback_entries")
        rollback_gated_against = prepared.state.get("rollback_gated_against")
        if candidate.recheck and decision.selected:
            metrics["rolled_back"] = True
            rollback_entries = None
            rollback_gated_against = None
        elif candidate.recheck:
            metrics["rolled_back"] = False
        elif decision.selected and self._recheck_every:
            rollback_entries = list(candidate.current_entries)
            rollback_gated_against = metrics["gated_against"]
        state = {**prepared.state, "failure_manifest": manifest.to_state(), "failure_streak": streak}
        state.pop("rollback_entries", None)
        state.pop("rollback_gated_against", None)
        if rollback_entries is not None:
            state["rollback_entries"] = rollback_entries
            state["rollback_gated_against"] = rollback_gated_against
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
            verdict = {"step": int(state["steps"]), "selected": decision.selected, "reason": decision.reason}
            self.proposals.settle(candidate.proposal_id, verdict)

        if decision.selected:
            entries = [dict(entry) for entry in candidate.candidate_entries]
            self._loader.root.update(entries)
            # The published tree carries its entries list too, so a resident process can mount it entry by entry.
            published = {**candidate.candidate_files, **tree_files(self._descriptor, entries)}
            artifact = Artifact.local(_write_rendered_files(published))
            step = int(state["steps"])
            replaced = self._rendered_publications.get(step)
            if replaced is not None:
                replaced.discard()
            self._rendered_publications[step] = artifact
            # Under review, or when a reviewed kind is touched, the release waits for a promote.
            pending = self._publish == "review" or bool(self._review_kinds & self._mutation_kinds(candidate))
            return TrainStepResult({**state, "entries": entries}, metrics, artifact=artifact, pending=pending)

        entries = [dict(entry) for entry in candidate.current_entries]
        self._loader.root.update(entries)
        return TrainStepResult({**state, "entries": entries}, metrics)

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Discard this backend's render source after its commit is durable."""
        artifact = self._rendered_publications.pop(int(state["steps"]), None)
        if artifact is not None:
            artifact.discard()

    def abort_step(self, prepared: PreparedStep) -> None:
        candidate = self._candidate_from(prepared)
        self._loader.root.update([dict(entry) for entry in candidate.current_entries])
        if candidate.proposal_id is not None and self.proposals is not None:
            # Filed, not left in claimed/ forever: the inbox never returns to a claimed file on its own.
            self.proposals.refuse(candidate.proposal_id, "step aborted before a verdict")

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
        scored = self._evaluation_pool.evaluate(pairings)
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

    def _gated_against(self) -> dict[str, Any]:
        """The served model and adapter version this step's gate runs against."""
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
    """One gate episode as the evaluation keeps it: the score, or why it has none, and what the trajectory showed."""

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
    """``episode.json`` beside the kept trajectory: what the scorer saw, so a verdict can be re-derived from the record."""
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


def _stage_path(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The root's ``stage/exit`` stage names in order and its ``turn/end`` reason kind; empty for other formats."""
    stages: list[str] = []
    reason: str | None = None
    agent: str | None = None
    for event in trajectory:
        type_, data = event.get("type"), event.get("data") or {}
        if type_ == "session":
            agent = str(data.get("agent") or "root")
        elif agent != "root":
            continue
        elif type_ == "stage/exit":
            stages.append(str(data.get("stage")))
        elif type_ == "turn/end":
            kind = (data.get("reason") or {}).get("kind")
            reason = None if kind is None else str(kind)
    return {"stages": stages, "reason": reason}


def _agent_work(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Turns, steps, tool calls and tool errors per agent of a native-jsonl trajectory; empty for other formats."""
    work: dict[str, dict[str, int]] = {}
    agent: str | None = None
    for event in trajectory:
        type_, data = event.get("type"), event.get("data") or {}
        if type_ == "session":
            agent = str(data.get("agent") or "root")
            work.setdefault(agent, {"turns": 0, "steps": 0, "tool_calls": 0, "tool_errors": 0})["turns"] += 1
        elif agent is None:
            continue
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
            sums = total.setdefault(agent, {"turns": 0, "steps": 0, "tool_calls": 0, "tool_errors": 0})
            for key, value in counts.items():
                sums[key] += value
    return {agent: total[agent] for agent in sorted(total)}


def _write_rendered_files(files: Mapping[str, str]) -> Path:
    """Write a rendered file mapping to a temporary directory."""
    directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return directory
