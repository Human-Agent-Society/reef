"""Harness evolution backend: candidate selection scored by real episodes.

Per training step, ``propose`` reads the current composition and the batched
traces and returns one ``Mutation`` or a sequence of them; the backend
applies the proposal to the compose Entry tree under one snapshot; the
candidate and current compositions render through the adapter descriptor and
each run one headless episode per task; the method's ``EpisodeScorer`` scores
individual results. The recipe composes this evaluator and its configured
``CandidateSelector`` into one ``DefaultCandidateEvaluationPlugin`` for ``Trainer``.
A sequence is one
composite proposal: it applies
atomically and receives one selection decision, never one per mutation. The
default policy selects a candidate with more task wins than losses.

Versioning goes through reef's native artifact stack: a selected mutation
renders to a directory and returns a ``TrainStepResult`` with the artifact
set, so ``ScenarioCommitProtocol`` stages and publishes it through
``Repository``. The composition tree state travels in the algorithm state
(``"entries"`` key), which the commit log and snapshot metadata persist and
recover.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.harness.descriptor import AdapterDescriptor
from reef.harness.episode import EpisodeError, run_episode
from reef.harness.executor import EpisodeExecutor, LocalExecutor
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.harness.nodes import NODE_KINDS, secret_shaped
from reef.harness.render import RenderError, render_composition
from reef.harness.trajectory import TrajectoryError
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.cordis_backend.manifest import FailureManifest, FailureObservation
from reef.train.cordis_backend.manifest import FailureRecord as FailureRecord  # re-export: manifest entry type
from reef.train.cordis_backend.manifest import advance
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

    def __post_init__(self) -> None:
        super().__post_init__()


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


def _admit_promoted(existing: Sequence[str], candidates: Sequence[str], seed: frozenset[str], cap: int) -> list[str]:
    """Append new candidate prompts under the cap; a secret-shaped or malformed prompt is skipped, never failed."""
    promoted = list(existing)
    seen = set(seed) | set(promoted)
    for prompt in candidates:
        if len(promoted) >= cap:
            break
        if not isinstance(prompt, str) or not prompt.strip() or prompt in seen or secret_shaped(prompt):
            continue
        promoted.append(prompt)
        seen.add(prompt)
    return promoted


class _BudgetedBinding(ModelBinding):
    """A ModelBinding that delegates ``chat`` to a wrapped binding under a
    shared per-step call budget.

    A subclass so the proposer still receives ModelBinding values, but it
    holds the real binding and forwards to its ``chat`` (never the base
    implementation), so a method's own binding behavior is preserved. The
    shared counter is a mutable one-element list so every binding in the set
    decrements the same budget.
    """

    _inner: ModelBinding
    _spent: list[int]
    _cap: int

    def __init__(self, inner: ModelBinding, spent: list[int], cap: int) -> None:
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

    def chat(self, *args: Any, **kwargs: Any) -> str:
        if self._spent[0] >= self._cap:
            raise RuntimeError(f"model call budget of {self._cap} per evolve step exhausted")
        self._spent[0] += 1
        return self._inner.chat(*args, **kwargs)


def _budgeted_bindings(models: ModelBindings, cap: int) -> ModelBindings:
    """The proposer's view: every ``chat`` shares one per-step budget.

    With ``cap`` 0 the bindings pass through unchanged. The counter is per
    prepare_step call, so a cap bounds one step's model bill, never the
    campaign's.
    """
    if not cap:
        return models
    spent: list[int] = [0]

    def wrap(binding: ModelBinding) -> ModelBinding:
        return _BudgetedBinding(binding, spent, cap)

    return ModelBindings(
        served=wrap(models.served), named={name: wrap(models[name]) for name in models if name != "served"}
    )


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
        promote: Promoter | None = None,
        recheck_every: int = 0,
        max_rejected_history: int = 25,
        publish: str = "auto",
        review_kinds: tuple[str, ...] = (),
        seed: tuple[Mapping[str, Any], ...] = (),
        episode_workers: int = 1,
    ) -> None:
        if not tasks:
            raise ValueError("harness evolution requires a non-empty task set")
        if episode_workers < 1:
            raise ValueError("harness evolution requires at least one episode worker")
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
        self._binary = binary
        self._episode_workers = episode_workers
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
        self._promote_failures = promote_failures
        self._max_promoted_tasks = max_promoted_tasks
        self._promote_task = promote
        self._promote_accepts_manifest = promote is not None and accepts_manifest(promote.__call__)
        self._recheck_every = recheck_every
        self._max_rejected_history = max_rejected_history
        if publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")
        self._publish = publish
        self._review_kinds = frozenset(review_kinds)
        self._seed = tuple(dict(entry) for entry in seed)
        self._validate_seed()

    def _validate_seed(self) -> None:
        """Load the seed into the tree once so a bad seed refuses construction.

        This is the same load path ``prepare_step`` runs on the
        state entries, so a node validation failure surfaces here - at
        recipe build time, naming the failing entry - instead of an empty
        or broken composition tying every comparison at run time.
        """
        for options in self._seed:
            for key in ("id", "name"):
                value = options.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"seed entry {options!r} requires a non-empty string {key!r}")
        self._loader.root.update([dict(options) for options in self._seed])
        for options in self._seed:
            error = self._load_error(str(options["id"]))
            if error is not None:
                raise ValueError(f"seed entry {options['id']!r} rejected: {error}")

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
            promoted = _admit_promoted(promoted, candidates, frozenset(self._tasks), self._max_promoted_tasks)
            if promoted:
                carried["promoted_tasks"] = promoted
        gate_tasks = (*self._tasks, *(task for task in promoted if task not in frozenset(self._tasks)))
        if self._promote_failures:
            metrics["gate_tasks"] = len(gate_tasks)
            metrics["promoted_tasks"] = len(gate_tasks) - len(self._tasks)
        # Re-gate the last-good tree against the published one on cadence or when the served model changed.
        drifted = rollback_gated_against is not None and rollback_gated_against != self._gated_against()
        due = bool(self._recheck_every) and steps % self._recheck_every == 0
        if self._recheck_every and rollback_entries is not None and (due or drifted):
            target = [dict(entry) for entry in rollback_entries]
            published = [dict(entry) for entry in self._entries()]
            metrics["recheck"] = True
            metrics["recheck_reason"] = "drift" if drifted else "cadence"
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
                ),
                state={"steps": steps, **carried},
                metrics=metrics,
            )
        models = _budgeted_bindings(self._models, self._max_model_calls_per_step)
        extra: dict[str, Any] = {}
        if self._propose_accepts_manifest:
            extra["manifest"] = manifest
        if self._propose_accepts_rejected:
            extra["rejected"] = tuple(rejected)
        proposal = self._propose(self._nodes(), batch.samples, models, **extra)
        mutations = (proposal,) if isinstance(proposal, Mutation) else tuple(proposal or ())
        if not mutations:
            return PreparedStep.skipped(
                state={"steps": steps, "entries": self._entries(), **carried},
                metrics={**metrics, "skipped": "no proposal"},
            )

        current_files = render_composition(self._nodes(), self._descriptor)
        snapshot = tuple(dict(entry) for entry in self._entries())
        try:
            for mutation in mutations:
                self._apply(mutation)
        except MutationError as error:
            self._loader.root.update([dict(entry) for entry in snapshot])
            return PreparedStep.skipped(
                state={"steps": steps, "entries": [dict(entry) for entry in snapshot], **carried},
                metrics={**metrics, "skipped": str(error)},
            )
        except BaseException:
            self._loader.root.update([dict(entry) for entry in snapshot])
            raise

        try:
            candidate_files = render_composition(self._nodes(), self._descriptor)
        except RenderError as error:
            self._loader.root.update([dict(entry) for entry in snapshot])
            return PreparedStep.skipped(
                state={"steps": steps, "entries": [dict(entry) for entry in snapshot], **carried},
                metrics={**metrics, "skipped": str(error)},
            )
        except BaseException:
            self._loader.root.update([dict(entry) for entry in snapshot])
            raise

        return PreparedStep.with_candidate(
            HarnessCandidate(
                candidate_id=f"{batch.batch_id}:candidate",
                candidate_files=candidate_files,
                current_files=current_files,
                candidate_entries=tuple(dict(entry) for entry in self._entries()),
                current_entries=snapshot,
                mutations=mutations,
                gate_tasks=gate_tasks,
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
        pairings = [
            (files, task)
            for task in gate_tasks
            for _ in range(self._episode_repeats)
            for files in (candidate_files, current_files)
        ]
        if self._episode_workers > 1 and len(pairings) > 1:
            with ThreadPoolExecutor(max_workers=min(self._episode_workers, len(pairings))) as pool:
                scored = list(pool.map(lambda pairing: self._run_and_score(*pairing), pairings))
        else:
            scored = [self._run_and_score(files, task) for files, task in pairings]
        candidate_runs = scored[0::2]
        current_runs = scored[1::2]
        candidate_scores = tuple(score for score, _, _ in candidate_runs)
        current_scores = tuple(score for score, _, _ in current_runs)
        return EvaluationResult(
            evaluator="harness_episode_pairs",
            evaluator_version="1",
            metrics={
                "candidate_scores": candidate_scores,
                "current_scores": current_scores,
                # Failure observations ride the evaluation so settlement can
                # build the committed side's manifest from the decision alone.
                "candidate_failures": tuple(f.to_dict() for _, f, _ in candidate_runs if f is not None),
                "current_failures": tuple(f.to_dict() for _, f, _ in current_runs if f is not None),
                "episode_failures": sum(score is None for score in candidate_scores + current_scores),
                "episode_repeats": self._episode_repeats,
                "candidate_residue": sum(residue for _, _, residue in candidate_runs),
                "current_residue": sum(residue for _, _, residue in current_runs),
                "candidate_score": float(sum(score for score in candidate_scores if score is not None)),
                "current_score": float(sum(score for score in current_scores if score is not None)),
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
        # A real rejection joins a bounded ledger the proposer can read back.
        if not candidate.recheck and not decision.selected and self._max_rejected_history:
            rejected = list(prepared.state.get("rejected_proposals", ()))
            rejected.append(
                {
                    "step": int(prepared.state["steps"]),
                    "mutations": [{"op": mutation.op, "id": mutation.id} for mutation in candidate.mutations],
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
            mutation = candidate.mutations[0]
            metrics["mutation"] = {"op": mutation.op, "id": mutation.id}
        else:
            metrics["mutations"] = [{"op": mutation.op, "id": mutation.id} for mutation in candidate.mutations]

        if decision.selected:
            entries = [dict(entry) for entry in candidate.candidate_entries]
            self._loader.root.update(entries)
            artifact = Artifact.local(_write_rendered_files(candidate.candidate_files))
            # Under review, or when a reviewed kind is touched, the release waits for a promote.
            if self._publish == "review" or self._review_kinds & self._mutation_kinds(candidate):
                metrics["pending"] = True
            return TrainStepResult({**state, "entries": entries}, metrics, artifact=artifact)

        entries = [dict(entry) for entry in candidate.current_entries]
        self._loader.root.update(entries)
        return TrainStepResult({**state, "entries": entries}, metrics)

    def abort_step(self, prepared: PreparedStep) -> None:
        candidate = self._candidate_from(prepared)
        self._loader.root.update([dict(entry) for entry in candidate.current_entries])

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

    def _run_and_score(
        self, files: Mapping[str, str], task: str
    ) -> tuple[float | None, FailureObservation | None, int]:
        """Score one side's episode; a ``None`` score marks an episode that
        could not run. The observation keeps what the exception handling
        would otherwise discard: the failure's stage and cause. A nonzero
        exit still scores, as before, and is observed alongside the score.
        The third element counts files the episode left outside the cleanup
        whitelist; with ``forbid_residue`` a littering episode scores as one
        that could not run."""
        try:
            result = run_episode(
                self._descriptor,
                files,
                task,
                binary=self._binary,
                timeout=self._episode_timeout_s,
                executor=self._executor,
            )
        except EpisodeError as error:
            return None, FailureObservation(task=task, stage="launch", cause=str(error)), 0
        except TrajectoryError as error:
            return None, FailureObservation(task=task, stage="trajectory", cause=str(error)), 0
        residue = len(result.residue)
        if residue and self._forbid_residue:
            cause = f"{residue} file(s) outside the cleanup whitelist: {result.residue[0]}"
            return None, FailureObservation(task=task, stage="residue", cause=cause), residue
        score = float(self._score_episode(task, result))
        if not math.isfinite(score):
            raise ValueError(f"episode scorer returned a non-finite score {score!r} for task {task!r}")
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return score, FailureObservation(task=task, stage="exit", cause=cause), residue
        return score, None, residue

    @staticmethod
    def _mutation_kinds(candidate: HarnessCandidate) -> frozenset[str]:
        """Node kinds the mutations touch; update and remove read the kind off the pre-mutation tree."""
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
        return render_composition((*self._nodes_from(entries), *self._binding_nodes), self._descriptor)

    def _nodes(self) -> tuple[tuple[str, Any], ...]:
        """The enabled composition in tree order, as (kind, config) pairs."""
        return self._nodes_from(self._entries())

    @staticmethod
    def _nodes_from(entries: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (str(options["name"]), options.get("config")) for options in entries if not options.get("disabled")
        )

    def _entries(self) -> list[EntryOptions]:
        """Live entry options in tree order."""
        result = []
        for options in self._loader.root.data:
            entry = self._loader.store.get(str(options.get("id")))
            if entry is not None:
                result.append(dict(entry.options))
        return result

    def _apply(self, mutation: Mutation) -> None:
        if mutation.op == "create":
            if self._exists(mutation.id):
                raise MutationError(f"entry {mutation.id!r} already exists")
            if mutation.options is None:
                raise MutationError("create mutation must carry options")
            self._loader.create({**mutation.options, "id": mutation.id})
        elif mutation.op == "update":
            self._resolve(mutation.id)
            if mutation.options is None:
                raise MutationError("update mutation must carry options")
            self._loader.update(mutation.id, dict(mutation.options))
        else:
            self._resolve(mutation.id)
            self._loader.remove(mutation.id)
            self._loader.root.data[:] = [
                options for options in self._loader.root.data if str(options.get("id")) != mutation.id
            ]

        if mutation.op != "remove":
            error = self._load_error(mutation.id)
            if error is not None:
                raise MutationError(f"mutation {mutation.op} {mutation.id!r} rejected: {error}")

    def _exists(self, id_: str) -> bool:
        try:
            self._loader.resolve(id_)
        except LookupError:
            return False
        return True

    def _resolve(self, id_: str) -> Any:
        try:
            return self._loader.resolve(id_)
        except LookupError as exc:
            raise MutationError(str(exc)) from exc

    def _load_error(self, id_: str) -> str | None:
        entry = self._loader.resolve(id_)
        if entry.disabled:
            # Disabled is a serving state, not a validation bypass (#476): a
            # disabled entry builds no fiber, but its options persist in the
            # state verbatim, so its kind's admission gate runs directly here.
            plugin = NODE_KINDS.get(str(entry.options.get("name")))
            if plugin is None:
                return f"unknown node kind {entry.options.get('name')!r}"
            try:
                plugin(None, entry.options.get("config"))
            except ValueError as error:
                return str(error)
            return None
        fiber = entry.fiber
        if fiber is None:
            return f"unknown node kind {entry.options.get('name')!r}"
        if fiber.error is not None:
            return str(fiber.error)
        if fiber.state is not FiberState.ACTIVE:
            return f"node fiber is {fiber.state.name}"
        return None


def _write_rendered_files(files: Mapping[str, str]) -> Path:
    """Write a rendered file mapping to a temporary directory."""
    directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return directory


# Re-exports of the engine's method shell; the ``as`` form marks them as
# public per PEP 484, which keeps the unused-import hooks off them. Imported
# last: recipe.py imports CordisBackend from this module.


# Re-exports of the engine's method shell, imported last because recipe.py
# imports CordisBackend from this module.
from reef.train.cordis_backend.processor import CordisProcessor  # noqa: F401
from reef.train.cordis_backend.recipe import CordisRecipe  # noqa: F401
