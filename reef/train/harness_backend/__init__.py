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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.harness.descriptor import AdapterDescriptor
from reef.harness.episode import EpisodeError, run_episode
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.harness.nodes import NODE_KINDS
from reef.harness.render import RenderError, render_composition
from reef.harness.trajectory import TrajectoryError
from reef.train.backend import PreparedStep, TrainingBackend
from reef.train.evaluation.contracts import CandidateSelector, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.harness_backend.manifest import FailureManifest, FailureObservation
from reef.train.harness_backend.manifest import FailureRecord as FailureRecord  # re-export: manifest entry type
from reef.train.harness_backend.manifest import advance
from reef.train.harness_backend.strategies import EpisodeScorer, Mutation, MutationError, Proposer, accepts_manifest
from reef.train.types import TraceBatch, TrainingBatch, TrainStepResult

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

    def __post_init__(self) -> None:
        super().__post_init__()


class ScoreComparisonSelector(CandidateSelector):
    """Select a candidate when it wins more task comparisons than it loses."""

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        candidate_scores, current_scores = _score_vectors(evaluation)
        wins, losses = _score_comparison_tally(candidate_scores, current_scores)
        ties = len(candidate_scores) - wins - losses
        selected = wins > losses
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
            metrics={"wins": wins, "losses": losses, "ties": ties},
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


class HarnessEvolveBackend(TrainingBackend):
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
        seed: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        if not tasks:
            raise ValueError("harness evolution requires a non-empty task set")
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
        self._binary = binary
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
        carried = {} if previous_manifest is None else {"failure_manifest": previous_manifest}

        metrics: dict[str, Any] = {"steps": steps, "traces": len(batch.samples)}
        if self._propose_accepts_manifest:
            manifest = None if previous_manifest is None else FailureManifest.from_state(previous_manifest)
            proposal = self._propose(self._nodes(), batch.samples, self._models, manifest=manifest)
        else:
            proposal = self._propose(self._nodes(), batch.samples, self._models)
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
        candidate_runs = tuple(self._run_and_score(candidate_files, task) for task in self._tasks)
        current_runs = tuple(self._run_and_score(current_files, task) for task in self._tasks)
        candidate_scores = tuple(score for score, _ in candidate_runs)
        current_scores = tuple(score for score, _ in current_runs)
        return EvaluationResult(
            evaluator="harness_episode_pairs",
            evaluator_version="1",
            metrics={
                "candidate_scores": candidate_scores,
                "current_scores": current_scores,
                # Failure observations ride the evaluation so settlement can
                # build the committed side's manifest from the decision alone.
                "candidate_failures": tuple(f.to_dict() for _, f in candidate_runs if f is not None),
                "current_failures": tuple(f.to_dict() for _, f in current_runs if f is not None),
                "episode_failures": sum(score is None for score in candidate_scores + current_scores),
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
        state = {**prepared.state, "failure_manifest": manifest.to_state()}

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

    def _run_and_score(self, files: Mapping[str, str], task: str) -> tuple[float | None, FailureObservation | None]:
        """Score one side's episode; a ``None`` score marks an episode that
        could not run. The observation keeps what the exception handling
        would otherwise discard: the failure's stage and cause. A nonzero
        exit still scores, as before, and is observed alongside the score."""
        try:
            result = run_episode(self._descriptor, files, task, binary=self._binary)
        except EpisodeError as error:
            return None, FailureObservation(task=task, stage="launch", cause=str(error))
        except TrajectoryError as error:
            return None, FailureObservation(task=task, stage="trajectory", cause=str(error))
        score = float(self._score_episode(task, result))
        if not math.isfinite(score):
            raise ValueError(f"episode scorer returned a non-finite score {score!r} for task {task!r}")
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return score, FailureObservation(task=task, stage="exit", cause=cause)
        return score, None

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
