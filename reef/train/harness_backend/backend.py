"""The harness backend: one evolution step, from the head tree to a release.

One step runs in the trainer's three phases, and each phase hands its work to
the module that owns it:

``prepare_step``
    Read the head tree from git through ``ComponentReleases`` and the step's
    evaluation tasks. The step's job data is its material: the traces of
    traffic, and on an offline step the train tasks, which the step rolls
    out on the head tree first. Then the method's candidate
    (``evolve.Evolution.propose``) in a fresh ``Workspace``.
``evaluate``
    Score the sides that the publication policy names with
    ``HarnessEpisodeEvaluator`` (``reef.evaluation.harness``), which reuses
    the scores the history holds.
``settle_step``
    Fold the decision into the state and the history, then make the step's
    release by ``reef.train.release.ReleaseRules``, as a weight trainer does:
    a selected tree is the artifact that the scenario committer publishes.

Nothing is restored on a rejection or an abort: the candidate is a new value,
and the head stays the committed release.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.core.job_data import TaskSample, TrajectorySample
from reef.evaluation.harness import (
    EVALUATION_SIDES,
    DecisionRecord,
    EpisodeRecord,
    EvaluationTasks,
    HarnessCandidate,
    HarnessCandidateEvaluator,
    HarnessEpisodeEvaluator,
    History,
)
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings, ModelBindingsResolver
from reef.harness.tree import HarnessTree, TreePatch
from reef.train.backend import CandidateBackend, ComponentReleases, PreparedStep
from reef.train.harness_backend.calls import StepCalls
from reef.train.harness_backend.capabilities import StepProgress, StepProgressReader, StepRecords
from reef.train.harness_backend.evolve import Evolution, skipped
from reef.train.harness_backend.records import StepRecord, bounded
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.workspace import Workspace
from reef.train.release import ReleaseRules
from reef.train.types import JobData, TrainStepResult


@dataclass(frozen=True)
class StepLimits:
    """The deployment's budgets for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` caps one step's model calls. The episodes a
    method runs are capped in ``evolve.Evolution``.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 turns the limit off)")


class HarnessBackend(CandidateBackend, HarnessCandidateEvaluator, StepRecords, StepProgressReader):
    """Evolve one harness component: one candidate per step, evaluated against the head tree.

    Methods never subclass it. A method's behavior arrives in ``evolution``,
    and the scorer and the publication policy arrive in the evaluator and the
    trainer's evaluation plugin. ``evaluator`` is ``None`` when the
    deployment configured no scorer; then no episode runs. ``rules`` make
    each decision a release, and ``review_parts`` name the parts whose change
    waits for a person besides the parts the adapter always reviews.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        releases: ComponentReleases,
        evolution: Evolution,
        evaluator: HarnessEpisodeEvaluator | None,
        tasks: EvaluationTasks,
        history: History,
        models: ModelBindings,
        model_resolver: ModelBindingsResolver | None = None,
        limits: StepLimits,
        rules: ReleaseRules,
        review_parts: frozenset[str] = frozenset(),
        records: StepRecord,
    ) -> None:
        self.descriptor = descriptor
        self.releases = releases
        self.evolution = evolution
        self.evaluator = evaluator
        self.tasks = tasks
        self.history = history
        self.models = models
        self.model_resolver = model_resolver
        self.limits = limits
        self.rules = rules
        self.review_parts = review_parts
        self.records = records
        # The selected tree of the step in flight, until the committer stored it.
        self.pending_artifact: Artifact | None = None
        #: Written by the training thread, read by the request page: each write is one assignment of a frozen value.
        self.progress: StepProgress | None = None
        self.step_calls: StepCalls | None = None
        #: What the step that raised had reached, for the failed request's row.
        self.failed_step: dict[str, Any] = {}

    # -- the candidate lifecycle ------------------------------------------------------------------------------------

    def initial_state(self) -> Mapping[str, Any]:
        return HarnessState().to_json()

    def close(self) -> None:
        if self.evaluator is not None:
            self.evaluator.close()

    def prepare_step(self, data: JobData, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        self.progress = None
        self.failed_step = {}
        try:
            prepared = self.prepare(data, HarnessState.from_json(state), scenario_step + 1)
        except BaseException:
            if self.progress is not None:
                # The failed row keeps the phase it reached and the record directory the attempt wrote.
                self.failed_step = {"failed_stage": self.progress.phase, "step_record": self.progress.step_record}
            self.progress = None
            raise
        if prepared.outcome != "candidate" or self.progress is None:
            # A skip ends the step here: neither settle_step nor abort_step follows it.
            self.progress = None
        else:
            self.progress = replace(self.progress, phase="evaluating")
        return prepared

    def prepare(self, data: JobData, state: HarnessState, step: int) -> PreparedStep:
        """Step ``step``'s candidate, or the skip and its reason; ``step`` is the scenario step it will commit as."""
        # The step's material: the traces of traffic, and on an offline step the train tasks it rolls out.
        samples = tuple(sample for sample in data.samples if isinstance(sample, TrajectorySample))
        rollout_tasks = tuple(str(sample.task_path) for sample in data.samples if isinstance(sample, TaskSample))
        if self.model_resolver is not None:
            self.models = self.model_resolver.resolve()
            if self.evaluator is not None:
                self.evaluator.use_models(self.models)
        metrics: dict[str, Any] = {"traces": len(samples)}
        if rollout_tasks:
            metrics["rollouts"] = len(rollout_tasks)
        automatic = data.request is None
        if automatic and self.limits.max_steps and step > self.limits.max_steps:
            return skipped(state, metrics, f"step budget of {self.limits.max_steps} exhausted")
        if automatic and self.limits.max_failure_streak and state.failure_streak >= self.limits.max_failure_streak:
            return skipped(state, metrics, f"failure streak of {state.failure_streak} rejections")
        head = self.releases.head()
        head_tree = HarnessTree(head.files)
        metrics["head_release_id"] = head.release_id
        self.history.add_candidate(head_tree, step=step, origin="release" if self.history.candidates else "seed")
        step_tasks = self.tasks.for_step(samples, state.collected_tasks, state.collected_clients)
        state = replace(
            state, collected_tasks=step_tasks.collected_tasks, collected_clients=step_tasks.collected_clients
        )
        metrics.update({"evaluation_task_count": len(step_tasks.tasks), "screened_tasks": step_tasks.screened})
        step_dir = self.records.claim(step)
        metrics["step_record"] = str(step_dir)
        calls = StepCalls(self.limits.max_model_calls_per_step, [])
        self.step_calls = calls
        self.progress = StepProgress(
            request_id=None if data.request is None else data.request.id,
            phase="proposing",
            started_at=time.time(),
            step_record=str(step_dir),
        )
        rollouts = self.roll_out(head_tree, rollout_tasks, step) if rollout_tasks else ()
        workspace = Workspace.create(
            step_dir,
            history_dir=self.history.root,
            step=step,
            tree=head_tree,
            descriptor=self.descriptor,
            tasks=step_tasks.tasks,
            repeats=0 if self.evaluator is None else self.evaluator.episode_repeats,
            samples=samples,
            rollouts=rollouts,
            request=data.request,
            state_files=state.method_files,
        )
        return self.evolution.propose(
            workspace,
            head=head_tree,
            evaluation_tasks=step_tasks.tasks,
            data=data,
            state=state,
            evaluator=self.evaluator,
            models=self.models,
            calls=calls,
            metrics=metrics,
        )

    def roll_out(self, head: HarnessTree, tasks: Sequence[str], step: int) -> tuple[EpisodeRecord, ...]:
        """One episode record of ``head`` on each of an offline step's train ``tasks``, in task order.

        The episodes go through the score cache, as an evaluation's do: an episode of the same tree, task and
        evaluation context that the history holds is reused. ``RuntimeError`` without a scorer.
        """
        if self.evaluator is None:
            raise RuntimeError("an offline step rolls out its train tasks, which needs evolution.evaluation.scorer")
        if self.progress is not None:
            self.progress = replace(self.progress, phase="rolling out", episodes_total=len(tasks))
        records = self.evaluator.episodes(head, tasks, repeats=1, step=step)
        if self.progress is not None:
            self.progress = replace(self.progress, phase="proposing", episodes_total=None)
        return records

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        harness_candidate = require_harness_candidate(candidate)
        if self.evaluator is None:
            raise RuntimeError("the deployment configured no evolution.evaluation.scorer, so no episode can run")
        if self.progress is not None:
            total = self.evaluator.episode_count(harness_candidate, len(tuple(sides)))
            self.progress = replace(self.progress, phase="evaluating", episodes_total=total)
        return self.evaluator.evaluate(harness_candidate, sides=sides)

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        """Fold ``decision`` into the state and the history, then make the step's release by ``rules``.

        The head the step prepared against is ``prepared.metrics["head_release_id"]``; no rollback, promote or
        import moves it while a step runs. The commit keeps the evaluation's sums, not its score vectors: the
        history holds every episode.
        """
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        state = HarnessState.from_json(prepared.state)
        state = replace(state, failure_streak=0 if decision.selected else state.failure_streak + 1)
        self.history.add_decision(
            DecisionRecord(
                step=candidate.step,
                tree=candidate.candidate_tree.checksum(),
                head=candidate.current_tree.checksum(),
                selected=decision.selected,
                policy=decision.policy,
                reason=decision.reason,
            )
        )
        evaluation = {
            key: value
            for key, value in decision.evaluation.metrics.items()
            if key not in ("candidate_scores", "current_scores")
        }
        metrics = {
            **prepared.metrics,
            **evaluation,
            **decision.metrics,
            "evaluation_context": None if self.evaluator is None else self.evaluator.context().key(),
            "patch": bounded(candidate.patch.to_json()),
            "patch_parts": self.descriptor.layout.part_names(candidate.patch.changes),
        }
        artifact = None
        if self.rules.released(decision):
            directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
            candidate.candidate_tree.write(directory)
            artifact = Artifact.local(directory)
        result = self.rules.release(
            candidate.candidate_id,
            decision,
            state=state.to_json(),
            metrics=metrics,
            artifact=artifact,
            held=self.reviewed(candidate.patch),
        )
        if result.artifact is not None:
            if self.pending_artifact is not None:
                self.pending_artifact.discard()
            self.pending_artifact = result.artifact
        return result

    def abort_step(self, prepared: PreparedStep) -> None:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        # The candidate reached its evaluation: the failed row keeps what changed, the method's notes and the
        # record directory, and says where it stopped.
        self.failed_step = {
            "failed_stage": "evaluating",
            "patch": bounded(candidate.patch.to_json()),
            "patch_parts": self.descriptor.layout.part_names(candidate.patch.changes),
            **{
                key: prepared.metrics[key]
                for key in ("method_summary", "method_notes", "step_record")
                if key in prepared.metrics
            },
        }

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Discard the step's local source tree once the committer stored it in git."""
        if self.pending_artifact is not None:
            self.pending_artifact.discard()
            self.pending_artifact = None

    def failed_step_metrics(self) -> Mapping[str, Any]:
        return dict(self.failed_step)

    # -- the service capabilities ------------------------------------------------------------------------------------

    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]:
        return self.records.read(directory, relative)

    @property
    def step_progress(self) -> StepProgress | None:
        progress, calls = self.progress, self.step_calls
        if progress is None or calls is None:
            return progress
        return replace(progress, activity=calls.activity())

    # -- helpers -------------------------------------------------------------------------------------------------------

    def reviewed(self, patch: TreePatch) -> bool:
        """Whether ``patch`` changes a part that waits for a person: one the adapter reviews, or one that
        ``review_parts`` names."""
        for path in patch.changes:
            location = self.descriptor.layout.locate(path)
            if location is not None and (location.part.review or location.part.name in self.review_parts):
                return True
        return False


def require_harness_candidate(candidate: UpdateCandidate | None) -> HarnessCandidate:
    if not isinstance(candidate, HarnessCandidate):
        raise TypeError(f"harness evolution requires a HarnessCandidate, got {type(candidate).__name__}")
    return candidate


__all__ = ["HarnessBackend", "StepLimits"]
