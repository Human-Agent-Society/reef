"""The harness backend: one evolution step, from the head tree to a release.

One step runs in the trainer's three phases:

``prepare_step``
    Read the head tree from git through ``ComponentReleases``. Take a
    client's patch from the inbox, or ask the method's ``Proposer`` for one,
    retrying once with the validation problems. Validate the patch, and make a
    ``HarnessCandidate`` of ``head.apply(patch)`` against ``head``.
``evaluate``
    Run the evaluation tasks on both trees with ``HarnessEpisodeEvaluator``.
``settle_step``
    Fold the result into ``HarnessState``. A selected candidate becomes an
    ``Artifact.local`` of its source tree, which the scenario committer
    publishes as a git commit, held for review when the deployment or the
    changed kinds ask for it.

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
from reef.core.batches import TrajectoryItem
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.core.requirements import MAX_REQUIRES, merge_requires
from reef.core.training_request import TrainingRequest
from reef.evaluation.harness import (
    EVALUATION_SIDES,
    FailureObservation,
    HarnessCandidate,
    HarnessEpisodeEvaluator,
)
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings, ModelBindingsResolver
from reef.harness.tree import HarnessTree, TreeKind, TreePatch, ValidationResult, locate, validate_patch
from reef.harness.tree.layout import ALWAYS_REVIEWED_KINDS
from reef.sandbox.process import EpisodeExecutor
from reef.train.backend import CandidateBackend, ComponentReleases, PreparedStep
from reef.train.harness_backend.calls import StepCalls, budgeted_bindings, proposer_tokens
from reef.train.harness_backend.capabilities import PatchAdmission, StepProgress, StepProgressReader, StepRecords
from reef.train.harness_backend.contract import (
    AgentHost,
    ProposalContext,
    Proposer,
    RejectedProposal,
    StepOutcome,
    TreeProposal,
)
from reef.train.harness_backend.inbox import ProposalInbox
from reef.train.harness_backend.manifest import advance
from reef.train.harness_backend.records import PATCH_FILE, PROPOSER_FILE, StepRecord, bounded
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.tasks import EvaluationTasks, source_of
from reef.train.harness_backend.trials import EpisodeTrials
from reef.train.types import TrainingBatch, TrainStepResult, trajectories

#: How many proposals a step asks the proposer for: the first, and one retry with the validation problems.
PROPOSAL_ATTEMPTS = 2


@dataclass(frozen=True)
class StepLimits:
    """The deployment's limits and publication policy for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` caps one step's proposer calls;
    ``recheck_every`` re-evaluates the latest publish against the tree it
    replaced every that many steps; ``max_rejected_history`` is how many
    rejected proposals the state keeps. ``publish`` is ``"auto"``, which
    serves a selected tree at once, or ``"review"``, which holds every win for
    a person. ``review_kinds`` are the tree kinds whose change waits for a
    person even under ``"auto"``.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution:``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    recheck_every: int = 0
    max_rejected_history: int = 25
    publish: str = "auto"
    review_kinds: frozenset[TreeKind] = frozenset()

    def __post_init__(self) -> None:
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("recheck_every", self.recheck_every),
            ("max_rejected_history", self.max_rejected_history),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 turns the limit off)")
        if self.publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")


@dataclass(frozen=True)
class AgentExecution:
    """Where an agent proposer runs: the executor built for it, the harness binary, and its two timeouts (seconds).

    Purpose: the recipe builds it from ``evolution.proposer_agent``; the
    backend makes an ``AgentHost`` from it for each step.
    """

    executor: EpisodeExecutor
    binary: str
    timeout_s: float
    trial_timeout_s: float


class HarnessBackend(CandidateBackend, PatchAdmission, StepRecords, StepProgressReader):
    """Evolve one harness component: one proposal per step, evaluated against the head tree.

    Methods never subclass it. A method's behavior arrives as ``proposer``
    (and the scorer and selection the recipe built the evaluator and the
    plugin from); a method's state rides ``TreeProposal.method_state``.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        releases: ComponentReleases,
        proposer: Proposer,
        evaluator: HarnessEpisodeEvaluator,
        tasks: EvaluationTasks,
        models: ModelBindings,
        model_resolver: ModelBindingsResolver | None = None,
        limits: StepLimits,
        records: StepRecord,
        inbox: ProposalInbox | None = None,
        agent: AgentExecution | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.releases = releases
        self.proposer = proposer
        self.evaluator = evaluator
        self.tasks = tasks
        self.models = models
        self.model_resolver = model_resolver
        self.limits = limits
        self.inbox = inbox
        self.records = records
        self.agent = agent
        #: The source trees of selected candidates, by step, until their commit is durable.
        self.local_artifacts: dict[int, Artifact] = {}
        #: Written by the training thread, read by the request page: each write is one assignment of a frozen value.
        self.progress: StepProgress | None = None
        self.step_calls: StepCalls | None = None
        #: What the step that raised had reached, for the failed request's row.
        self.failed_step: dict[str, Any] = {}

    # -- the candidate lifecycle ------------------------------------------------------------------------------------

    def initial_state(self) -> Mapping[str, Any]:
        return HarnessState().to_json()

    def close(self) -> None:
        self.evaluator.close()

    def prepare_step(self, batch: TrainingBatch, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        self.progress = None
        self.failed_step = {}
        try:
            prepared = self.prepare(batch, HarnessState.from_json(state))
        except BaseException:
            if self.progress is not None:
                self.failed_step = {"failed_stage": self.progress.phase}
            self.progress = None
            raise
        if prepared.outcome != "candidate" or self.progress is None:
            # A skip ends the step here: neither settle_step nor abort_step follows it.
            self.progress = None
        else:
            self.progress = replace(self.progress, phase="evaluating")
        return prepared

    def prepare(self, batch: TrainingBatch, previous: HarnessState) -> PreparedStep:
        """The step's candidate, or the skip and its reason."""
        samples = trajectories(batch)
        if self.model_resolver is not None:
            self.models = self.model_resolver.resolve()
            self.evaluator.use_models(self.models)
        state = replace(previous, steps=previous.steps + 1)
        metrics: dict[str, Any] = {"steps": state.steps, "traces": len(samples)}
        automatic = batch.request is None
        if automatic and self.limits.max_steps and state.steps > self.limits.max_steps:
            return skipped(state, metrics, f"step budget of {self.limits.max_steps} exhausted")
        if automatic and self.limits.max_failure_streak and state.failure_streak >= self.limits.max_failure_streak:
            return skipped(state, metrics, f"failure streak of {state.failure_streak} rejections")
        head = self.releases.head()
        head_tree = HarnessTree(head.files)
        metrics["head_release_id"] = head.release_id
        step_tasks = self.tasks.for_step(samples, state, state.manifest)
        state = replace(state, promoted_tasks=step_tasks.promoted_tasks, promoted_clients=step_tasks.promoted_clients)
        metrics.update({"evaluation_task_count": len(step_tasks.tasks), "screened_tasks": step_tasks.screened})
        step_dir = self.records.claim(state.steps)
        if step_dir is not None:
            metrics["step_record"] = str(step_dir)
        self.step_calls = StepCalls(self.limits.max_model_calls_per_step, [])
        self.progress = StepProgress(
            request_id=None if batch.request is None else batch.request.id,
            phase="proposing",
            started_at=time.time(),
            step_record=None if step_dir is None else str(step_dir),
        )
        if automatic and self.recheck_due(state):
            raise NotImplementedError("the recheck of a published tree is not ported to the tree backend yet")
        claimed = None if self.inbox is None or not automatic else self.inbox.claim()
        if claimed is not None:
            metrics["proposal"] = {"id": claimed.id, "session": claimed.session, "reason": claimed.reason}
            proposal: TreeProposal | None = TreeProposal(claimed.patch)
            validation = validate_patch(head_tree, claimed.patch, self.descriptor)
            if not validation.passed:
                self.inbox_refuse(claimed.id, validation.message())
        else:
            proposal, validation = self.propose(batch.request, samples, state, head_tree, step_dir, metrics)
        if proposal is None:
            return skipped(state, metrics, "no proposal")
        self.records.write(step_dir, PATCH_FILE, bounded(proposal.patch.to_json()))
        if not validation.passed:
            return skipped(state, metrics, validation.message())
        if proposal.notes:
            metrics["proposal_notes"] = bounded(proposal.notes)
        if batch.request is not None:
            requires = merge_requires(batch.request.requires, proposal.requires)
            if len(requires) > MAX_REQUIRES:
                return skipped(state, metrics, f"proposal requires {len(requires)} items; at most {MAX_REQUIRES}")
            metrics["training_request"] = {"id": batch.request.id, **batch.request.to_dict(), "requires": requires}
        method_state = state.method_state if proposal.method_state is None else proposal.method_state
        state = replace(state, method_state=method_state)
        candidate = HarnessCandidate(
            candidate_id=f"{batch.batch_id}:candidate",
            candidate_tree=head_tree.apply(proposal.patch),
            current_tree=head_tree,
            patch=proposal.patch,
            evaluation_tasks=step_tasks.tasks,
            proposal_id=None if claimed is None else claimed.id,
            record_dir=step_dir,
            method_state=method_state,
        )
        return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)

    def propose(
        self,
        request: TrainingRequest | None,
        samples: tuple[TrajectoryItem, ...],
        state: HarnessState,
        head_tree: HarnessTree,
        step_dir: Path | None,
        metrics: dict[str, Any],
    ) -> tuple[TreeProposal | None, ValidationResult]:
        """Ask the proposer for a patch, and once more with the problems when the first fails validation."""
        if request is not None and not self.proposer.reads_requests:
            raise ValueError("an instruction step requires a proposer whose reads_requests is true")
        calls = self.step_calls if self.step_calls is not None else StepCalls(0, [])
        context = ProposalContext(
            tree=head_tree,
            samples=samples,
            sources=tuple(source_of(sample) for sample in samples),
            request=request,
            models=budgeted_bindings(self.models, calls),
            manifest=state.manifest,
            rejected=state.rejected,
            method_state=state.method_state,
            last_outcome=state.last_outcome,
            trials=EpisodeTrials(self.evaluator, calls, step_dir),
            agent_host=self.agent_host(step_dir, calls) if self.proposer.runs_agent else None,
        )
        proposal: TreeProposal | None = None
        validation = ValidationResult()
        try:
            for _ in range(PROPOSAL_ATTEMPTS):
                proposal = self.proposer.propose(context)
                if proposal is None:
                    break
                validation = validate_patch(head_tree, proposal.patch, self.descriptor)
                if validation.passed:
                    break
                context = replace(context, problems=validation.problems)
        finally:
            # Written even when propose raised: the calls before the failure are the step's record.
            self.records.write(step_dir, PROPOSER_FILE, calls.entries)
            metrics["proposer_calls"] = len(calls.entries)
            metrics["proposer_seconds"] = round(sum(float(entry.get("seconds", 0.0)) for entry in calls.entries), 3)
            metrics["proposer_input_tokens"], metrics["proposer_output_tokens"] = proposer_tokens(calls.entries)
        return proposal, validation

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        harness_candidate = require_harness_candidate(candidate)
        if self.progress is not None:
            total = len(harness_candidate.evaluation_tasks) * self.evaluator.episode_repeats * len(tuple(sides))
            self.progress = replace(self.progress, phase="evaluating", episodes_total=total)
        return self.evaluator.evaluate(harness_candidate, sides=sides)

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        state = HarnessState.from_json(prepared.state)
        metrics = dict(prepared.metrics)
        evaluation = dict(decision.evaluation.metrics)
        candidate_scores = tuple(evaluation.pop("candidate_scores", ()))
        current_scores = tuple(evaluation.pop("current_scores", ()))
        candidate_failures = evaluation.pop("candidate_failures", ())
        current_failures = evaluation.pop("current_failures", ())
        # The manifest describes the tree this step commits: the candidate when selected, else the current tree.
        committed_side = "candidate" if decision.selected else "current"
        manifest = state.manifest
        if committed_side in evaluation.get("evaluation_sides", EVALUATION_SIDES):
            observed = candidate_failures if decision.selected else current_failures
            manifest = advance(state.manifest, state.steps, tuple(FailureObservation.from_dict(v) for v in observed))
            metrics["failures"] = {
                "new": len(manifest.new),
                "persisting": len(manifest.persisting),
                "fixed": len(manifest.fixed),
            }
        context = self.evaluation_context()
        rejected = state.rejected
        if not decision.selected and self.limits.max_rejected_history:
            rejected = (*rejected, RejectedProposal(state.steps, candidate.patch, decision.reason))
            rejected = rejected[-self.limits.max_rejected_history :]
        rollback_release_id, rollback_context = state.rollback_release_id, state.rollback_context
        if decision.selected and self.limits.recheck_every:
            rollback_release_id, rollback_context = str(metrics["head_release_id"]), context
        state = replace(
            state,
            failure_streak=0 if decision.selected else state.failure_streak + 1,
            manifest=manifest,
            rejected=rejected,
            rollback_release_id=rollback_release_id,
            rollback_context=rollback_context,
            last_outcome=StepOutcome(
                state.steps, candidate.patch, decision.selected, decision.reason, candidate_scores, current_scores
            ),
        )
        metrics.update(
            {
                **evaluation,
                **decision.metrics,
                "evaluation_context": context,
                "selected": decision.selected,
                "selection": {"candidate_id": candidate.candidate_id, **decision.to_dict()},
                "patch": bounded(candidate.patch.to_json()),
            }
        )
        if candidate.proposal_id is not None and self.inbox is not None:
            self.inbox.settle(
                candidate.proposal_id, {"step": state.steps, "selected": decision.selected, "reason": decision.reason}
            )
        if not decision.selected:
            return TrainStepResult(state.to_json(), metrics)
        directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
        candidate.candidate_tree.write(directory)
        artifact = Artifact.local(directory)
        self.local_artifacts[state.steps] = artifact
        return TrainStepResult(state.to_json(), metrics, artifact=artifact, pending=self.held_for_review(candidate.patch))

    def abort_step(self, prepared: PreparedStep) -> None:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        self.failed_step = {"failed_stage": "evaluating", "patch": bounded(candidate.patch.to_json())}
        if candidate.proposal_id is not None:
            self.inbox_refuse(candidate.proposal_id, "step aborted before a result")

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Discard the step's local source tree once the committer stored it in git."""
        artifact = self.local_artifacts.pop(int(state["steps"]), None)
        if artifact is not None:
            artifact.discard()

    def failed_step_metrics(self) -> Mapping[str, Any]:
        return dict(self.failed_step)

    # -- the service capabilities ------------------------------------------------------------------------------------

    @property
    def proposals(self) -> ProposalInbox | None:
        return self.inbox

    def admit(self, patch: TreePatch) -> ValidationResult:
        return validate_patch(HarnessTree(self.releases.head().files), patch, self.descriptor)

    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]:
        return self.records.read(directory, relative)

    @property
    def step_progress(self) -> StepProgress | None:
        progress, calls = self.progress, self.step_calls
        if progress is None or calls is None:
            return progress
        return replace(progress, activity=calls.activity())

    # -- helpers -------------------------------------------------------------------------------------------------------

    def evaluation_context(self) -> dict[str, Any]:
        """The served model and adapter version an evaluation runs against; a recheck runs again when they change."""
        return {
            "model": self.models.served.model,
            "adapter": self.descriptor.name,
            "adapter_version": None if self.descriptor.install is None else self.descriptor.install.version,
        }

    def recheck_due(self, state: HarnessState) -> bool:
        """Whether this automatic step re-evaluates the latest publish against the tree it replaced."""
        if not self.limits.recheck_every or state.rollback_release_id is None:
            return False
        return state.steps % self.limits.recheck_every == 0 or state.rollback_context != self.evaluation_context()

    def held_for_review(self, patch: TreePatch) -> bool:
        """Whether a selected patch waits for a person: under ``publish: review``, or when it changes a reviewed kind."""
        if self.limits.publish == "review":
            return True
        reviewed = self.limits.review_kinds | ALWAYS_REVIEWED_KINDS
        return any((location := locate(path)) is not None and location.kind in reviewed for path in patch.changes)

    def agent_host(self, step_dir: Path | None, calls: StepCalls) -> AgentHost | None:
        """The agent run this step offers, or ``None`` when the deployment configured no agent executor."""
        if self.agent is None:
            return None
        return AgentHost(
            descriptor=self.descriptor,
            binary=self.agent.binary,
            executor=self.agent.executor,
            step_dir=step_dir,
            calls=calls,
            timeout_s=self.agent.timeout_s,
            trial_timeout_s=self.agent.trial_timeout_s,
        )

    def inbox_refuse(self, proposal_id: str, reason: str) -> None:
        if self.inbox is not None:
            self.inbox.refuse(proposal_id, reason)


def skipped(state: HarnessState, metrics: Mapping[str, Any], reason: str) -> PreparedStep:
    """A step that ends without a candidate, keeping ``state`` and saying why."""
    return PreparedStep.skipped(state=state.to_json(), metrics={**metrics, "skipped": reason})


def require_harness_candidate(candidate: UpdateCandidate | None) -> HarnessCandidate:
    if not isinstance(candidate, HarnessCandidate):
        raise TypeError(f"harness evolution requires a HarnessCandidate, got {type(candidate).__name__}")
    return candidate


__all__ = ["AgentExecution", "HarnessBackend", "StepLimits"]
