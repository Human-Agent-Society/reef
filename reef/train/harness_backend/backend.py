"""The harness backend: one evolution step, from the head tree to a release.

One step runs in the trainer's three phases:

``prepare_step``
    Read the head tree from git through ``ComponentReleases``. Take a
    client's patch from the inbox, or ask the method's ``Proposer`` for one,
    retrying once with the validation problems. Validate the patch, and make a
    ``HarnessCandidate`` of ``head.apply(patch)`` against ``head``.
``evaluate``
    Run the evaluation tasks on both trees with ``HarnessRolloutEvaluator``.
``settle_step``
    Fold the result into ``HarnessState``. A selected candidate becomes an
    ``Artifact.local`` of its source tree, which the scenario committer
    publishes as a git commit. The scenario's activation rule decides whether
    it serves, from the tree kinds it changes.

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
from reef.core.evaluation import EvaluationResult, RolloutScores, SelectionDecision, UpdateCandidate
from reef.core.job_data import TrajectorySample
from reef.core.requirements import MAX_REQUIRES, merge_requires
from reef.core.training_request import TrainingRequest
from reef.evaluation.harness import EVALUATION_SIDES, FailureObservation, HarnessCandidate, HarnessRolloutEvaluator
from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.rollouts.model_binding import ModelBindings, ModelBindingsResolver
from reef.harness.tree import HarnessTree, TreePatch, ValidationResult, locate, validate_patch, validate_tree
from reef.harness.tree.checks import directive_shaped, secret_shaped
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
from reef.train.harness_backend.trials import RolloutTrials
from reef.train.types import JobData, TrainStepResult, trajectories

#: How many proposals a step asks the proposer for: the first, and one retry with the validation problems.
PROPOSAL_ATTEMPTS = 2


@dataclass(frozen=True)
class StepLimits:
    """The deployment's limits for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` caps one step's proposer calls;
    ``recheck_every`` re-evaluates the latest publish against the tree it
    replaced every that many steps; ``max_rejected_history`` is how many
    rejected proposals the state keeps.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution:``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    recheck_every: int = 0
    max_rejected_history: int = 25

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


@dataclass(frozen=True)
class AgentExecution:
    """Where an agent proposer runs.

    Values: ``executor`` is the one the recipe built for the agent,
    ``binary`` the installed harness binary it runs, ``timeout_s`` > 0 the
    seconds of the whole agent run, and ``trial_timeout_s`` > 0 the seconds
    of each trial.
    Purpose: the recipe builds it from ``evolution.proposer_agent``; the
    backend makes an ``AgentHost`` from it for each step.
    """

    executor: ProgramExecutor
    binary: str
    timeout_s: float
    trial_timeout_s: float

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.trial_timeout_s <= 0:
            raise ValueError("AgentExecution timeouts must be positive")


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
        evaluator: HarnessRolloutEvaluator,
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

    def started_from(self, state: Mapping[str, Any], artifact: Artifact) -> Mapping[str, Any]:
        """``state`` without its recheck target, once ``artifact``'s tree passes validation.

        The next step builds on the head that ``ComponentReleases`` reads, which
        the ``"train_from"`` record moves to ``artifact``. The recheck target
        belongs to the old start, so it is dropped. ``ValueError`` when
        ``artifact`` has no local files or its tree does not pass validation.
        """
        if artifact.local_path is None:
            raise ValueError(f"release {artifact.ref.release_id!r} has no local files to start from")
        validation = validate_tree(HarnessTree.read(artifact.local_path), self.descriptor)
        if not validation.passed:
            raise ValueError(f"the tree of release {artifact.ref.release_id!r} is refused: {validation.message()}")
        return replace(HarnessState.from_json(state), recheck_release_id=None, recheck_context=None).to_json()

    def close(self) -> None:
        self.evaluator.close()

    def prepare_step(self, data: JobData, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        self.progress = None
        self.failed_step = {}
        try:
            prepared = self.prepare(data, HarnessState.from_json(state))
        except BaseException:
            if self.progress is not None:
                # The failed row keeps the phase it reached and the record directory the attempt wrote.
                record = {} if self.progress.step_record is None else {"step_record": self.progress.step_record}
                self.failed_step = {"failed_stage": self.progress.phase, **record}
            self.progress = None
            raise
        if prepared.outcome != "candidate" or self.progress is None:
            # A skip ends the step here: neither settle_step nor abort_step follows it.
            self.progress = None
        else:
            self.progress = replace(self.progress, phase="evaluating")
        return prepared

    def prepare(self, data: JobData, previous: HarnessState) -> PreparedStep:
        """The step's candidate, or the skip and its reason."""
        samples = trajectories(data)
        if self.model_resolver is not None:
            self.models = self.model_resolver.resolve()
            self.evaluator.use_models(self.models)
        state = replace(previous, steps=previous.steps + 1)
        metrics: dict[str, Any] = {"steps": state.steps, "traces": len(samples)}
        automatic = data.request is None
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
            request_id=None if data.request is None else data.request.id,
            phase="proposing",
            started_at=time.time(),
            step_record=None if step_dir is None else str(step_dir),
        )
        if automatic and self.recheck_due(state) and state.recheck_release_id is not None:
            # The candidate is the tree the latest publish replaced; it wins only when the head got worse.
            target = self.releases.release(state.recheck_release_id)
            target_tree = HarnessTree(target.files)
            drifted = state.recheck_context != self.evaluation_context()
            metrics.update(
                {
                    "recheck": True,
                    "recheck_reason": "drift" if drifted else "cadence",
                    "recheck_release_id": target.release_id,
                    # A recheck asks the proposer nothing, so its record holds rollouts only.
                    "proposer_calls": 0,
                }
            )
            candidate = HarnessCandidate(
                candidate_id=f"{data.data_id}:recheck",
                candidate_tree=target_tree,
                current_tree=head_tree,
                patch=head_tree.diff(target_tree),
                evaluation_tasks=step_tasks.tasks,
                recheck=True,
                record_dir=step_dir,
                method_state=state.method_state,
            )
            return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)
        claimed = None if self.inbox is None or not automatic else self.inbox.claim()
        if claimed is not None:
            metrics["proposal"] = {
                "id": claimed.id,
                "session": claimed.session,
                "release_id": claimed.release_id,
                "reason": claimed.reason,
            }
            proposal: TreeProposal | None = TreeProposal(claimed.patch)
            validation = validate_patch(head_tree, claimed.patch, self.descriptor)
            if not validation.passed:
                self.inbox_refuse(claimed.id, validation.message())
        else:
            proposal, validation = self.propose(
                data.request, samples, step_tasks.tasks, state, head_tree, step_dir, metrics
            )
        if proposal is None:
            return skipped(state, metrics, "no proposal")
        if proposal.notes:
            metrics["proposal_notes"] = bounded(proposal.notes)
        if proposal.method_state is not None:
            state = replace(state, method_state=proposal.method_state)
        if not proposal.patch.changes:
            return skipped(state, metrics, "the proposal changes nothing")
        self.records.write(step_dir, PATCH_FILE, bounded(proposal.patch.to_json()))
        if not validation.passed:
            return skipped(state, metrics, validation.message())
        if data.request is not None:
            added, refused = screened_requires(data.request.requires, proposal.requires)
            requires = merge_requires(data.request.requires, added)
            if len(requires) > MAX_REQUIRES:
                return skipped(state, metrics, f"proposal requires {len(requires)} items; at most {MAX_REQUIRES}")
            metrics["training_request"] = {"id": data.request.id, **data.request.to_dict(), "requires": requires}
            if refused:
                metrics["training_request"]["refused_requires"] = refused
        candidate = HarnessCandidate(
            candidate_id=f"{data.data_id}:candidate",
            candidate_tree=head_tree.apply(proposal.patch),
            current_tree=head_tree,
            patch=proposal.patch,
            evaluation_tasks=step_tasks.tasks,
            proposal_id=None if claimed is None else claimed.id,
            record_dir=step_dir,
            method_state=state.method_state,
        )
        return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)

    def propose(
        self,
        request: TrainingRequest | None,
        samples: tuple[TrajectorySample, ...],
        evaluation_tasks: tuple[str, ...],
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
            step=state.steps,
            tree=head_tree,
            descriptor=self.descriptor,
            samples=samples,
            sources=tuple(source_of(sample) for sample in samples),
            request=request,
            evaluation_tasks=evaluation_tasks,
            models=budgeted_bindings(self.models, calls),
            manifest=state.manifest,
            rejected=state.rejected,
            method_state=state.method_state,
            last_outcome=state.last_outcome,
            trials=RolloutTrials(self.evaluator, calls, step_dir),
            agent_host=self.agent_host(step_dir, calls) if self.proposer.runs_agent else None,
        )
        proposal: TreeProposal | None = None
        validation = ValidationResult()
        try:
            for _ in range(PROPOSAL_ATTEMPTS):
                proposal = self.proposer.propose(context)
                if proposal is None or not proposal.patch.changes:
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
            total = len(harness_candidate.evaluation_tasks) * self.evaluator.rollout_repeats * len(tuple(sides))
            self.progress = replace(self.progress, phase="evaluating", rollouts_total=total)
        return self.evaluator.evaluate(harness_candidate, sides=sides)

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        state = HarnessState.from_json(prepared.state)
        metrics = dict(prepared.metrics)
        evaluation = dict(decision.evaluation.metrics)
        candidate_scores = rollout_values(evaluation.pop("candidate_scores", []))
        current_scores = rollout_values(evaluation.pop("current_scores", []))
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
        if not candidate.recheck and not decision.selected and self.limits.max_rejected_history:
            rejected = (*rejected, RejectedProposal(state.steps, candidate.patch, decision.reason))
            rejected = rejected[-self.limits.max_rejected_history :]
        # A publish keeps the tree it replaced as the recheck's target; a recheck that selects that tree consumes it.
        recheck_release_id, recheck_context = state.recheck_release_id, state.recheck_context
        if candidate.recheck:
            metrics["rolled_back"] = decision.selected
            if decision.selected:
                recheck_release_id, recheck_context = None, None
        elif decision.selected and self.limits.recheck_every:
            recheck_release_id, recheck_context = str(metrics["head_release_id"]), context
        # A recheck is not a proposal, so it leaves the failure streak alone.
        if candidate.recheck:
            failure_streak = state.failure_streak
        else:
            failure_streak = 0 if decision.selected else state.failure_streak + 1
        state = replace(
            state,
            failure_streak=failure_streak,
            manifest=manifest,
            rejected=rejected,
            recheck_release_id=recheck_release_id,
            recheck_context=recheck_context,
            # A recheck and a client's proposal evaluated nothing the method proposed, so its latest outcome stays.
            last_outcome=(
                state.last_outcome
                if candidate.recheck or candidate.proposal_id is not None
                else StepOutcome(
                    state.steps, candidate.patch, decision.selected, decision.reason, candidate_scores, current_scores
                )
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
        # The scenario's activation rule decides whether the kept tree serves, from the kinds it changes.
        changed_kinds = frozenset(
            str(location.kind) for path in candidate.patch.changes if (location := locate(path)) is not None
        )
        return TrainStepResult(state.to_json(), metrics, artifact=artifact, changed_kinds=changed_kinds)

    def abort_step(self, prepared: PreparedStep) -> None:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        # The candidate reached its evaluation: the failed row keeps what was proposed, the method's notes and the
        # record directory, and says where it stopped.
        self.failed_step = {
            "failed_stage": "evaluating",
            "patch": bounded(candidate.patch.to_json()),
            **{key: prepared.metrics[key] for key in ("proposal_notes", "step_record") if key in prepared.metrics},
        }
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
        if not self.limits.recheck_every or state.recheck_release_id is None:
            return False
        return state.steps % self.limits.recheck_every == 0 or state.recheck_context != self.evaluation_context()

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


def screened_requires(
    base: Sequence[Mapping[str, Any]], added: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The items a proposer added that pass the screens a person's items pass, and the ones they refuse.

    An added item whose name a person's item in ``base`` already has is dropped: the person's copy wins. An item
    whose name, check or prompt carries a credential-shaped literal or an instruction override is refused, each
    refusal as ``{item, reason}``, so the step records it; the others are kept in order.
    """
    names = {str(item.get("name")) for item in base}
    kept: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for item in added:
        if str(item.get("name")) in names:
            continue
        texts = (str(item.get("name") or ""), str(item.get("check") or ""), str(item.get("prompt") or ""))
        if any(secret_shaped(text) for text in texts):
            refused.append({"item": bounded(dict(item)), "reason": "carries a credential shaped literal"})
        elif any(directive_shaped(text) for text in texts):
            refused.append({"item": bounded(dict(item)), "reason": "carries an instruction override phrasing"})
        else:
            kept.append(dict(item))
    return kept, refused


def rollout_values(scores: object) -> tuple[float | None, ...]:
    """The value of each rollout in ``scores``, ``RolloutScores`` JSON or an empty list, in run order; ``None`` for a
    failed rollout."""
    if not scores:
        return ()
    return tuple(score.value for score in RolloutScores.from_json(scores).rollouts())


def skipped(state: HarnessState, metrics: Mapping[str, Any], reason: str) -> PreparedStep:
    """A step that ends without a candidate, keeping ``state`` and saying why."""
    return PreparedStep.skipped(state=state.to_json(), metrics={**metrics, "skipped": reason})


def require_harness_candidate(candidate: UpdateCandidate | None) -> HarnessCandidate:
    if not isinstance(candidate, HarnessCandidate):
        raise TypeError(f"harness evolution requires a HarnessCandidate, got {type(candidate).__name__}")
    return candidate


__all__ = ["AgentExecution", "HarnessBackend", "StepLimits"]
