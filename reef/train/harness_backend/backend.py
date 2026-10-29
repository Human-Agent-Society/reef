"""The harness backend: one evolution step, from the head tree to a release.

One step runs in the trainer's three phases:

``prepare_step``
    Read the head tree from git through ``ComponentReleases``. The candidate
    is the tree the latest kept tree replaced (a recheck), a client's patch from
    the inbox, or the tree the method leaves in a fresh step workspace.
    Validate it, and make a ``HarnessCandidate`` of it against the head.
``evaluate``
    Score the sides that the publication policy names with
    ``HarnessRolloutEvaluator``, which reuses the scores the history holds.
``settle_step``
    Record the decision in the history and fold it into ``HarnessState``. A
    selected candidate becomes an ``Artifact.local`` of its source tree, which
    the scenario committer stores as a git commit. The scenario's activation
    rule decides whether it serves, from the tree kinds it changes.

Nothing is restored on a rejection or an abort: the candidate is a new value,
and the head stays the committed release.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.core.requirements import MAX_REQUIRES, merge_requires
from reef.evaluation.harness import (
    EVALUATION_SIDES,
    DecisionRecord,
    EvaluationTasks,
    FailureObservation,
    HarnessCandidate,
    HarnessCandidateEvaluator,
    HarnessRolloutEvaluator,
    History,
)
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.rollouts.model_binding import ModelBindings, ModelBindingsResolver
from reef.harness.tree import HarnessTree, TreePatch, ValidationResult, locate, validate_patch, validate_tree
from reef.harness.tree.checks import clip_record_text, directive_shaped, secret_shaped
from reef.train.backend import CandidateBackend, ComponentReleases, PreparedStep
from reef.train.harness_backend.calls import StepCalls, call_tokens
from reef.train.harness_backend.capabilities import PatchAdmission, StepProgress, StepProgressReader, StepRecords
from reef.train.harness_backend.contract import Method, StepContext, StepResult
from reef.train.harness_backend.inbox import ProposalInbox
from reef.train.harness_backend.manifest import advance
from reef.train.harness_backend.records import CALLS_FILE, RESULT_FILE, StepRecord, bounded
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.tools import StepTools
from reef.train.harness_backend.workspace import Workspace
from reef.train.types import JobData, TrainStepResult, trajectories


@dataclass(frozen=True)
class StepLimits:
    """The deployment's limits for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` and ``max_rollouts_per_step`` cap one step's
    model calls and the rollouts its method runs; ``recheck_every``
    re-evaluates the latest kept tree against the tree it replaced every that
    many steps.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    max_rollouts_per_step: int = 0
    recheck_every: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("max_rollouts_per_step", self.max_rollouts_per_step),
            ("recheck_every", self.recheck_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 turns the limit off)")


class HarnessBackend(CandidateBackend, HarnessCandidateEvaluator, PatchAdmission, StepRecords, StepProgressReader):
    """Evolve one harness component: one candidate per step, evaluated against the head tree.

    Methods never subclass it. A method's behavior arrives as ``method``, and
    the scorer and the publication policy arrive in the evaluator and the
    trainer's evaluation plugin. ``evaluator`` is ``None`` when the
    deployment configured no scorer; then no rollout runs.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        releases: ComponentReleases,
        method: Method,
        evaluator: HarnessRolloutEvaluator | None,
        tasks: EvaluationTasks,
        history: History,
        models: ModelBindings,
        model_resolver: ModelBindingsResolver | None = None,
        limits: StepLimits,
        records: StepRecord,
        inbox: ProposalInbox | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.releases = releases
        self.method = method
        self.evaluator = evaluator
        self.tasks = tasks
        self.history = history
        self.models = models
        self.model_resolver = model_resolver
        self.limits = limits
        self.records = records
        self.inbox = inbox
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
        if self.evaluator is not None:
            self.evaluator.close()

    def prepare_step(self, data: JobData, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        self.progress = None
        self.failed_step = {}
        try:
            prepared = self.prepare(data, HarnessState.from_json(state))
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

    def prepare(self, data: JobData, previous: HarnessState) -> PreparedStep:
        """The step's candidate, or the skip and its reason."""
        samples = trajectories(data)
        if self.model_resolver is not None:
            self.models = self.model_resolver.resolve()
            if self.evaluator is not None:
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
        head_record = self.history.add_candidate(
            head_tree, step=state.steps, origin="release" if self.history.candidates else "seed"
        )
        step_tasks = self.tasks.for_step(samples, state.promoted_tasks, state.promoted_clients)
        state = replace(state, promoted_tasks=step_tasks.promoted_tasks, promoted_clients=step_tasks.promoted_clients)
        metrics.update({"evaluation_task_count": len(step_tasks.tasks), "screened_tasks": step_tasks.screened})
        step_dir = self.records.claim(state.steps)
        metrics["step_record"] = str(step_dir)
        calls = StepCalls(self.limits.max_model_calls_per_step, [])
        self.step_calls = calls
        self.progress = StepProgress(
            request_id=None if data.request is None else data.request.id,
            phase="proposing",
            started_at=time.time(),
            step_record=str(step_dir),
        )
        if automatic and self.recheck_due(state) and state.recheck_release_id is not None:
            # The candidate is the tree the latest kept tree replaced; it wins only when the head got worse.
            target = self.releases.release(state.recheck_release_id)
            target_tree = HarnessTree(target.files)
            metrics.update(
                {
                    "recheck": True,
                    "recheck_reason": "drift" if state.recheck_context != self.context_key() else "cadence",
                    "recheck_release_id": target.release_id,
                }
            )
            validation = validate_tree(target_tree, self.descriptor)
            if not validation.passed:
                # The current checks refuse the replaced tree, so no recheck can bring it back: the step drops it.
                dropped = replace(state, recheck_release_id=None, recheck_context=None)
                return skipped(dropped, metrics, validation.message())
            self.history.add_candidate(target_tree, step=state.steps, origin="release")
            candidate = HarnessCandidate(
                candidate_id=f"{data.data_id}:recheck",
                candidate_tree=target_tree,
                current_tree=head_tree,
                evaluation_tasks=step_tasks.tasks,
                step=state.steps,
                recheck=True,
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
            validation = validate_patch(head_tree, claimed.patch, self.descriptor)
            if not validation.passed:
                self.inbox_refuse(claimed.id, validation.message())
                return skipped(state, metrics, validation.message())
            candidate_tree = head_tree.apply(claimed.patch)
            # No method ran: the client's reason describes the candidate.
            self.history.add_candidate(
                candidate_tree, step=state.steps, origin="proposal", parent=head_record.tree, summary=claimed.reason
            )
        else:
            workspace = Workspace.create(
                step_dir,
                history_dir=self.history.root,
                step=state.steps,
                tree=head_tree,
                descriptor=self.descriptor,
                tasks=step_tasks.tasks,
                repeats=0 if self.evaluator is None else self.evaluator.rollout_repeats,
                samples=samples,
                request=data.request,
                manifest=state.manifest,
                state_files=state.method_files,
            )
            result = self.run_method(workspace, head_record.tree, metrics, calls)
            try:
                state = replace(state, method_files=workspace.state_files())
                candidate_tree = workspace.tree()
            except ValueError as exc:
                return skipped(state, metrics, str(exc))
            if candidate_tree == head_tree:
                return skipped(state, metrics, "the method left the head tree unchanged")
            validation = validate_tree(candidate_tree, self.descriptor)
            if not validation.passed:
                return skipped(state, metrics, validation.message())
            self.history.add_candidate(
                candidate_tree,
                step=state.steps,
                origin="method",
                parent=head_record.tree,
                summary=result.summary,
                notes=result.notes,
            )
            # An instruction step never claims a proposal: its method's requires join the request's.
            if data.request is not None:
                added, refused = screened_requires(data.request.requires, result.requires)
                requires = merge_requires(data.request.requires, added)
                if len(requires) > MAX_REQUIRES:
                    return skipped(state, metrics, f"the step requires {len(requires)} items; at most {MAX_REQUIRES}")
                metrics["training_request"] = {"id": data.request.id, **data.request.to_dict(), "requires": requires}
                if refused:
                    metrics["training_request"]["refused_requires"] = refused
        candidate = HarnessCandidate(
            candidate_id=f"{data.data_id}:candidate",
            candidate_tree=candidate_tree,
            current_tree=head_tree,
            evaluation_tasks=step_tasks.tasks,
            step=state.steps,
            proposal_id=None if claimed is None else claimed.id,
        )
        return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)

    def run_method(self, workspace: Workspace, head: str, metrics: dict[str, Any], calls: StepCalls) -> StepResult:
        """Run the method's step in ``workspace`` with ``calls`` as its call meter, and record its calls and its
        result with the step."""
        tools = StepTools(
            descriptor=self.descriptor,
            evaluator=self.evaluator,
            history=self.history,
            head=head,
            step=workspace.step,
            models=self.models,
            step_calls=calls,
            max_rollouts=self.limits.max_rollouts_per_step,
        )
        try:
            result = self.method.step(StepContext(workspace, tools))
        finally:
            # Written even when the step raised: the calls before the failure are the step's record.
            self.records.write(workspace.root, CALLS_FILE, calls.entries)
            shutil.rmtree(workspace.scratch, ignore_errors=True)
            metrics["model_calls"] = len(calls.entries)
            metrics["model_seconds"] = round(sum(float(entry.get("seconds", 0.0)) for entry in calls.entries), 3)
            metrics["model_input_tokens"], metrics["model_output_tokens"] = call_tokens(calls.entries)
            metrics["method_rollouts"] = tools.rollouts_run
        if not isinstance(result, StepResult):
            raise TypeError(f"{type(self.method).__name__}.step must return a StepResult")
        self.records.write(
            workspace.root,
            RESULT_FILE,
            bounded({"summary": result.summary, "notes": dict(result.notes), "requires": list(result.requires)}),
        )
        if result.summary:
            metrics["method_summary"] = clip_record_text(result.summary)
        if result.notes:
            metrics["method_notes"] = bounded(result.notes)
        return result

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        harness_candidate = require_harness_candidate(candidate)
        if self.evaluator is None:
            raise RuntimeError("the deployment configured no evolution.evaluation.scorer, so no rollout can run")
        if self.progress is not None:
            total = self.evaluator.rollout_count(harness_candidate, len(tuple(sides)))
            self.progress = replace(self.progress, phase="evaluating", rollouts_total=total)
        return self.evaluator.evaluate(harness_candidate, sides=sides)

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        state = HarnessState.from_json(prepared.state)
        metrics = dict(prepared.metrics)
        evaluation = dict(decision.evaluation.metrics)
        # The history holds every rollout: the commit keeps the evaluation's sums, not its score and failure vectors.
        evaluation.pop("candidate_scores", None)
        evaluation.pop("current_scores", None)
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
        context = self.context_key()
        # A selected candidate makes the tree it replaced the recheck's target; a recheck that selects that tree
        # consumes it.
        recheck_release_id, recheck_context = state.recheck_release_id, state.recheck_context
        if candidate.recheck:
            metrics["rolled_back"] = decision.selected
            if decision.selected:
                recheck_release_id, recheck_context = None, None
            else:
                # The head held in this context, so the next recheck waits for the cadence or another context change.
                recheck_context = context
        elif decision.selected and self.limits.recheck_every and context is not None:
            recheck_release_id, recheck_context = str(metrics["head_release_id"]), context
        # A recheck is not a candidate the method made, so it leaves the failure streak alone.
        if candidate.recheck:
            failure_streak = state.failure_streak
        else:
            failure_streak = 0 if decision.selected else state.failure_streak + 1
        state = replace(
            state,
            failure_streak=failure_streak,
            manifest=manifest,
            recheck_release_id=recheck_release_id,
            recheck_context=recheck_context,
        )
        self.history.add_decision(
            DecisionRecord(
                step=state.steps,
                tree=candidate.candidate_tree.checksum(),
                head=candidate.current_tree.checksum(),
                selected=decision.selected,
                policy=decision.policy,
                reason=decision.reason,
            )
        )
        metrics.update(
            {
                **evaluation,
                **decision.metrics,
                "evaluation_context": context,
                "selected": decision.selected,
                # The decision without its evaluation, whose metrics are above without the vectors.
                "selection": {
                    "candidate_id": candidate.candidate_id,
                    "selected": decision.selected,
                    "policy": decision.policy,
                    "reason": decision.reason,
                    "metrics": dict(decision.metrics),
                },
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
        # The candidate reached its evaluation: the failed row keeps what changed, the method's notes and the
        # record directory, and says where it stopped.
        self.failed_step = {
            "failed_stage": "evaluating",
            "patch": bounded(candidate.patch.to_json()),
            **{
                key: prepared.metrics[key]
                for key in ("method_summary", "method_notes", "step_record")
                if key in prepared.metrics
            },
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

    def context_key(self) -> str | None:
        """The key of the current evaluation context; ``None`` when no rollout can run."""
        return None if self.evaluator is None else self.evaluator.context().key()

    def recheck_due(self, state: HarnessState) -> bool:
        """Whether this automatic step re-evaluates the latest kept tree against the tree it replaced."""
        if not self.limits.recheck_every or state.recheck_release_id is None:
            return False
        return state.steps % self.limits.recheck_every == 0 or state.recheck_context != self.context_key()

    def inbox_refuse(self, proposal_id: str, reason: str) -> None:
        if self.inbox is not None:
            self.inbox.refuse(proposal_id, reason)


def screened_requires(
    base: Sequence[Mapping[str, Any]], added: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The items a method added that pass the screens a person's items pass, and the ones they refuse.

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


def skipped(state: HarnessState, metrics: Mapping[str, Any], reason: str) -> PreparedStep:
    """A step that ends without a candidate, keeping ``state`` and saying why."""
    return PreparedStep.skipped(state=state.to_json(), metrics={**metrics, "skipped": reason})


def require_harness_candidate(candidate: UpdateCandidate | None) -> HarnessCandidate:
    if not isinstance(candidate, HarnessCandidate):
        raise TypeError(f"harness evolution requires a HarnessCandidate, got {type(candidate).__name__}")
    return candidate


__all__ = ["HarnessBackend", "StepLimits"]
