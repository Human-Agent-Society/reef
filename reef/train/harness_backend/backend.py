"""The harness backend: one evolution step, from the head tree to a release.

One step runs in the trainer's three phases:

``prepare_step``
    Read the head tree from git through ``ComponentReleases``. The candidate
    is the tree the latest publish replaced (a recheck), a client's patch from
    the inbox, or the tree the method leaves in a fresh step workspace. An
    offline step's job data holds train tasks: the step rolls them out on the
    head tree first, and the method reads the episodes as rollout traces.
    Validate the candidate, and make a ``HarnessCandidate`` of it against the
    head.
``evaluate``
    Score the sides that the publication policy names with
    ``HarnessEpisodeEvaluator``, which reuses the scores the history holds.
``settle_step``
    Record the decision in the history and fold it into ``HarnessState``. A
    selected candidate becomes an ``Artifact.local`` of its source tree, which
    the scenario committer publishes as a git commit, held for review when
    the deployment or the changed kinds ask for it.

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
from reef.core.job_data import TaskSample, TrajectorySample
from reef.core.requirements import MAX_REQUIRES, merge_requires
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
from reef.harness.tree import HarnessTree, TreePatch, ValidationResult, validate_patch, validate_tree
from reef.harness.tree.checks import clip_record_text, directive_shaped, secret_shaped
from reef.train.backend import CandidateBackend, ComponentReleases, PreparedStep
from reef.train.harness_backend.agent.run import AgentExecution
from reef.train.harness_backend.calls import StepCalls, call_tokens
from reef.train.harness_backend.capabilities import PatchAdmission, StepProgress, StepProgressReader, StepRecords
from reef.train.harness_backend.contract import Method, StepContext
from reef.train.harness_backend.inbox import ProposalInbox
from reef.train.harness_backend.records import CALLS_FILE, StepRecord, bounded
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.tools import StepTools
from reef.train.harness_backend.workspace import StepOutput, Workspace
from reef.train.types import JobData, TrainStepResult


@dataclass(frozen=True)
class StepLimits:
    """The deployment's limits and publication settings for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` and ``max_episodes_per_step`` cap one step's
    model calls and the episodes its method runs; ``recheck_every``
    re-evaluates the latest publish against the tree it replaced every that
    many steps. ``publish`` is ``"auto"``, which serves a selected tree at
    once, or ``"review"``, which holds every selected tree for a person.
    ``review_parts`` name the parts of the adapter's tree whose change waits
    for a person even under ``"auto"``, besides the parts that always do.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    max_episodes_per_step: int = 0
    recheck_every: int = 0
    publish: str = "auto"
    review_parts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("max_episodes_per_step", self.max_episodes_per_step),
            ("recheck_every", self.recheck_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 turns the limit off)")
        if self.publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")


class HarnessBackend(CandidateBackend, HarnessCandidateEvaluator, PatchAdmission, StepRecords, StepProgressReader):
    """Evolve one harness component: one candidate per step, evaluated against the head tree.

    Methods never subclass it. A method's behavior arrives as ``method``, and
    the scorer and the publication policy arrive in the evaluator and the
    trainer's evaluation plugin. ``evaluator`` is ``None`` when the
    deployment configured no scorer; then no episode runs. ``agent`` is the
    deployment's coding agent, which ``Tools.run_agent`` runs, ``None`` when
    it configured none.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        releases: ComponentReleases,
        method: Method,
        evaluator: HarnessEpisodeEvaluator | None,
        tasks: EvaluationTasks,
        history: History,
        models: ModelBindings,
        model_resolver: ModelBindingsResolver | None = None,
        limits: StepLimits,
        records: StepRecord,
        inbox: ProposalInbox | None = None,
        agent: AgentExecution | None = None,
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
        self.agent = agent
        #: The source trees of selected candidates, by step, until their commit is durable.
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
        head_record = self.history.add_candidate(
            head_tree, step=step, origin="release" if self.history.candidates else "seed"
        )
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
        # A recheck and a client's proposal wait for a step without rollouts: the method reads every rolled-out chunk.
        if automatic and not rollout_tasks and self.recheck_due(state, step) and state.rollback_release_id is not None:
            # The candidate is the tree the latest publish replaced; it wins only when the head got worse.
            target = self.releases.release(state.rollback_release_id)
            target_tree = HarnessTree(target.files)
            metrics.update(
                {
                    "recheck": True,
                    "recheck_reason": "drift" if state.rollback_context != self.context_key() else "cadence",
                    "recheck_release_id": target.release_id,
                }
            )
            validation = validate_tree(target_tree, self.descriptor)
            if not validation.passed:
                # The current checks refuse the replaced tree, so no recheck can roll back to it: the step drops it.
                dropped = replace(state, rollback_release_id=None, rollback_context=None)
                return skipped(dropped, metrics, validation.message())
            self.history.add_candidate(target_tree, step=step, origin="release")
            candidate = HarnessCandidate(
                candidate_id=f"{data.data_id}:recheck",
                candidate_tree=target_tree,
                current_tree=head_tree,
                evaluation_tasks=step_tasks.tasks,
                step=step,
                origin="release",
            )
            return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)
        claimed = None if self.inbox is None or not automatic or rollout_tasks else self.inbox.claim()
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
                candidate_tree, step=step, origin="proposal", parent=head_record.tree, summary=claimed.reason
            )
        else:
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
            output = self.run_method(workspace, head_record.tree, metrics, calls)
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
                step=step,
                origin="method",
                parent=head_record.tree,
                summary=output.summary,
                notes=output.notes,
            )
            # An instruction step never claims a proposal: its method's requires join the request's.
            if data.request is not None:
                added, refused = screened_requires(data.request.requires, output.requires)
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
            step=step,
            origin="method" if claimed is None else "proposal",
        )
        return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)

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

    def run_method(self, workspace: Workspace, head: str, metrics: dict[str, Any], calls: StepCalls) -> StepOutput:
        """Run the method's step in ``workspace`` with ``calls`` as its call meter, record its calls with the step,
        and read what it wrote to ``output/``."""
        tools = StepTools(
            workspace=workspace,
            descriptor=self.descriptor,
            evaluator=self.evaluator,
            history=self.history,
            head=head,
            models=self.models,
            step_calls=calls,
            max_episodes=self.limits.max_episodes_per_step,
            agent=self.agent,
        )
        try:
            self.method.step(StepContext(workspace, tools))
        finally:
            # Written even when the step raised: the calls before the failure are the step's record.
            self.records.write(workspace.root, CALLS_FILE, calls.entries)
            shutil.rmtree(workspace.scratch, ignore_errors=True)
            metrics["model_calls"] = len(calls.entries)
            metrics["model_seconds"] = round(sum(float(entry.get("seconds", 0.0)) for entry in calls.entries), 3)
            metrics["model_input_tokens"], metrics["model_output_tokens"] = call_tokens(calls.entries)
            metrics["method_episodes"] = tools.episodes_run
        output = workspace.read_output()
        # The record keeps the method's words redacted and clipped, as it keeps its model calls.
        workspace.write_output(clip_record_text(output.summary), bounded(output.notes), output.requires)
        if output.summary:
            metrics["method_summary"] = clip_record_text(output.summary)
        if output.notes:
            metrics["method_notes"] = bounded(output.notes)
        if output.refused:
            metrics["method_output_refused"] = dict(output.refused)
        return output

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        harness_candidate = require_harness_candidate(candidate)
        if self.evaluator is None:
            raise RuntimeError("the deployment configured no evolution.evaluation.scorer, so no episode can run")
        if self.progress is not None:
            total = self.evaluator.episode_count(harness_candidate, len(tuple(sides)))
            self.progress = replace(self.progress, phase="evaluating", episodes_total=total)
        return self.evaluator.evaluate(harness_candidate, sides=sides)

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        state = HarnessState.from_json(prepared.state)
        metrics = dict(prepared.metrics)
        evaluation = dict(decision.evaluation.metrics)
        # The history holds every episode: the commit keeps the evaluation's sums, not its score vectors.
        evaluation.pop("candidate_scores", None)
        evaluation.pop("current_scores", None)
        context = self.context_key()
        # A publish keeps the tree it replaced as the recheck's target; a recheck that rolls back consumes it.
        rollback_release_id, rollback_context = state.rollback_release_id, state.rollback_context
        if candidate.origin == "release":
            metrics["rolled_back"] = decision.selected
            if decision.selected:
                rollback_release_id, rollback_context = None, None
            else:
                # The head held in this context, so the next recheck waits for the cadence or another context change.
                rollback_context = context
        elif decision.selected and self.limits.recheck_every and context is not None:
            rollback_release_id, rollback_context = str(metrics["head_release_id"]), context
        # A recheck is not a candidate the method made, so it leaves the failure streak alone.
        if candidate.origin == "release":
            failure_streak = state.failure_streak
        else:
            failure_streak = 0 if decision.selected else state.failure_streak + 1
        state = replace(
            state,
            failure_streak=failure_streak,
            rollback_release_id=rollback_release_id,
            rollback_context=rollback_context,
        )
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
                "patch_parts": self.patch_parts(candidate.patch),
            }
        )
        if candidate.origin == "proposal" and self.inbox is not None:
            self.inbox.settle(
                metrics["proposal"]["id"],
                {"step": candidate.step, "selected": decision.selected, "reason": decision.reason},
            )
        if not decision.selected:
            return TrainStepResult(state.to_json(), metrics)
        directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
        candidate.candidate_tree.write(directory)
        artifact = Artifact.local(directory)
        if self.pending_artifact is not None:
            self.pending_artifact.discard()
        self.pending_artifact = artifact
        # A rollback serves a tree that was published before, so only publish: review holds it.
        pending = (
            self.limits.publish == "review" if candidate.origin == "release" else self.held_for_review(candidate.patch)
        )
        return TrainStepResult(state.to_json(), metrics, artifact=artifact, pending=pending)

    def abort_step(self, prepared: PreparedStep) -> None:
        self.progress = None
        candidate = require_harness_candidate(prepared.candidate)
        # The candidate reached its evaluation: the failed row keeps what changed, the method's notes and the
        # record directory, and says where it stopped.
        self.failed_step = {
            "failed_stage": "evaluating",
            "patch": bounded(candidate.patch.to_json()),
            "patch_parts": self.patch_parts(candidate.patch),
            **{
                key: prepared.metrics[key]
                for key in ("method_summary", "method_notes", "step_record")
                if key in prepared.metrics
            },
        }
        if candidate.origin == "proposal":
            self.inbox_refuse(prepared.metrics["proposal"]["id"], "step aborted before a result")

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Discard the step's local source tree once the committer stored it in git."""
        if self.pending_artifact is not None:
            self.pending_artifact.discard()
            self.pending_artifact = None

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
        """The key of the current evaluation context; ``None`` when no episode can run."""
        return None if self.evaluator is None else self.evaluator.context().key()

    def recheck_due(self, state: HarnessState, step: int) -> bool:
        """Whether automatic step ``step`` re-evaluates the latest publish against the tree it replaced."""
        if not self.limits.recheck_every or state.rollback_release_id is None:
            return False
        return step % self.limits.recheck_every == 0 or state.rollback_context != self.context_key()

    def held_for_review(self, patch: TreePatch) -> bool:
        """Whether a selected patch waits for a person: under ``publish: review``, or when it changes a part that the
        deployment's ``review_parts`` name or that the adapter always reviews."""
        if self.limits.publish == "review":
            return True
        for path in patch.changes:
            location = self.descriptor.layout.locate(path)
            if location is not None and (location.part.review or location.part.name in self.limits.review_parts):
                return True
        return False

    def patch_parts(self, patch: TreePatch) -> dict[str, str]:
        """The part of the adapter's tree each path of ``patch`` is, for the step pages."""
        located = {path: self.descriptor.layout.locate(path) for path in patch.changes}
        return {path: location.part.name for path, location in located.items() if location is not None}

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
