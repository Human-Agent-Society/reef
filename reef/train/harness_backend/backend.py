"""The harness backend: one evolution step, from the head tree to a release.

One step runs in the trainer's three phases:

``prepare_step``
    Read the head tree from git through ``ComponentReleases`` and the step's
    evaluation tasks. The step's job data is its material: the traces of
    traffic, and on an offline step the train tasks, which the step rolls
    out on the head tree first. Write them to a fresh ``Workspace``, then
    ``propose`` runs the method in it:

    1. Call ``Method.step`` once with the step's services, and keep its
       model calls with the step record, even when it raises.
    2. Read what the method says about its step from ``output/``, clipped
       and redacted as the record keeps it, and its files from ``state/``.
    3. Read the tree it left in ``harness/``. An unchanged tree, or one that
       fails validation for the adapter, skips the step.
    4. Add the tree to the history with its parent and the method's
       summary, and make it a ``HarnessCandidate`` against the head. On an
       instruction step the method's ``requires`` join the request's, after
       the screens a person's items pass.
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
from reef.harness.tree import HarnessTree, TreePatch, validate_tree
from reef.harness.tree.checks import clip_record_text, clip_record_value, directive_shaped, secret_shaped
from reef.train.backend import CandidateBackend, ComponentReleases, PreparedStep
from reef.train.harness_backend.agent.run import AgentExecution
from reef.train.harness_backend.calls import StepCalls, call_tokens
from reef.train.harness_backend.contract import Method, StepContext
from reef.train.harness_backend.records import CALLS_FILE, StepRecord
from reef.train.harness_backend.services import StepServices
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.workspace import StepOutput, Workspace
from reef.train.release import ReleaseRules
from reef.train.types import JobData, TrainStepResult

#: The phases of a running step, in order; only an offline step, which rolls out train tasks, has the first.
STEP_PHASES = ("rolling out", "proposing", "evaluating")


@dataclass(frozen=True)
class StepProgress:
    """Where the running step stands, for a page a person watches.

    Values: ``request_id`` names the instruction the step answers, ``None``
    for an automatic step. ``phase`` is one of
    ``STEP_PHASES``: ``"rolling out"`` while an offline step runs the head
    tree on its train tasks, ``"evaluating"`` while the evaluation's
    episodes run, and ``"proposing"`` at any other time from the moment the
    step claims its record directory until a candidate exists.
    ``started_at`` is the step's start in seconds since the epoch.
    ``step_record`` is the step's record directory, which the step claims
    before it reports any progress.
    ``episodes_total`` ≥ 0 is the episode count of the rollouts or of the
    evaluation once its episodes are laid out, else ``None``. ``activity``
    is what the method did so far, oldest first, each ``{at, kind, text}``
    with ``failed`` on a line that went wrong.
    Purpose: the backend replaces it at each phase change; the request page
    reads it from another thread through ``HarnessBackend.step_progress``.
    """

    request_id: str | None
    phase: str
    started_at: float
    step_record: str
    episodes_total: int | None = None
    activity: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in STEP_PHASES:
            raise ValueError(f"StepProgress.phase must be one of {', '.join(STEP_PHASES)}")
        if not isinstance(self.step_record, str) or not self.step_record:
            raise ValueError("StepProgress.step_record must name the step's record directory")
        total = self.episodes_total
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ValueError("StepProgress.episodes_total must be an integer of at least 0 or None")


@dataclass(frozen=True)
class StepLimits:
    """The deployment's budgets for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` caps one step's model calls;
    ``max_episodes_per_step`` caps the episodes the method's
    ``services.evaluate`` runs in one step.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    max_episodes_per_step: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("max_episodes_per_step", self.max_episodes_per_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 turns the limit off)")


class HarnessBackend(CandidateBackend, HarnessCandidateEvaluator):
    """Evolve one harness component: one candidate per step, evaluated against the head tree.

    Values: ``method`` is the deployment's ``Method``, the one part of a step
    that is not Reef's; methods never subclass the backend. ``agent`` is the
    deployment's coding agent, ``None`` when it configured none or the method
    does not run it. ``evaluator`` is ``None`` when the deployment configured
    no scorer; then no episode runs. ``rules`` make each decision a release,
    and ``review_parts`` name the parts whose change waits for a person
    besides the parts the adapter always reviews. ``task_directories`` says
    that the tasks are Harbor task directories.
    Purpose: ``HarnessEvolutionRecipe.build_backend`` makes one per scenario
    component; the trainer runs its steps, and the service reads its step
    records and its ``step_progress``.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        releases: ComponentReleases,
        method: Method,
        agent: AgentExecution | None,
        evaluator: HarnessEpisodeEvaluator | None,
        tasks: EvaluationTasks,
        history: History,
        models: ModelBindings,
        model_resolver: ModelBindingsResolver | None = None,
        limits: StepLimits,
        rules: ReleaseRules,
        review_parts: frozenset[str] = frozenset(),
        records: StepRecord,
        task_directories: bool,
    ) -> None:
        self.descriptor = descriptor
        self.releases = releases
        self.method = method
        self.agent = agent
        self.evaluator = evaluator
        self.tasks = tasks
        self.history = history
        self.models = models
        self.model_resolver = model_resolver
        self.limits = limits
        self.rules = rules
        self.review_parts = review_parts
        self.records = records
        self.task_directories = task_directories
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
            task_directories=self.task_directories,
            tasks=step_tasks.tasks,
            repeats=0 if self.evaluator is None else self.evaluator.episode_repeats,
            samples=samples,
            rollouts=rollouts,
            request=data.request,
            state_files=state.method_files,
        )
        return self.propose(
            workspace,
            head=head_tree,
            evaluation_tasks=step_tasks.tasks,
            data=data,
            state=state,
            calls=calls,
            metrics=metrics,
        )

    def propose(
        self,
        workspace: Workspace,
        *,
        head: HarnessTree,
        evaluation_tasks: tuple[str, ...],
        data: JobData,
        state: HarnessState,
        calls: StepCalls,
        metrics: dict[str, Any],
    ) -> PreparedStep:
        """Run the method in ``workspace`` on ``head``: the step's candidate, or the skip and its reason.

        The candidate is evaluated on ``evaluation_tasks``. ``state`` is the step's state, and the returned step's
        state carries the method's files from ``state/``. ``calls`` is the step's call record, and ``metrics``
        gains the method's calls, episodes and words.
        """
        services = StepServices(
            workspace=workspace,
            descriptor=self.descriptor,
            evaluator=self.evaluator,
            history=self.history,
            head=head.checksum(),
            models=self.models,
            step_calls=calls,
            max_episodes=self.limits.max_episodes_per_step,
            agent=self.agent,
        )
        try:
            self.method.step(StepContext(workspace, services))
        finally:
            # Written even when the step raised: the calls before the failure are the step's record.
            self.records.write(workspace.root, CALLS_FILE, calls.entries)
            shutil.rmtree(workspace.scratch, ignore_errors=True)
            metrics["model_calls"] = len(calls.entries)
            metrics["model_seconds"] = round(sum(float(entry.get("seconds", 0.0)) for entry in calls.entries), 3)
            metrics["model_input_tokens"], metrics["model_output_tokens"] = call_tokens(calls.entries)
            metrics["method_episodes"] = services.episodes_run
        output = self.read_output(workspace, metrics)
        try:
            state = replace(state, method_files=workspace.state_files())
            candidate_tree = workspace.tree()
        except ValueError as exc:
            return skipped(state, metrics, str(exc))
        if candidate_tree == head:
            return skipped(state, metrics, "the method left the head tree unchanged")
        validation = validate_tree(candidate_tree, self.descriptor)
        if not validation.passed:
            return skipped(state, metrics, validation.message())
        self.history.add_candidate(
            candidate_tree,
            step=workspace.step,
            origin="method",
            parent=head.checksum(),
            summary=output.summary,
            notes=output.notes,
        )
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
            current_tree=head,
            evaluation_tasks=evaluation_tasks,
            step=workspace.step,
        )
        return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)

    @staticmethod
    def read_output(workspace: Workspace, metrics: dict[str, Any]) -> StepOutput:
        """What the method wrote to ``output/``, rewritten as the record keeps it: redacted and clipped."""
        output = workspace.read_output()
        workspace.write_output(clip_record_text(output.summary), clip_record_value(output.notes), output.requires)
        if output.summary:
            metrics["method_summary"] = clip_record_text(output.summary)
        if output.notes:
            metrics["method_notes"] = clip_record_value(output.notes)
        if output.refused:
            metrics["method_output_refused"] = dict(output.refused)
        return output

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
            "patch": clip_record_value(candidate.patch.to_json()),
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
            "patch": clip_record_value(candidate.patch.to_json()),
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

    @property
    def step_progress(self) -> StepProgress | None:
        """Where the running step stands, with the method's activity so far; ``None`` when no step runs."""
        progress, calls = self.progress, self.step_calls
        if progress is None or calls is None:
            return progress
        return replace(progress, activity=calls.activity())

    def reviewed(self, patch: TreePatch) -> bool:
        """Whether ``patch`` changes a part that waits for a person: one the adapter reviews, or one that
        ``review_parts`` names."""
        for path in patch.changes:
            location = self.descriptor.layout.locate(path)
            if location is not None and (location.part.review or location.part.name in self.review_parts):
                return True
        return False


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
            refused.append({"item": clip_record_value(dict(item)), "reason": "carries a credential shaped literal"})
        elif any(directive_shaped(text) for text in texts):
            refused.append(
                {"item": clip_record_value(dict(item)), "reason": "carries an instruction override phrasing"}
            )
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


__all__ = ["STEP_PHASES", "HarnessBackend", "StepLimits", "StepProgress", "screened_requires"]
