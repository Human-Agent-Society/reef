"""The harness backend: one evolution step, from the head tree to a release.

One step runs in three phases, one training job of the harness training
component (``reef.train.harness_backend.component``):

``prepare_step``
    Read the head tree, which the job names, and the step's evaluation tasks. The step's job data is its material: the traces of
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
    ``HarnessTaskEvaluator`` (``reef.evaluation.harness``), which reuses
    the scores the history holds.
``settle_step``
    Fold the decision into the state and the history. A selected tree is
    the step's kept tree: an ``Artifact.local`` of its source tree, which the
    component saves as the job's checkpoint and the scenario committer stores
    as a git commit. The scenario's activation rule
    decides whether it serves, from the tree kinds it changes.

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
    EvaluationTasks,
    HarnessCandidate,
    HarnessCandidateEvaluator,
    HarnessTaskEvaluator,
    History,
    RunRecord,
)
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tasks.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, validate_tree
from reef.harness.tree.checks import clip_record_text, clip_record_value, directive_shaped, secret_shaped
from reef.train.backend import PreparedStep, ReleaseFiles
from reef.train.harness_backend.agent.run import AgentExecution
from reef.train.harness_backend.calls import StepCalls, call_tokens
from reef.train.harness_backend.contract import Method, StepContext
from reef.train.harness_backend.records import CALLS_FILE, StepRecord
from reef.train.harness_backend.services import StepServices
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.workspace import StepOutput, Workspace
from reef.train.types import JobData, TrainStepResult

#: The phases of a running step, in order; only an offline step, which rolls out train tasks, has the first.
STEP_PHASES = ("running tasks", "proposing", "evaluating")


@dataclass(frozen=True)
class StepProgress:
    """Where the running step stands, for a page a person watches.

    Values: ``request_id`` names the instruction the step answers, ``None``
    for an automatic step. ``phase`` is one of
    ``STEP_PHASES``: ``"running tasks"`` while an offline step runs the head
    tree on its train tasks, ``"evaluating"`` while the evaluation's
    tasks run, and ``"proposing"`` at any other time from the moment the
    step claims its record directory until a candidate exists.
    ``started_at`` is the step's start in seconds since the epoch.
    ``step_record`` is the step's record directory, which the step claims
    before it reports any progress.
    ``runs_total`` ≥ 0 is the run count of the runs or of the
    evaluation once its runs are laid out, else ``None``. ``activity``
    is what the method did so far, oldest first, each ``{at, kind, text}``
    with ``failed`` on a line that went wrong.
    Purpose: the backend replaces it at each phase change; the harness training component writes it to the step's
    progress file (``to_json``), and the request page reads it back (``from_json``).
    """

    request_id: str | None
    phase: str
    started_at: float
    step_record: str
    runs_total: int | None = None
    activity: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in STEP_PHASES:
            raise ValueError(f"StepProgress.phase must be one of {', '.join(STEP_PHASES)}")
        if not isinstance(self.step_record, str) or not self.step_record:
            raise ValueError("StepProgress.step_record must name the step's record directory")
        total = self.runs_total
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ValueError("StepProgress.runs_total must be an integer of at least 0 or None")

    def to_json(self) -> dict[str, Any]:
        """The progress file's form of this value; ``from_json`` reads it back as an equal value."""
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "started_at": self.started_at,
            "step_record": self.step_record,
            "runs_total": self.runs_total,
            "activity": [dict(line) for line in self.activity],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> StepProgress:
        """The value that ``to_json`` wrote; ``ValueError`` or ``KeyError`` for another mapping."""
        request_id = value["request_id"]
        return cls(
            request_id=None if request_id is None else str(request_id),
            phase=str(value["phase"]),
            started_at=float(value["started_at"]),
            step_record=str(value["step_record"]),
            runs_total=value["runs_total"],
            activity=tuple(dict(line) for line in value["activity"]),
        )


@dataclass(frozen=True)
class StepLimits:
    """The deployment's budgets for harness steps.

    Values: every count is an integer ≥ 0, where 0 turns its limit off.
    ``max_steps`` caps automatic steps; ``max_failure_streak`` skips
    automatic steps after that many rejections in a row;
    ``max_model_calls_per_step`` caps one step's model calls;
    ``max_runs_per_step`` caps the runs the method's
    ``services.evaluate`` runs in one step.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution``; the
    backend reads it in every step.
    """

    max_steps: int = 0
    max_failure_streak: int = 0
    max_model_calls_per_step: int = 0
    max_runs_per_step: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("max_runs_per_step", self.max_runs_per_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be an integer of at least 0 (0 turns the limit off)")


class HarnessBackend(HarnessCandidateEvaluator):
    """Evolve one harness component: one candidate per step, evaluated against the head tree.

    Values: ``method`` is the deployment's ``Method``, the one part of a step that is not Reef's; methods never
    subclass the backend. ``models`` are the scenario's model bindings for every step. ``agent`` is the
    deployment's coding agent, ``None`` when it configured none or the method
    does not run it. ``evaluator`` is ``None`` when the deployment configured
    no scorer; then no task runs. ``task_directories`` says
    that the tasks are Harbor task directories.
    Purpose: ``HarnessEvolutionRecipe.build_backend`` makes one per scenario
    component; the harness training component (``reef.train.harness_backend.component``) runs one step per job.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        method: Method,
        agent: AgentExecution | None,
        evaluator: HarnessTaskEvaluator | None,
        tasks: EvaluationTasks,
        history: History,
        models: ModelBindings,
        limits: StepLimits,
        records: StepRecord,
        task_directories: bool,
    ) -> None:
        self.descriptor = descriptor
        self.method = method
        self.agent = agent
        self.evaluator = evaluator
        self.tasks = tasks
        self.history = history
        self.models = models
        self.limits = limits
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

    def close(self) -> None:
        if self.evaluator is not None:
            self.evaluator.close()

    def prepare_step(
        self,
        data: JobData,
        state: Mapping[str, Any],
        scenario_step: int,
        *,
        head: ReleaseFiles,
        seed_tasks: tuple[str, ...] | None = None,
    ) -> PreparedStep:
        """Step ``scenario_step + 1``'s candidate, or its skip, on ``head``. ``seed_tasks`` replaces the
        evaluation's seed tasks for this step when it is set."""
        self.progress = None
        self.failed_step = {}
        try:
            prepared = self.prepare(data, HarnessState.from_json(state), scenario_step + 1, head, seed_tasks)
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

    def prepare(
        self, data: JobData, state: HarnessState, step: int, head: ReleaseFiles, seed_tasks: tuple[str, ...] | None
    ) -> PreparedStep:
        """Step ``step``'s candidate on ``head``, or the skip and its reason; ``seed_tasks`` as ``prepare_step``."""
        # The step's material: the traces of traffic, and on an offline step the train tasks it rolls out.
        samples = tuple(sample for sample in data.samples if isinstance(sample, TrajectorySample))
        train_tasks = tuple(str(sample.task_path) for sample in data.samples if isinstance(sample, TaskSample))
        metrics: dict[str, Any] = {"traces": len(samples)}
        if train_tasks:
            metrics["runs"] = len(train_tasks)
        automatic = data.request is None
        if automatic and self.limits.max_steps and step > self.limits.max_steps:
            return skipped(state, metrics, f"step budget of {self.limits.max_steps} exhausted")
        if automatic and self.limits.max_failure_streak and state.failure_streak >= self.limits.max_failure_streak:
            return skipped(state, metrics, f"failure streak of {state.failure_streak} rejections")
        head_tree = HarnessTree(head.files)
        metrics["head_release_id"] = head.release_id
        self.history.add_candidate(head_tree, step=step, origin="release" if self.history.candidates else "seed")
        step_tasks = self.tasks.for_step(samples, state.collected_tasks, state.collected_clients, seed=seed_tasks)
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
        task_runs = self.run_train_tasks(head_tree, train_tasks, step) if train_tasks else ()
        workspace = Workspace.create(
            step_dir,
            history_dir=self.history.root,
            step=step,
            tree=head_tree,
            descriptor=self.descriptor,
            task_directories=self.task_directories,
            tasks=step_tasks.tasks,
            repeats=0 if self.evaluator is None else self.evaluator.task_repeats,
            samples=samples,
            runs=task_runs,
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
        gains the method's calls, runs and words.
        """
        services = StepServices(
            workspace=workspace,
            descriptor=self.descriptor,
            evaluator=self.evaluator,
            history=self.history,
            head=head.checksum(),
            models=self.models,
            step_calls=calls,
            max_runs=self.limits.max_runs_per_step,
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
            metrics["method_runs"] = services.runs_run
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

    def run_train_tasks(self, head: HarnessTree, tasks: Sequence[str], step: int) -> tuple[RunRecord, ...]:
        """One run record of ``head`` on each of an offline step's train ``tasks``, in task order.

        The runs go through the score cache, as an evaluation's do: a run of the same tree, task and
        evaluation context that the history holds is reused. ``RuntimeError`` without a scorer.
        """
        if self.evaluator is None:
            raise RuntimeError("an offline step rolls out its train tasks, which needs evolution.evaluation.scorer")
        if self.progress is not None:
            self.progress = replace(self.progress, phase="running tasks", runs_total=len(tasks))
        records = self.evaluator.runs(head, tasks, repeats=1, step=step)
        if self.progress is not None:
            self.progress = replace(self.progress, phase="proposing", runs_total=None)
        return records

    def evaluate(self, candidate: UpdateCandidate, *, sides: Sequence[str] = EVALUATION_SIDES) -> EvaluationResult:
        harness_candidate = require_harness_candidate(candidate)
        if self.evaluator is None:
            raise RuntimeError("the deployment configured no evolution.evaluation.scorer, so no run can run")
        if self.progress is not None:
            total = self.evaluator.run_count(harness_candidate, len(tuple(sides)))
            self.progress = replace(self.progress, phase="evaluating", runs_total=total)
        return self.evaluator.evaluate(harness_candidate, sides=sides)

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        """Fold ``decision`` into the state and the history; a selected tree is the step's kept tree.

        The head the step prepared against is ``prepared.metrics["head_release_id"]``; no activation, new training
        start, import or sync moves it while a step runs. The commit keeps the evaluation's sums, not its score
        vectors: the history holds every run. The metrics gain ``selected`` and ``selection``: the candidate
        id, the decision's ``selected``, ``policy``, ``reason`` and ``metrics``, and the ``evaluation`` that the
        decision read without its metrics (``evaluator``, ``evaluator_version``, ``metadata``). A rejected
        candidate makes no artifact. A selected one is the result's artifact, and its ``changed_kinds`` are the
        names of the parts its patch changes, from which the scenario's activation rule decides whether it serves.
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
        patch_parts = self.descriptor.layout.part_names(candidate.patch.changes)
        metrics = {
            **prepared.metrics,
            **evaluation,
            **decision.metrics,
            "evaluation_context": None if self.evaluator is None else self.evaluator.context().key(),
            "patch": clip_record_value(candidate.patch.to_json()),
            "patch_parts": patch_parts,
            "selected": decision.selected,
            "selection": {
                "candidate_id": candidate.candidate_id,
                "selected": decision.selected,
                "policy": decision.policy,
                "reason": decision.reason,
                "metrics": dict(decision.metrics),
                "evaluation": {
                    "evaluator": decision.evaluation.evaluator,
                    "evaluator_version": decision.evaluation.evaluator_version,
                    "metadata": dict(decision.evaluation.metadata),
                },
            },
        }
        if not decision.selected:
            return TrainStepResult(state.to_json(), metrics)
        directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
        candidate.candidate_tree.write(directory)
        artifact = Artifact.local(directory)
        if self.pending_artifact is not None:
            self.pending_artifact.discard()
        self.pending_artifact = artifact
        return TrainStepResult(
            state.to_json(), metrics, artifact=artifact, changed_kinds=frozenset(patch_parts.values())
        )

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

    def discard_kept_tree(self) -> None:
        """Discard the local source tree of the step's kept tree, once its job's checkpoint holds a copy."""
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
