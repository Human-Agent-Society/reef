"""A step's candidate: the tree the method leaves in the step's workspace.

``Evolution.propose`` runs the method:

1. Run the method in the step's workspace with the step's tools, and keep its
   model calls with the step record, even when it raises.
2. Read what the method says about its step from ``output/``, clipped and
   redacted as the record keeps it.
3. Read the tree it left in ``harness/``. An unchanged tree, or one that
   fails validation for the adapter, skips the step.
4. Add the tree to the history with its parent and the method's summary, and
   make it a ``HarnessCandidate`` against the head.

On an instruction step the method's ``requires`` join the request's, after the
screens a person's items pass.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from reef.core.requirements import MAX_REQUIRES, merge_requires
from reef.evaluation.harness import HarnessCandidate, HarnessEpisodeEvaluator, History
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, validate_tree
from reef.harness.tree.checks import clip_record_text, directive_shaped, secret_shaped
from reef.train.backend import PreparedStep
from reef.train.harness_backend.agent.run import AgentExecution
from reef.train.harness_backend.calls import StepCalls, call_tokens
from reef.train.harness_backend.contract import Method, StepContext
from reef.train.harness_backend.records import CALLS_FILE, StepRecord, bounded
from reef.train.harness_backend.state import HarnessState
from reef.train.harness_backend.tools import StepTools
from reef.train.harness_backend.workspace import StepOutput, Workspace
from reef.train.types import JobData


@dataclass(frozen=True)
class Evolution:
    """The method of one scenario's harness, and what each of its steps runs with.

    Values: ``method`` is the deployment's ``Method``; ``descriptor`` is the
    adapter whose tree it edits; ``history`` and ``records`` are the
    scenario's history and step records; ``agent`` is the deployment's
    coding agent, ``None`` when it configured none; ``max_episodes`` ≥ 0
    caps the episodes the method's ``tools.evaluate`` runs in one step, 0 for
    no cap.
    Purpose: ``HarnessBackend`` makes one and calls ``propose`` in each step.
    """

    method: Method
    descriptor: AdapterDescriptor
    history: History
    records: StepRecord
    agent: AgentExecution | None
    max_episodes: int

    def __post_init__(self) -> None:
        value = self.max_episodes
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("max_episodes must be an integer of at least 0 (0 turns it off)")

    def propose(
        self,
        workspace: Workspace,
        *,
        head: HarnessTree,
        evaluation_tasks: tuple[str, ...],
        data: JobData,
        state: HarnessState,
        evaluator: HarnessEpisodeEvaluator | None,
        models: ModelBindings,
        calls: StepCalls,
        metrics: dict[str, Any],
    ) -> PreparedStep:
        """Run the method in ``workspace`` on ``head``: the step's candidate, or the skip and its reason.

        The candidate is evaluated on ``evaluation_tasks``. ``state`` is the step's state, and the returned step's state carries the method's files from ``state/``.
        ``metrics`` gains the method's calls, episodes and words.
        """
        tools = StepTools(
            workspace=workspace,
            descriptor=self.descriptor,
            evaluator=evaluator,
            history=self.history,
            head=head.checksum(),
            models=models,
            step_calls=calls,
            max_episodes=self.max_episodes,
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
        workspace.write_output(clip_record_text(output.summary), bounded(output.notes), output.requires)
        if output.summary:
            metrics["method_summary"] = clip_record_text(output.summary)
        if output.notes:
            metrics["method_notes"] = bounded(output.notes)
        if output.refused:
            metrics["method_output_refused"] = dict(output.refused)
        return output


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


__all__ = ["Evolution", "screened_requires", "skipped"]
