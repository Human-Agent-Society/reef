"""How a step's decision becomes a release: review, recheck, and the state that crosses steps.

The evaluation decides whether a candidate tree is better than the head
(``reef.evaluation.harness``). This module turns that decision into what the
scenario committer publishes:

``settle``
    Fold the decision into ``HarnessState`` and the history, and make a
    selected tree an ``Artifact.local`` of its source tree. The committer
    publishes it as a git commit; a pending one waits for a promote.
``ReleaseRules.held_for_review``
    Whether a selected tree waits for a person: under ``publish: review``, or
    when it changes a part that the deployment or the adapter reviews.
``recheck_candidate``
    Every ``recheck_every`` steps, or when the evaluation context changed,
    the tree that the latest publish replaced is a candidate again, so that
    selecting it rolls back.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import SelectionDecision
from reef.evaluation.harness import DecisionRecord, HarnessCandidate, History
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree import HarnessTree, TreePatch, validate_tree
from reef.train.backend import ComponentReleases, PreparedStep
from reef.train.harness_backend.records import bounded
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData, TrainStepResult


@dataclass(frozen=True)
class ReleaseRules:
    """The deployment's rules for turning a selected tree into a release.

    Values: ``publish`` is ``"auto"``, which serves a selected tree at once,
    or ``"review"``, which holds every selected tree for a person.
    ``review_parts`` name the parts of the adapter's tree whose change waits
    for a person even under ``"auto"``, besides the parts that always do.
    ``recheck_every`` ≥ 0 re-evaluates the latest publish against the tree
    it replaced every that many steps; 0 turns the recheck off.
    Purpose: ``HarnessEvolutionRecipe`` builds it from ``evolution.publish``,
    ``evolution.review_parts`` and ``evolution.evaluation.recheck_every``;
    the backend reads it when it prepares and settles a step.
    """

    publish: str = "auto"
    review_parts: frozenset[str] = frozenset()
    recheck_every: int = 0

    def __post_init__(self) -> None:
        if self.publish not in ("auto", "review"):
            raise ValueError("publish must be 'auto' or 'review'")
        if isinstance(self.recheck_every, bool) or not isinstance(self.recheck_every, int) or self.recheck_every < 0:
            raise ValueError("recheck_every must be an integer of at least 0 (0 turns the recheck off)")

    def held_for_review(self, patch: TreePatch, descriptor: AdapterDescriptor) -> bool:
        """Whether a selected ``patch`` waits for a person: under ``publish: review``, or when it changes a part
        that ``review_parts`` names or that ``descriptor`` always reviews."""
        if self.publish == "review":
            return True
        for path in patch.changes:
            location = descriptor.layout.locate(path)
            if location is not None and (location.part.review or location.part.name in self.review_parts):
                return True
        return False

    def recheck_due(self, state: HarnessState, step: int, context: str | None) -> bool:
        """Whether automatic step ``step`` re-evaluates the latest publish against the tree it replaced: on the
        cadence, or at once when the evaluation context is no longer the one the publish was decided in."""
        if not self.recheck_every or state.rollback_release_id is None:
            return False
        return step % self.recheck_every == 0 or state.rollback_context != context


def recheck_candidate(
    *,
    releases: ComponentReleases,
    history: History,
    descriptor: AdapterDescriptor,
    data: JobData,
    state: HarnessState,
    head: HarnessTree,
    evaluation_tasks: tuple[str, ...],
    step: int,
    context: str | None,
    metrics: dict[str, Any],
) -> PreparedStep:
    """The step whose candidate is the tree the latest publish replaced; it wins only when the head got worse.

    ``state.rollback_release_id`` names that tree. When today's checks refuse it, no recheck can roll back to it:
    the step drops the target and skips.
    """
    if state.rollback_release_id is None:
        raise ValueError("a recheck needs the release the latest publish replaced")
    target = releases.release(state.rollback_release_id)
    target_tree = HarnessTree(target.files)
    metrics.update(
        {
            "recheck": True,
            "recheck_reason": "drift" if state.rollback_context != context else "cadence",
            "recheck_release_id": target.release_id,
        }
    )
    validation = validate_tree(target_tree, descriptor)
    if not validation.passed:
        dropped = replace(state, rollback_release_id=None, rollback_context=None)
        return PreparedStep.skipped(state=dropped.to_json(), metrics={**metrics, "skipped": validation.message()})
    history.add_candidate(target_tree, step=step, origin="release")
    candidate = HarnessCandidate(
        candidate_id=f"{data.data_id}:recheck",
        candidate_tree=target_tree,
        current_tree=head,
        evaluation_tasks=evaluation_tasks,
        step=step,
        origin="release",
    )
    return PreparedStep.with_candidate(candidate, state=state.to_json(), metrics=metrics)


def settle(
    *,
    candidate: HarnessCandidate,
    decision: SelectionDecision,
    prepared_state: Mapping[str, Any],
    prepared_metrics: Mapping[str, Any],
    rules: ReleaseRules,
    history: History,
    descriptor: AdapterDescriptor,
    context: str | None,
) -> TrainStepResult:
    """The step's result: ``decision`` folded into the state and the history, and a selected tree as the artifact.

    The head the step prepared against is ``prepared_metrics["head_release_id"]``; no rollback, promote or import
    moves it while a step runs. The commit keeps the evaluation's sums, not its score vectors: the history holds
    every episode.
    """
    state = HarnessState.from_json(prepared_state)
    metrics = dict(prepared_metrics)
    evaluation = dict(decision.evaluation.metrics)
    evaluation.pop("candidate_scores", None)
    evaluation.pop("current_scores", None)
    # A publish keeps the tree it replaced as the recheck's target; a recheck that rolls back consumes it.
    rollback_release_id, rollback_context = state.rollback_release_id, state.rollback_context
    if candidate.origin == "release":
        metrics["rolled_back"] = decision.selected
        if decision.selected:
            rollback_release_id, rollback_context = None, None
        else:
            # The head held in this context, so the next recheck waits for the cadence or another context change.
            rollback_context = context
    elif decision.selected and rules.recheck_every and context is not None:
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
    history.add_decision(
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
            "patch_parts": patch_parts(candidate.patch, descriptor),
        }
    )
    if not decision.selected:
        return TrainStepResult(state.to_json(), metrics)
    directory = Path(tempfile.mkdtemp(prefix="reef-harness-"))
    candidate.candidate_tree.write(directory)
    # A rollback serves a tree that was published before, so only publish: review holds it.
    pending = (
        rules.publish == "review"
        if candidate.origin == "release"
        else rules.held_for_review(candidate.patch, descriptor)
    )
    return TrainStepResult(state.to_json(), metrics, artifact=Artifact.local(directory), pending=pending)


def patch_parts(patch: TreePatch, descriptor: AdapterDescriptor) -> dict[str, str]:
    """The part of ``descriptor``'s tree each path of ``patch`` is, for the step pages."""
    located = {path: descriptor.layout.locate(path) for path in patch.changes}
    return {path: location.part.name for path, location in located.items() if location is not None}


__all__ = ["ReleaseRules", "patch_parts", "recheck_candidate", "settle"]
