"""How a step's decision becomes a release, for model weights and harness trees alike.

The candidate evaluation (``reef.evaluation``) measures a candidate and decides
whether Reef serves it. ``ReleaseRules.release`` turns that decision into the
step's result, and the scenario committer publishes the result:

==========  ===============  =====================================================
decision    held for review  release
==========  ===============  =====================================================
selected    no               served at once
selected    yes              pending: published, served when a person promotes it
rejected    any              pending when the rules keep rejections, else none
==========  ===============  =====================================================

A weight trainer keeps rejections (``reef.train.jobs``): a checkpoint is the
only copy of a job's training, and a person may still promote it. A harness
backend drops them, because its history keeps every candidate tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import SelectionDecision
from reef.train.types import TrainStepResult

#: The ``publish`` values: ``auto`` serves a selected candidate at once, ``review`` holds it for a person.
PUBLISH_MODES = ("auto", "review")


@dataclass(frozen=True)
class ReleaseRules:
    """The rules that turn one trainer's decisions into releases.

    Values: ``publish`` is one of ``PUBLISH_MODES``. ``keep_rejected`` is true
    when a rejected candidate is still a pending release.
    Purpose: the weight job loop uses ``WEIGHT_RELEASES``; the harness recipe
    builds its rules from ``evolution.publish``. Each calls ``release`` once
    per decided candidate.
    """

    publish: str = "auto"
    keep_rejected: bool = False

    def __post_init__(self) -> None:
        if self.publish not in PUBLISH_MODES:
            raise ValueError(f"publish must be one of {', '.join(PUBLISH_MODES)}")
        if not isinstance(self.keep_rejected, bool):
            raise TypeError("keep_rejected must be a boolean")

    def released(self, decision: SelectionDecision) -> bool:
        """Whether ``decision`` makes a release: a selected candidate does, a rejected one when the rules keep it."""
        return decision.selected or self.keep_rejected

    def release(
        self,
        candidate_id: str,
        decision: SelectionDecision,
        *,
        state: Mapping[str, Any],
        metrics: Mapping[str, Any],
        artifact: Artifact | None,
        held: bool = False,
        training_job_id: str | None = None,
    ) -> TrainStepResult:
        """The step's result for ``decision`` on candidate ``candidate_id``, by the table of this module.

        The result's metrics are ``metrics`` with ``selected`` and ``selection``: the candidate id, the
        decision's ``selected``, ``policy``, ``reason`` and ``metrics``, and the ``evaluation`` that the decision
        read without its metrics (``evaluator``, ``evaluator_version``, ``metadata``). The caller records the
        evaluation's metrics in ``metrics``, as much of them as it keeps. ``held`` holds a selected candidate for a person even under ``publish: auto``. ``artifact`` is
        the candidate's bytes, given exactly when ``released(decision)``. Raise ``ValueError`` when it is not,
        or when ``metrics`` already holds ``selected`` or ``selection``.
        """
        released = self.released(decision)
        if (artifact is not None) != released:
            raise ValueError(
                "a released candidate needs its artifact, and a candidate that makes no release takes none"
            )
        if "selected" in metrics or "selection" in metrics:
            raise ValueError("the step's metrics already hold a decision")
        recorded = {
            **metrics,
            "selected": decision.selected,
            "selection": {
                "candidate_id": candidate_id,
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
        pending = released and (not decision.selected or held or self.publish == "review")
        return TrainStepResult(state, recorded, artifact=artifact, training_job_id=training_job_id, pending=pending)


#: A weight trainer's rules: a selected checkpoint serves at once, and a rejected one waits for a promote.
WEIGHT_RELEASES = ReleaseRules(publish="auto", keep_rejected=True)


__all__ = ["PUBLISH_MODES", "WEIGHT_RELEASES", "ReleaseRules"]
