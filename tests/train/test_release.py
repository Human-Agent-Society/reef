"""The release table: how a decision becomes a release, for weights and harness trees alike."""

from __future__ import annotations

import pytest

from reef.artifact.artifact import Artifact
from reef.core.evaluation import EvaluationResult, SelectionDecision
from reef.train.release import WEIGHT_RELEASES, ReleaseRules

EVALUATION = EvaluationResult(evaluator="episodes", evaluator_version="1", metrics={}, metadata={"tasks": 3})


@pytest.mark.parametrize(
    ("rules", "selected", "held", "release"),
    [
        (ReleaseRules(), True, False, "served"),
        (ReleaseRules(), True, True, "pending"),
        (ReleaseRules(publish="review"), True, False, "pending"),
        (ReleaseRules(), False, False, "none"),
        (WEIGHT_RELEASES, True, False, "served"),
        (WEIGHT_RELEASES, False, False, "pending"),
    ],
)
def test_a_decision_is_served_held_or_dropped(rules, selected, held, release, tmp_path) -> None:
    decision = SelectionDecision(selected, "compare", "a reason", EVALUATION, {"wins": 2})
    artifact = Artifact.local(tmp_path) if rules.released(decision) else None
    result = rules.release("c1", decision, state={}, metrics={"traces": 1}, artifact=artifact, held=held)
    outcome = "none" if result.artifact is None else ("pending" if result.pending else "served")
    assert outcome == release
    assert result.metrics["selected"] is selected and result.metrics["traces"] == 1
    assert result.metrics["selection"] == {
        "candidate_id": "c1",
        "selected": selected,
        "policy": "compare",
        "reason": "a reason",
        "metrics": {"wins": 2},
        "evaluation": {"evaluator": "episodes", "evaluator_version": "1", "metadata": {"tasks": 3}},
    }


def test_the_artifact_comes_exactly_with_a_release(tmp_path) -> None:
    rejected = SelectionDecision(False, "compare", "a reason", EVALUATION)
    with pytest.raises(ValueError, match="takes none"):
        ReleaseRules().release("c1", rejected, state={}, metrics={}, artifact=Artifact.local(tmp_path))
    with pytest.raises(ValueError, match="needs its artifact"):
        WEIGHT_RELEASES.release("c1", rejected, state={}, metrics={}, artifact=None)
