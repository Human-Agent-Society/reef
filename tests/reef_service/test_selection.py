"""Contracts for evaluating and selecting produced updates."""

from __future__ import annotations

import pytest

from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.train.backend import TrainingBackend
from reef.train.cordis_backend import ScoreComparisonSelector
from reef.train.evaluation import (
    AlwaysSelect,
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    DefaultCandidateEvaluationPlugin,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)


def test_built_ins_explicitly_implement_their_public_contracts() -> None:
    assert issubclass(AlwaysSelect, CandidateSelector)
    assert issubclass(ScoreComparisonSelector, CandidateSelector)
    assert issubclass(TrainingBackend, CandidateEvaluator)
    assert issubclass(DefaultCandidateEvaluationPlugin, CandidateEvaluationPlugin)


def evaluation() -> EvaluationResult:
    return EvaluationResult(
        evaluator="held-out-suite",
        evaluator_version="2026-08-21",
        metrics={"candidate_reward": 0.8, "current_reward": 0.7},
    )


def test_always_select_returns_an_explainable_structured_decision() -> None:
    candidate = UpdateCandidate("job-7", current_version="version-6")
    decision = AlwaysSelect().decide(candidate, evaluation())

    assert decision.selected is True
    assert decision.outcome == "select"
    assert decision.policy == "always"
    assert decision.to_dict()["evaluation"]["metrics"] == {
        "candidate_reward": 0.8,
        "current_reward": 0.7,
    }


def test_default_plugin_composes_evaluator_and_selector() -> None:
    calls: list[tuple[str, object]] = []

    class Backend:
        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            calls.append(("evaluate", candidate))
            return evaluation()

    class Selector:
        def decide(self, candidate: UpdateCandidate, result: EvaluationResult) -> SelectionDecision:
            calls.append(("decide", result))
            return SelectionDecision(
                outcome="select",
                policy="demo",
                policy_version="1",
                reason=f"selected {candidate.candidate_id}",
                evaluation=result,
            )

    candidate = UpdateCandidate("job-7")
    evaluator = DefaultCandidateEvaluationPlugin(Backend(), Selector())
    result = evaluator.evaluate(candidate)
    decision = evaluator.decide(candidate, result)

    assert decision.selected is True
    assert calls == [("evaluate", candidate), ("decide", decision.evaluation)]


def test_default_plugin_uses_always_select_when_selector_is_omitted() -> None:
    calls = []

    class Backend:
        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            calls.append(candidate)
            return evaluation()

    candidate = UpdateCandidate("job-7")
    backend = Backend()
    evaluator = DefaultCandidateEvaluationPlugin(backend)

    result = evaluator.evaluate(candidate)
    decision = evaluator.decide(candidate, result)

    assert evaluator.evaluator is backend
    assert calls == [candidate]
    assert decision.selected is True
    assert decision.policy == "always"


def test_rejected_decision_is_not_selected() -> None:
    decision = SelectionDecision(
        outcome="reject",
        policy="minimum-improvement",
        policy_version="2",
        reason="candidate did not clear the configured delta",
        evaluation=evaluation(),
    )
    assert decision.selected is False


@pytest.mark.parametrize(
    ("candidate_id", "current_version", "message"),
    [
        ("", None, "candidate_id"),
        ("job-1", "", "current_version"),
    ],
)
def test_candidate_identity_is_validated(candidate_id, current_version, message) -> None:
    with pytest.raises(ValueError, match=message):
        UpdateCandidate(candidate_id, current_version=current_version)


def test_decision_rejects_an_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="selection outcome"):
        SelectionDecision(
            outcome="defer",  # type: ignore[arg-type]
            policy="demo",
            policy_version="1",
            reason="not terminal",
            evaluation=evaluation(),
        )


def test_model_candidate_records_the_unactivated_checkpoint() -> None:
    candidate = ModelCandidate(
        candidate_id="job-7",
        training_job_id="job-7",
        checkpoint_path="/checkpoints/job-7",
        current_weight_version="inc:6",
    )

    assert candidate.current_weight_version == "inc:6"
    assert ActivatedModel(candidate.candidate_id, "inc:7").weight_version == "inc:7"
