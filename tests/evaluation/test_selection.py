"""Contracts for evaluating and selecting produced updates.

A candidate-evaluation plugin is one class that both measures (``evaluate``)
and decides (``decide``). The built-in policies are abstract mixins a plugin
composes at class-definition time — :class:`AlwaysSelectMixin`,
:class:`RegressionCheckMixin`, cordis's :class:`ScoreComparisonMixin` — paired
with :class:`BackendEvaluateMixin` when the measurement comes from the training
backend. Each implements one half of the contract and leaves the other
abstract, so only the pairing is instantiable.
"""

from __future__ import annotations

import pytest

from reef.evaluation import (
    AlwaysSelectMixin,
    BackendAlwaysSelectPlugin,
    BackendEvaluateMixin,
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    RegressionCheckMixin,
    SelectionDecision,
    UpdateCandidate,
)
from reef.evaluation.harness import FloorMixin, FloorPlugin, ScoreComparisonMixin, ScoreComparisonPlugin
from reef.runtime.interfaces import ActivatedModel, ModelCandidate
from reef.train.backend import CandidateBackend


class DecideOnlyBackend(CandidateEvaluator):
    """Stands in for the training backend: these cases exercise ``decide`` only."""

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise AssertionError("this case exercises decide(), not evaluate()")


def evaluation() -> EvaluationResult:
    return EvaluationResult(
        evaluator="held-out-suite",
        evaluator_version="2026-08-21",
        metrics={"candidate_reward": 0.8, "current_reward": 0.7},
    )


def test_evaluation_and_selection_remain_independent_nominal_capabilities() -> None:
    class Selector(CandidateSelector):
        def decide(self, candidate, result):
            return SelectionDecision(True, "test", "accepted", result)

    evaluator = DecideOnlyBackend()
    selector = Selector()
    assert isinstance(evaluator, CandidateEvaluator)
    assert not isinstance(evaluator, (CandidateSelector, CandidateEvaluationPlugin))
    assert isinstance(selector, CandidateSelector)
    assert not isinstance(selector, (CandidateEvaluator, CandidateEvaluationPlugin))

    class DuckPlugin:
        def evaluate(self, candidate):
            return evaluation()

        def decide(self, candidate, result):
            return selector.decide(candidate, result)

    assert not isinstance(DuckPlugin(), (CandidateEvaluator, CandidateSelector, CandidateEvaluationPlugin))


def test_built_ins_explicitly_implement_their_public_contracts() -> None:
    assert issubclass(CandidateBackend, CandidateEvaluator)
    # The shipped plugins are whole plugins: they evaluate and decide.
    for plugin in (BackendAlwaysSelectPlugin, ScoreComparisonPlugin, FloorPlugin):
        assert issubclass(plugin, CandidateEvaluationPlugin)
        assert callable(plugin.evaluate)
        assert callable(plugin.decide)
    # The policies are mixins: they supply decide() and stay abstract on the half
    # they do not implement, so a mixin cannot stand up as a plugin on its own.
    for mixin in (AlwaysSelectMixin, RegressionCheckMixin, ScoreComparisonMixin, FloorMixin):
        assert callable(mixin.decide)
        assert mixin.__abstractmethods__ == frozenset({"evaluate"})
    assert BackendEvaluateMixin.__abstractmethods__ == frozenset({"decide"})
    with pytest.raises(TypeError, match="abstract"):
        AlwaysSelectMixin()  # type: ignore[abstract]


def test_always_select_returns_an_explainable_structured_decision() -> None:
    candidate = UpdateCandidate("job-7")
    decision = BackendAlwaysSelectPlugin(DecideOnlyBackend()).decide(candidate, evaluation())

    assert decision.selected is True
    assert decision.policy == "always"
    assert decision.to_dict()["evaluation"]["metrics"] == {
        "candidate_reward": 0.8,
        "current_reward": 0.7,
    }


def test_a_plugin_mixes_its_measurement_and_its_decision() -> None:
    calls: list[tuple[str, object]] = []

    class DemoPlugin(CandidateEvaluationPlugin):
        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            calls.append(("evaluate", candidate))
            return evaluation()

        def decide(self, candidate: UpdateCandidate, result: EvaluationResult) -> SelectionDecision:
            calls.append(("decide", result))
            return SelectionDecision(
                selected=True,
                policy="demo",
                reason=f"selected {candidate.candidate_id}",
                evaluation=result,
            )

    candidate = UpdateCandidate("job-7")
    plugin = DemoPlugin()
    result = plugin.evaluate(candidate)
    decision = plugin.decide(candidate, result)

    assert decision.selected is True
    assert calls == [("evaluate", candidate), ("decide", decision.evaluation)]


def test_the_backend_always_select_plugin_measures_through_the_backend() -> None:
    calls = []

    class Backend:
        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            calls.append(candidate)
            return evaluation()

    candidate = UpdateCandidate("job-7")
    backend = Backend()
    plugin = BackendAlwaysSelectPlugin(backend)

    result = plugin.evaluate(candidate)
    decision = plugin.decide(candidate, result)

    assert calls == [candidate]
    assert decision.selected is True
    assert decision.policy == "always"


def test_a_policy_mixin_pairs_with_the_backend_evaluate_mixin() -> None:
    class Backend:
        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            return EvaluationResult(
                evaluator="pairs",
                evaluator_version="1",
                metrics={"candidate_scores": (1.0, 1.0), "current_scores": (0.0, 0.0)},
            )

    plugin = ScoreComparisonPlugin(Backend())
    candidate = UpdateCandidate("job-7")
    decision = plugin.decide(candidate, plugin.evaluate(candidate))

    assert decision.selected is True
    assert decision.policy == "score_comparison"


def test_rejected_decision_is_not_selected() -> None:
    decision = SelectionDecision(
        selected=False,
        policy="minimum-improvement",
        reason="candidate did not clear the configured delta",
        evaluation=evaluation(),
    )
    assert decision.selected is False


@pytest.mark.parametrize("selected", [True, False])
def test_selection_decision_serializes_selected(selected: bool) -> None:
    result = evaluation()
    decision = SelectionDecision(
        selected=selected,
        policy="demo",
        reason="compared candidate scores",
        evaluation=result,
        metrics={"wins": 2, "losses": 1},
    )

    assert decision.selected is selected
    assert decision.to_dict() == {
        "selected": selected,
        "policy": "demo",
        "reason": "compared candidate scores",
        "evaluation": result.to_dict(),
        "metrics": {"wins": 2, "losses": 1},
    }


def test_candidate_identity_is_validated() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        UpdateCandidate("")


@pytest.mark.parametrize("selected", ["select", "reject", 1, None])
def test_decision_rejects_non_boolean_selected(selected: object) -> None:
    with pytest.raises(ValueError, match="selected must be a boolean"):
        SelectionDecision(
            selected=selected,  # type: ignore[arg-type]
            policy="demo",
            reason="invalid selection flag",
            evaluation=evaluation(),
        )


def test_model_candidate_records_the_unactivated_checkpoint() -> None:
    candidate = ModelCandidate(
        candidate_id="job-7",
        training_job_id="job-7",
        checkpoint_path="/checkpoints/job-7",
        current_runtime_load_id="inc:6",
    )

    assert candidate.current_runtime_load_id == "inc:6"
    assert ActivatedModel(candidate.candidate_id, "inc:7").runtime_load_id == "inc:7"


def test_backend_evaluate_mixin_delegates_to_the_plugins_backend() -> None:
    class Backend:
        def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
            return evaluation()

    class Plugin(AlwaysSelectMixin, BackendEvaluateMixin, CandidateEvaluationPlugin):
        def __init__(self, backend: object) -> None:
            super().__init__()
            self._candidate_backend = backend

    candidate = UpdateCandidate("job-7")
    plugin = Plugin(Backend())
    assert plugin.evaluate(candidate).evaluator == "held-out-suite"
    assert plugin.decide(candidate, plugin.evaluate(candidate)).selected is True
