"""Contracts for evaluating and selecting produced updates.

A candidate-evaluation plugin is one class that both measures (``evaluate``)
and decides (``decide``). The built-in policies are abstract mixins a plugin
composes at class-definition time — :class:`AlwaysSelectMixin` and
:class:`RegressionCheckMixin` — paired with :class:`BackendEvaluateMixin` when
the measurement comes from the training backend. Each implements one half of
the contract and leaves the other abstract, so only the pairing is
instantiable. The harness publication policies (``compare``, ``floor`` and
``none``) are whole plugins that measure through a harness evaluator.
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
from reef.evaluation.harness import (
    EVALUATION_SIDES,
    ComparePolicy,
    FloorPolicy,
    HarnessCandidateEvaluator,
    NoEvaluationPolicy,
    RolloutPolicy,
)
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
    for plugin in (BackendAlwaysSelectPlugin, ComparePolicy, FloorPolicy, NoEvaluationPolicy):
        assert issubclass(plugin, CandidateEvaluationPlugin)
        assert callable(plugin.evaluate)
        assert callable(plugin.decide)
    # The policies are mixins: they supply decide() and stay abstract on the half
    # they do not implement, so a mixin cannot stand up as a plugin on its own.
    for mixin in (AlwaysSelectMixin, RegressionCheckMixin):
        assert callable(mixin.decide)
        assert mixin.__abstractmethods__ == frozenset({"evaluate"})
    assert BackendEvaluateMixin.__abstractmethods__ == frozenset({"decide"})
    # The harness rollout policies share their measurement, and each one decides.
    assert RolloutPolicy.__abstractmethods__ == frozenset({"decide"})
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


def task_scores(*tasks: float | tuple[float | None, ...] | None) -> list[dict[str, object]]:
    """``RolloutScores`` JSON, one task per value: a score, ``None`` for a failed rollout, or a tuple of runs."""
    rows = []
    for index, runs in enumerate(tasks):
        values = runs if isinstance(runs, tuple) else (runs,)
        rows.append({"task": f"t{index}", "scores": ["failed" if value is None else value for value in values]})
    return rows


class PairedScores(HarnessCandidateEvaluator):
    """Stands in for the harness evaluator: every evaluation returns the same scores, as ``RolloutScores`` JSON."""

    def __init__(self, candidate_scores: list[dict[str, object]], current_scores: list[dict[str, object]]) -> None:
        self.metrics = {"candidate_scores": candidate_scores, "current_scores": current_scores}
        self.sides: list[tuple[str, ...]] = []

    def evaluate(self, candidate: UpdateCandidate, *, sides=EVALUATION_SIDES) -> EvaluationResult:
        self.sides.append(tuple(sides))
        return EvaluationResult(evaluator="pairs", evaluator_version="1", metrics=self.metrics)


def test_the_compare_policy_decides_by_wins_or_by_the_mean_score() -> None:
    # One win, and one pair whose candidate rollout failed: the failed pair is dropped, not lost.
    evaluator = PairedScores(task_scores(1.0, None), task_scores(0.4, 0.4))
    candidate = UpdateCandidate("job-7")
    wins = ComparePolicy(evaluator)
    decision = wins.decide(candidate, wins.evaluate(candidate))
    assert evaluator.sides == [EVALUATION_SIDES]
    assert (decision.selected, decision.policy) == (True, "compare")
    assert decision.metrics == {
        "statistic": "wins",
        "min_margin": 0.0,
        "wins": 1,
        "losses": 0,
        "ties": 0,
        "dropped": 1,
    }
    assert decision.reason == "candidate won 1 task pairings and lost 0; 1 pairings with a failed rollout were dropped"
    margin = ComparePolicy(evaluator, min_margin=1)
    assert margin.decide(candidate, margin.evaluate(candidate)).selected is False

    # The mean averages the rollouts that ran; a failed rollout does not count as 0.
    mean = ComparePolicy(evaluator, statistic="mean")
    decision = mean.decide(candidate, mean.evaluate(candidate))
    assert decision.selected is True
    assert (decision.metrics["candidate_mean"], decision.metrics["current_mean"]) == (1.0, 0.4)
    assert decision.reason.startswith("candidate scored 1.0000 against 0.4000")
    for min_margin, selected in ((0.5, True), (0.7, False)):
        policy = ComparePolicy(evaluator, statistic="mean", min_margin=min_margin)
        assert policy.decide(candidate, policy.evaluate(candidate)).selected is selected
    # A rollout that ran but gave no answer scores 0, which is a loss and lowers the mean.
    unanswered = PairedScores(task_scores(1.0, 0.0), task_scores(0.4, 0.4))
    assert ComparePolicy(unanswered).decide(candidate, unanswered.evaluate(candidate)).metrics["losses"] == 1
    policy = ComparePolicy(unanswered, statistic="mean", min_margin=0.2)
    assert policy.decide(candidate, policy.evaluate(candidate)).selected is False


def test_the_compare_policy_refuses_unpaired_scores_and_publishes_nothing_unscored() -> None:
    candidate = UpdateCandidate("job-7")
    unscored = ComparePolicy(PairedScores([], []), statistic="mean")
    decision = unscored.decide(candidate, unscored.evaluate(candidate))
    assert decision.selected is False and decision.reason == "no evaluation task was scored"
    # Every candidate rollout failed: no pair is left to compare.
    failed = PairedScores(task_scores(None, None), task_scores(0.4, 0.4))
    for statistic in ("wins", "mean"):
        policy = ComparePolicy(failed, statistic=statistic)
        assert policy.decide(candidate, policy.evaluate(candidate)).selected is False
    unpaired = PairedScores(task_scores(1.0), [{"task": "other", "scores": [1.0]}])
    with pytest.raises(ValueError, match="same tasks"):
        ComparePolicy(unpaired).decide(candidate, unpaired.evaluate(candidate))
    with pytest.raises(ValueError, match="statistic must be one of wins, mean"):
        ComparePolicy(PairedScores([], []), statistic="median")
    with pytest.raises(ValueError, match="min_margin must be a finite number of at least 0"):
        ComparePolicy(PairedScores([], []), min_margin=-1)


def test_the_floor_policy_scores_the_candidate_alone_and_selects_when_every_task_meets_the_floor() -> None:
    candidate = UpdateCandidate("job-7")

    def decide(scores: list[dict[str, object]], floor_score: float = 1.0) -> SelectionDecision:
        evaluator = PairedScores(scores, [])
        policy = FloorPolicy(evaluator, floor_score=floor_score)
        decision = policy.decide(candidate, policy.evaluate(candidate))
        assert evaluator.sides == [("candidate",)]
        return decision

    met = decide(task_scores(1.0, 1.5))
    assert met.selected and met.policy == "floor"
    assert met.metrics == {"passed": 2, "failed": 0, "failed_rollouts": 0, "floor_score": 1.0}
    assert met.reason == "candidate met the floor on all 2 tasks"

    # A rollout that gave no answer scores 0 and misses the floor.
    missed = decide(task_scores(1.0, 0.5, 0.0))
    assert not missed.selected
    assert missed.metrics == {"passed": 1, "failed": 2, "failed_rollouts": 0, "floor_score": 1.0}
    assert missed.reason == "candidate missed the floor on 2 of 3 tasks"

    # A failed rollout fails no task, but a task whose rollouts all failed does not pass.
    partly = decide(task_scores((1.0, None)))
    assert partly.selected and partly.metrics["failed_rollouts"] == 1
    unrun = decide(task_scores((1.0, 1.0), (None, None)))
    assert not unrun.selected
    assert unrun.metrics == {"passed": 1, "failed": 1, "failed_rollouts": 2, "floor_score": 1.0}

    assert decide(task_scores(0.5, 0.75), floor_score=0.5).selected
    assert decide(task_scores(0.5, 0.25), floor_score=0.5).metrics == {
        "passed": 1,
        "failed": 1,
        "failed_rollouts": 0,
        "floor_score": 0.5,
    }

    nothing = decide([])
    assert not nothing.selected and nothing.reason == "no evaluation task was scored"

    for bad in (0, -1.0, True, float("nan")):
        with pytest.raises(ValueError, match="floor_score must be a positive number"):
            FloorPolicy(PairedScores([], []), floor_score=bad)


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
