"""The ``RegressionGateMixin``: publish the climb, reject the regression.

The mixin is the reusable half of best-checkpoint selection — it supplies a
plugin's ``decide()`` as a pure policy over one scalar metric, with no probe,
model, or runtime. These tests mix it into a trivial plugin and drive it with
canned ``EvaluationResult``s, so its admit/reject decisions and the running best
it tracks are pinned directly.
"""

from __future__ import annotations

import pytest

from reef.core.evaluation import CandidateEvaluationPlugin
from reef.train.evaluation import EvaluationResult, RegressionGateMixin, UpdateCandidate


class _GatedPlugin(RegressionGateMixin, CandidateEvaluationPlugin):
    """Minimal plugin: the mixin's decide(), a stub evaluate the tests never call."""

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:  # pragma: no cover - unused
        raise NotImplementedError


def _evaluation(**metrics: float) -> EvaluationResult:
    return EvaluationResult(evaluator="probe", evaluator_version="1", metrics=metrics)


def _decide(plugin: _GatedPlugin, **metrics: float) -> str:
    candidate = UpdateCandidate(candidate_id="c")
    return plugin.decide(candidate, _evaluation(**metrics)).outcome


def test_the_first_candidate_seeds_the_bar_and_is_selected() -> None:
    plugin = _GatedPlugin(metric="score", margin=0.1)
    assert _decide(plugin, score=0.2) == "select"


def test_it_admits_the_climb_and_rejects_a_regression() -> None:
    plugin = _GatedPlugin(metric="score", margin=0.15)
    assert _decide(plugin, score=0.3) == "select"  # best -> 0.3
    assert _decide(plugin, score=0.6) == "select"  # best -> 0.6
    # 0.4 is 0.2 below the 0.6 best, past the 0.15 margin.
    assert _decide(plugin, score=0.4) == "reject"
    # A reject does not move the bar: 0.5 is measured against 0.6, and
    # 0.5 >= 0.6 - 0.15 = 0.45, so it is still admitted.
    assert _decide(plugin, score=0.5) == "select"
    # The best never fell below the peak the rejects protected.
    assert plugin._best == pytest.approx(0.6)


def test_a_within_margin_dip_is_admitted() -> None:
    plugin = _GatedPlugin(metric="score", margin=0.15)
    assert _decide(plugin, score=0.6) == "select"
    # 0.5 >= 0.6 - 0.15 = 0.45, so it stays admitted.
    assert _decide(plugin, score=0.5) == "select"


def test_a_new_high_raises_the_bar() -> None:
    plugin = _GatedPlugin(metric="score", margin=0.1)
    assert _decide(plugin, score=0.5) == "select"
    assert _decide(plugin, score=0.9) == "select"
    assert plugin._best == pytest.approx(0.9)
    assert _decide(plugin, score=0.7) == "reject"  # 0.7 < 0.9 - 0.1 = 0.8


def test_lower_is_better_gates_a_loss() -> None:
    plugin = _GatedPlugin(metric="loss", margin=0.2, higher_is_better=False)
    assert _decide(plugin, loss=1.0) == "select"  # best loss -> 1.0
    assert _decide(plugin, loss=0.7) == "select"  # improved -> best 0.7
    # A loss climbing back to 1.0 is 0.3 worse than the 0.7 best, past 0.2.
    assert _decide(plugin, loss=1.0) == "reject"
    # Within margin (0.85 <= 0.7 + 0.2) is still admitted.
    assert _decide(plugin, loss=0.85) == "select"


def test_the_reject_reason_reports_the_metric_and_best() -> None:
    plugin = _GatedPlugin(metric="clean_rate", margin=0.1)
    candidate = UpdateCandidate(candidate_id="c")
    plugin.decide(candidate, _evaluation(clean_rate=0.8))
    decision = plugin.decide(candidate, _evaluation(clean_rate=0.3))
    assert decision.outcome == "reject"
    assert "clean_rate" in decision.reason
    assert decision.metrics["best"] == pytest.approx(0.8)
    assert decision.metrics["value"] == pytest.approx(0.3)


def test_a_missing_or_non_numeric_metric_is_a_clear_error() -> None:
    plugin = _GatedPlugin(metric="score")
    candidate = UpdateCandidate(candidate_id="c")
    with pytest.raises(ValueError, match="missing or non-numeric"):
        plugin.decide(candidate, _evaluation(other=1.0))


def test_construction_validates_the_metric_and_margin() -> None:
    with pytest.raises(ValueError, match="non-empty metric"):
        _GatedPlugin(metric="")
    with pytest.raises(ValueError, match="non-negative"):
        _GatedPlugin(metric="score", margin=-0.1)
