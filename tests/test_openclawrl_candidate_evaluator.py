"""OpenClaw-RL's candidate probe and its wiring to reef's RegressionGate.

The probe is the OpenClaw-RL-specific half — it scores a candidate's replies
with the benchmark's style criterion. The generic select/reject policy is
reef's ``RegressionGate`` (covered in ``test_regression_gate``). These tests pin
the probe's scoring and that ``build`` pairs it with the gate into a working
evaluate-then-decide plugin, using a fake runtime so no model is needed.
"""

from __future__ import annotations

import pytest

from recipes.openclawrl.candidate_evaluator import build
from reef.core.evaluation import UpdateCandidate

_CLEAN = (
    "She sold 48 clips in April. In May she sold half as many, so 48 divided by 2 "
    "equals 24. Altogether that is 48 plus 24 equals 72 clips."
)
_MARKDOWN = "**Answer:** 72\n- April: 48\n- May: 24\n"
_EMPTY = ""  # a no-reply / tool-loop turn


class _FakeEngine:
    def render_prompt(self, messages):
        # The probe only forwards these to probe_candidate, which the fake
        # overrides, so any stable token stand-in is fine.
        return tuple(messages[0]["content"])


class _FakeRuntime:
    """A runtime whose candidate probe returns whatever the test scripts."""

    def __init__(self) -> None:
        self.engine = _FakeEngine()
        self.replies: list[str] = []

    def probe_candidate(self, candidate_id, prompts, *, max_tokens):
        return list(self.replies)


def _plugin(margin: float = 0.17):
    runtime = _FakeRuntime()
    plugin = build(
        {"probe_size": 6, "max_tokens": 64, "regression_margin": margin},
        runtime=runtime,
        scenario="gsm8k",
        environ={},
    )
    return plugin, runtime


def _decide(plugin, runtime, replies):
    runtime.replies = replies
    candidate = UpdateCandidate(candidate_id="c")
    evaluation = plugin.evaluate(candidate)
    decision = plugin.decide(candidate, evaluation)
    return decision, evaluation


def test_the_probe_scores_clean_markdown_and_empty_replies() -> None:
    plugin, runtime = _plugin()
    # Six probes: three clean, one markdown, one empty, one clean -> 4/6 clean.
    _, evaluation = _decide(plugin, runtime, [_CLEAN, _CLEAN, _MARKDOWN, _EMPTY, _CLEAN, _CLEAN])
    assert evaluation.metrics["n_clean"] == 4
    assert evaluation.metrics["n_total"] == 6
    assert evaluation.metrics["clean_rate"] == pytest.approx(4 / 6)
    # The empty turn is not even an answer; markdown answered but not clean.
    assert evaluation.metrics["answered_rate"] == pytest.approx(5 / 6)


def test_the_wired_plugin_admits_the_climb_and_rejects_the_collapse() -> None:
    plugin, runtime = _plugin()
    # Climb: the probe scores rising clean rates, the gate admits them.
    assert _decide(plugin, runtime, [_CLEAN] * 6)[0].selected  # best -> 1.0
    # Collapse: the policy stops answering; every probe is empty.
    collapse, _ = _decide(plugin, runtime, [_EMPTY] * 6)
    assert not collapse.selected
    assert "holding" in collapse.reason
    # A within-margin dip is still admitted (5/6 >= 1.0 - 0.17).
    assert _decide(plugin, runtime, [_CLEAN] * 5 + [_EMPTY])[0].selected


def test_build_refuses_a_runtime_that_cannot_probe_a_candidate() -> None:
    class _NoProbe:
        engine = _FakeEngine()

    with pytest.raises(ValueError, match="probe_candidate"):
        build({}, runtime=_NoProbe(), scenario="gsm8k", environ={})
