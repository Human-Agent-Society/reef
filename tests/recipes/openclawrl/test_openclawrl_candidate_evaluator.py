"""OpenClaw-RL's candidate probe and its regression check.

The tests run the plugin on a real MLX runtime around a fake engine, so the
probe goes through ``MLXRuntime.probe_candidate`` and needs no model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from tests.support.mlx_engine_stubs import FakeAdapterSnapshot, FakeMLXEngine

from recipes.openclawrl.candidate_evaluator import OpenClawRLCandidateEvaluation
from reef.core.evaluation import UpdateCandidate
from reef.core.plugins import PluginConfigError
from reef.train.mlx_backend.config import MLXConfig
from reef.train.mlx_backend.runtime import MLXRuntime, mlx_runtimes

CLEAN = (
    "She sold 48 clips in April. In May she sold half as many, so 48 divided by 2 "
    "equals 24. Altogether that is 48 plus 24 equals 72 clips."
)
MARKDOWN = "**Answer:** 72\n- April: 48\n- May: 24\n"
EMPTY = ""  # a turn with no reply
TEXT_ANSWER = "<think>\nThe file holds the problem.\n</think>\n\nShe sold 72 clips altogether."
TOOL_CALL = "<tool_call>\n<function=terminal>\n<parameter=command>\necho ymd\n</parameter>\n</function>\n</tool_call>"


@dataclass
class ScriptedEngine(FakeMLXEngine):
    """A fake engine that answers each probe prompt with the next scripted reply."""

    replies: list[str] = field(default_factory=list)

    def generate(self, prompt_tokens, *, max_tokens=None, temperature=None, listener=None):
        rollout = super().generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature, listener=listener)
        return replace(rollout, text=self.replies.pop(0))


def probe_runtime(tmp_path: Path) -> tuple[MLXRuntime, ScriptedEngine]:
    engine = ScriptedEngine()
    training, serving = mlx_runtimes(MLXConfig(checkpoint_dir=str(tmp_path)), engine)
    serving.stage_candidate("c", FakeAdapterSnapshot({"layer.lora_a": 1.0, "layer.lora_b": 1.0}))
    return training, engine


def build_plugin(training: MLXRuntime, **config) -> OpenClawRLCandidateEvaluation:
    return OpenClawRLCandidateEvaluation.from_config(
        {"probe_size": 6, "max_tokens": 64, **config},
        runtime=None,
        training_runtime=training,
        scenario="gsm8k",
        environ={},
    )


class Probe:
    """Evaluate and decide one candidate with scripted replies."""

    def __init__(self, tmp_path: Path) -> None:
        training, self.engine = probe_runtime(tmp_path)
        self.plugin = build_plugin(training)

    def __call__(self, replies: list[str]):
        self.engine.replies = list(replies)
        candidate = UpdateCandidate(candidate_id="c")
        evaluation = self.plugin.evaluate(candidate)
        return self.plugin.decide(candidate, evaluation), evaluation


@pytest.fixture
def probe(tmp_path: Path) -> Probe:
    return Probe(tmp_path)


def test_the_probe_scores_clean_markdown_and_empty_replies(probe) -> None:
    _, evaluation = probe([CLEAN, CLEAN, MARKDOWN, EMPTY, CLEAN, CLEAN])

    assert evaluation.metrics["clean_count"] == 4
    assert evaluation.metrics["probe_count"] == 6
    assert evaluation.metrics["clean_rate"] == pytest.approx(4 / 6)
    # The markdown reply is an answer; the empty reply is not.
    assert evaluation.metrics["answered_rate"] == pytest.approx(5 / 6)


def test_the_probe_generates_greedily_from_the_candidate_weights(probe) -> None:
    probe([CLEAN] * 6)

    assert [call["weights"] for call in probe.engine.generated] == [{"layer.lora_a": 1.0, "layer.lora_b": 1.0}] * 6
    assert {call["temperature"] for call in probe.engine.generated} == {0.0}
    assert {call["max_tokens"] for call in probe.engine.generated} == {64}
    # Serving gets its own weights back after the probe.
    assert probe.engine.weights == {"layer.lora_a": 0.0, "layer.lora_b": 0.0}


def test_the_probe_reports_how_far_the_replies_are_from_the_criterion(probe) -> None:
    """``mean_violations`` orders replies that all have a ``clean_rate`` of zero."""
    one_marker = "- April: 48 divided by 2 equals 24, and 48 plus 24 equals 72."

    _, near = probe([one_marker] * 6)
    _, far = probe([MARKDOWN] * 6)
    _, silent = probe([EMPTY] * 6)

    assert near.metrics["clean_rate"] == far.metrics["clean_rate"] == silent.metrics["clean_rate"] == 0.0
    assert near.metrics["mean_violations"] == pytest.approx(1.0)
    assert far.metrics["mean_violations"] == pytest.approx(3.0)
    # An empty reply fails only the shown-work check, so ``answered_rate`` shows it, not the distance.
    assert silent.metrics["mean_violations"] == pytest.approx(1.0)
    assert silent.metrics["answered_rate"] == 0.0
    assert near.metrics["answered_rate"] == 1.0


def test_the_plugin_accepts_the_climb_and_rejects_the_collapse(probe) -> None:
    assert probe([CLEAN] * 6)[0].selected
    collapse, _ = probe([EMPTY] * 6)
    assert not collapse.selected
    assert "holding" in collapse.reason
    # A drop within the margin is accepted: 5/6 >= 1.0 - 0.17.
    assert probe([CLEAN] * 5 + [EMPTY])[0].selected


def write_reply_probe(tmp_path: Path, conversation_count: int = 2, answers: list[str] | None = None) -> Path:
    path = tmp_path / "reply_probe.json"
    conversation = [
        {"role": "user", "content": "Hey, I have my homework in the file homework/0.txt."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "Problem: ..."},
    ]
    path.write_text(
        json.dumps(
            {
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
                "conversations": [conversation] * conversation_count,
                **({"answers": answers} if answers else {}),
            }
        )
    )
    return path


def test_the_reply_probe_counts_text_answers_and_refuses_a_policy_that_stops_answering(tmp_path: Path) -> None:
    training, engine = probe_runtime(tmp_path)
    plugin = build_plugin(training, reply_probe_file=str(write_reply_probe(tmp_path)))
    # The conversations are rendered with the harness tools.
    assert engine.rendered[-1]["tools"] == [{"type": "function", "function": {"name": "read_file"}}]

    def run(answers: list[str]):
        engine.replies = [CLEAN] * 6 + answers
        candidate = UpdateCandidate(candidate_id="c")
        evaluation = plugin.evaluate(candidate)
        return plugin.decide(candidate, evaluation), evaluation

    first, evaluation = run([TEXT_ANSWER, TEXT_ANSWER])
    assert first.selected
    assert evaluation.metrics["reply_rate"] == 1.0
    # A tool call or an empty turn is not a text answer, even with a clean style probe.
    collapse, evaluation = run([TOOL_CALL, EMPTY])
    assert evaluation.metrics["reply_rate"] == 0.0
    assert not collapse.selected
    assert collapse.reason.startswith("reply_rate 0 regressed")
    # The refused candidate does not lower the bar: 1/2 is still more than 0.25 below 1.0.
    assert not run([TEXT_ANSWER, TOOL_CALL])[0].selected


def test_reply_clean_rate_selects_on_the_style_of_correct_first_replies(tmp_path: Path) -> None:
    training, engine = probe_runtime(tmp_path)
    plugin = build_plugin(
        training,
        reply_probe_file=str(write_reply_probe(tmp_path, 3, answers=["72", "72", "72"])),
        regression_metric="reply_clean_rate",
        regression_margin=0.0,
        reply_margin=1.0,
    )
    plain = "She sold 48 in April and 48 / 2 = 24 in May, so 48 + 24 = 72 clips."
    bold = "She sold 48 in April and 48 / 2 = 24 in May, so 48 + 24 = **72** clips."

    def run(answers: list[str]):
        engine.replies = [MARKDOWN] * 6 + answers
        candidate = UpdateCandidate(candidate_id="c")
        evaluation = plugin.evaluate(candidate)
        return plugin.decide(candidate, evaluation), evaluation

    # A wrong answer, a styled answer and a tool call do not count; the style probe is ignored.
    first, evaluation = run([plain, bold, "She sold 70 clips."])
    assert evaluation.metrics["reply_clean_rate"] == pytest.approx(1 / 3)
    assert evaluation.metrics["clean_rate"] == 0.0
    assert first.selected
    regressed, _ = run([bold, bold, plain.replace("72", "70")])
    assert not regressed.selected
    assert regressed.reason.startswith("reply_clean_rate 0 regressed")
    assert run([plain, plain, bold])[0].selected


def test_reply_clean_rate_needs_reply_probe_answers(tmp_path: Path) -> None:
    training, _ = probe_runtime(tmp_path)
    with pytest.raises(PluginConfigError, match="needs a reply_probe_file with 'answers'"):
        build_plugin(training, reply_probe_file=str(write_reply_probe(tmp_path)), regression_metric="reply_clean_rate")


def test_min_reply_rate_refuses_the_first_candidate_too(tmp_path: Path) -> None:
    training, engine = probe_runtime(tmp_path)
    plugin = build_plugin(training, reply_probe_file=str(write_reply_probe(tmp_path)), min_reply_rate=0.75)
    engine.replies = [CLEAN] * 6 + [TEXT_ANSWER, TOOL_CALL]
    candidate = UpdateCandidate(candidate_id="c")

    decision = plugin.decide(candidate, plugin.evaluate(candidate))

    # Without a best selected rate (a fresh process), the floor still holds.
    assert not decision.selected
    assert decision.reason.startswith("reply_rate 0.5 is below min_reply_rate 0.75")


def test_a_turn_that_never_closes_an_opened_reasoning_block_is_not_a_text_answer() -> None:
    from recipes.openclawrl.candidate_evaluator import answers_in_text

    unclosed = "Here is the solution: 72 clips."
    assert answers_in_text(unclosed)
    assert not answers_in_text(unclosed, opens_reasoning=True)
    assert answers_in_text(f"{unclosed}</think>\n\nShe sold 72 clips.", opens_reasoning=True)
    # With a gold answer, the visible text must contain it.
    assert answers_in_text("She sold 72 clips.", answer="72")
    assert answers_in_text("That is 1,200 clips.", answer="1200")
    assert not answers_in_text("import numpy as np", answer="72")
    assert not answers_in_text("She sold 720 clips.", answer="72")


def test_from_config_refuses_a_runtime_that_cannot_probe_a_candidate() -> None:
    with pytest.raises(PluginConfigError, match="mlx training runtime"):
        OpenClawRLCandidateEvaluation.from_config(
            {}, runtime=None, training_runtime=object(), scenario="gsm8k", environ={}
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"probe_size": 0}, "probe_size must be between 1 and 10"),
        ({"probe_size": 11}, "probe_size must be between 1 and 10"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"regression_margin": -0.1}, "regression_margin must be non-negative"),
        ({"reply_max_tokens": 0}, "reply_max_tokens must be positive"),
        ({"reply_margin": -0.1}, "reply_margin must be non-negative"),
        ({"min_reply_rate": 1.5}, "min_reply_rate must be between 0 and 1"),
        ({"regression_metric": "gold_rate"}, "regression_metric must be one of"),
        ({"reply_probe_file": "/nonexistent/reply_probe.json"}, "cannot read reply_probe_file"),
    ],
)
def test_from_config_refuses_an_invalid_setting(tmp_path: Path, config, message) -> None:
    training, _ = probe_runtime(tmp_path)
    with pytest.raises(PluginConfigError, match=message):
        build_plugin(training, **config)
