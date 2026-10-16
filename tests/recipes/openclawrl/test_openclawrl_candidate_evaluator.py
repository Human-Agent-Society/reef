"""OpenClaw-RL's candidate probe and its regression check.

The tests run the plugin on a real MLX runtime around a fake engine, so the
probe goes through ``MLXRuntime.probe_candidate`` and needs no model.
"""

from __future__ import annotations

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
    ],
)
def test_from_config_refuses_an_invalid_setting(tmp_path: Path, config, message) -> None:
    training, _ = probe_runtime(tmp_path)
    with pytest.raises(PluginConfigError, match=message):
        build_plugin(training, **config)
