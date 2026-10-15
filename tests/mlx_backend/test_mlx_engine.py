"""The MLX engine on a small real model: generation, prompts, adapter files, and both loss families end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory

mx = pytest.importorskip("mlx.core", reason="the MLX engine needs the optional mlx extra")
pytest.importorskip("mlx_lm", reason="the MLX engine needs the optional mlx extra")

import recipes.openclawrl
import recipes.tttd  # noqa: F401  (registers the tttd loss family)
from reef.artifact.peft import ADAPTER_CONFIG, ADAPTER_METADATA, file_sha256
from reef.core.chat_protocol import ParsedToolCall
from reef.train.mlx_backend.config import MLXEngineConfig
from reef.train.mlx_backend.contracts import GenerationListener, LossOptions
from reef.train.mlx_backend.engine import MLX_ADAPTER_WEIGHTS, MLXAdapterSnapshot, MLXEngine
from reef.train.mlx_backend.training import MLXEngineError, ResponseTokens

pytestmark = pytest.mark.integration

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
QUESTION = [{"role": "user", "content": "Summarise the result."}]


@pytest.fixture(scope="module")
def engine():
    engine = MLXEngine(
        MODEL, MLXEngineConfig(lora_layers=4, max_tokens=24, capture_topk=8, learning_rate=1e-4, seed=0)
    )
    yield engine
    engine.close()


@pytest.fixture
def restored(engine: MLXEngine):
    """The engine, with its adapter put back after the test."""
    snapshot = engine.adapter_snapshot()
    yield engine
    engine.apply_adapter(snapshot)


class Collector(GenerationListener):
    """Keeps the pieces it hears and cancels after ``stop_after`` of them."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.pieces: list[str] = []
        self.stop_after = stop_after

    def emit(self, piece: str) -> None:
        self.pieces.append(piece)

    def cancelled(self) -> bool:
        return self.stop_after is not None and len(self.pieces) >= self.stop_after


def test_streamed_pieces_are_the_rollout_text_and_a_cancelled_stream_stops_early(engine: MLXEngine) -> None:
    prompt = engine.render_prompt([{"role": "user", "content": "Count from one to twenty in words."}])

    heard = Collector()
    rollout = engine.generate(prompt, temperature=0.0, listener=heard)
    stopped = Collector(stop_after=3)
    partial = engine.generate(prompt, temperature=0.0, listener=stopped)

    assert "".join(heard.pieces) == rollout.text
    assert len(heard.pieces) > 1
    assert rollout.finish_reason in ("stop", "length")
    assert len(rollout.rollout_log_probs) == len(rollout.output_tokens)
    assert len(rollout.topk_indices) == len(rollout.output_tokens)
    assert partial.finish_reason == "cancelled"
    assert len(partial.output_tokens) < len(rollout.output_tokens)
    assert partial.text.startswith("".join(stopped.pieces))


def test_string_tool_call_arguments_render_like_object_arguments(engine: MLXEngine) -> None:
    def conversation(arguments) -> list[dict]:
        call = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": arguments}}
        return [*QUESTION, {"role": "assistant", "content": "", "tool_calls": [call]}]

    assert engine.render_prompt(conversation('{"path": "a.txt"}')) == engine.render_prompt(
        conversation({"path": "a.txt"})
    )
    with pytest.raises(ValueError, match="valid JSON"):
        engine.render_prompt(conversation("not json"))
    with pytest.raises(ValueError, match="add_generation_prompt"):
        engine.render_prompt(QUESTION, template_kwargs={"add_generation_prompt": False})


def test_the_templates_tool_parser_reads_a_call(engine: MLXEngine) -> None:
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]

    assert engine.tool_call_markers == ("<tool_call>", "</tool_call>")
    assert engine.parse_tool_call('{"name": "read_file", "arguments": {"path": "a"}}', tools) == ParsedToolCall(
        "read_file", '{"path": "a"}'
    )
    assert engine.opens_reasoning(engine.render_prompt(QUESTION)) is False


def test_a_saved_adapter_loads_back_and_records_its_checksums(restored: MLXEngine, tmp_path: Path) -> None:
    engine = restored
    saved = engine.adapter_snapshot()
    destination = tmp_path / "candidate"

    engine.save_adapter(destination, source={"scenario_step": 3})
    engine.apply_adapter(drifted(engine, saved, 0.05))
    engine.load_adapter(destination)

    assert engine.adapter_delta(saved, engine.adapter_snapshot()) == (0.0, 0)
    metadata = json.loads((destination / ADAPTER_METADATA).read_text())
    assert metadata["base_model"] == {"name_or_path": MODEL}
    assert metadata["source"] == {"scenario_step": 3}
    assert metadata["files"] == {
        name: file_sha256(destination / name) for name in (ADAPTER_CONFIG, MLX_ADAPTER_WEIGHTS)
    }
    assert json.loads((destination / ADAPTER_CONFIG).read_text())["num_layers"] == 4
    with pytest.raises(MLXEngineError, match="already exists"):
        engine.save_adapter(destination, source={})


def drifted(engine: MLXEngine, snapshot: MLXAdapterSnapshot, amount: float) -> MLXAdapterSnapshot:
    """``snapshot`` with ``amount`` added to every ``lora_b``; built on the engine thread, where MLX arrays live."""
    return engine.run(
        lambda: MLXAdapterSnapshot(
            {name: value + amount if name.endswith("lora_b") else value for name, value in snapshot.tensors.items()}
        )
    )


def distillation_sample(engine: MLXEngine):
    """A real rollout with captured candidates and two teacher hints."""
    prompt = engine.render_prompt(QUESTION)
    rollout = engine.generate(prompt)

    def teacher_tokens(hint: str) -> list[int]:
        hinted = engine.render_prompt([{"role": "user", "content": f"Summarise the result.\n\n[hint] {hint}"}])
        return [*hinted, *rollout.output_tokens]

    return policy_trajectory(
        "turn-1",
        (*prompt, *rollout.output_tokens),
        (1,) * len(rollout.output_tokens),
        rollout.rollout_log_probs,
        1.0,
        topk_indices=rollout.topk_indices,
        topk_log_probs=rollout.topk_log_probs,
        extras={
            "teacher_cands": [
                {"hint": hint, "teacher_tokens": teacher_tokens(hint)} for hint in ("Be terse.", "Write plain prose.")
            ]
        },
    )


def openclawrl_loss(engine: MLXEngine, sample, **options) -> float:
    """The loss of one openclawrl step, measured from the current adapter, which is then put back."""
    before = engine.adapter_snapshot()
    metrics = engine.train(
        "openclawrl", [sample], [1.0], LossOptions(options.pop("kl_coef", 0.0), {"native_k": 8, **options})
    )
    engine.apply_adapter(before)
    return float(metrics["loss"])


def test_the_two_openclawrl_terms_add_up(restored: MLXEngine) -> None:
    sample = distillation_sample(restored)

    both = openclawrl_loss(restored, sample, w_rl=1.0, w_opd=1.0)
    reward_only = openclawrl_loss(restored, sample, w_rl=1.0, w_opd=0.0)
    distillation_only = openclawrl_loss(restored, sample, w_rl=0.0, w_opd=1.0)

    assert reward_only != 0.0
    assert distillation_only != 0.0
    assert both == pytest.approx(reward_only + distillation_only, rel=1e-3, abs=1e-4)


@pytest.mark.parametrize(("w_rl", "w_opd"), [(1.0, 0.0), (0.0, 1.0)], ids=["reward", "distillation"])
def test_each_openclawrl_term_alone_moves_every_adapter_tensor(restored: MLXEngine, w_rl: float, w_opd: float) -> None:
    sample = distillation_sample(restored)
    before = restored.adapter_snapshot()

    restored.train("openclawrl", [sample], [1.0], LossOptions(0.0, {"w_rl": w_rl, "w_opd": w_opd, "native_k": 8}))

    delta, changed = restored.adapter_delta(before, restored.adapter_snapshot())
    assert delta > 0.0
    assert changed == len(before.tensors)


def test_the_kl_term_is_negligible_at_the_base_and_real_after_drift(restored: MLXEngine) -> None:
    """``lora_b`` starts at zero, so the policy is the base model. The reference comes from a second
    forward pass, so the penalty at the base is small rather than zero."""
    sample = distillation_sample(restored)
    at_base = openclawrl_loss(restored, sample, kl_coef=1.0) - openclawrl_loss(restored, sample)
    restored.apply_adapter(drifted(restored, restored.adapter_snapshot(), 0.05))
    after_drift = openclawrl_loss(restored, sample, kl_coef=1.0) - openclawrl_loss(restored, sample)

    # k3 is non-negative, so the penalty only raises the loss.
    assert after_drift > 0.0
    assert abs(at_base) < 1e-3
    assert after_drift > 100 * abs(at_base)


def test_the_frozen_base_window_leaves_the_adapter_where_it_found_it(restored: MLXEngine) -> None:
    restored.apply_adapter(drifted(restored, restored.adapter_snapshot(), 0.05))
    before = restored.adapter_snapshot()
    trainer = restored.trainer
    tokens = (*restored.render_prompt(QUESTION), 100, 101)

    def score_base() -> None:
        with trainer.base_model():
            trainer.response_logits(trainer.pack([ResponseTokens(tokens, 2)]))

    restored.run(score_base)

    assert restored.adapter_delta(before, restored.adapter_snapshot()) == (0.0, 0)


def test_a_tttd_step_trains_from_a_fresh_rollout(restored: MLXEngine) -> None:
    prompt = restored.render_prompt(QUESTION)
    rollout = restored.generate(prompt)
    sample = policy_trajectory(
        "turn-1", (*prompt, *rollout.output_tokens), (1,) * len(rollout.output_tokens), rollout.rollout_log_probs, 1.0
    )
    before = restored.adapter_snapshot()

    metrics = restored.train("tttd", [sample, sample], [1.0, -0.5], LossOptions(kl_coef=0.1))

    assert metrics["importance_ratio"] == pytest.approx(1.0, abs=0.05)
    assert metrics["response_tokens"] == 2 * len(rollout.output_tokens)
    assert restored.adapter_delta(before, restored.adapter_snapshot())[0] > 0.0
