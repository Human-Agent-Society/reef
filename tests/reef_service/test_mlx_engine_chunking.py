"""The chunked log-prob path must compute the same gradients as the whole one.

Chunking exists to keep the ``[batch, length, vocab]`` logits tensor from
being materialised, which is what decides whether a long sequence trains at
all. It is only a memory optimisation if it is also numerically the same
optimisation, so this pins the equivalence rather than trusting it.

Needs real MLX and a real model, so it skips everywhere except Apple Silicon
with the optional extra installed. The runtime's own contract tests
(``test_mlx_runtime.py``) run everywhere.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="the MLX engine needs the optional mlx extra")
pytest.importorskip("mlx_lm", reason="the MLX engine needs the optional mlx extra")

from reef.train.mlx_backend.engine import (
    MLXEngine,
    MLXEngineConfig,
    TrainingRow,
    _head_holder,
    _head_logits,
    _ObjectiveWeights,
)
from reef.train.mlx_backend.rows import DistillationRow, TeacherCandidate

#: Small enough to load quickly, real enough to exercise a genuine head.
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LENGTH = 192
RESPONSE = 160


def rows() -> list[TrainingRow]:
    # Advantages of opposing sign on purpose: the objective is a sum over
    # tokens that cancels heavily, which is the regime where a sloppy
    # reduction shows up.
    return [
        TrainingRow(
            tokens=tuple((index * 7 + position) % 9000 + 100 for position in range(LENGTH)),
            loss_mask=(1,) * RESPONSE,
            rollout_log_probs=tuple(-0.5 - 0.001 * position for position in range(RESPONSE)),
            advantages=tuple(1.0 if index % 2 else -0.7 for _ in range(RESPONSE)),
        )
        for index in range(2)
    ]


def gradients(chunk: int, *, float32: bool):
    engine = MLXEngine(
        MLXEngineConfig(
            model_path=MODEL,
            lora_layers=4,
            lora_rank=8,
            micro_batch_size=2,
            log_probs_chunk_size=chunk,
            seed=0,
        )
    )
    try:
        if float32:
            engine._run(lambda: engine._model.set_dtype(mx.float32))
        loss, grads = engine._run(lambda: engine._micro_batch_gradients(engine._pack(rows())))
        return loss, {name: mx.array(value).astype(mx.float32) for name, value in grads.items()}
    finally:
        engine.close()


def worst_relative_deviation(reference, candidate) -> float:
    worst = 0.0
    for name, expected in reference.items():
        scale = mx.maximum(mx.max(mx.abs(expected)), mx.array(1e-12))
        deviation = float((mx.max(mx.abs(expected - candidate[name])) / scale).item())
        worst = max(worst, deviation)
    return worst


@pytest.mark.integration
def test_the_head_split_reproduces_the_model_logits_exactly() -> None:
    # Chunked scoring runs the body and the head separately. If that split is
    # not exact, every number below is measuring the wrong thing.
    engine = MLXEngine(MLXEngineConfig(model_path=MODEL, lora_layers=4, seed=0))
    try:

        def probe() -> float:
            holder = _head_holder(engine._model)
            tokens = mx.array([[101, 202, 303, 404, 505]])
            whole = engine._model(tokens)
            split = _head_logits(holder, holder.model(tokens))
            mx.eval(whole, split)
            return float(mx.max(mx.abs(whole - split)).item())

        assert engine._run(probe) == 0.0
    finally:
        engine.close()


@pytest.mark.integration
def test_chunked_gradients_match_the_whole_sequence_in_float32() -> None:
    # float32 removes the reduced-precision noise, leaving only whether the
    # chunked decomposition is the same mathematics.
    reference_loss, reference = gradients(0, float32=True)
    chunked_loss, chunked = gradients(64, float32=True)

    assert abs(reference_loss - chunked_loss) / abs(reference_loss) < 1e-3
    assert worst_relative_deviation(reference, chunked) < 1e-3


def _distillation_row(response: int, k: int, prompt: int = 8):
    mx.random.seed(1)
    tokens = tuple(int(t) for t in mx.random.randint(0, 1000, (prompt + response,)).tolist())
    topk_indices = tuple(tuple(int(x) for x in r) for r in mx.random.randint(0, 1000, (response, k)).tolist())
    topk_log_probs = tuple(tuple(float(x) - 2.0 for x in r) for r in mx.random.normal((response, k)).tolist())
    row = DistillationRow(
        tokens=tokens,
        loss_mask=(1,) * response,
        rollout_log_probs=(-1.0,) * response,
        reward=1.0,
        topk_indices=topk_indices,
        topk_log_probs=topk_log_probs,
        candidates=(TeacherCandidate(hint="h", tokens=(*tokens, 0)),),
    )
    teacher_vals = [[v - 0.5 for v in r] for r in topk_log_probs]
    reference_vals = [float(x) - 2.0 for x in mx.random.normal((response,)).tolist()]
    return row, topk_indices, teacher_vals, reference_vals


def _opd_gradients(chunk: int):
    row, topk_indices, teacher_vals, reference_vals = _distillation_row(response=40, k=5)
    weights = _ObjectiveWeights(w_rl=1.0, w_opd=1.0, eps_lo=0.2, eps_hi=0.28, diff_clip=None, kl_coef=0.05)
    engine = MLXEngine(
        MLXEngineConfig(model_path=MODEL, lora_layers=4, lora_rank=8, log_probs_chunk_size=chunk, seed=0)
    )

    def run():
        engine._model.set_dtype(mx.float32)
        # MLX streams are per-thread, so the teacher tensors are built here.
        student_indices = mx.array([list(r) for r in topk_indices])
        return engine._row_gradients(row, student_indices, mx.array(teacher_vals), weights, mx.array(reference_vals))

    try:
        loss, grads = engine._run(run)
        return loss, {name: mx.array(value).astype(mx.float32) for name, value in grads.items()}
    finally:
        engine.close()


@pytest.mark.integration
def test_chunked_opd_gradients_match_the_whole_response_in_float32() -> None:
    # The OpenClaw-RL path (_row_gradients) has its own chunking, over response
    # positions, carrying the RL + distillation + KL terms. Same equivalence
    # bar as the TTT-Discover path: float32, so only the decomposition is under
    # test, not rounding.
    reference_loss, reference = _opd_gradients(0)
    chunked_loss, chunked = _opd_gradients(16)

    assert abs(reference_loss - chunked_loss) / abs(reference_loss) < 1e-3
    assert worst_relative_deviation(reference, chunked) < 1e-3


@pytest.mark.integration
@pytest.mark.parametrize("chunk", [32, 64, 128])
def test_chunking_adds_no_meaningful_error_in_float16(chunk: int) -> None:
    # The model serves in float16, so the useful question is not whether the
    # two paths agree with each other — neither is exact — but whether
    # chunking makes the answer worse. Measured against the float32 reference
    # both sit around 4e-2, so the bound below is deliberately tight: it
    # fails if chunking ever becomes materially less accurate, while
    # tolerating the slice-boundary rounding it legitimately introduces.
    _, reference = gradients(0, float32=True)
    _, whole = gradients(0, float32=False)
    _, chunked = gradients(chunk, float32=False)

    whole_error = worst_relative_deviation(reference, whole)
    chunked_error = worst_relative_deviation(reference, chunked)
    assert chunked_error <= max(whole_error * 1.5, 1e-3)


class _Collector:
    """A GenerationListener that keeps what it hears and can be told to stop after some pieces."""

    def __init__(self, stop_after: int | None = None) -> None:
        self.pieces: list[str] = []
        self.stop_after = stop_after

    def emit(self, piece: str) -> None:
        self.pieces.append(piece)

    def cancelled(self) -> bool:
        return self.stop_after is not None and len(self.pieces) >= self.stop_after


@pytest.mark.integration
def test_streamed_pieces_are_the_rollout_text_and_a_cancelled_stream_stops_early() -> None:
    """The pieces the detokenizer releases concatenate to exactly the text the rollout decodes
    whole, so a client that watched the stream saw what the record holds; and a listener that
    cancels ends the generation within a token, with a finish reason nothing will record."""
    engine = MLXEngine(MLXEngineConfig(model_path=MODEL, lora_layers=2, max_tokens=24, seed=0))
    try:
        prompt = engine.render_prompt([{"role": "user", "content": "Count from one to twenty in words."}])
        heard = _Collector()
        rollout = engine.generate_stream(prompt, listener=heard, temperature=0.0)
        assert "".join(heard.pieces) == rollout.text
        assert len(heard.pieces) > 1
        assert rollout.finish_reason in ("stop", "length")

        stopped = _Collector(stop_after=3)
        partial = engine.generate_stream(prompt, listener=stopped, temperature=0.0)
        assert partial.finish_reason == "cancelled"
        assert len(partial.output_tokens) < len(rollout.output_tokens)
        assert "".join(stopped.pieces) == partial.text[: len("".join(stopped.pieces))]
    finally:
        engine.close()


def test_generate_batch_yields_the_same_tokens_as_the_single_path() -> None:
    """One batched decode over several prompts produces, per prompt, exactly the
    tokens the single path would — including the left-padding of the shorter
    prompt. Greedy makes it deterministic; the recorded log-probs can differ
    slightly (the batched-attention cache is a different kernel), so only the
    tokens are pinned. The GatedDeltaNet hybrid is exercised out-of-band; this
    pins the batching logic on a model that loads fast."""
    engine = MLXEngine(MLXEngineConfig(model_path=MODEL, lora_layers=2, max_tokens=16, seed=0))
    try:
        short = engine.render_prompt([{"role": "user", "content": "Say hi."}])
        longer = engine.render_prompt([{"role": "user", "content": "Name three colors, then count to five."}])
        assert len(short) != len(longer)  # the left-padding path is what this hits

        batched = engine.generate_batch([short, longer], max_tokens=16, temperature=0.0)
        singles = [engine.generate(p, max_tokens=16, temperature=0.0) for p in (short, longer)]

        assert len(batched) == 2
        for batch_rollout, single_rollout in zip(batched, singles, strict=True):
            assert batch_rollout.output_tokens == single_rollout.output_tokens

        assert engine.generate_batch([], temperature=0.0) == []
    finally:
        engine.close()


def test_a_submission_mid_batch_defers_to_the_next_window() -> None:
    """Static batching: a rollout submitted while a batch is decoding is not
    spliced into it, but waits for that batch to drain and then decodes in the
    next window — correctly, matching the single path. Splicing a fresh prompt
    into a live batch is what corrupts the Qwen3.5 hybrid cache; this pins that
    the serving batch never does it, on a model that loads fast."""
    engine = MLXEngine(MLXEngineConfig(model_path=MODEL, lora_layers=2, max_tokens=32, seed=0))
    try:
        # A runs long enough to stay live across the B submission; B is short.
        a = engine.render_prompt([{"role": "user", "content": "Count from one to twenty."}])
        b = engine.render_prompt([{"role": "user", "content": "Say hi."}])
        single_b = engine.generate(b, max_tokens=32, temperature=0.0)

        batcher = engine._serving_batcher()
        engine.submit_rollout(a, temperature=0.0)
        engine.pump_rollouts()  # batch idle -> A admitted, now decoding
        assert engine._run(lambda: len(batcher._live)) == 1

        ticket_b = engine.submit_rollout(b, temperature=0.0)  # queued while A is live
        engine.pump_rollouts()  # must NOT splice B into the live batch
        assert engine._run(lambda: len(batcher._live)) == 1  # still just A
        assert engine._run(batcher._has_queued)  # B still waiting its window

        resolved: dict[object, object] = {}
        while engine.rollouts_pending():
            for ticket, rollout in engine.pump_rollouts():
                resolved[ticket] = rollout
        assert len(resolved) == 2
        # B decoded in its own later window, identical to the single path.
        assert resolved[ticket_b].output_tokens == single_b.output_tokens
    finally:
        engine.close()
