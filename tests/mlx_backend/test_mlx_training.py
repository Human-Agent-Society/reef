"""The trainer's memory-saving paths compute the same scores and gradients as the direct path.

Chunked scoring keeps the ``[rows, positions, vocab]`` logits from existing
at once, and a cached prompt bounds the attention size of a long prompt.
Each is a memory change only if the results stay the same. The tests load
a small real model on the test thread and drive :class:`MLXTrainer` there.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="the MLX trainer needs the optional mlx extra")
pytest.importorskip("mlx_lm", reason="the MLX trainer needs the optional mlx extra")

from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers

from recipes.openclawrl.mlx import DistillationRow, OpenClawRLMLXSettings, OpenClawRLTokenLoss, TeacherCandidate
from recipes.tttd.mlx import ImportanceSamplingSurrogate
from reef.train.mlx_backend.config import MLXEngineConfig
from reef.train.mlx_backend.training import MLXTrainer, ModelParts, ResponseTokens, TrainingRow

pytestmark = pytest.mark.integration

#: Small enough to load quickly, real enough to have a real output head.
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LENGTH = 192
RESPONSE = 160
#: The only configuration that permits a cached prompt: the last layer, query projection only.
LAST_LAYER = {"lora_layers": 1, "lora_keys": ("self_attn.q_proj",)}


def adapted_parts(*, float32: bool, lora_layers: int = 4, lora_keys=("self_attn.q_proj", "self_attn.v_proj")):
    model, _ = load(MODEL)[:2]
    mx.random.seed(0)
    model.eval()
    model.freeze()
    linear_to_lora_layers(model, lora_layers, {"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": list(lora_keys)})
    if float32:
        model.set_dtype(mx.float32)
    return ModelParts.of(model)


@pytest.fixture(scope="module")
def parts16() -> ModelParts:
    return adapted_parts(float32=False)


@pytest.fixture(scope="module")
def parts32() -> ModelParts:
    return adapted_parts(float32=True)


@pytest.fixture(scope="module")
def last_layer_parts() -> dict[bool, ModelParts]:
    return {float32: adapted_parts(float32=float32, **LAST_LAYER) for float32 in (False, True)}


def trainer(parts: ModelParts, **options) -> MLXTrainer:
    lora = {key: options.pop(key) for key in ("lora_layers", "lora_keys") if key in options}
    return MLXTrainer(parts, MLXEngineConfig(micro_batch_size=2, **lora, **options), 0)


def tttd_rows() -> list[TrainingRow]:
    # Advantages of opposite sign: the sum cancels heavily, where a careless reduction shows.
    return [
        TrainingRow(
            tokens=tuple((index * 7 + position) % 9000 + 100 for position in range(LENGTH)),
            response_length=RESPONSE,
            loss_mask=(1,) * RESPONSE,
            rollout_log_probs=tuple(-0.5 - 0.001 * position for position in range(RESPONSE)),
            advantages=(1.0 if index % 2 else -0.7,) * RESPONSE,
        )
        for index in range(2)
    ]


def tttd_gradients(parts: ModelParts, rows: list[TrainingRow], **options):
    rows_trainer = trainer(parts, **options)
    packed = rows_trainer.pack(rows)
    loss, gradients = rows_trainer.gradients(packed, ImportanceSamplingSurrogate(packed, rows))
    return loss, {name: value.astype(mx.float32) for name, value in gradients.items()}


def worst_relative_deviation(reference, candidate) -> float:
    worst = 0.0
    for name, expected in reference.items():
        scale = mx.maximum(mx.max(mx.abs(expected)), mx.array(1e-12))
        worst = max(worst, float((mx.max(mx.abs(expected - candidate[name])) / scale).item()))
    return worst


def test_the_body_and_head_split_reproduces_the_model_logits(parts16: ModelParts) -> None:
    tokens = mx.array([[101, 202, 303, 404, 505]])

    whole = parts16.model(tokens)
    split = parts16.logits(parts16.body(tokens))

    assert float(mx.max(mx.abs(whole - split)).item()) == 0.0


def test_chunked_gradients_match_the_whole_sequence_in_float32(parts32: ModelParts) -> None:
    reference_loss, reference = tttd_gradients(parts32, tttd_rows())
    chunked_loss, chunked = tttd_gradients(parts32, tttd_rows(), log_probs_chunk_size=64)

    assert abs(reference_loss - chunked_loss) / abs(reference_loss) < 1e-3
    assert worst_relative_deviation(reference, chunked) < 1e-3


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_chunking_adds_no_meaningful_error_in_float16(
    parts16: ModelParts, parts32: ModelParts, chunk_size: int
) -> None:
    # Neither float16 path is exact; chunking must not make the result
    # materially worse than the whole-sequence float16 path.
    _, reference = tttd_gradients(parts32, tttd_rows())
    _, whole = tttd_gradients(parts16, tttd_rows())
    _, chunked = tttd_gradients(parts16, tttd_rows(), log_probs_chunk_size=chunk_size)

    whole_error = worst_relative_deviation(reference, whole)
    assert worst_relative_deviation(reference, chunked) <= max(whole_error * 1.5, 1e-3)


def openclawrl_gradients(parts: ModelParts, **options):
    mx.random.seed(1)
    response, k, prompt = 40, 5, 8
    tokens = tuple(int(token) for token in mx.random.randint(0, 1000, (prompt + response,)).tolist())
    topk_indices = tuple(tuple(int(x) for x in row) for row in mx.random.randint(0, 1000, (response, k)).tolist())
    topk_log_probs = tuple(tuple(float(x) - 2.0 for x in row) for row in mx.random.normal((response, k)).tolist())
    row = DistillationRow(
        policy=TrainingRow(tokens, response, (1,) * response, (-1.0,) * response, (1.0,) * response),
        topk_indices=topk_indices,
        topk_log_probs=topk_log_probs,
        candidates=(TeacherCandidate(hint="h", tokens=(*tokens, 0)),),
    )
    teacher = mx.array([[value - 0.5 for value in values] for values in topk_log_probs])
    reference = mx.array([float(x) - 2.0 for x in mx.random.normal((response,)).tolist()])
    settings = OpenClawRLMLXSettings(w_rl=1.0, w_opd=1.0, diff_clip=None, kl_coef=0.05)
    row_trainer = trainer(parts, **options)
    packed = row_trainer.pack([row.policy])
    loss, gradients = row_trainer.gradients(packed, OpenClawRLTokenLoss(packed, row, teacher, reference, settings))
    return loss, {name: value.astype(mx.float32) for name, value in gradients.items()}


def test_chunked_openclawrl_gradients_match_the_whole_response_in_float32(parts32: ModelParts) -> None:
    reference_loss, reference = openclawrl_gradients(parts32)
    chunked_loss, chunked = openclawrl_gradients(parts32, log_probs_chunk_size=16)

    assert abs(reference_loss - chunked_loss) / abs(reference_loss) < 1e-3
    assert worst_relative_deviation(reference, chunked) < 1e-3


LONG_TOKENS = tuple(100 + (index * 7) % 9000 for index in range(600))


@pytest.mark.parametrize(("float32", "tolerance"), [(True, 1e-3), (False, 0.1)], ids=["float32", "float16"])
def test_a_cached_prompt_scores_the_response_the_same_way(
    last_layer_parts: dict[bool, ModelParts], float32: bool, tolerance: float
) -> None:
    # float16 differs by the accumulation order of the chunked attention.
    parts = last_layer_parts[float32]

    def log_probs(prefill_step_size: int) -> mx.array:
        scorer = trainer(parts, prefill_step_size=prefill_step_size, **LAST_LAYER)
        logits = scorer.response_logits(scorer.pack([ResponseTokens(LONG_TOKENS, 32)]))
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    whole, cached = log_probs(0), log_probs(128)

    assert whole.shape == cached.shape
    assert float(mx.max(mx.abs(whole - cached)).item()) < tolerance


def test_a_cached_prompt_gives_the_whole_gradient_when_only_the_last_layer_trains(
    last_layer_parts: dict[bool, ModelParts],
) -> None:
    rows = [
        TrainingRow(LONG_TOKENS, 32, (1,) * 32, tuple(-0.5 - 0.01 * position for position in range(32)), (1.0,) * 32)
    ]

    _, whole = tttd_gradients(last_layer_parts[True], rows, **LAST_LAYER)
    _, cached = tttd_gradients(last_layer_parts[True], rows, prefill_step_size=128, **LAST_LAYER)

    assert worst_relative_deviation(whole, cached) < 1e-3


def test_a_restored_optimizer_state_repeats_the_step_that_it_was_taken_before(parts32: ModelParts) -> None:
    steps = trainer(parts32)
    rows = tttd_rows()
    packed = steps.pack(rows)
    untrained = steps.adapter_tensors()

    def step() -> dict[str, mx.array]:
        steps.optimizer_step([(packed, ImportanceSamplingSurrogate(packed, rows))])
        return steps.adapter_tensors()

    try:
        step()
        weights, state = steps.adapter_tensors(), steps.optimizer_state()
        rejected = step()
        # The copy did not change when the second step updated the live state.
        assert int(state["step"].item()) == 1

        steps.load_adapter_tensors(weights)
        steps.load_optimizer_state(state)
        repeated = step()

        assert worst_relative_deviation(rejected, repeated) == 0.0
        assert int(steps.optimizer_state()["step"].item()) == 2
    finally:
        steps.load_adapter_tensors(untrained)
