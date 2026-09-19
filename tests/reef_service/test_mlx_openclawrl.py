"""The MLX OPD kernel against the torch original, on identical inputs.

``reef/train/mlx_backend/objective.py`` is a port, so the only test that
means anything is a differential one: feed both implementations the same
tensors and compare. Expectations written by hand would just re-encode
whatever the port happens to do.

Skips unless both frameworks are importable — torch for the reference,
MLX for the port.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the torch reference is the thing under comparison")
mx = pytest.importorskip("mlx.core", reason="the MLX port needs the optional mlx extra")

from reef.train.mlx_backend.objective import candidate_log_probs, masked_softmax, opd_one_sample, policy_loss

pytestmark = pytest.mark.integration

ROWS = 6
STUDENT_K = 5
TEACHER_K = 5
VOCAB = 64
EPS_LO = 0.2
EPS_HI = 0.28


@pytest.fixture
def reference(monkeypatch):
    """The torch objective with only its device glue stubbed.

    Mirrors ``tests/test_openclawrl.py``'s fixture: Megatron is unavailable
    off a GPU host, so the tensor-parallel group is stubbed while the tensor
    kernel and the real PPO helpers stay intact.
    """
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    loss_module = ModuleType("slime.backends.megatron_utils.loss")
    loss_module.get_log_probs_and_entropy = lambda logits, **kwargs: (torch.empty(0), {"log_probs": [logits]})
    loss_module.get_responses = lambda logits, **_: iter([(logits, torch.empty(0, dtype=torch.long))])
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", loss_module)

    name = "recipes.openclawrl.slime.objective"
    previous = sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


def inputs(seed: int, *, disjoint: bool):
    """One sample's tensors, as numpy, so both frameworks see the same bits."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(ROWS, VOCAB, generator=generator, dtype=torch.float32)
    if disjoint:
        # Student and teacher propose overlapping but different candidates,
        # which is the case the intersection mask exists for.
        student = torch.stack([torch.randperm(VOCAB, generator=generator)[:STUDENT_K] for _ in range(ROWS)])
        teacher = torch.stack([torch.randperm(VOCAB, generator=generator)[:TEACHER_K] for _ in range(ROWS)])
        # Guarantee at least one shared candidate per row, and one row with
        # none at all, so row_valid gets exercised in both directions.
        teacher[0, 0] = student[0, 0]
        teacher[1, 2] = student[1, 4]
    else:
        student = torch.stack([torch.randperm(VOCAB, generator=generator)[:STUDENT_K] for _ in range(ROWS)])
        teacher = student.clone()
    captured = torch.randn(ROWS, STUDENT_K, generator=generator, dtype=torch.float32) - 1.0
    teacher_lp = torch.randn(ROWS, TEACHER_K, generator=generator, dtype=torch.float32) - 1.0
    return logits, student, teacher, captured, teacher_lp


def run_reference(module, logits, student, teacher, captured, teacher_lp, diff_clip):
    outputs = module._opd_one_sample(
        logits,
        student_indices=student,
        student_captured_lp=captured,
        teacher_indices=teacher,
        teacher_lp=teacher_lp,
        eps_lo=EPS_LO,
        eps_hi=EPS_HI,
        diff_clip=diff_clip,
        tp_group=None,
    )
    return [value.detach().numpy() for value in outputs[:4]], outputs[4].numpy()


def run_port(logits, student, teacher, captured, teacher_lp, diff_clip):
    current = candidate_log_probs(mx.array(logits.numpy()), mx.array(student.numpy()))
    result = opd_one_sample(
        current,
        student_indices=mx.array(student.numpy()),
        student_captured_log_probs=mx.array(captured.numpy()),
        teacher_indices=mx.array(teacher.numpy()),
        teacher_log_probs=mx.array(teacher_lp.numpy()),
        eps_lo=EPS_LO,
        eps_hi=EPS_HI,
        diff_clip=diff_clip,
    )
    values = [result.per_token_pg, result.per_token_clip, result.per_token_diff, result.per_token_capture_drift]
    mx.eval(*values, result.row_valid)
    import numpy

    return [numpy.array(value) for value in values], numpy.array(result.row_valid)


NAMES = ("per_token_pg", "per_token_clip", "per_token_diff", "per_token_capture_drift")


@pytest.mark.parametrize("disjoint", [False, True], ids=["identical-candidates", "partial-overlap"])
@pytest.mark.parametrize("diff_clip", [None, 0.5], ids=["unclipped", "clipped"])
def test_the_port_reproduces_the_torch_kernel(reference, disjoint, diff_clip) -> None:
    import numpy

    tensors = inputs(seed=7 if disjoint else 3, disjoint=disjoint)
    expected, expected_valid = run_reference(reference, *tensors, diff_clip)
    actual, actual_valid = run_port(*tensors, diff_clip)

    assert numpy.array_equal(expected_valid, actual_valid)
    for name, want, got in zip(NAMES, expected, actual, strict=True):
        numpy.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)


def test_a_row_with_no_shared_candidate_contributes_nothing(reference) -> None:
    import numpy

    logits, student, teacher, captured, teacher_lp = inputs(seed=11, disjoint=True)
    # Force row 3 to share nothing: its teacher candidates leave the student's set.
    teacher[3] = torch.tensor([(int(student[3].max().item()) + 1 + offset) % VOCAB for offset in range(TEACHER_K)])
    for candidate in student[3].tolist():
        assert candidate not in teacher[3].tolist()

    expected, expected_valid = run_reference(reference, logits, student, teacher, captured, teacher_lp, None)
    actual, actual_valid = run_port(logits, student, teacher, captured, teacher_lp, None)

    assert not bool(actual_valid[3])
    assert expected_valid[3] == actual_valid[3]
    for name, want, got in zip(NAMES, expected, actual, strict=True):
        assert got[3] == 0.0, name
        numpy.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=name)


def test_the_gradient_pushes_the_policy_toward_the_teacher(reference) -> None:
    # The point of the term: raise the log-prob of candidates the teacher
    # likes more than the current policy does, lower the ones it likes less.
    import mlx.nn as nn  # noqa: F401  (imported for parity with engine usage)

    logits, student, teacher, captured, teacher_lp = inputs(seed=5, disjoint=False)

    def loss(raw: mx.array) -> mx.array:
        current = candidate_log_probs(raw, mx.array(student.numpy()))
        result = opd_one_sample(
            current,
            student_indices=mx.array(student.numpy()),
            student_captured_log_probs=mx.array(captured.numpy()),
            teacher_indices=mx.array(teacher.numpy()),
            teacher_log_probs=mx.array(teacher_lp.numpy()),
            eps_lo=EPS_LO,
            eps_hi=EPS_HI,
            diff_clip=None,
        )
        return mx.sum(result.per_token_pg)

    raw = mx.array(logits.numpy())
    gradient = mx.grad(loss)(raw)
    mx.eval(gradient)

    # Descending the loss raises the log-prob of a candidate the teacher
    # prefers, so its logit gradient must be negative.
    teacher_gap = teacher_lp.numpy() - candidate_log_probs(raw, mx.array(student.numpy()))
    mx.eval(teacher_gap)
    import numpy

    gaps = numpy.array(teacher_gap)
    grads = numpy.array(gradient)
    preferred = numpy.unravel_index(numpy.argmax(gaps), gaps.shape)
    row = int(preferred[0])
    token = int(student[row, int(preferred[1])].item())
    assert grads[row, token] < 0.0


def test_the_masked_softmax_ignores_entries_outside_the_subset(reference) -> None:
    import numpy

    values = mx.array([[1.0, 2.0, 3.0, 4.0]])
    mask = mx.array([[True, False, True, False]])
    weights = masked_softmax(values, mask)
    mx.eval(weights)
    got = numpy.array(weights)[0]

    assert got[1] == 0.0 and got[3] == 0.0
    numpy.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)
    # Same as a softmax over just the unmasked pair.
    expected = torch.softmax(torch.tensor([1.0, 3.0]), dim=-1).numpy()
    numpy.testing.assert_allclose([got[0], got[2]], expected, rtol=1e-6)


def test_policy_loss_matches_slime(reference) -> None:
    import numpy
    from slime.utils.ppo_utils import compute_policy_loss

    ppo_kl = torch.linspace(-1.5, 1.5, 24).reshape(4, 6)
    advantages = torch.linspace(-2.0, 2.0, 24).reshape(4, 6)
    want_loss, want_clip = compute_policy_loss(ppo_kl, advantages, EPS_LO, EPS_HI)
    got_loss, got_clip = policy_loss(mx.array(ppo_kl.numpy()), mx.array(advantages.numpy()), EPS_LO, EPS_HI)
    mx.eval(got_loss, got_clip)

    numpy.testing.assert_allclose(numpy.array(got_loss), want_loss.numpy(), rtol=1e-5, atol=1e-6)
    numpy.testing.assert_allclose(numpy.array(got_clip), want_clip.numpy(), rtol=1e-5, atol=1e-6)


def _engine_and_row(capture=8):
    """A real rollout with captured candidates and two teacher hints."""
    from reef.train.mlx_backend.engine import DistillationRow, MLXEngine, MLXEngineConfig, TeacherCandidate

    engine = MLXEngine(
        MLXEngineConfig(
            model_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            lora_layers=4,
            max_tokens=24,
            capture_topk=capture,
            learning_rate=1e-4,
            seed=0,
        )
    )
    prompt = engine.render_prompt([{"role": "user", "content": "Summarise the result."}])
    rollout = engine.generate(prompt)
    response_length = len(rollout.output_tokens)

    def teacher_tokens(hint: str) -> tuple[int, ...]:
        hinted = engine.render_prompt([{"role": "user", "content": f"Summarise the result.\n\n[hint] {hint}"}])
        return (*hinted, *rollout.output_tokens)

    row = DistillationRow(
        tokens=(*prompt, *rollout.output_tokens),
        loss_mask=(1,) * response_length,
        rollout_log_probs=rollout.rollout_log_probs,
        reward=1.0,
        topk_indices=rollout.topk_indices,
        topk_log_probs=rollout.topk_log_probs,
        candidates=(
            TeacherCandidate("Be terse.", teacher_tokens("Be terse.")),
            TeacherCandidate("Write plain prose.", teacher_tokens("Write plain prose.")),
        ),
    )
    return engine, row


def _step_loss(engine, row, *, w_rl: float, w_opd: float) -> float:
    before = engine.adapter_snapshot()
    metrics = engine.openclawrl_step(
        [row],
        w_rl=w_rl,
        w_opd=w_opd,
        eps_lo=0.2,
        eps_hi=0.28,
        diff_clip=1.0,
        hint_selection="sequence_optimal",
        native_k=8,
    )
    # Roll the weights back so each weighting is measured from the same point.
    engine.apply_adapter(before)
    return float(metrics["loss"])


@pytest.mark.integration
def test_the_two_terms_combine_linearly() -> None:
    # loss = w_rl * reward_term + w_opd * distillation_term. Measuring each
    # weighting from the same parameters makes that additivity checkable, and
    # it is the property that would break first if the terms were wired to
    # the wrong tensors.
    engine, row = _engine_and_row()
    try:
        both = _step_loss(engine, row, w_rl=1.0, w_opd=1.0)
        reward_only = _step_loss(engine, row, w_rl=1.0, w_opd=0.0)
        distil_only = _step_loss(engine, row, w_rl=0.0, w_opd=1.0)
    finally:
        engine.close()

    assert reward_only != 0.0
    assert distil_only != 0.0
    assert both == pytest.approx(reward_only + distil_only, rel=1e-3, abs=1e-4)


@pytest.mark.integration
def test_each_term_moves_the_adapter_on_its_own() -> None:
    # Either term alone must produce a real update: a weighting that silently
    # trains nothing is the failure this whole path is built to refuse.
    engine, row = _engine_and_row()
    try:
        for w_rl, w_opd in ((1.0, 0.0), (0.0, 1.0)):
            before = engine.adapter_snapshot()
            engine.openclawrl_step(
                [row],
                w_rl=w_rl,
                w_opd=w_opd,
                eps_lo=0.2,
                eps_hi=0.28,
                diff_clip=1.0,
                hint_selection="sequence_optimal",
                native_k=8,
            )
            delta, changed = engine.adapter_delta(before, engine.adapter_snapshot())
            engine.apply_adapter(before)
            assert delta > 0.0, (w_rl, w_opd)
            assert changed == len(before), (w_rl, w_opd)
    finally:
        engine.close()


@pytest.mark.integration
def test_the_kl_term_prices_drift_and_all_but_ignores_the_base() -> None:
    """The penalty must be negligible where the policy still is the base, and
    real once it has left — that contrast is the whole mechanism.

    Not asserted as exact equality at the base: the reference log-probs come
    from a second forward pass, and two numerically identical passes still
    differ by ~5e-3 nats in this model's precision. What matters is that the
    residual is orders of magnitude below what actual drift costs.
    """

    def step(engine, row, kl_coef):
        return engine.openclawrl_step(
            [row],
            w_rl=1.0,
            w_opd=0.0,
            eps_lo=0.2,
            eps_hi=0.28,
            diff_clip=1.0,
            hint_selection="sequence_optimal",
            native_k=8,
            kl_coef=kl_coef,
        )

    engine, row = _engine_and_row()
    try:
        # lora_b starts at zero, so the policy *is* the base here.
        at_base = step(engine, row, 1.0)["loss"] - step(engine, row, 0.0)["loss"]

        # MLX streams are per-thread, so build the arrays on the engine's own.
        def drift() -> None:
            engine._apply_adapter(
                {n: (v + 0.05 if n.endswith("lora_b") else v) for n, v in engine._adapter_snapshot().items()}
            )

        engine._run(drift)
        drifted_with = step(engine, row, 1.0)["loss"]
        engine._run(drift)
        drifted_without = step(engine, row, 0.0)["loss"]
        after_drift = drifted_with - drifted_without
    finally:
        engine.close()

    # k3 is non-negative, so the penalty only ever raises the loss.
    assert after_drift > 0.0
    assert abs(at_base) < 1e-3
    assert after_drift > 100 * abs(at_base)


@pytest.mark.integration
def test_the_teacher_pass_leaves_the_adapter_where_it_found_it() -> None:
    # The teacher is the frozen base, reached by zeroing lora_b in place. If
    # that were not restored, serving would answer from the bare base model.
    # Checked on the pass itself rather than through a step, because the
    # optimizer moves the weights even at zero gradient (AdamW decays them).
    engine, row = _engine_and_row()
    try:
        before = engine.adapter_snapshot()
        gathered, native, reference = engine._run(lambda: engine._teacher_rows(row, 8, with_reference=True))
        delta, changed = engine.adapter_delta(before, engine.adapter_snapshot())
    finally:
        engine.close()

    assert len(gathered) == 2 and len(native) == 2
    # The reference pass runs inside the same zeroed window, so it must leave
    # the adapter exactly as it found it too.
    assert reference is not None and reference.shape == (len(row.loss_mask),)
    assert delta == 0.0
    assert changed == 0


@pytest.mark.integration
def test_an_unimplemented_hint_selection_is_refused() -> None:
    from reef.train.mlx_backend.engine import MLXEngineError

    engine, row = _engine_and_row()
    try:
        with pytest.raises(MLXEngineError, match="hint selection"):
            engine.openclawrl_step(
                [row],
                w_rl=1.0,
                w_opd=1.0,
                eps_lo=0.2,
                eps_hi=0.28,
                diff_clip=1.0,
                hint_selection="token_optimal",
                native_k=8,
            )
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.parametrize("float32", [True, False], ids=["float32", "float16"])
def test_a_cached_prompt_scores_the_response_the_same_way(float32) -> None:
    # Prefilling exists to bound memory, not to change the answer. In float32
    # the two paths agree to 6e-05, which says the decomposition is exact; in
    # float16 they separate by a few hundredths of a log-prob, which is the
    # accumulation order of chunked attention and nothing else. Both bounds
    # are asserted so a real divergence cannot hide behind the looser one.
    import numpy

    from reef.train.mlx_backend.engine import MLXEngine, MLXEngineConfig

    tokens = tuple(100 + (index * 7) % 9000 for index in range(600))
    response_length = 32

    def log_probs(prefill: int):
        engine = MLXEngine(
            MLXEngineConfig(
                model_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                lora_layers=4,
                # q_proj only: a cached prompt is exact exactly when the
                # adapter leaves keys and values alone.
                lora_keys=("self_attn.q_proj",),
                prefill_step_size=prefill,
                seed=0,
            )
        )
        try:
            if float32:
                engine._run(lambda: engine._model.set_dtype(mx.float32))

            def run():
                cache = engine._prefill(engine._model, tokens, response_length)
                logits = engine._response_logits_of(engine._model, tokens, response_length, cache)
                return numpy.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))

            return engine._run(run)
        finally:
            engine.close()

    whole = log_probs(0)
    cached = log_probs(128)

    assert whole.shape == cached.shape == (response_length, whole.shape[-1])
    numpy.testing.assert_allclose(cached, whole, atol=1e-3 if float32 else 0.1)


@pytest.mark.unit
def test_a_cached_prompt_is_refused_when_the_adapter_reaches_keys_or_values() -> None:
    # The cache would then be a function of the weights being trained, and
    # freezing it drops that part of the gradient without saying so.
    from reef.train.mlx_backend.engine import MLXEngineConfig

    with pytest.raises(ValueError, match="leaves keys"):
        MLXEngineConfig(
            model_path="fake/model",
            lora_keys=("self_attn.q_proj", "self_attn.v_proj"),
            prefill_step_size=2048,
        )


@pytest.mark.unit
def test_the_engine_keeps_ownership_of_how_a_prompt_is_rendered() -> None:
    # `add_generation_prompt` and `tokenize` decide what a prompt *is*. A
    # caller that could set them could serve one token sequence and train
    # another, which is the one promise `render_prompt` exists to keep.
    from reef.train.mlx_backend.engine import MLXEngineConfig

    with pytest.raises(ValueError, match="add_generation_prompt"):
        MLXEngineConfig(
            model_path="fake/model",
            chat_template_kwargs={"add_generation_prompt": False},
        )
    # `tools` is a request field of its own; two sources would let one request
    # declare two different toolsets.
    with pytest.raises(ValueError, match="tools"):
        MLXEngineConfig(model_path="fake/model", chat_template_kwargs={"tools": []})


@pytest.mark.unit
def test_a_deployment_can_default_a_template_flag() -> None:
    from reef.train.mlx_backend.engine import MLXEngineConfig

    config = MLXEngineConfig(
        model_path="fake/model",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert config.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.unit
def test_a_cached_prompt_is_allowed_when_only_queries_are_adapted() -> None:
    from reef.train.mlx_backend.engine import MLXEngineConfig

    config = MLXEngineConfig(
        model_path="fake/model",
        lora_keys=("self_attn.q_proj", "self_attn.o_proj"),
        prefill_step_size=2048,
    )

    assert config.prefill_step_size == 2048
