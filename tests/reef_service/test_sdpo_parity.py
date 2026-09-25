"""SDPO top-K plus tail: independent dense value and gradient checks."""

import pytest

torch = pytest.importorskip("torch")

from reef.train.slime_backend.distill.objective import (
    gather_log_probs_at_ids,
    native_topk_ids,
    restricted_divergence,
    token_importance_weights,
)


@pytest.mark.unit
@pytest.mark.parametrize("divergence", ["forward", "reverse", "jsd"])
@pytest.mark.parametrize("k", [1, 4, 11])
def test_student_selected_topk_tail_value_and_gradient(divergence, k):
    rng = torch.Generator().manual_seed(219)
    logits = torch.randn(5, 11, generator=rng, dtype=torch.float64).requires_grad_()
    teacher = torch.randn(5, 11, generator=rng, dtype=torch.float64).log_softmax(-1)
    ids = native_topk_ids(logits.detach(), k, None, 1, 0)
    student_at = gather_log_probs_at_ids(logits, ids, None, 1, 0)
    teacher_at = teacher.gather(-1, ids)
    got = restricted_divergence(student_at, teacher_at, divergence=divergence, distribution="tail")
    grad = torch.autograd.grad(got.sum(), logits)[0]

    dense = logits.detach().clone().requires_grad_()
    p = dense.softmax(-1).gather(-1, ids)
    q = teacher.exp().gather(-1, ids)
    # The author's 1e-7 log-mass cap is only material at full support.
    floor = -torch.expm1(torch.tensor(-1e-7, dtype=torch.float64))
    p = torch.cat((p, (1 - p.sum(-1, keepdim=True)).clamp_min(floor)), -1)
    q = torch.cat((q, (1 - q.sum(-1, keepdim=True)).clamp_min(floor)), -1)
    if divergence == "forward":
        expected = (q * (q.log() - p.log())).sum(-1)
    elif divergence == "reverse":
        expected = (p * (p.log() - q.log())).sum(-1)
    else:
        mixture = (p + q) / 2
        expected = ((p * (p.log() - mixture.log()) + q * (q.log() - mixture.log())) / 2).sum(-1)
    expected_grad = torch.autograd.grad(expected.sum(), dense)[0]
    torch.testing.assert_close(got, expected, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-8, atol=1e-8)


@pytest.mark.unit
def test_tail_mass_is_not_renormalized_away():
    student = torch.tensor([[0.7, 0.2]]).log()
    teacher = torch.tensor([[0.35, 0.1]]).log()
    assert restricted_divergence(student, teacher, divergence="jsd").item() == pytest.approx(0, abs=1e-7)
    assert restricted_divergence(student, teacher, divergence="jsd", distribution="tail").item() > 0.1


@pytest.mark.unit
def test_token_importance_weights_are_detached_and_clipped_independently():
    student = torch.tensor([0.0, -2.0, -100.0], requires_grad=True)
    rollout = torch.tensor([-2.0, -1.0, 0.0])
    weights = token_importance_weights(student, rollout, 2.0)
    torch.testing.assert_close(weights, torch.tensor([2.0, 1.0 / torch.e, torch.exp(torch.tensor(-20.0))]))
    assert not weights.requires_grad


@pytest.mark.unit
def test_student_support_survives_teacher_packing_and_duplicate_sequences(monkeypatch):
    """The teacher scores each student's chosen ids, through packing and for identical teacher prompts."""
    import sys
    from types import SimpleNamespace

    from recipes.sdpo.slime import SdpoAlgorithm, SdpoSettings
    from reef.train.slime_backend.distill import teacher

    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_group=lambda: None,
        get_virtual_pipeline_model_parallel_world_size=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(teacher, "_TEACHER", None)
    settings = SdpoSettings(top_k=1, teacher_update_rate=1)
    # A six-token budget packs both three-token sequences into one microbatch.
    args = SimpleNamespace(seq_length=32, max_tokens_per_gpu=6, rollout_temperature=1.0)
    SdpoAlgorithm().apply_driver_options(args, settings)

    class Iterator:
        def __init__(self, data, indices):
            self.data = data
            self.indices = indices

    passes = []

    def forward_only(callback, args, model, iterators, num_microbatches):
        """Slime's pass: the callback sees the microbatches in schedule order, their samples packed end to end."""
        data = iterators[0].data
        student_pass = not passes
        passes.append([])
        collected = {}
        for sample_indices in iterators[0].indices:
            unconcat_tokens = [data["tokens"][index] for index in sample_indices]
            total_lengths = [len(tokens) for tokens in unconcat_tokens]
            logits = torch.zeros(1, sum(total_lengths), 7)
            offset = 0
            for tokens, length in zip(unconcat_tokens, total_lengths, strict=True):
                # The row predicting a sequence's one response token is its second to last.
                if student_pass:
                    logits[0, offset + length - 2, int(tokens[0])] = 3.0
                else:
                    logits[0, offset + length - 2] = torch.arange(7).float()
                offset += length
            passes[-1].append(list(sample_indices))
            _, output = callback(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=[1] * len(unconcat_tokens),
            )
            for key, rows in output.items():
                collected.setdefault(key, []).extend(rows)
        return collected

    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.data", SimpleNamespace(DataIterator=Iterator))
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.model", SimpleNamespace(forward_only=forward_only))
    identical_teacher = torch.tensor([0, 2, 3])
    batch = {
        "tokens": [torch.tensor([1, 2, 3]), torch.tensor([2, 2, 3])],
        "teacher_tokens": [identical_teacher, identical_teacher],
        "response_lengths": [1, 1],
        "loss_masks": [torch.ones(1), torch.ones(1)],
    }
    teacher.compute_teacher_rows(SimpleNamespace(args=args, model=object()), batch, settings)
    # One student pass and one teacher pass, each one packed microbatch of both samples.
    assert passes == [[[0, 1]], [[0, 1]]]
    assert [ids.item() for ids in batch["distill_teacher_topk_ids"]] == [1, 2]
    expected = torch.arange(7).float().log_softmax(-1)
    for index, log_probs in enumerate(batch["distill_teacher_topk_log_probs"]):
        assert log_probs.item() == pytest.approx(expected[index + 1].item())


@pytest.mark.unit
def test_selected_ids_follow_the_packing_schedule():
    from reef.train.slime_backend.distill.teacher import TeacherTopKAtSelectedIds

    callback = TeacherTopKAtSelectedIds([[0, 1], [2]], [torch.zeros(1, 1, dtype=torch.long)] * 3)
    with pytest.raises(ValueError, match="packing schedule"):
        callback(
            torch.zeros(1, 3, 7),
            args=None,
            unconcat_tokens=[torch.tensor([1, 2, 3])],
            total_lengths=[3],
            response_lengths=[1],
        )
