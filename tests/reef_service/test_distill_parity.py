"""The distillation package's tensor code on CPU torch: the kernels against the pure-Python reference, and the teacher's weights.

The reference (torch free) is the source of truth for the per-token
divergences and the importance-sampling weight of
``reef/train/slime_backend/distill/objective.py``. These tests run both on
the same inputs at tensor-parallel world size one, on CPU tensors, so they
are cheap enough for the minimal CI gate that installs CPU torch; the
sharded tests spawn four gloo ranks on the CPU. The teacher's weights are
exercised through a fake actor: a dict of bfloat16 tensors as the model, a
backuper that copies and restores them by tag, and a tagged checkpoint
loader that fills the model. No Megatron is needed: the kernels take the
tensor-parallel group as an argument, and the teacher module imports
Megatron only where the pass runs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reef.train.slime_backend.distill import DistillSettings
from reef.train.slime_backend.distill.objective import (
    chunked_token_divergence,
    gather_log_probs_at_ids,
    global_log_sum_exp,
    native_topk_ids,
    restricted_divergence,
    sampled_reverse_kl,
    sequence_importance_weight,
    token_divergence,
)
from reef.train.slime_backend.distill.teacher import (
    ACTOR_TAG,
    TEACHER_TAG,
    CurrentWeights,
    MovingCopy,
    SeparateCheckpoint,
    mix_teacher_weights,
    teacher_weights,
)

from .reference_algorithms import distill

_ROWS, _VOCAB = 5, 11
_DIVERGENCES = ("forward", "reverse", "jsd")


def _rows(seed: int, vocab: int = _VOCAB, scale: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(_ROWS, vocab, generator=generator, dtype=torch.float64) * scale
    teacher = torch.log_softmax(torch.randn(_ROWS, vocab, generator=generator, dtype=torch.float64) * scale, dim=-1)
    return student, teacher


def _reference_gradient(student: torch.Tensor, teacher: torch.Tensor, divergence: str) -> torch.Tensor:
    """Autograd through the dense divergence written out on the full vocabulary."""
    dense = student.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(dense, dim=-1)
    if divergence == "forward":
        value = (teacher.exp() * (teacher - log_probs)).sum()
    elif divergence == "reverse":
        value = (log_probs.exp() * (log_probs - teacher)).sum()
    else:
        mixture = torch.log(0.5 * teacher.exp() + 0.5 * log_probs.exp())
        value = (
            0.5 * (teacher.exp() * (teacher - mixture)).sum() + 0.5 * (log_probs.exp() * (log_probs - mixture)).sum()
        )
    value.backward()
    return dense.grad


@pytest.mark.unit
@pytest.mark.parametrize("divergence", _DIVERGENCES)
@pytest.mark.parametrize("seed", [0, 1])
def test_token_divergence_matches_reference(divergence: str, seed: int) -> None:
    student, teacher = _rows(seed)

    value = token_divergence(student, teacher, divergence=divergence, tp_group=None, tp_world=1)

    expected = [distill.divergence(student[row].tolist(), teacher[row].tolist(), divergence) for row in range(_ROWS)]
    assert value.tolist() == pytest.approx(expected, abs=1e-9)
    assert all(entry >= 0 for entry in value.tolist())


@pytest.mark.unit
@pytest.mark.parametrize("divergence", _DIVERGENCES)
def test_token_divergence_is_zero_when_student_equals_teacher(divergence: str) -> None:
    student, _ = _rows(3)
    teacher = torch.log_softmax(student, dim=-1)

    value = token_divergence(student, teacher, divergence=divergence, tp_group=None, tp_world=1)

    assert value.tolist() == pytest.approx([0.0] * _ROWS, abs=1e-12)


@pytest.mark.unit
def test_forward_kl_gradient_is_student_minus_teacher() -> None:
    student, teacher = _rows(4)
    student = student.clone().requires_grad_(True)

    token_divergence(student, teacher, divergence="forward", tp_group=None, tp_world=1).sum().backward()

    expected = torch.softmax(student.detach(), dim=-1) - teacher.exp()
    assert torch.allclose(student.grad, expected, atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("divergence", ["reverse", "jsd"])
def test_explicit_gradients_match_autograd_of_the_dense_form(divergence: str) -> None:
    student, teacher = _rows(5)
    student = student.clone().requires_grad_(True)

    token_divergence(student, teacher, divergence=divergence, tp_group=None, tp_world=1).sum().backward()

    assert torch.allclose(student.grad, _reference_gradient(student, teacher, divergence), atol=1e-9)


@pytest.mark.unit
def test_jsd_beta_weights_the_teacher_in_the_mixture() -> None:
    student, teacher = _rows(9)

    value = token_divergence(student, teacher, divergence="jsd", jsd_beta=0.2, tp_group=None, tp_world=1)

    expected = [distill.divergence(student[row].tolist(), teacher[row].tolist(), "jsd", 0.2) for row in range(_ROWS)]
    assert value.tolist() == pytest.approx(expected, abs=1e-9)
    with pytest.raises(ValueError, match="divergence must be one of"):
        token_divergence(student, teacher, divergence="hellinger", tp_group=None, tp_world=1)


@pytest.mark.unit
@pytest.mark.parametrize("divergence", _DIVERGENCES)
def test_chunked_divergence_matches_whole_and_backpropagates_through_checkpoint(divergence: str) -> None:
    student, teacher = _rows(6)
    whole = student.clone().requires_grad_(True)
    chunked = student.clone().requires_grad_(True)

    whole_value = token_divergence(whole, teacher, divergence=divergence, tp_group=None, tp_world=1)
    chunked_value = chunked_token_divergence(
        chunked, teacher, divergence=divergence, chunk_size=2, tp_group=None, tp_world=1
    )
    whole_value.sum().backward()
    chunked_value.sum().backward()

    assert torch.allclose(chunked_value, whole_value)
    assert torch.allclose(chunked.grad, whole.grad)


@pytest.mark.unit
@pytest.mark.parametrize("divergence", _DIVERGENCES)
def test_float16_teacher_rows_with_the_storage_floor_keep_the_divergence_finite(divergence: str) -> None:
    # The teacher pass stores float16 rows clamped at -1e4; exp of the floor
    # is exactly zero, so a vocabulary the teacher rules out costs nothing.
    student, teacher = _rows(7)
    teacher[:, 0] = -float("inf")
    stored = teacher.clamp_min(-1.0e4).to(torch.float16)

    value = token_divergence(student, stored, divergence=divergence, tp_group=None, tp_world=1)

    finite_reference = teacher.clone()
    finite_reference[:, 0] = -1.0e4
    expected = [
        distill.divergence(student[row].tolist(), finite_reference[row].tolist(), divergence) for row in range(_ROWS)
    ]
    assert torch.isfinite(value).all()
    # float16 keeps about three significant digits of each stored log-prob.
    assert value.tolist() == pytest.approx(expected, rel=2e-2, abs=2e-2)


@pytest.mark.unit
def test_global_log_sum_exp_matches_torch_at_world_size_one() -> None:
    student, _ = _rows(8)
    assert torch.allclose(global_log_sum_exp(student, None, 1), torch.logsumexp(student, dim=-1))


# --- the top-K representation ------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("divergence", _DIVERGENCES)
def test_restricted_divergence_renormalizes_both_distributions_over_the_ids(divergence: str) -> None:
    student, teacher = _rows(10)
    ids = native_topk_ids(teacher, 4, None, 1, 0)
    student_at = gather_log_probs_at_ids(student, ids, None, 1, 0)
    teacher_at = torch.gather(teacher, -1, ids)

    value = restricted_divergence(student_at, teacher_at, divergence=divergence)

    expected = [
        distill.restricted_divergence(student_at[row].tolist(), teacher_at[row].tolist(), divergence)
        for row in range(_ROWS)
    ]
    assert value.tolist() == pytest.approx(expected, abs=1e-9)
    assert ids.tolist() == torch.topk(teacher, 4, dim=-1).indices.tolist()
    assert torch.allclose(student_at, torch.gather(torch.log_softmax(student, -1), -1, ids))


@pytest.mark.unit
def test_sampled_reverse_kl_is_the_gap_with_the_score_function_gradient() -> None:
    student, teacher = _rows(11)
    sampled = torch.tensor([0, 3, 5, 7, 10])
    logits = student.clone().requires_grad_(True)
    student_at = torch.log_softmax(logits, -1).gather(-1, sampled[:, None]).squeeze(-1)
    teacher_at = teacher.gather(-1, sampled[:, None]).squeeze(-1)

    value = sampled_reverse_kl(student_at, teacher_at)
    value.sum().backward()

    gap = (student_at - teacher_at).detach()
    assert torch.allclose(value, gap)
    # d/dz [gap * log p(y)] = gap * (onehot(y) - p): the estimator's gradient.
    probs = torch.softmax(student, -1)
    onehot = torch.zeros_like(probs).scatter_(-1, sampled[:, None], 1.0)
    assert torch.allclose(logits.grad, gap[:, None] * (onehot - probs), atol=1e-9)


# --- the importance weight and the teacher copy -------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("cap", [2.0, 0.5])
def test_sequence_importance_weight_matches_reference(cap: float) -> None:
    student = torch.tensor([-0.5, -1.0, -0.2, -2.0], dtype=torch.float64)
    rollout = torch.tensor([-0.4, -2.0, -0.3, -0.1], dtype=torch.float64)
    mask = torch.tensor([1, 1, 0, 1])

    weight = sequence_importance_weight(student, rollout, mask, cap)

    expected = distill.sequence_importance_weight(student.tolist(), rollout.tolist(), mask.tolist(), cap)
    assert weight.item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
def test_sequence_importance_weight_of_an_untrained_sample_is_zero() -> None:
    zeros = torch.zeros(3, dtype=torch.float64)
    assert sequence_importance_weight(zeros, zeros, torch.zeros(3, dtype=torch.int64), 2.0).item() == 0.0


@pytest.mark.unit
def test_teacher_weights_move_toward_the_actor_in_float32() -> None:
    # The reference's ref = 0.99 * ref + 0.01 * policy, accumulated where a
    # 1% step of a small change survives (bfloat16 would round it away).
    teacher = {"w": torch.full((4,), 1.0, dtype=torch.float32), "steps": torch.tensor([3])}
    actor = {"w": torch.full((4,), 1.0 + 1e-3, dtype=torch.bfloat16), "steps": torch.tensor([9])}

    mix_teacher_weights(teacher, actor, 0.01)

    expected = 0.99 * 1.0 + 0.01 * torch.full((4,), 1.0 + 1e-3, dtype=torch.bfloat16).float()
    assert torch.allclose(teacher["w"], expected)
    assert teacher["w"].dtype == torch.float32
    assert teacher["steps"].tolist() == [3]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_teacher_weights(teacher, actor, 1.5)


# --- the vocab shards -----------------------------------------------------------


def _sharded_worker(rank: int, world: int, port: int, divergence: str, seed: int) -> None:
    """One tensor-parallel rank: its vocab shard's divergences and gradients must match the dense computation."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        group = dist.group.WORLD
        vocab = 24
        student, teacher = _rows(seed, vocab=vocab, scale=3.0)
        shard = slice(rank * vocab // world, (rank + 1) * vocab // world)

        # The exact representation: the shard's rows against the dense value and gradient.
        dense = student.clone().requires_grad_(True)
        dense_value = token_divergence(dense, teacher, divergence=divergence, tp_group=None, tp_world=1)
        dense_value.sum().backward()
        local = student[:, shard].clone().requires_grad_(True)
        value = token_divergence(local, teacher[:, shard], divergence=divergence, tp_group=group, tp_world=world)
        value.sum().backward()
        assert torch.allclose(value, dense_value.detach(), atol=1e-9), (rank, value, dense_value)
        assert torch.allclose(local.grad, dense.grad[:, shard], atol=1e-9), (rank, local.grad, dense.grad[:, shard])

        # The top-K representation: the gathered log-probs are replicated, their gradient lands on the owner.
        ids = native_topk_ids(teacher[:, shard], 4, group, world, rank)
        assert ids.tolist() == torch.topk(teacher, 4, dim=-1).indices.tolist()
        dense = student.clone().requires_grad_(True)
        dense_at = torch.log_softmax(dense, -1).gather(-1, ids)
        dense_at.sum().backward()
        local = student[:, shard].clone().requires_grad_(True)
        at_ids = gather_log_probs_at_ids(local, ids, group, world, rank)
        at_ids.sum().backward()
        assert torch.allclose(at_ids, dense_at.detach(), atol=1e-9), (rank, at_ids, dense_at)
        assert torch.allclose(local.grad, dense.grad[:, shard], atol=1e-9), (rank, local.grad, dense.grad[:, shard])
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("divergence", _DIVERGENCES)
def test_sharded_divergences_match_the_dense_computation_across_vocab_shards(divergence: str) -> None:
    # Every rank sees one vocab shard and the all-reduced totals, the way
    # tensor-parallel Megatron hands the loss its logits; the gradient on a
    # shard must be the dense gradient's slice, coupling through the global
    # log-sum-exp included.
    import socket

    import torch.multiprocessing as multiprocessing

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    multiprocessing.spawn(_sharded_worker, args=(4, port, divergence, 3), nprocs=4, join=True)


# --- the teacher's weights -----------------------------------------------------------


class FakeBackuper:
    def __init__(self, model: dict[str, torch.Tensor]) -> None:
        self._model = model
        self._backups: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def backup_tags(self) -> set[str]:
        return set(self._backups)

    def backup(self, tag: str) -> None:
        self._backups[tag] = {name: tensor.detach().clone() for name, tensor in self._model.items()}

    def get(self, tag: str) -> dict[str, torch.Tensor]:
        return self._backups[tag]

    def restore(self, tag: str) -> None:
        for name, tensor in self._backups[tag].items():
            self._model[name].copy_(tensor)


class FakeActor:
    def __init__(self) -> None:
        self.model = {"w": torch.full((3,), 1.0, dtype=torch.bfloat16), "steps": torch.tensor([0])}
        self.weights_backuper = FakeBackuper(self.model)
        self.active = ACTOR_TAG
        self.loaded: list[tuple[str, str]] = []

    def _switch_model(self, tag: str) -> None:
        self.weights_backuper.restore(tag)
        self.active = tag

    def load_other_checkpoint(self, tag: str, path: str) -> None:
        self.loaded.append((tag, path))
        self.model["w"].fill_(7.0)
        self.weights_backuper.backup(tag)
        self.active = tag

    def train_step(self, delta: float) -> None:
        self.model["w"].add_(delta)
        self.model["steps"].add_(1)


@pytest.mark.unit
def test_the_moving_copy_is_seeded_from_the_actor_then_moves_toward_it_in_float32() -> None:
    actor = FakeActor()
    teacher = MovingCopy(0.01)

    teacher.switch_in(actor)
    assert actor.active == TEACHER_TAG
    assert actor.model["w"].tolist() == [1.0, 1.0, 1.0]  # the seed is the actor's weights
    teacher.switch_out(actor)
    assert actor.active == ACTOR_TAG

    actor.train_step(1.0)
    teacher.switch_in(actor)
    # ref = 0.99 * ref + 0.01 * policy: a step the bfloat16 copy alone would round away.
    expected = 0.99 * 1.0 + 0.01 * 2.0
    assert torch.allclose(actor.model["w"].float(), torch.full((3,), expected), atol=1e-2)
    assert actor.weights_backuper.get(TEACHER_TAG)["w"].dtype == torch.bfloat16
    assert actor.model["steps"].tolist() == [0]  # integer buffers stay the seed's, unmixed
    teacher.switch_out(actor)
    assert actor.model["w"].tolist() == [2.0, 2.0, 2.0]  # the trained weights are back
    assert actor.model["steps"].tolist() == [1]


@pytest.mark.unit
def test_a_frozen_copy_keeps_the_seed() -> None:
    actor = FakeActor()
    teacher = MovingCopy(0.0)
    teacher.switch_in(actor)
    teacher.switch_out(actor)
    actor.train_step(5.0)

    teacher.switch_in(actor)
    assert actor.model["w"].tolist() == [1.0, 1.0, 1.0]
    teacher.switch_out(actor)
    assert actor.model["w"].tolist() == [6.0, 6.0, 6.0]
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MovingCopy(1.0)


@pytest.mark.unit
def test_a_separate_checkpoint_is_loaded_once_beside_the_actor() -> None:
    actor = FakeActor()
    teacher = SeparateCheckpoint("/models/teacher")

    teacher.switch_in(actor)
    assert actor.loaded == [(TEACHER_TAG, "/models/teacher")]
    assert actor.model["w"].tolist() == [7.0, 7.0, 7.0]
    teacher.switch_out(actor)
    assert actor.model["w"].tolist() == [1.0, 1.0, 1.0]

    actor.train_step(1.0)
    teacher.switch_in(actor)
    assert actor.loaded == [(TEACHER_TAG, "/models/teacher")]  # not loaded again
    assert actor.model["w"].tolist() == [7.0, 7.0, 7.0]
    teacher.switch_out(actor)
    assert actor.model["w"].tolist() == [2.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="checkpoint"):
        SeparateCheckpoint(" ")


@pytest.mark.unit
def test_the_current_weights_need_no_switch() -> None:
    actor = FakeActor()
    teacher = CurrentWeights()
    teacher.switch_in(actor)
    teacher.switch_out(actor)
    assert actor.weights_backuper.backup_tags == set()
    assert actor.active == ACTOR_TAG


@pytest.mark.unit
def test_the_settings_pick_the_teacher_weights() -> None:
    assert isinstance(teacher_weights(DistillSettings(teacher_update_rate=1.0)), CurrentWeights)
    assert isinstance(teacher_weights(DistillSettings(teacher_update_rate=0.02)), MovingCopy)
    assert isinstance(teacher_weights(DistillSettings(teacher_update_rate=0.0)), MovingCopy)
    separate = teacher_weights(DistillSettings(teacher="separate", teacher_checkpoint="/models/teacher"))
    assert isinstance(separate, SeparateCheckpoint)
