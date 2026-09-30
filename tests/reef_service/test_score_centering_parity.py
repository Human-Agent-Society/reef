"""Score centering's term on CPU torch: added to a weighted policy-gradient loss, against a full-vocabulary reference.

A loss ``-A * sg[f(r_y)] * log p_y`` plus the term must have the gradient
``-A (f(r_y) s_y - sum_v q_v f(r_v) s_v)`` on the trainer's logits, with
``s_v = e_v - p`` the score of token ``v`` and ``r = p / q``: the loss's
weighted score, centered under the sampler. With the head covering the
vocabulary the sum must match it exactly; with a smaller head it must match
exactly when the sampler's tail is proportional to the trainer's (the
approximation's assumption), which separates approximation error from
implementation error. SAO's own loss, with the weight SAO declares, must
compose the same way. The trainer's log-probs come from the tensor-parallel
gather at world size one, and across four ranks in the last test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from recipes.sao.slime import SaoAlgorithm
from recipes.sao.slime.objective import compute_sao_loss
from reef.train.slime_backend.algorithm import PolicyGradientWeight
from reef.train.slime_backend.score_centering import ScoreCenteringSettings
from reef.train.slime_backend.score_centering.heads import PADDING_LOG_PROB
from reef.train.slime_backend.score_centering.term import centering_term, importance_weight
from reef.train.slime_backend.vocab_parallel import gather_log_probs_at_ids

VOCAB = 7
MIN_TAIL_MASS = ScoreCenteringSettings().min_tail_mass
WEIGHTS = (
    PolicyGradientWeight("none"),
    PolicyGradientWeight("truncated", upper=1.2),
    PolicyGradientWeight("masked", lower=0.7, upper=1.4),
)


def distributions(seed: int, rows: int = 4, mismatch: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    trainer_logits = torch.randn(rows, VOCAB, generator=generator, dtype=torch.float64)
    sampler_log_probs = torch.log_softmax(
        trainer_logits + mismatch * torch.randn(rows, VOCAB, generator=generator, dtype=torch.float64), dim=-1
    )
    return trainer_logits, sampler_log_probs


def loss_with_term_gradient(
    trainer_logits: torch.Tensor,
    sampler_log_probs: torch.Tensor,
    head_ids: torch.Tensor,
    sampled: torch.Tensor,
    advantages: torch.Tensor,
    weight: PolicyGradientWeight,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient on the logits of ``-A sg[f(r_y)] log p_y`` plus the term, and the term's correction."""
    logits = trainer_logits.detach().clone().requires_grad_(True)
    trainer_at = gather_log_probs_at_ids(logits, torch.cat([head_ids, sampled[:, None]], -1), None, 1, 0)
    with torch.no_grad():
        sampled_ratio = (trainer_at[:, -1] - sampler_log_probs.gather(-1, sampled[:, None])[:, 0]).exp()
    base = -advantages * importance_weight(sampled_ratio, weight) * trainer_at[:, -1]
    result = centering_term(
        trainer_at[:, :-1], torch.gather(sampler_log_probs, -1, head_ids), advantages, weight, MIN_TAIL_MASS
    )
    (base + result.term).sum().backward()
    return logits.grad, result.correction


def reference_gradient(
    trainer_logits: torch.Tensor,
    sampler_log_probs: torch.Tensor,
    sampled: torch.Tensor,
    advantages: torch.Tensor,
    weight: PolicyGradientWeight,
) -> torch.Tensor:
    """``-A (f(r_y) s_y - sum_v q_v f(r_v) s_v)`` per row, over the whole vocabulary."""
    probabilities = torch.softmax(trainer_logits, dim=-1)
    sampler = sampler_log_probs.exp()
    weights = importance_weight(probabilities / sampler, weight)
    scores = torch.eye(VOCAB, dtype=torch.float64)[None, :, :] - probabilities[:, None, :]  # [R, v, logit]
    rows = torch.arange(trainer_logits.size(0))
    sampled_term = weights[rows, sampled][:, None] * scores[rows, sampled]
    expected_term = ((sampler * weights)[:, :, None] * scores).sum(dim=1)
    return -advantages[:, None] * (sampled_term - expected_term)


@pytest.mark.unit
@pytest.mark.parametrize("weight", WEIGHTS, ids=lambda w: w.kind)
def test_full_vocabulary_head_centers_the_weighted_score_exactly(weight: PolicyGradientWeight) -> None:
    trainer_logits, sampler_log_probs = distributions(0)
    rows = trainer_logits.size(0)
    head_ids = torch.arange(VOCAB).expand(rows, VOCAB).contiguous()
    sampled = torch.tensor([0, 3, 6, 2])
    advantages = torch.tensor([1.0, -0.5, 2.0, 0.25], dtype=torch.float64)

    gradient, _ = loss_with_term_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, weight)

    torch.testing.assert_close(
        gradient, reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, weight)
    )


@pytest.mark.unit
@pytest.mark.parametrize("weight", WEIGHTS, ids=lambda w: w.kind)
def test_constant_reward_has_zero_expected_update_under_the_exact_sampler(weight: PolicyGradientWeight) -> None:
    # The drift the term removes: E_{y ~ q}[gradient] is zero for any sampler
    # once the weighted score is centered, whatever the mismatch.
    trainer_logits, sampler_log_probs = distributions(1, rows=1)
    head_ids = torch.arange(VOCAB)[None, :]
    expected = torch.zeros_like(trainer_logits)
    for token in range(VOCAB):
        gradient, _ = loss_with_term_gradient(
            trainer_logits,
            sampler_log_probs,
            head_ids,
            torch.tensor([token]),
            torch.ones(1, dtype=torch.float64),
            weight,
        )
        expected += sampler_log_probs[0, token].exp() * gradient
    torch.testing.assert_close(expected, torch.zeros_like(expected), atol=1e-12, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("weight", WEIGHTS, ids=lambda w: w.kind)
def test_matched_distributions_add_nothing(weight: PolicyGradientWeight) -> None:
    trainer_logits, _ = distributions(2)
    sampler_log_probs = torch.log_softmax(trainer_logits, dim=-1)
    head_ids = torch.topk(sampler_log_probs, k=3, dim=-1).indices
    sampled = head_ids[:, 0]
    advantages = torch.ones(trainer_logits.size(0), dtype=torch.float64)

    gradient, correction = loss_with_term_gradient(
        trainer_logits, sampler_log_probs, head_ids, sampled, advantages, weight
    )

    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=1e-12, rtol=0)
    # Without a correction the gradient is the plain on-policy -A s_y.
    scores = torch.eye(VOCAB, dtype=torch.float64)[sampled] - torch.softmax(trainer_logits, dim=-1)
    torch.testing.assert_close(gradient, -scores)


def proportional_tail_sampler(trainer_logits: torch.Tensor, head_ids: torch.Tensor, seed: int) -> torch.Tensor:
    """A sampler that differs from the trainer on the head and is ``rho p`` on the tail."""
    probabilities = torch.softmax(trainer_logits, dim=-1)
    generator = torch.Generator().manual_seed(seed)
    head_probabilities = torch.rand(head_ids.shape, generator=generator, dtype=torch.float64) + 0.1
    head_probabilities = 0.8 * head_probabilities / head_probabilities.sum(dim=-1, keepdim=True)
    in_head = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, head_ids, True)
    tail = torch.where(in_head, torch.zeros_like(probabilities), probabilities)
    sampler = 0.2 * tail / tail.sum(dim=-1, keepdim=True)
    return sampler.scatter(-1, head_ids, head_probabilities).log()


@pytest.mark.unit
@pytest.mark.parametrize("weight", WEIGHTS, ids=lambda w: w.kind)
def test_top_k_head_is_exact_when_the_sampler_tail_is_proportional(weight: PolicyGradientWeight) -> None:
    trainer_logits, _ = distributions(3)
    head_ids = torch.tensor([[0, 1, 2], [4, 5, 6], [1, 3, 5], [6, 0, 2]])
    sampler_log_probs = proportional_tail_sampler(trainer_logits, head_ids, seed=3)
    # Sampled tokens inside and outside the head.
    sampled = torch.tensor([0, 3, 6, 2])
    advantages = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=torch.float64)

    gradient, _ = loss_with_term_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, weight)

    torch.testing.assert_close(
        gradient, reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, weight)
    )


@pytest.mark.unit
def test_top_k_head_approximates_an_arbitrary_tail() -> None:
    # An arbitrary sampler tail is not rho p: the head-only estimate differs
    # from the exact one, but only by the tail's share.
    weight = PolicyGradientWeight("none")
    trainer_logits, sampler_log_probs = distributions(4)
    head_ids = torch.topk(sampler_log_probs, k=5, dim=-1).indices
    sampled = head_ids[:, 0]
    advantages = torch.ones(trainer_logits.size(0), dtype=torch.float64)

    approximate, _ = loss_with_term_gradient(trainer_logits, sampler_log_probs, head_ids, sampled, advantages, weight)
    exact = reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, weight)

    assert not torch.allclose(approximate, exact)
    tail_mass = 1.0 - torch.gather(sampler_log_probs.exp(), -1, head_ids).sum(dim=-1)
    assert torch.all((approximate - exact).abs().sum(dim=-1) <= 4 * tail_mass)


@pytest.mark.unit
def test_sao_loss_plus_its_declared_weight_centers_its_own_score() -> None:
    # SAO's surrogate masks the ratio to its trust region; the weight it
    # declares must be that mask, so its loss plus the term is the paper's
    # MIS + SC gradient with SAO's bounds.
    weight = SaoAlgorithm().policy_gradient_weight(SimpleNamespace(eps_clip=0.2, eps_clip_high=0.28))
    assert weight == PolicyGradientWeight("masked", lower=0.8, upper=1.28)
    trainer_logits, sampler_log_probs = distributions(7, rows=6, mismatch=0.3)
    rows = trainer_logits.size(0)
    head_ids = torch.arange(VOCAB).expand(rows, VOCAB).contiguous()
    sampled = torch.tensor([0, 1, 2, 3, 4, 5])
    advantages = torch.tensor([1.0, -1.0, 0.5, 2.0, -0.3, 1.5], dtype=torch.float64)
    ratios = (torch.softmax(trainer_logits, -1) / sampler_log_probs.exp()).gather(-1, sampled[:, None])[:, 0]
    assert bool(((ratios > 0.8) & (ratios < 1.28)).any()) and bool(((ratios <= 0.8) | (ratios >= 1.28)).any())

    logits = trainer_logits.detach().clone().requires_grad_(True)
    trainer_at = gather_log_probs_at_ids(logits, torch.cat([head_ids, sampled[:, None]], -1), None, 1, 0)
    sampled_rollout = sampler_log_probs.gather(-1, sampled[:, None])[:, 0]
    sao_losses, _ = compute_sao_loss(sampled_rollout - trainer_at[:, -1], trainer_at[:, -1], advantages, 0.2, 0.28)
    result = centering_term(
        trainer_at[:, :-1], torch.gather(sampler_log_probs, -1, head_ids), advantages, weight, MIN_TAIL_MASS
    )
    (sao_losses + result.term).sum().backward()

    torch.testing.assert_close(
        logits.grad, reference_gradient(trainer_logits, sampler_log_probs, sampled, advantages, weight)
    )


@pytest.mark.unit
def test_coefficients_are_detached() -> None:
    trainer_logits, sampler_log_probs = distributions(5, rows=1)
    head = torch.log_softmax(trainer_logits, dim=-1)[:, :3].clone().requires_grad_(True)
    advantages = torch.tensor([2.0], dtype=torch.float64)
    weight = PolicyGradientWeight("truncated", upper=5.0)

    result = centering_term(head, sampler_log_probs[:, :3], advantages, weight, MIN_TAIL_MASS)
    result.term.sum().backward()

    # d term / d log p_v = A (q_v f(r_v) - alpha p_v), the coefficients held fixed.
    probabilities = head.detach().exp()
    sampler = sampler_log_probs[:, :3].exp()
    tail_ratio = (1 - sampler.sum(-1)) / (1 - probabilities.sum(-1))
    alpha = tail_ratio * importance_weight(1 / tail_ratio, weight)
    expected = advantages[:, None] * (
        sampler * importance_weight(probabilities / sampler, weight) - alpha[:, None] * probabilities
    )
    torch.testing.assert_close(head.grad, expected)


@pytest.mark.unit
def test_declared_weights_evaluate_elementwise() -> None:
    ratios = torch.tensor([0.4, 0.5, 1.0, 2.0, 2.5], dtype=torch.float64)
    assert importance_weight(ratios, PolicyGradientWeight("none")).tolist() == [1.0] * 5
    assert importance_weight(ratios, PolicyGradientWeight("truncated", upper=2.0)).tolist() == [
        0.4,
        0.5,
        1.0,
        2.0,
        2.0,
    ]
    # Masked bounds are strict, as SAO's trust region is.
    masked = PolicyGradientWeight("masked", lower=0.5, upper=2.0)
    assert importance_weight(ratios, masked).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.mark.unit
def test_nearly_empty_tails_are_clipped_and_counted() -> None:
    trainer_logits = torch.tensor([[8.0, 7.0, -30.0, -30.0]], dtype=torch.float64)
    trainer_log_probs = torch.log_softmax(trainer_logits, dim=-1)
    sampler_log_probs = torch.log(torch.tensor([[0.6, 0.4 - 1e-12, 5e-13, 5e-13]], dtype=torch.float64))

    result = centering_term(
        trainer_log_probs[:, :2].requires_grad_(True),
        sampler_log_probs[:, :2],
        torch.ones(1, dtype=torch.float64),
        PolicyGradientWeight("truncated", upper=2.0),
        MIN_TAIL_MASS,
    )

    assert result.tail_clipped.tolist() == [1.0]
    assert torch.isfinite(result.term).all()
    assert torch.isfinite(result.tail_ratio).all()


@pytest.mark.unit
def test_padded_positions_stay_finite() -> None:
    # The bridge's placeholder head for an untrained position: probability zero everywhere.
    trainer_logits, _ = distributions(6, rows=1)
    logits = trainer_logits.clone().requires_grad_(True)
    trainer_at = gather_log_probs_at_ids(logits, torch.zeros(1, 3, dtype=torch.long), None, 1, 0)
    for weight in WEIGHTS:
        result = centering_term(
            trainer_at,
            torch.full((1, 3), PADDING_LOG_PROB, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float64),
            weight,
            MIN_TAIL_MASS,
        )
        assert torch.isfinite(result.term).all()
        (result.term * 0).sum().backward(retain_graph=True)
        assert torch.isfinite(logits.grad).all()


def sharded_worker(rank: int, world: int, port: int) -> None:
    """One tensor-parallel rank: the gradient on its vocab shard must be the dense gradient's slice."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        vocab = 24
        generator = torch.Generator().manual_seed(7)
        trainer_logits = torch.randn(3, vocab, generator=generator, dtype=torch.float64) * 2
        sampler_log_probs = torch.log_softmax(
            trainer_logits + torch.randn(3, vocab, generator=generator, dtype=torch.float64), dim=-1
        )
        head_ids = torch.topk(sampler_log_probs, k=5, dim=-1).indices
        advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        weight = PolicyGradientWeight("truncated", upper=2.0)
        shard = slice(rank * vocab // world, (rank + 1) * vocab // world)

        def _term(logits: torch.Tensor, group: object, tp_world: int, tp_rank: int) -> torch.Tensor:
            trainer_at = gather_log_probs_at_ids(logits, head_ids, group, tp_world, tp_rank)
            return centering_term(
                trainer_at, torch.gather(sampler_log_probs, -1, head_ids), advantages, weight, MIN_TAIL_MASS
            ).term

        dense = trainer_logits.clone().requires_grad_(True)
        dense_term = _term(dense, None, 1, 0)
        dense_term.sum().backward()
        local = trainer_logits[:, shard].clone().requires_grad_(True)
        local_term = _term(local, dist.group.WORLD, world, rank)
        local_term.sum().backward()
        assert torch.allclose(local_term, dense_term.detach(), atol=1e-9), (rank, local_term, dense_term)
        assert torch.allclose(local.grad, dense.grad[:, shard], atol=1e-9), (rank, local.grad, dense.grad[:, shard])
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_sharded_term_matches_the_dense_computation_across_vocab_shards() -> None:
    import socket

    import torch.multiprocessing as multiprocessing

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    multiprocessing.spawn(sharded_worker, args=(4, port), nprocs=4, join=True)


def test_float32_inputs_track_the_float64_term_when_the_head_covers_nearly_everything() -> None:
    """The coefficients are detached, so they are evaluated wide enough to survive ``1 - head_mass``.

    A head that leaves a tail near ``min_tail_mass`` is the case the correction is most useful in and the
    one float32 handles worst: the subtraction cancels, and the tail ratio and coefficients inherit the
    error. Evaluating the same already-rounded float32 log-probs both ways pins that drift down.
    """
    tail = torch.tensor([0.8e-6, 1.2e-6, 3e-6, 1e-9], dtype=torch.float64)
    head = torch.linspace(1, 3, 128, dtype=torch.float64)
    head /= head.sum()
    trainer = ((1 - tail[:, None]) * head).log().float()
    sampler = ((1 - tail.flip(0)[:, None]) * head.flip(0)).log().float()
    advantages = torch.tensor([0.2, -0.5, 0.1, -0.2])
    weight = PolicyGradientWeight("truncated", upper=2.0)

    def run(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = trainer.to(dtype).detach().requires_grad_()
        term = centering_term(inputs, sampler.to(dtype), advantages.to(dtype), weight, 1e-6).term
        (gradient,) = torch.autograd.grad(term.sum(), inputs)
        return term.detach().double(), gradient.double()

    narrow_value, narrow_gradient = run(torch.float32)
    wide_value, wide_gradient = run(torch.float64)

    # The independent algebra the wide path must agree with, straight from the paper's coefficients.
    p, q = trainer.double().exp(), sampler.double().exp()
    rho = (1 - q.sum(-1)).clamp_min(1e-6) / (1 - p.sum(-1)).clamp_min(1e-6)
    alpha = torch.minimum(2 * rho, torch.ones_like(rho))
    expected_gradient = advantages.double()[:, None] * (torch.minimum(p, 2 * q) - alpha[:, None] * p)
    torch.testing.assert_close(wide_gradient, expected_gradient, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(wide_value, (expected_gradient * trainer.double()).sum(-1), atol=1e-12, rtol=1e-10)

    # Float32 storage still rounds the result, but the arithmetic behind it must not add error of its own.
    assert (narrow_value - wide_value).abs().max() < 1e-6
    assert (narrow_gradient - wide_gradient).norm() / wide_gradient.norm() < 1e-6
