"""Pin the Slime SAO tensor code to the independent pure-Python test oracle.

The reference (torch/numpy free) is the source of truth for SAO numerics and is
exercised by ``test_sao_reference.py``. The tensor code in
``recipes/sao/slime/objective.py`` is a tensor
transcription of it, reached by Slime through its generic custom-function
hooks. These tests run both on the same inputs and assert equality, so a
divergence between the transcription and the spec fails loudly.

They need real ``torch`` but *not* Megatron — a single-rank ``megatron.core.mpu``
shim supplies the context-parallel size — and no GPU: everything runs on CPU
tensors at cp_size=1. That makes them cheap enough for the minimal CI gate, which
installs CPU torch, so they are not ignored there. Keeping them in the gate
matters because SAO's masked-token counter shares the ``pg_clipfrac`` key with
PPO's clamp counter; without this parity check a regression from hard mask back
to clamp would report the same metric name and hide.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _install_single_rank_mpu() -> None:
    """Provide a cp_size=1 ``megatron.core.mpu`` so the helpers take the
    no-context-parallel path without a real Megatron install.

    Only ``megatron.core`` is registered, never the top-level ``megatron``
    package: ``from megatron.core import mpu`` resolves straight out of
    ``sys.modules``, while ``import megatron`` keeps failing. Other modules in
    the suite gate on ``importorskip("megatron")`` and must still skip — a fake
    parent package would make that gate pass and then break on the first real
    submodule import.
    """
    try:
        from megatron.core import mpu as real_mpu  # noqa: F401
    except ImportError:
        pass
    else:
        return
    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    sys.modules["megatron.core"] = core


_install_single_rank_mpu()

from argparse import Namespace

from recipes.sao.slime.objective import compute_sao_loss, get_skip_observation_advantages_and_returns, sao_advantages
from reef.train.slime_backend.loss_families import resolve_loss_family

from .reference_algorithms import sao


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "action_mask", "reward", "gamma", "lambd"),
    [
        ([1.0, 2.0, 3.0], [1, 1, 1], 5.0, 1.0, 1.0),
        ([10.0, 99.0, 99.0, 4.0], [1, 0, 0, 1], 1.0, 1.0, 1.0),
        ([1.0, 1.0, 1.0], [1, 1, 1], 8.0, 0.5, 0.5),
        ([0.5, -0.5, 2.0, -1.0, 0.0], [1, 0, 1, 0, 1], -2.0, 0.99, 0.95),
    ],
)
def test_skip_observation_gae_matches_reference(values, action_mask, reward, gamma, lambd) -> None:
    response_len = len(values)
    values_t = torch.tensor(values, dtype=torch.float64)
    action_mask_t = torch.tensor(action_mask, dtype=torch.float64)

    adv_t, ret_t = get_skip_observation_advantages_and_returns(
        total_len=response_len,
        response_len=response_len,
        values=values_t,
        reward=reward,
        action_mask=action_mask_t,
        gamma=gamma,
        lambd=lambd,
    )

    # Reference builds per-token rewards with the terminal reward on the last
    # action token — mirror that placement to compare like-for-like.
    ref_rewards = [0.0] * response_len
    action_positions = [i for i, m in enumerate(action_mask) if m]
    if action_positions:
        ref_rewards[action_positions[-1]] = reward
    ref_adv, ref_ret = sao.skip_observation_gae(values, ref_rewards, action_mask, gamma=gamma, lambd=lambd)

    assert adv_t.tolist() == pytest.approx(ref_adv, abs=1e-9)
    assert ret_t.tolist() == pytest.approx(ref_ret, abs=1e-9)


@pytest.mark.unit
def test_skip_observation_gae_advantages_are_detached() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    action_mask = torch.tensor([1.0, 1.0, 1.0])

    adv, _ = get_skip_observation_advantages_and_returns(3, 3, values, 5.0, action_mask, 1.0, 1.0)

    # Advantages are a stop-gradient signal into the policy loss.
    assert not adv.requires_grad


@pytest.mark.unit
@pytest.mark.parametrize("eps_low", [0.3, 0.8])
@pytest.mark.parametrize("eps_high", [5.0, 3.0])
def test_compute_sao_loss_calibration_matches_reference(eps_low, eps_high) -> None:
    # Rollout (behaviour) log-probs and current-policy log-probs. ppo_kl is the
    # caller's convention: logπrollout - logπθ, so the loss recovers r_t.
    rollout_log_probs = torch.tensor([0.0, -1.0, 0.5, -2.0, 0.1], dtype=torch.float64)
    policy_log_probs = torch.tensor([0.0, 0.4, -0.7, 1.5, 0.2], dtype=torch.float64)
    advantages = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0], dtype=torch.float64)
    ppo_kl = rollout_log_probs - policy_log_probs

    pg_losses, clipfrac = compute_sao_loss(ppo_kl, policy_log_probs, advantages, eps_low, eps_high)

    # Reference path: ratios -> calibration weights -> per-token surrogate.
    ratios = sao.importance_ratios(policy_log_probs.tolist(), rollout_log_probs.tolist())
    weights = sao.calibration_weights(ratios, eps_low=eps_low, eps_high=eps_high)
    ref_losses = [
        -w * a * lp for w, a, lp in zip(weights, advantages.tolist(), policy_log_probs.tolist(), strict=True)
    ]
    ref_clip = [0.0 if w != 0.0 else 1.0 for w in weights]

    assert pg_losses.tolist() == pytest.approx(ref_losses, abs=1e-9)
    assert clipfrac.tolist() == pytest.approx(ref_clip, abs=1e-9)


@pytest.mark.unit
def test_compute_sao_loss_is_a_hard_mask_not_a_clamp() -> None:
    # A token far outside the trust region must contribute exactly zero, not a
    # clamped-to-bound gradient.
    rollout_log_probs = torch.tensor([0.0], dtype=torch.float64)
    policy_log_probs = torch.tensor([5.0], dtype=torch.float64)  # ratio = exp(5) ~ 148
    advantages = torch.tensor([10.0], dtype=torch.float64)
    ppo_kl = rollout_log_probs - policy_log_probs

    pg_losses, clipfrac = compute_sao_loss(ppo_kl, policy_log_probs, advantages, 0.3, 5.0)

    assert pg_losses.item() == pytest.approx(0.0)
    assert clipfrac.item() == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_sao_loss_calibration_carries_no_gradient() -> None:
    # The calibration weight is detached; gradient flows only through logπθ.
    rollout_log_probs = torch.tensor([0.0, 0.0], dtype=torch.float32)
    policy_log_probs = torch.tensor([0.1, -0.1], dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([1.0, 1.0], dtype=torch.float32)
    ppo_kl = rollout_log_probs - policy_log_probs.detach()

    pg_losses, _ = compute_sao_loss(ppo_kl, policy_log_probs, advantages, 0.3, 5.0)
    pg_losses.sum().backward()

    # d/dlogπθ [ -sg(f(r))*A*logπθ ] = -sg(f(r))*A. For r in-band, f(r)=r.
    ratios = [torch.exp(policy_log_probs[i].detach() - rollout_log_probs[i]).item() for i in range(2)]
    expected_grad = [-r * 1.0 for r in ratios]
    assert policy_log_probs.grad.tolist() == pytest.approx(expected_grad, abs=1e-6)


@pytest.mark.unit
def test_explained_variance_tensor_matches_reference() -> None:
    # The value-loss metric is computed inline in loss.py; check the formula it
    # uses against the reference on the same inputs.
    returns = torch.tensor([1.0, 2.0, 5.0, 4.0], dtype=torch.float64)
    old_values = torch.tensor([1.2, 1.8, 4.0, 4.5], dtype=torch.float64)

    var_returns = returns.var(unbiased=False)
    ev_tensor = (1.0 - (returns - old_values).var(unbiased=False) / var_returns).item()
    ev_ref = sao.explained_variance(returns.tolist(), old_values.tolist())

    assert ev_tensor == pytest.approx(ev_ref, abs=1e-9)


@pytest.mark.unit
def test_explained_variance_action_mask_restriction_matches_reference() -> None:
    # Production (loss.py value_loss_function) restricts the metric to action
    # tokens when the batch carries action masks: observation tokens have zero
    # returns by construction and would dilute the variance. Mirror the tensor
    # path (boolean select, then the same formula) against the masked reference.
    returns = torch.tensor([1.0, 0.0, 5.0, 4.0, 0.0], dtype=torch.float64)
    old_values = torch.tensor([1.2, 9.0, 4.0, 4.5, -9.0], dtype=torch.float64)
    action_mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32)

    selected = action_mask.bool()
    ev_returns, ev_values = returns[selected], old_values[selected]
    var_returns = ev_returns.var(unbiased=False)
    ev_tensor = (1.0 - (ev_returns - ev_values).var(unbiased=False) / var_returns).item()

    ev_ref = sao.explained_variance(returns.tolist(), old_values.tolist(), action_mask.tolist())
    ev_unmasked = sao.explained_variance(returns.tolist(), old_values.tolist())

    assert ev_tensor == pytest.approx(ev_ref, abs=1e-9)
    # The observation tokens' garbage values must actually change the answer,
    # or this test would not distinguish the masked metric from the old one.
    assert ev_tensor != pytest.approx(ev_unmasked, abs=1e-6)


def _sao_hook_args(**overrides) -> Namespace:
    values = {
        "gamma": 1.0,
        "lambd": 1.0,
        "sao_length_adaptive_lambda": False,
        "sao_lambda_alpha": 1.5,
    }
    values.update(overrides)
    return Namespace(**values)


def _hook_rollout_data(values_lists, rewards, action_masks=None):
    values = [torch.tensor(sample, dtype=torch.float64) for sample in values_lists]
    data = {
        "values": values,
        "rewards": list(rewards),
        "response_lengths": [len(sample) for sample in values_lists],
        "total_lengths": [len(sample) for sample in values_lists],
    }
    if action_masks is not None:
        data["action_masks"] = [torch.tensor(mask, dtype=torch.float64) for mask in action_masks]
    return data


def _reference_gae(values, reward, action_mask, gamma, lambd):
    rewards = [0.0] * len(values)
    action_positions = [i for i, m in enumerate(action_mask) if m]
    if action_positions:
        rewards[action_positions[-1]] = reward
    return sao.skip_observation_gae(values, rewards, action_mask, gamma=gamma, lambd=lambd)


@pytest.mark.unit
def test_sao_advantages_hook_fixed_lambda_matches_reference() -> None:
    args = _sao_hook_args(lambd=0.9, gamma=0.99)
    data = _hook_rollout_data([[1.0, 2.0, 3.0], [0.5, -0.5]], [5.0, -1.0], [[1, 0, 1], [1, 1]])

    sao_advantages(args, data)

    for i, (values, mask, reward) in enumerate([([1.0, 2.0, 3.0], [1, 0, 1], 5.0), ([0.5, -0.5], [1, 1], -1.0)]):
        ref_adv, ref_ret = _reference_gae(values, reward, mask, 0.99, 0.9)
        assert data["advantages"][i].tolist() == pytest.approx(ref_adv, abs=1e-9)
        assert data["returns"][i].tolist() == pytest.approx(ref_ret, abs=1e-9)


@pytest.mark.unit
def test_sao_advantages_hook_length_adaptive_lambda_matches_reference() -> None:
    # Two samples of different lengths: each gets its own lambda from
    # 1 - 1/(alpha*l), matching the reference selection per sample.
    args = _sao_hook_args(sao_length_adaptive_lambda=True, sao_lambda_alpha=1.5)
    samples = [([1.0, 2.0, 3.0, 4.0], 2.0), ([0.5, 1.5], -3.0)]
    data = _hook_rollout_data([values for values, _ in samples], [reward for _, reward in samples])

    sao_advantages(args, data)

    for i, (values, reward) in enumerate(samples):
        lambd = sao.role_lambda("actor", len(values), length_adaptive=True, alpha=1.5)
        ref_adv, ref_ret = _reference_gae(values, reward, [1] * len(values), 1.0, lambd)
        assert data["advantages"][i].tolist() == pytest.approx(ref_adv, abs=1e-9)
        assert data["returns"][i].tolist() == pytest.approx(ref_ret, abs=1e-9)


@pytest.mark.unit
def test_sao_advantages_hook_clamps_nonpositive_length_to_lambda_zero() -> None:
    # response_lengths[i] <= 0 must select lambda 0.0, not divide by zero.
    args = _sao_hook_args(sao_length_adaptive_lambda=True, sao_lambda_alpha=1.5)
    data = _hook_rollout_data([[1.0, 2.0]], [4.0])
    data["response_lengths"] = [0]

    sao_advantages(args, data)

    # The recurrence runs over range(0): no token is reached, everything zero.
    assert data["advantages"][0].tolist() == [0.0, 0.0]
    assert data["returns"][0].tolist() == [0.0, 0.0]


@pytest.mark.unit
def test_sao_advantages_hook_defaults_to_all_action_tokens_without_masks() -> None:
    # A rollout without an explicit action mask is a pure model trajectory:
    # the hook must behave exactly as if every response token were an action.
    args = _sao_hook_args()
    values, reward = [1.0, 2.0, 3.0], 5.0
    without_mask = _hook_rollout_data([values], [reward])
    with_mask = _hook_rollout_data([values], [reward], [[1, 1, 1]])

    sao_advantages(args, without_mask)
    sao_advantages(args, with_mask)

    assert without_mask["advantages"][0].tolist() == pytest.approx(with_mask["advantages"][0].tolist(), abs=1e-12)
    assert without_mask["returns"][0].tolist() == pytest.approx(with_mask["returns"][0].tolist(), abs=1e-12)


@pytest.mark.unit
def test_sao_advantages_hook_requires_critic_values() -> None:
    args = _sao_hook_args()
    data = _hook_rollout_data([[1.0]], [1.0])
    del data["values"]

    with pytest.raises(ValueError, match="requires critic values"):
        sao_advantages(args, data)


@pytest.mark.unit
def test_critic_role_uses_lambda_one_while_actor_adapts_on_the_same_batch() -> None:
    # Role separation (paper: lambda_critic = 1, lambda_policy length-adaptive):
    # the critic role's args come out of configure_critic_args with the
    # adaptive lambda disabled and lambd pinned to sao_critic_lambda, so the
    # same hook produces MC-flavoured value targets for the critic and
    # length-adaptive advantages for the actor on the same synthetic batch.
    values, reward, mask = [1.0, -2.0, 3.0, 0.5], 4.0, [1, 0, 1, 1]
    actor_args = _sao_hook_args(sao_length_adaptive_lambda=True, sao_lambda_alpha=1.5, sao_critic_lambda=1.0)
    critic_args = _sao_hook_args(sao_length_adaptive_lambda=True, sao_lambda_alpha=1.5, sao_critic_lambda=1.0)
    resolve_loss_family("sao").configure_critic_args(critic_args)

    actor_data = _hook_rollout_data([values], [reward], [mask])
    critic_data = _hook_rollout_data([values], [reward], [mask])
    sao_advantages(actor_args, actor_data)
    sao_advantages(critic_args, critic_data)

    actor_lambda = sao.role_lambda("actor", len(values), length_adaptive=True, alpha=1.5)
    critic_lambda = sao.role_lambda("critic", len(values), length_adaptive=True, alpha=1.5)
    ref_actor_adv, _ = _reference_gae(values, reward, mask, 1.0, actor_lambda)
    _, ref_critic_ret = _reference_gae(values, reward, mask, 1.0, critic_lambda)

    assert actor_data["advantages"][0].tolist() == pytest.approx(ref_actor_adv, abs=1e-9)
    assert critic_data["returns"][0].tolist() == pytest.approx(ref_critic_ret, abs=1e-9)
    # The separation must be observable: the two lambdas differ, so the
    # critic's value targets differ from what the actor's lambda would give.
    _, actor_lambda_ret = _reference_gae(values, reward, mask, 1.0, actor_lambda)
    assert critic_data["returns"][0].tolist() != pytest.approx(actor_lambda_ret, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
