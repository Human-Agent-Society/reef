"""Pure-Python reference numerics for SAO (arXiv:2607.07508).

``reference_algorithms.sao`` is an independent, torch/numpy-free oracle for the
Slime SAO objective. The Slime GPU code is a tensor transcription of these
functions; a separate parity test
(``test_sao_parity.py``) pins the two together. These tests fix the expected
values by hand so a change to the reference has to be a deliberate edit here,
not an accident.
"""

from __future__ import annotations

from math import exp, isclose
from types import SimpleNamespace

import pytest

from reef.train.slime_backend.loss_families import resolve_loss_family

from .reference_algorithms import sao


def _sao_backend_args(**overrides):
    values = {
        "loss_type": "policy_loss",
        "use_rollout_logprobs": True,
        "custom_advantage_function_path": "recipes.sao.slime.objective.sao_advantages",
        "custom_pg_loss_function_path": "recipes.sao.slime.objective.sao_loss",
        "use_critic": True,
        "eps_clip": 0.3,
        "eps_clip_high": 5.0,
        "critic_steps_per_actor": 2,
        "num_critic_only_steps": 0,
        "sao_length_adaptive_lambda": True,
        "sao_lambda_alpha": 1.5,
        "sao_critic_lambda": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_sao_backend_contract_accepts_the_exact_objective() -> None:
    resolve_loss_family("sao").validate_backend_args(_sao_backend_args())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_sao_backend_args(use_critic=False), "requires a value model"),
        (_sao_backend_args(eps_clip=1.0), "eps-clip in"),
        (_sao_backend_args(eps_clip_high=-0.1), "eps-clip-high"),
        (_sao_backend_args(critic_steps_per_actor=0), "critic-steps-per-actor"),
        (_sao_backend_args(num_critic_only_steps=-1), "num-critic-only-steps"),
        (_sao_backend_args(sao_lambda_alpha=0.0), "sao-lambda-alpha"),
        (_sao_backend_args(sao_critic_lambda=1.5), "sao-critic-lambda"),
        (_sao_backend_args(sao_critic_lambda=-0.1), "sao-critic-lambda"),
    ],
)
def test_sao_backend_contract_rejects_objective_drift(args, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("sao").validate_backend_args(args)


@pytest.mark.unit
def test_sao_backend_contract_ignores_adaptive_alpha_when_disabled() -> None:
    resolve_loss_family("sao").validate_backend_args(
        _sao_backend_args(sao_length_adaptive_lambda=False, sao_lambda_alpha=0.0)
    )


@pytest.mark.unit
def test_sao_backend_contract_rejects_the_eps_clip_high_fallback() -> None:
    # Slime's parser falls an unset --eps-clip-high back to --eps-clip
    # (symmetric narrow band); slime_validate_args records the fallback in
    # eps_clip_high_explicit and the SAO contract refuses to run on it.
    args = _sao_backend_args(eps_clip_high_explicit=False)
    with pytest.raises(RuntimeError, match=r"explicit --eps-clip-high.*0\.3/5\.0.*0\.8/3\.0"):
        resolve_loss_family("sao").validate_backend_args(args)


@pytest.mark.unit
def test_sao_backend_contract_accepts_an_explicitly_recorded_eps_clip_high() -> None:
    resolve_loss_family("sao").validate_backend_args(_sao_backend_args(eps_clip_high_explicit=True))


@pytest.mark.unit
def test_critic_role_args_disable_the_adaptive_lambda_and_pin_critic_lambda() -> None:
    sao = resolve_loss_family("sao")

    critic_args = _sao_backend_args(lambd=0.95)
    sao.configure_critic_args(critic_args)

    assert critic_args.sao_length_adaptive_lambda is False
    assert critic_args.lambd == 1.0


@pytest.mark.unit
def test_critic_role_args_are_a_no_op_without_the_sao_projection() -> None:
    sao = resolve_loss_family("sao")

    non_sao = SimpleNamespace(lambd=0.95)
    sao.configure_critic_args(non_sao)

    assert non_sao.lambd == 0.95
    assert not hasattr(non_sao, "sao_length_adaptive_lambda")


@pytest.mark.unit
def test_role_lambda_separates_critic_from_length_adaptive_actor() -> None:
    # Same sample, both roles: the actor's lambda adapts to length while the
    # critic's value targets stay at lambda=1 (paper's lambda_critic).
    actor = sao.role_lambda("actor", 100, length_adaptive=True, alpha=1.5)
    critic = sao.role_lambda("critic", 100, length_adaptive=True, alpha=1.5)

    assert actor == pytest.approx(1.0 - 1.0 / (1.5 * 100))
    assert critic == 1.0
    assert actor != critic


@pytest.mark.unit
def test_role_lambda_actor_uses_fixed_lambda_when_adaptive_is_off() -> None:
    assert sao.role_lambda("actor", 100, lambd=0.9) == 0.9
    with pytest.raises(ValueError, match="unknown role"):
        sao.role_lambda("ref", 100)


@pytest.mark.unit
def test_importance_ratios_are_exp_of_logprob_difference() -> None:
    policy = [0.0, -1.0, 2.0]
    rollout = [0.0, 0.0, 0.0]

    ratios = sao.importance_ratios(policy, rollout)

    assert ratios[0] == pytest.approx(1.0)
    assert ratios[1] == pytest.approx(exp(-1.0))
    assert ratios[2] == pytest.approx(exp(2.0))


@pytest.mark.unit
def test_importance_ratios_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        sao.importance_ratios([0.0, 0.0], [0.0])


@pytest.mark.unit
def test_calibration_masks_tokens_outside_the_trust_region() -> None:
    # Lower bound 1-0.3 = 0.7, upper bound 1+5.0 = 6.0.
    ratios = [0.7, 0.5, 1.0, 5.9, 6.0, 100.0]

    weights = sao.calibration_weights(ratios, eps_low=0.3, eps_high=5.0)

    # 0.7 is *on* the lower bound (strict <) -> masked. 6.0 on the upper bound
    # -> masked. Interior values keep their ratio.
    assert weights == [0.0, 0.0, 1.0, 5.9, 0.0, 0.0]


@pytest.mark.unit
def test_calibration_is_a_hard_mask_not_a_clamp() -> None:
    # A clamp would push an out-of-band ratio to the nearest bound; SAO zeroes
    # it. This is the load-bearing difference from CISPO/PPO clipping.
    weights = sao.calibration_weights([0.6, 7.0], eps_low=0.3, eps_high=5.0)

    assert weights == [0.0, 0.0]


@pytest.mark.unit
def test_clip_mask_is_the_boolean_shadow_of_calibration() -> None:
    ratios = [0.7, 0.71, 1.0, 6.0, 5.99]

    mask = sao.clip_mask(ratios, eps_low=0.3, eps_high=5.0)

    assert mask == [0, 1, 1, 0, 1]


@pytest.mark.unit
def test_clip_ratio_counts_only_trained_tokens() -> None:
    # Four tokens, two masked by the clip (indices 0 and 3). Restrict the
    # denominator to trained (loss_mask) tokens: token 3 is not trained, so it
    # must not count toward the clip fraction.
    ratios = [0.5, 1.0, 2.0, 50.0]
    loss_mask = [1, 1, 1, 0]

    rate = sao.clip_ratio(ratios, eps_low=0.3, eps_high=5.0, loss_mask=loss_mask)

    # Trained tokens are 0,1,2; only token 0 is clipped -> 1/3.
    assert rate == pytest.approx(1.0 / 3.0)


@pytest.mark.unit
def test_clip_ratio_empty_denominator_is_zero() -> None:
    assert sao.clip_ratio([1.0, 2.0], loss_mask=[0, 0]) == 0.0


@pytest.mark.unit
def test_policy_gradient_coeffs_zero_out_masked_tokens() -> None:
    ratios = [1.0, 0.5, 2.0]
    advantages = [3.0, 3.0, -1.0]

    coeffs = sao.policy_gradient_coeffs(ratios, advantages, eps_low=0.3, eps_high=5.0)

    # Token 1 (ratio 0.5 < 0.7) is masked -> zero gradient regardless of its
    # advantage. The others are ratio*advantage.
    assert coeffs == pytest.approx([3.0, 0.0, -2.0])


@pytest.mark.unit
def test_length_adaptive_lambda_grows_toward_one_with_length() -> None:
    short = sao.length_adaptive_lambda(2, alpha=1.5)
    long = sao.length_adaptive_lambda(1000, alpha=1.5)

    assert short == pytest.approx(1.0 - 1.0 / (1.5 * 2))
    assert long > short
    assert long < 1.0


@pytest.mark.unit
def test_length_adaptive_lambda_clamps_nonpositive_length() -> None:
    assert sao.length_adaptive_lambda(0) == 0.0


@pytest.mark.unit
def test_skip_observation_gae_all_action_tokens_matches_plain_gae() -> None:
    # With every token an action token and gamma=lambda=1 the recurrence is a
    # plain reward-to-go GAE. Values [1,2,3], reward only on the last token.
    values = [1.0, 2.0, 3.0]
    rewards = [0.0, 0.0, 5.0]
    action_mask = [1, 1, 1]

    adv, ret = sao.skip_observation_gae(values, rewards, action_mask, gamma=1.0, lambd=1.0)

    # Work backward (next_value starts at 0 for the terminal action token):
    #   t=2: delta = 5 + 0 - 3 = 2;         gae = 2
    #   t=1: delta = 0 + 3 - 2 = 1;         gae = 1 + 2 = 3
    #   t=0: delta = 0 + 2 - 1 = 1;         gae = 1 + 3 = 4
    assert adv == pytest.approx([4.0, 3.0, 2.0])
    # returns = advantages + values on action tokens
    assert ret == pytest.approx([5.0, 5.0, 5.0])


@pytest.mark.unit
def test_skip_observation_gae_bridges_across_observation_tokens() -> None:
    # Action tokens at 0 and 3; observation tokens at 1 and 2 (e.g. a tool
    # result). The terminal reward lands on the last action token. Advantage
    # must bridge from the action at index 3 back to the action at index 0
    # *without* regressing the value model against the observation tokens.
    values = [10.0, 99.0, 99.0, 4.0]
    rewards = [0.0, 0.0, 0.0, 1.0]
    action_mask = [1, 0, 0, 1]

    adv, ret = sao.skip_observation_gae(values, rewards, action_mask, gamma=1.0, lambd=1.0)

    # Reverse over action tokens only (next_value starts 0):
    #   t=3 (action): delta = 1 + 0 - 4 = -3;    gae = -3;  next_value = 4
    #   t=2 (obs): skipped
    #   t=1 (obs): skipped
    #   t=0 (action): delta = 0 + 4 - 10 = -6;   gae = -6 + (-3) = -9
    # Observation tokens carry 0 in both outputs (never trained).
    assert adv == pytest.approx([-9.0, 0.0, 0.0, -3.0])
    assert ret == pytest.approx([1.0, 0.0, 0.0, 1.0])


@pytest.mark.unit
def test_skip_observation_gae_discount_and_lambda_apply_only_across_actions() -> None:
    # gamma and lambda both 0.5, three action tokens, terminal reward 8.
    values = [1.0, 1.0, 1.0]
    rewards = [0.0, 0.0, 8.0]
    action_mask = [1, 1, 1]

    adv, ret = sao.skip_observation_gae(values, rewards, action_mask, gamma=0.5, lambd=0.5)

    #   t=2: delta = 8 + 0.5*0 - 1 = 7;               gae = 7;    nv = 1
    #   t=1: delta = 0 + 0.5*1 - 1 = -0.5;            gae = -0.5 + 0.25*7 = 1.25; nv = 1
    #   t=0: delta = 0 + 0.5*1 - 1 = -0.5;            gae = -0.5 + 0.25*1.25 = -0.1875
    assert adv == pytest.approx([-0.1875, 1.25, 7.0])
    assert ret == pytest.approx([0.8125, 2.25, 8.0])


@pytest.mark.unit
def test_skip_observation_gae_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        sao.skip_observation_gae([1.0, 2.0], [0.0], [1, 1])


@pytest.mark.unit
def test_explained_variance_is_one_for_a_perfect_critic() -> None:
    returns = [1.0, 2.0, 3.0]
    values = [1.0, 2.0, 3.0]

    assert sao.explained_variance(returns, values) == pytest.approx(1.0)


@pytest.mark.unit
def test_explained_variance_is_zero_when_returns_are_constant() -> None:
    # No variance to explain -> defined as 0.0 rather than a division by zero.
    assert sao.explained_variance([5.0, 5.0], [1.0, 2.0]) == 0.0


@pytest.mark.unit
def test_explained_variance_restricts_to_action_tokens() -> None:
    # Observation tokens carry garbage values; with the action mask they must
    # not enter the metric. Action tokens 0 and 2 have a perfect critic.
    returns = [1.0, 999.0, 3.0]
    values = [1.0, -999.0, 3.0]
    action_mask = [1, 0, 1]

    assert sao.explained_variance(returns, values, action_mask) == pytest.approx(1.0)


@pytest.mark.unit
def test_explained_variance_can_go_negative() -> None:
    # A critic worse than predicting the mean explains negative variance.
    ev = sao.explained_variance([1.0, -1.0], [-5.0, 5.0])

    assert ev < 0.0


@pytest.mark.unit
def test_eps_validation_guards_the_lower_trust_bound() -> None:
    # eps_low >= 1 would make the lower bound non-positive and admit negative
    # ratios that exp() can never produce.
    with pytest.raises(ValueError, match="eps_low"):
        sao.calibration_weights([1.0], eps_low=1.0, eps_high=5.0)
    with pytest.raises(ValueError, match="non-negative"):
        sao.calibration_weights([1.0], eps_low=-0.1, eps_high=5.0)


@pytest.mark.unit
def test_coding_domain_bounds_are_narrower_than_reasoning() -> None:
    # Section 4: coding uses eps_low=0.8, eps_high=3.0. A ratio of 4.5 is kept
    # under reasoning bounds (upper 6.0) but masked under coding bounds (4.0).
    assert isclose(sao.calibration_weights([4.5], eps_low=0.3, eps_high=5.0)[0], 4.5)
    assert sao.calibration_weights([4.5], eps_low=0.8, eps_high=3.0)[0] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
