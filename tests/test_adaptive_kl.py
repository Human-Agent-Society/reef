from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from reef.train.adaptive_kl import AdaptiveKLConfig, AdaptiveKLController


def ppo_args(**overrides):
    values = {
        "adaptive_kl_mode": "telemetry",
        "adaptive_kl_target": 0.1,
        "adaptive_kl_initial_beta": None,
        "adaptive_kl_min_beta": 1e-6,
        "adaptive_kl_max_beta": 10.0,
        "adaptive_kl_adaptation_rate": 0.05,
        "adaptive_kl_ema_decay": 0.9,
        "adaptive_kl_max_update_ratio": 2.0,
        "adaptive_kl_cooldown_steps": 0,
        "adaptive_kl_spike_threshold": None,
        "adaptive_kl_reference_policy_id": "ref-v1",
        "kl_coef": 0.1,
        "loss_type": "policy_loss",
        "compute_advantages_and_returns": True,
        "advantage_estimator": "ppo",
        "use_kl_loss": False,
        "use_opd": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_adapter_accepts_standard_ppo_telemetry_and_reuses_fixed_beta() -> None:
    from reef.train.slime_backend.reef_adapters.adaptive_kl import validate_adaptive_kl_args

    args = ppo_args()
    config = validate_adaptive_kl_args(args, loss_family="ppo_rlhf_reference_reward")

    assert config is not None
    assert config.initial_beta == 0.1
    assert args.adaptive_kl_initial_beta == 0.1


@pytest.mark.unit
def test_adapter_accepts_wired_reward_mode() -> None:
    from reef.train.slime_backend.reef_adapters.adaptive_kl import validate_adaptive_kl_args

    config = validate_adaptive_kl_args(
        ppo_args(adaptive_kl_mode="reward"), loss_family="ppo_rlhf_reference_reward"
    )
    assert config is not None and config.mode == "reward"


@pytest.mark.unit
@pytest.mark.parametrize("family", ["sao", "tttd", "openclawrl", "custom_ppo"])
def test_adapter_rejects_other_kl_families(family: str) -> None:
    from reef.train.slime_backend.reef_adapters.adaptive_kl import validate_adaptive_kl_args

    with pytest.raises(RuntimeError):
        validate_adaptive_kl_args(ppo_args(), loss_family=family)


def make_config(**overrides) -> AdaptiveKLConfig:
    values = {
        "target_kl": 1.0,
        "initial_beta": 1.0,
        "min_beta": 0.1,
        "max_beta": 10.0,
        "adaptation_rate": 0.5,
        "ema_decay": 0.0,
        "max_update_ratio": 2.0,
        "reference_policy_id": "ref-v1",
        "mode": "reward",
    }
    values.update(overrides)
    return AdaptiveKLConfig(**values)


def observe(controller: AdaptiveKLController, observed_kl: float, **kwargs):
    return controller.observe(
        observed_kl,
        training_step_succeeded=True,
        loss_is_finite=True,
        valid_token_count=1,
        **kwargs,
    )


@pytest.mark.unit
def test_above_target_increases_beta_and_below_target_decreases_beta() -> None:
    above = AdaptiveKLController(make_config())
    below = AdaptiveKLController(make_config())

    assert observe(above, 2.0).beta_next > 1.0
    assert observe(below, 0.5).beta_next < 1.0


@pytest.mark.unit
def test_verl_controller_matches_clipped_proportional_update() -> None:
    controller = AdaptiveKLController(make_config(controller_type="verl", horizon=10, error_clip=0.2))

    decision = observe(controller, 2.0, n_steps=2)

    # current_kl / target - 1 = 1.0, clipped to 0.2:
    # beta_next = 1 * (1 + 0.2 * 2 / 10) = 1.04.
    assert decision.beta_next == pytest.approx(1.04)
    assert decision.controller_kl == 2.0
    assert decision.ema_kl is None


@pytest.mark.unit
def test_fixed_controller_does_not_update_beta() -> None:
    controller = AdaptiveKLController(make_config(controller_type="fixed"))

    decision = observe(controller, -0.5, n_steps=8)

    assert decision.beta_next == 1.0
    assert decision.observed_kl == -0.5
    assert decision.update_applied is False
    assert decision.update_reason == "fixed"


@pytest.mark.unit
def test_verl_controller_accepts_finite_negative_sampled_kl() -> None:
    controller = AdaptiveKLController(make_config(controller_type="verl", horizon=10))

    decision = observe(controller, -0.5)

    # The sampled log-ratio estimate can be negative; VERL clips the
    # proportional error before applying its horizon-scaled update.
    assert decision.observed_kl == -0.5
    assert decision.controller_kl == -0.5
    assert decision.beta_next == pytest.approx(0.98)
    assert decision.update_reason == "updated"


@pytest.mark.unit
def test_first_observation_initializes_ema_and_target_is_stable() -> None:
    controller = AdaptiveKLController(make_config(ema_decay=0.5))

    first = observe(controller, 1.0)
    second = observe(controller, 1.0)

    assert first.ema_kl == 1.0
    assert second.ema_kl == 1.0
    assert second.beta_next == 1.0
    assert second.update_reason == "target_reached"


@pytest.mark.unit
def test_ema_uses_configured_decay() -> None:
    controller = AdaptiveKLController(make_config(ema_decay=0.5))

    observe(controller, 2.0)
    decision = observe(controller, 4.0)

    assert decision.ema_kl == 3.0


@pytest.mark.unit
def test_absolute_and_per_update_bounds_are_enforced() -> None:
    controller = AdaptiveKLController(
        make_config(min_beta=0.8, max_beta=1.2, max_update_ratio=1.1, adaptation_rate=100.0)
    )

    assert observe(controller, 100.0).beta_next == pytest.approx(1.1)
    assert observe(controller, 0.0).beta_next == pytest.approx(1.0)

    controller = AdaptiveKLController(make_config(min_beta=0.8, max_beta=1.2, max_update_ratio=10.0))
    assert observe(controller, 100.0).beta_next == pytest.approx(1.2)
    assert observe(controller, 0.0).beta_next == pytest.approx(0.8)


@pytest.mark.unit
def test_spike_enters_cooldown_without_updating_beta() -> None:
    controller = AdaptiveKLController(make_config(spike_threshold=5.0, cooldown_steps=2))

    spike = observe(controller, 10.0)
    cooldown_one = observe(controller, 1.0)
    cooldown_two = observe(controller, 1.0)
    eligible = observe(controller, 2.0)

    assert spike.update_reason == "kl_spike"
    assert spike.beta_next == 1.0
    assert spike.cooldown_remaining == 2
    assert cooldown_one.update_reason == "cooldown"
    assert cooldown_two.update_reason == "cooldown"
    assert cooldown_two.cooldown_remaining == 0
    assert eligible.beta_next > 1.0


@pytest.mark.unit
def test_failed_step_loss_and_empty_mask_leave_state_unchanged() -> None:
    controller = AdaptiveKLController(make_config())
    initial = controller.state_dict()

    assert controller.observe(2.0, training_step_succeeded=False, loss_is_finite=True).update_reason == (
        "training_step_failed"
    )
    assert controller.state_dict() == initial
    assert controller.observe(2.0, training_step_succeeded=True, loss_is_finite=False).update_reason == (
        "non_finite_loss"
    )
    assert controller.state_dict() == initial
    assert (
        controller.observe(2.0, training_step_succeeded=True, loss_is_finite=True, valid_token_count=0).update_reason
        == "empty_valid_token_mask"
    )
    assert controller.state_dict() == initial


@pytest.mark.unit
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0])
def test_invalid_observed_kl_leaves_state_unchanged(value: float) -> None:
    controller = AdaptiveKLController(make_config())
    initial = controller.state_dict()

    decision = controller.observe(value, training_step_succeeded=True, loss_is_finite=True, valid_token_count=1)

    assert decision.update_reason == "invalid_observed_kl"
    assert controller.state_dict() == initial


@pytest.mark.unit
def test_off_mode_is_disabled_without_mutating_state() -> None:
    controller = AdaptiveKLController(make_config(mode="off"))
    initial = controller.state_dict()

    decision = controller.observe(2.0, training_step_succeeded=True, loss_is_finite=True, valid_token_count=1)

    assert decision.update_reason == "disabled"
    assert controller.state_dict() == initial


@pytest.mark.unit
def test_state_round_trip_is_deterministic() -> None:
    config = make_config(ema_decay=0.25, cooldown_steps=1)
    original = AdaptiveKLController(config)
    observe(original, 2.0)
    state = original.state_dict()
    restored = AdaptiveKLController.from_state_dict(state, config)

    assert restored.state_dict() == state
    original_decision = observe(original, 0.5)
    restored_decision = observe(restored, 0.5)
    assert restored_decision == original_decision
    assert restored.state_dict() == original.state_dict()


@pytest.mark.unit
@pytest.mark.parametrize(
    "state_update, match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"reference_policy_id": "other"}, "reference policy identity"),
        ({"controller_version": "other"}, "controller version"),
        ({"beta": 100.0}, "state.beta"),
    ],
)
def test_incompatible_state_is_rejected(state_update, match: str) -> None:
    config = make_config()
    state = AdaptiveKLController(config).state_dict()
    state.update(state_update)

    with pytest.raises(ValueError, match=match):
        AdaptiveKLController.from_state_dict(state, config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, value",
    [
        ("target_kl", 0.0),
        ("initial_beta", 0.0),
        ("min_beta", 0.0),
        ("max_beta", 0.0),
        ("adaptation_rate", 0.0),
        ("ema_decay", 1.0),
        ("max_update_ratio", 0.0),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: float) -> None:
    values = {
        "target_kl": 1.0,
        "initial_beta": 1.0,
        "min_beta": 0.1,
        "max_beta": 10.0,
        "adaptation_rate": 0.5,
        "ema_decay": 0.0,
        "max_update_ratio": 2.0,
        "reference_policy_id": "ref-v1",
        "mode": "reward",
    }
    values[field] = value

    with pytest.raises(ValueError):
        AdaptiveKLConfig(**values)
