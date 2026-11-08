"""Group-relative advantages: SPADE's population z-score and Slime's own rule."""

from __future__ import annotations

import math

import pytest

from reef.train.algos.group_advantages import GroupAdvantageRule, group_advantages, slime_rule

SPADE = GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)


@pytest.mark.unit
def test_spades_rule_is_the_population_z_score_of_the_group() -> None:
    advantages = group_advantages([1, 0, 0, 1, 1, 0, 0, 0], SPADE)
    assert advantages[0] == pytest.approx(1.290994, abs=1e-6)
    assert advantages[1] == pytest.approx(-0.774597, abs=1e-6)
    assert advantages == pytest.approx(
        [1.290994, -0.774597, -0.774597, 1.290994, 1.290994] + [-0.774597] * 3, abs=1e-6
    )


@pytest.mark.unit
@pytest.mark.parametrize("rule", [SPADE, GroupAdvantageRule(True, "sample", 1e-6), GroupAdvantageRule(False)])
def test_a_constant_group_and_a_group_of_one_have_no_advantage(rule: GroupAdvantageRule) -> None:
    assert group_advantages([0.5, 0.5, 0.5], rule) == [0.0, 0.0, 0.0]
    assert group_advantages([1.0], rule) == [0.0]


@pytest.mark.unit
def test_the_sample_deviation_divides_by_the_group_size_minus_one() -> None:
    rewards = [1.0, 0.0, 0.0, 1.0]
    scale = math.sqrt(4 * 0.25 / 3) + 1e-6
    assert group_advantages(rewards, GroupAdvantageRule(True, "sample", 1e-6)) == pytest.approx(
        [0.5 / scale, -0.5 / scale, -0.5 / scale, 0.5 / scale]
    )
    # Without the deviation the rule only centers.
    assert group_advantages(rewards, GroupAdvantageRule(False)) == [0.5, -0.5, -0.5, 0.5]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("estimator", "normalization", "deviation", "rule"),
    [
        ("grpo", True, True, GroupAdvantageRule(True, "sample", 1e-6)),
        ("gspo", True, True, GroupAdvantageRule(True, "sample", 1e-6)),
        ("cispo", True, False, GroupAdvantageRule(False, "sample", 1e-6)),
        ("reinforce_plus_plus_baseline", True, True, GroupAdvantageRule(False, "sample", 1e-6)),
        ("grpo", False, True, None),
        ("ppo", True, True, None),
        ("reinforce_plus_plus", True, True, None),
    ],
)
def test_slimes_rule_follows_its_estimator_and_its_normalization_options(
    estimator: str, normalization: bool, deviation: bool, rule: GroupAdvantageRule | None
) -> None:
    assert slime_rule(estimator, rewards_normalization=normalization, std_normalization=deviation) == rule


@pytest.mark.unit
def test_a_rule_reads_back_from_its_dict_and_refuses_values_outside_its_domain() -> None:
    assert GroupAdvantageRule.from_dict(SPADE.to_dict()) == SPADE
    for value in ({"divide_by_std": True, "std": "biased", "epsilon": 1e-8}, {"divide_by_std": True}):
        with pytest.raises(ValueError):
            GroupAdvantageRule.from_dict(value)
    with pytest.raises(ValueError, match="'population' or 'sample'"):
        GroupAdvantageRule(True, "biased", 1e-8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive and finite"):
        GroupAdvantageRule(True, "population", 0.0)
    with pytest.raises(ValueError, match="at least one reward"):
        group_advantages([], SPADE)
    with pytest.raises(ValueError, match="finite"):
        group_advantages([1.0, math.nan], SPADE)
