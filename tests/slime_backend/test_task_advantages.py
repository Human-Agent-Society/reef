"""Reef's reward step of a Slime rollout: delayed rows keep their advantages, task rows follow one group rule.

These tests need no Slime, except the last one, which compares the step with Slime's own at the pin where Slime is
installed.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest

from reef.train.algos.group_advantages import GroupAdvantageRule, group_advantages
from reef.train.slime_backend.task_advantages import (
    ADVANTAGE_RULE_KEY,
    DELAYED_ADVANTAGE_KEY,
    RewardRow,
    post_process_rewards,
    rollout_advantages,
)

SPADE = GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)


class FakeSample:
    """The fields of Slime's ``Sample`` that the reward step reads."""

    def __init__(
        self,
        reward: float,
        group_index: int | None,
        index: int,
        *,
        rollout_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.reward = reward
        self.group_index = group_index
        self.index = index
        self.rollout_id = rollout_id
        self.metadata = metadata or {}

    def get_reward_value(self, args: Any) -> float:
        return self.reward


def slime_args(estimator: str = "grpo", *, normalization: bool = True, deviation: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator=estimator,
        rewards_normalization=normalization,
        grpo_std_normalization=deviation,
        reward_key=None,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
    )


def task(reward: float, group: int, index: int, rule: GroupAdvantageRule | None = SPADE) -> FakeSample:
    return FakeSample(reward, group, index, metadata={} if rule is None else {ADVANTAGE_RULE_KEY: rule.to_dict()})


@pytest.mark.unit
def test_a_mixed_rollout_keeps_the_delayed_advantages_and_centers_the_tasks_by_their_rule() -> None:
    delayed = [
        FakeSample(0.0, 0, 0, metadata={DELAYED_ADVANTAGE_KEY: 2.5}),
        FakeSample(0.0, 0, 1, metadata={DELAYED_ADVANTAGE_KEY: -1.0}),
        # A padding row: removed, in the rollout of its group's first entry.
        FakeSample(0.0, -3, 2, rollout_id=0, metadata={DELAYED_ADVANTAGE_KEY: 0.0}),
    ]
    plays = [task(reward, 1, 3 + offset) for offset, reward in enumerate([1, 0, 0, 1, 1, 0, 0, 0])]
    # A removed play leaves its group: it has a group of its own.
    removed = task(0.0, -1 - 11, 11)
    samples = [*delayed, *plays, removed]

    raw, advantages = post_process_rewards(slime_args(), samples)

    assert raw == [sample.reward for sample in samples]
    assert advantages[:3] == [2.5, -1.0, 0.0]
    assert advantages[3:11] == pytest.approx([1.290994, -0.774597, -0.774597, 1.290994, 1.290994] + [-0.774597] * 3)
    assert advantages[11] == 0.0


@pytest.mark.unit
def test_without_a_stamped_rule_the_task_rows_follow_slimes_own_rule() -> None:
    samples = [task(reward, 0, index, rule=None) for index, reward in enumerate([1.0, 0.0, 0.0, 1.0])]
    # Two segments of one rollout count once: the second group holds the rollouts 4 and 6.
    samples += [
        FakeSample(1.0, 1, 4, rollout_id=4),
        FakeSample(1.0, 1, 5, rollout_id=4),
        FakeSample(0.0, 1, 6),
    ]
    _, advantages = post_process_rewards(slime_args(), samples)
    sample_rule = GroupAdvantageRule(True, "sample", 1e-6)
    assert advantages[:4] == pytest.approx(group_advantages([1.0, 0.0, 0.0, 1.0], sample_rule))
    first, second = group_advantages([1.0, 0.0], sample_rule)
    assert advantages[4:] == pytest.approx([first, first, second])
    # An estimator that does not center trains the raw rewards.
    assert post_process_rewards(slime_args("ppo"), samples)[1] == [sample.reward for sample in samples]


@pytest.mark.unit
def test_a_rollout_whose_task_rows_carry_two_rules_is_refused() -> None:
    other = GroupAdvantageRule(divide_by_std=False)
    with pytest.raises(ValueError, match="2 different advantage rules"):
        post_process_rewards(slime_args(), [task(1.0, 0, 0), task(0.0, 0, 1, rule=other)])
    with pytest.raises(ValueError, match="2 different advantage rules"):
        post_process_rewards(slime_args(), [task(1.0, 0, 0), task(0.0, 0, 1, rule=None)])


@pytest.mark.unit
def test_rows_without_groups_follow_slimes_positions() -> None:
    plain = GroupAdvantageRule(divide_by_std=False)
    rows = [RewardRow(reward, None, index) for index, reward in enumerate([1.0, 0.0, 1.0, 1.0])]
    # Two prompts of two: the complete batch groups by position, as Slime does.
    assert rollout_advantages(rows, plain, flat_batch=(2, 2)) == [0.5, -0.5, 0.0, 0.0]
    # Any other count of rollouts is one group.
    assert rollout_advantages(rows, plain, flat_batch=(2, 3)) == [0.25, -0.75, 0.25, 0.25]


@pytest.mark.unit
def test_rows_that_slimes_grouping_cannot_center_are_refused() -> None:
    with pytest.raises(ValueError, match="or none does"):
        rollout_advantages([RewardRow(1.0, None, 0), RewardRow(0.0, 0, 1)], SPADE)
    with pytest.raises(ValueError, match="needs its prompt group"):
        rollout_advantages([RewardRow(1.0, None, 0, rule=SPADE), RewardRow(0.0, None, 1, rule=SPADE)], None)
    with pytest.raises(ValueError, match="segments of rollout 3"):
        rollout_advantages([RewardRow(1.0, 0, 3), RewardRow(0.0, 0, 3)], SPADE)
    with pytest.raises(ValueError, match="no rule"):
        RewardRow(0.0, 0, 0, rule=SPADE, delayed_advantage=1.0)


def random_batch(types: Any, rng: random.Random) -> list[Any]:
    """A rollout of Slime ``Sample`` values: groups of plays, removed plays and rollouts of several segments."""
    samples = []
    index = 0
    for group in range(rng.randint(1, 5)):
        for _ in range(rng.randint(1, 6)):
            reward = rng.choice([0.0, 1.0, rng.random()])
            if rng.random() < 0.15:
                samples.append(types.Sample(group_index=-1 - index, index=index, reward=0.0, remove_sample=True))
                index += 1
                continue
            segments = rng.choice([1, 1, 1, 2, 3])
            rollout = index
            for _ in range(segments):
                samples.append(
                    types.Sample(
                        group_index=group,
                        index=index,
                        rollout_id=rollout if segments > 1 else None,
                        reward=reward,
                    )
                )
                index += 1
    return samples


@pytest.mark.unit
@pytest.mark.parametrize(
    ("estimator", "normalization", "deviation"),
    [
        ("grpo", True, True),
        ("gspo", True, True),
        ("cispo", True, True),
        ("grpo", True, False),
        ("reinforce_plus_plus_baseline", True, True),
        ("grpo", False, True),
        ("ppo", True, True),
    ],
)
def test_without_a_rule_the_reward_step_equals_slimes_own_at_the_pin(
    estimator: str, normalization: bool, deviation: bool
) -> None:
    batch_builder = pytest.importorskip("slime.data.batch_builder", reason="requires the Slime runtime")
    types = pytest.importorskip("slime.utils.types", reason="requires the Slime runtime")
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator=estimator,
        rewards_normalization=normalization,
        grpo_std_normalization=deviation,
        reward_key=None,
        n_samples_per_prompt=4,
        rollout_batch_size=4,
    )
    builder = batch_builder.BatchBuilder(args)
    rng = random.Random(1234)
    # Flat batches without group indexes too: a complete batch of 4 x 4 groups by position, any other is one group.
    flat = [
        [types.Sample(index=index, reward=rng.choice([0.0, 1.0, rng.random()])) for index in range(count)]
        for count in (16, 16, 7)
    ]
    for samples in [random_batch(types, rng) for _ in range(50)] + flat:
        expected_raw, expected = builder._post_process_rewards(samples)
        raw, advantages = post_process_rewards(args, samples)
        assert raw == expected_raw
        # Slime computes in float32.
        assert advantages == pytest.approx(expected, rel=1e-5, abs=1e-5)
