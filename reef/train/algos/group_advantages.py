"""Group-relative advantages: each reward minus the mean of its group, optionally over the group's deviation.

A GRPO-style method centers the reward of each rollout on the rewards of the other rollouts of the same prompt. Slime's
batch builder divides by the unbiased sample deviation plus 1e-6 (``slime_rule``). SPADE's own trainer divides by the
population deviation (NumPy's ``np.std``) plus 1e-8. A training objective declares its rule, and the training backend
applies it with ``group_advantages``. This module imports only the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

#: How a rule divides the squared deviations of a group of ``n`` rewards: by ``n`` or by ``n - 1``.
DeviationKind = Literal["population", "sample"]
#: The estimators for which Slime's batch builder centers each reward on its prompt group, at the Slime pin.
SLIME_CENTERING_ESTIMATORS = ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline")
#: The estimators for which Slime's batch builder also divides by the group's deviation, at the Slime pin.
SLIME_DEVIATION_ESTIMATORS = ("grpo", "gspo", "cispo")


@dataclass(frozen=True)
class GroupAdvantageRule:
    """How the rewards of one group become advantages.

    Values: ``divide_by_std`` says whether a centered reward is divided by the group's standard deviation plus
    ``epsilon``. ``std`` is ``"population"``, the deviation that divides the squared deviations by the group size, or
    ``"sample"``, the unbiased one that divides them by the group size minus 1. ``epsilon`` is a positive finite
    float. ``std`` and ``epsilon`` take no part when ``divide_by_std`` is false. Purpose: a training objective
    declares a rule (``SlimeObjective.rollout_advantages``) and ``slime_rule`` gives Slime's own; ``group_advantages``
    applies it. ``to_dict`` and ``from_dict`` carry it in a task prompt's metadata.
    """

    divide_by_std: bool
    std: DeviationKind = "population"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.divide_by_std, bool):
            raise TypeError("GroupAdvantageRule.divide_by_std must be a bool")
        if self.std not in get_args(DeviationKind):
            raise ValueError("GroupAdvantageRule.std must be 'population' or 'sample'")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise TypeError("GroupAdvantageRule.epsilon must be a number")
        if not (math.isfinite(self.epsilon) and self.epsilon > 0):
            raise ValueError("GroupAdvantageRule.epsilon must be positive and finite")
        object.__setattr__(self, "epsilon", float(self.epsilon))

    def to_dict(self) -> dict[str, object]:
        """``{divide_by_std, std, epsilon}``, which ``from_dict`` reads back to an equal rule."""
        return {"divide_by_std": self.divide_by_std, "std": self.std, "epsilon": self.epsilon}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GroupAdvantageRule:
        """The rule that ``to_dict`` wrote; ``ValueError`` for another value."""
        if not isinstance(value, Mapping) or set(value) != {"divide_by_std", "std", "epsilon"}:
            raise ValueError("a group advantage rule is an object with divide_by_std, std and epsilon")
        divide_by_std, std, epsilon = value["divide_by_std"], value["std"], value["epsilon"]
        if not isinstance(divide_by_std, bool) or std not in get_args(DeviationKind):
            raise ValueError("a group advantage rule needs a bool divide_by_std and a std of population or sample")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("a group advantage rule needs a numeric epsilon")
        return cls(divide_by_std, "population" if std == "population" else "sample", epsilon)


def group_advantages(rewards: Sequence[float], rule: GroupAdvantageRule) -> list[float]:
    """The advantage of each reward of one group, in order.

    Each advantage is the reward minus the group's mean. With ``rule.divide_by_std`` and at least two rewards, it is
    then divided by the group's deviation of kind ``rule.std`` plus ``rule.epsilon``. A group of one reward has
    advantage 0.0, and so does a group whose rewards are all equal. ``ValueError`` for an empty group or a reward that
    is not finite.
    """
    values = [float(reward) for reward in rewards]
    if not values:
        raise ValueError("a group needs at least one reward")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("group rewards must be finite")
    mean = math.fsum(values) / len(values)
    centered = [value - mean for value in values]
    if not rule.divide_by_std or len(values) < 2:
        return centered
    squares = math.fsum(value * value for value in centered)
    count = len(values) if rule.std == "population" else len(values) - 1
    scale = math.sqrt(squares / count) + rule.epsilon
    return [value / scale for value in centered]


def slime_rule(
    advantage_estimator: str, *, rewards_normalization: bool, std_normalization: bool
) -> GroupAdvantageRule | None:
    """The rule of Slime's own reward step (``BatchBuilder._post_process_rewards``) at the Slime pin, or None.

    The arguments are Slime's ``--advantage-estimator``, its ``rewards_normalization`` and its
    ``grpo_std_normalization``. None means that Slime trains the raw rewards: the estimator is not one that centers,
    or ``--disable-rewards-normalization`` is set. Slime divides by the unbiased deviation plus 1e-6 for the
    estimators of ``SLIME_DEVIATION_ESTIMATORS``, unless ``--disable-grpo-std-normalization`` is set.
    """
    if advantage_estimator not in SLIME_CENTERING_ESTIMATORS or not rewards_normalization:
        return None
    return GroupAdvantageRule(
        divide_by_std=advantage_estimator in SLIME_DEVIATION_ESTIMATORS and std_normalization,
        std="sample",
        epsilon=1e-6,
    )


__all__ = [
    "SLIME_CENTERING_ESTIMATORS",
    "SLIME_DEVIATION_ESTIMATORS",
    "DeviationKind",
    "GroupAdvantageRule",
    "group_advantages",
    "slime_rule",
]
