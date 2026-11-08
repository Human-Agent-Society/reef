"""The reward step of a Slime rollout: Slime's ``--custom-reward-post-process-path``, which the slime family installs.

Slime's batch builder calls ``post_process_rewards`` once per rollout with the rollout's flat samples, before it
splits them over the data-parallel ranks. It trains the returned values as the samples' rewards; for the grpo, gspo
and cispo estimators these are the advantages. Slime's rollout manager builds its batch builder once for the serving
session, so the slime family (``reef.train.slime_backend.native``) installs the function for every job, and the
function decides by the rows of each rollout:

- A rollout whose rows carry no rule and no delayed advantage gets Slime's own reward step
  (``BatchBuilder._post_process_rewards``), so a dataset job, and a task job of an objective without its own rule,
  trains as Slime trains it.
- Otherwise a delayed row, whose metadata holds ``DELAYED_ADVANTAGE_KEY``, trains with that advantage. The task job's
  driver carries these rows (``reef.train.slime_backend.harbor_rollouts.generate_delayed_samples``); a padding row of
  the delayed groups has advantage 0.0 and no trained token. Each other row is centered on its prompt group with the
  rule that the task job's objective stamps on its prompt (``ADVANTAGE_RULE_KEY``). The groups are those of Slime's
  own reward step: segments of one rollout count once, and a removed sample, which has a prompt group of its own, has
  advantage 0.0.

The function returns the raw rewards unchanged. This module imports Slime only to run Slime's own reward step, in
the rollout manager's process; otherwise it reads the fields of Slime's ``Sample``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.train.algos.group_advantages import GroupAdvantageRule, group_advantages
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments

#: The function that the slime family names as Slime's ``--custom-reward-post-process-path``.
TASK_ADVANTAGES_PATH = "reef.train.slime_backend.task_advantages.post_process_rewards"
#: The metadata key of a delayed row's advantage, which ``generate_delayed_samples`` sets.
DELAYED_ADVANTAGE_KEY = "delayed_advantage"
#: The metadata key of the rule that a task job's objective stamps on each task prompt, as ``to_dict`` writes it.
ADVANTAGE_RULE_KEY = "advantage_rule"


@dataclass(frozen=True)
class RewardRow:
    """One sample of a Slime rollout, as the reward step reads it.

    Values: ``reward`` is the sample's raw reward, a float. ``group_index`` is its prompt group, or None.
    ``rollout_id`` is the rollout the sample is a segment of (Slime's ``rollout_id``, else its ``index``), or None.
    ``rule`` is the rule stamped on its prompt, or None. ``delayed_advantage`` is the advantage of a delayed row, a
    finite float, or None for every other row; a delayed row has no rule. Purpose: ``post_process_rewards`` makes one
    row per Slime sample, and ``rollout_advantages`` reads them.
    """

    reward: float
    group_index: int | None
    rollout_id: int | None
    rule: GroupAdvantageRule | None = None
    delayed_advantage: float | None = None

    def __post_init__(self) -> None:
        if self.delayed_advantage is not None and (self.rule is not None or not math.isfinite(self.delayed_advantage)):
            raise ValueError("a delayed row has a finite advantage and no rule")


def rollout_advantages(rows: Sequence[RewardRow]) -> list[float]:
    """The advantage of each row of one rollout, in order.

    A delayed row keeps its advantage. The other rows share the one rule stamped on them, and they group as Slime's
    reward step groups them: a rollout ID (else the row's position) names one rollout, whose segments share their
    group and reward and count once; ``group_advantages`` centers the rollouts of each prompt group, and each segment
    gets its rollout's advantage. ``ValueError`` when the other rows carry two rules or none, when some of them have no
    group index, or when the segments of one rollout disagree.
    """
    advantages = [0.0] * len(rows)
    others: list[int] = []
    for position, row in enumerate(rows):
        if row.delayed_advantage is not None:
            advantages[position] = row.delayed_advantage
        else:
            others.append(position)
    if not others:
        return advantages
    rules = {rows[position].rule for position in others}
    if len(rules) > 1:
        raise ValueError(f"the task rows of one rollout carry {len(rules)} different advantage rules")
    (rule,) = rules
    if rule is None:
        raise ValueError("the task rows beside delayed rows need the advantage rule of their objective")
    if any(rows[position].group_index is None for position in others):
        raise ValueError("a task row with an advantage rule needs its prompt group to be centered")
    rollouts: dict[Hashable, tuple[int | None, float, int]] = {}
    rollout_of: dict[int, int] = {}
    for position in others:
        row = rows[position]
        key: Hashable = ("rollout", row.rollout_id) if row.rollout_id is not None else ("position", position)
        if key in rollouts:
            group, reward, _ = rollouts[key]
            if group != row.group_index or reward != row.reward:
                raise ValueError(f"the segments of rollout {row.rollout_id} must share a prompt group and a reward")
        else:
            rollouts[key] = (row.group_index, row.reward, len(rollouts))
        rollout_of[position] = rollouts[key][2]
    groups: dict[int | None, list[int]] = {}
    rewards = [0.0] * len(rollouts)
    for group, reward, number in rollouts.values():
        groups.setdefault(group, []).append(number)
        rewards[number] = reward
    rollout_values = [0.0] * len(rollouts)
    for numbers in groups.values():
        for number, value in zip(
            numbers, group_advantages([rewards[number] for number in numbers], rule), strict=True
        ):
            rollout_values[number] = value
    for position in others:
        advantages[position] = rollout_values[rollout_of[position]]
    return advantages


def slime_rewards(args: SlimeArguments, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Slime's own reward step of one rollout (``BatchBuilder._post_process_rewards``), as without this hook."""
    # Imported here: only the rollout manager's process has Slime.
    from slime.data.batch_builder import BatchBuilder

    own_args = copy.copy(args)
    own_args.custom_reward_post_process_path = None
    return BatchBuilder(own_args)._post_process_rewards(samples)


def post_process_rewards(args: SlimeArguments, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Slime's reward step of one rollout: the raw rewards of ``samples`` and the values that Slime trains.

    ``samples`` are the rollout's Slime ``Sample`` values in batch order, flat. A rollout whose rows carry no rule and
    no delayed advantage gets Slime's own step (``slime_rewards``); another gets ``rollout_advantages``.
    """
    rows = []
    for sample in samples:
        metadata: Mapping[str, Any] = sample.metadata
        rule = metadata.get(ADVANTAGE_RULE_KEY)
        delayed = metadata.get(DELAYED_ADVANTAGE_KEY)
        rows.append(
            RewardRow(
                reward=sample.get_reward_value(args),
                group_index=sample.group_index,
                rollout_id=sample.rollout_id if sample.rollout_id is not None else sample.index,
                rule=None if rule is None else GroupAdvantageRule.from_dict(rule),
                delayed_advantage=None if delayed is None else float(delayed),
            )
        )
    if all(row.rule is None and row.delayed_advantage is None for row in rows):
        return slime_rewards(args, samples)
    return [row.reward for row in rows], rollout_advantages(rows)


__all__ = [
    "ADVANTAGE_RULE_KEY",
    "DELAYED_ADVANTAGE_KEY",
    "TASK_ADVANTAGES_PATH",
    "RewardRow",
    "post_process_rewards",
    "rollout_advantages",
    "slime_rewards",
]
