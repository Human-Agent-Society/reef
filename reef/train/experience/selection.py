"""The units that a processor can batch, and the policies that choose and order them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, kw_only=True)
class ExperienceUnit:
    """One unit that a processor can put in a batch: one sample, or one ready group.

    A processor subclasses this type to add its own data, such as the sample or
    the reports of a group. The buffer, the selection policies, and the
    eligibility checks read only these fields:

    * ``arrival_index`` tells when the processor received the unit. A group
      uses the index of its oldest member.
    * ``group_key`` names the group of a unit that is a group.
    * ``runtime_load_id`` is the runtime load ID that produced the unit, if known.
    * ``source_index`` tells when the source record of the unit arrived, if it
      differs from ``arrival_index``.
    """

    unit_id: Hashable
    arrival_index: int
    group_key: Hashable | None = None
    runtime_load_id: str | None = None
    source_index: int | None = None


class SelectionPolicy(ABC):
    """Choose the units for one batch, and put them in batch order.

    The processor chooses the policy. The buffer applies it to the units that it holds.
    """

    @abstractmethod
    def select(self, candidates: Sequence[ExperienceUnit], max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        """Return at most ``max_unit_count`` units from ``candidates``, in batch order."""


class ArrivalOrder(SelectionPolicy):
    """Take the oldest units first."""

    def select(self, candidates: Sequence[ExperienceUnit], max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        return tuple(sorted(candidates, key=lambda unit: unit.arrival_index)[:max_unit_count])


class GroupKeyOrder(SelectionPolicy):
    """Take ungrouped units in arrival order, then groups in ``group_key`` order.

    Group keys must be sortable, such as step indices.
    """

    def select(self, candidates: Sequence[ExperienceUnit], max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        ungrouped = sorted(
            (unit for unit in candidates if unit.group_key is None), key=lambda unit: unit.arrival_index
        )
        # The unit contract requires a hashable group key. This policy also requires a sortable key.
        grouped = sorted(
            (unit for unit in candidates if unit.group_key is not None), key=lambda unit: cast(Any, unit.group_key)
        )
        return tuple([*ungrouped, *grouped][:max_unit_count])
