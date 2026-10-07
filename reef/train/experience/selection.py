"""The units that a processor can batch, and the policies that choose and order them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

KeyT = TypeVar("KeyT", bound=Hashable)
MemberT = TypeVar("MemberT")


@dataclass(frozen=True)
class ExperienceUnit(Generic[KeyT, MemberT]):
    """One unit that a processor can put in a batch: one sample, or one ready group.

    ``members`` are values of the processor. The buffer and the selection
    policies do not read them. ``arrival_index`` tells when the processor
    received the unit. A group uses the index of its oldest member. A unit
    that is a group has a ``group_key``.
    """

    unit_id: KeyT
    members: tuple[MemberT, ...]
    arrival_index: int
    group_key: Hashable | None = None

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("an experience unit needs at least one member")


class SelectionPolicy(ABC):
    """Choose the units for one batch, and put them in batch order.

    The processor chooses the policy. The buffer applies it to the units that it holds.
    """

    @abstractmethod
    def select(
        self, candidates: Sequence[ExperienceUnit[KeyT, MemberT]], max_unit_count: int
    ) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        """Return at most ``max_unit_count`` units from ``candidates``, in batch order."""


class ArrivalOrder(SelectionPolicy):
    """Take the oldest units first."""

    def select(
        self, candidates: Sequence[ExperienceUnit[KeyT, MemberT]], max_unit_count: int
    ) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        return tuple(sorted(candidates, key=lambda unit: unit.arrival_index)[:max_unit_count])


class GroupKeyOrder(SelectionPolicy):
    """Take ungrouped units in arrival order, then groups in ``group_key`` order.

    Group keys must be sortable, such as step indices.
    """

    def select(
        self, candidates: Sequence[ExperienceUnit[KeyT, MemberT]], max_unit_count: int
    ) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        ungrouped = sorted(
            (unit for unit in candidates if unit.group_key is None), key=lambda unit: unit.arrival_index
        )
        # The unit contract requires a hashable group key. This policy also requires a sortable key.
        grouped = sorted(
            (unit for unit in candidates if unit.group_key is not None), key=lambda unit: cast(Any, unit.group_key)
        )
        return tuple([*ungrouped, *grouped][:max_unit_count])
