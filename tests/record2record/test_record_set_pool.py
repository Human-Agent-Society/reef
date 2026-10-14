"""Record set pool contracts: selection order, reservations, and the staleness check."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from reef.record2record import ArrivalOrder, GroupKeyOrder, RecordSet, RecordSetPool, SelectionPolicy
from reef.record2record.staleness import stale_sets


@dataclass(frozen=True, kw_only=True)
class NamedSet(RecordSet):
    """A set that holds only its name."""

    def record_ids(self) -> tuple[str, ...]:
        return (str(self.set_id),)


def named(set_id: str, arrival_index: int, group_key: int | None = None) -> NamedSet:
    return NamedSet(set_id=set_id, arrival_index=arrival_index, group_key=group_key)


class ForeignSet(SelectionPolicy):
    def select(self, candidates: Sequence[RecordSet], max_set_count: int) -> tuple[RecordSet, ...]:
        return (named("not-held", 1),)


def test_arrival_order_takes_the_oldest_sets_up_to_the_maximum_count() -> None:
    candidates = (named("c", 3), named("a", 1), named("b", 2))
    assert [selected.set_id for selected in ArrivalOrder().select(candidates, 2)] == ["a", "b"]


def test_group_key_order_takes_ungrouped_sets_before_groups_in_key_order() -> None:
    candidates = (named("late-group", 1, group_key=2), named("single", 3), named("early-group", 2, group_key=1))
    selected = GroupKeyOrder().select(candidates, 3)
    assert [chosen.set_id for chosen in selected] == ["single", "early-group", "late-group"]


def test_a_reservation_does_not_change_until_it_is_consumed() -> None:
    pool = RecordSetPool()
    for name in ("a", "b", "c"):
        pool.put(named(name, pool.next_arrival_index()))
    reserved = pool.reserve(2)
    pool.remove("a")
    pool.put(named("d", pool.next_arrival_index()))
    assert pool.reserved_sets() == reserved
    # The consumed batch includes a reserved set that was removed after the reservation.
    assert [consumed.set_id for consumed in pool.consume_reserved()] == ["a", "b"]
    assert [remaining.set_id for remaining in pool.sets()] == ["c", "d"]
    assert pool.reserved_sets() == ()


def test_a_reservation_refuses_a_negative_count_and_sets_the_pool_does_not_hold() -> None:
    with pytest.raises(ValueError, match="max_set_count must not be negative"):
        RecordSetPool().reserve(-1)
    pool = RecordSetPool(ForeignSet())
    pool.put(named("held", pool.next_arrival_index()))
    with pytest.raises(ValueError, match="does not hold"):
        pool.reserve(1)


def test_staleness_check_keeps_sets_within_the_window_of_the_last_source_record() -> None:
    def version(set_id: str, runtime_load_id: str, source_index: int) -> NamedSet:
        return NamedSet(
            set_id=set_id,
            arrival_index=10 - source_index,
            runtime_load_id=runtime_load_id,
            source_index=source_index,
        )

    # The reference is the last source record ("newest", e:5), not the last set in the pool.
    record_sets = (
        version("newest", "e:5", 6),
        version("one-behind", "e:4", 5),
        version("two-behind", "e:3", 4),
        version("ahead", "e:6", 3),
        version("other-incarnation", "f:5", 2),
        version("unparsed", "v1", 1),
    )
    # Too far behind, ahead, another incarnation, and an unparsed version are stale.
    stale = {"two-behind", "ahead", "other-incarnation", "unparsed"}
    assert {record_set.set_id for record_set in stale_sets(record_sets, 1)} == stale
    # Window 0 keeps only the reference version.
    assert {record_set.set_id for record_set in stale_sets(record_sets, 0)} == stale | {"one-behind"}


def test_staleness_check_compares_non_canonical_ids_by_equality() -> None:
    late_judgment = NamedSet(set_id="late", arrival_index=3, runtime_load_id="v1", source_index=1)
    newest = NamedSet(set_id="newest", arrival_index=2, runtime_load_id="v2", source_index=3)
    same_version = NamedSet(set_id="same", arrival_index=1, runtime_load_id="v2", source_index=2)
    assert [record_set.set_id for record_set in stale_sets((late_judgment, newest, same_version), 5)] == ["late"]
