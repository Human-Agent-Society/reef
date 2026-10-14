"""Record set pool contracts: selection order, reservations, and eligibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from reef.record2record import (
    ArrivalOrder,
    EligibilityCheck,
    GroupKeyOrder,
    IneligibleSet,
    RecordSet,
    RecordSetPool,
    SelectionPolicy,
    StalenessCheck,
)


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
    reasons = {result.record_set.set_id: result.reason for result in StalenessCheck(1).ineligible_sets(record_sets)}
    assert reasons == {
        "two-behind": "policy_lag_exceeded",
        "ahead": "future_producing_runtime_load_id",
        "other-incarnation": "cross_incarnation",
        "unparsed": "malformed_producing_runtime_load_id",
    }
    # Window 0 keeps only the reference version.
    assert {result.record_set.set_id for result in StalenessCheck().ineligible_sets(record_sets)} == set(reasons) | {
        "one-behind"
    }


def test_staleness_check_compares_non_canonical_ids_by_equality() -> None:
    late_judgment = NamedSet(set_id="late", arrival_index=3, runtime_load_id="v1", source_index=1)
    newest = NamedSet(set_id="newest", arrival_index=2, runtime_load_id="v2", source_index=3)
    same_version = NamedSet(set_id="same", arrival_index=1, runtime_load_id="v2", source_index=2)
    dropped = StalenessCheck(5).ineligible_sets((late_judgment, newest, same_version))
    assert [(result.record_set.set_id, result.reason) for result in dropped] == [("late", "different_runtime_load_id")]
    with pytest.raises(ValueError, match="non-negative"):
        StalenessCheck(-1)


def test_drop_ineligible_removes_and_returns_the_sets_that_fail_each_check() -> None:
    pool = RecordSetPool(checks=(StalenessCheck(),))
    for name, version in (("old", "e:1"), ("new", "e:2")):
        pool.put(NamedSet(set_id=name, arrival_index=pool.next_arrival_index(), runtime_load_id=version))
    assert [result.record_set.set_id for result in pool.drop_ineligible()] == ["old"]
    assert [held.set_id for held in pool.sets()] == ["new"]

    class ForeignCheck(EligibilityCheck):
        def ineligible_sets(self, record_sets: Sequence[RecordSet]) -> tuple[IneligibleSet, ...]:
            return (IneligibleSet(named("not-held", 1), "foreign"),)

    with pytest.raises(ValueError, match="does not hold"):
        RecordSetPool(checks=(ForeignCheck(),)).drop_ineligible()
