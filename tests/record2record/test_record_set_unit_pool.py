"""Record sets, batch units and their pool: the type definitions, selection order, reservations, and staleness."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.artifact_ref import RuntimeLoadSpan
from reef.core.batches import TaskItem
from reef.record2record import (
    ArrivalOrder,
    GroupKeyOrder,
    InferenceRecord2Record,
    RecordSet,
    RecordSetUnit,
    RecordSetUnitPool,
    SelectionPolicy,
    SingleRecordSet,
    SingleSetUnit,
    TaskStatus,
)
from reef.record2record.staleness import producing_runtime_load_ids, stale_units
from reef.record2task import TaskResult


def inference(record_id: str) -> AgentRecord:
    return AgentRecord.create(scenario="s", request_type=RequestType.INFERENCE, payload={}, agent_record_id=record_id)


def report(record_id: str, *references: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, payload={}, agent_record_id=record_id, references=references
    )


@dataclass(frozen=True, kw_only=True)
class RecordsSet(RecordSet):
    held: tuple[AgentRecord, ...]

    def records(self) -> tuple[AgentRecord, ...]:
        return self.held


@dataclass(frozen=True, kw_only=True)
class ManySetUnit(RecordSetUnit):
    held: tuple[RecordSet, ...]

    def record_sets(self) -> tuple[RecordSet, ...]:
        return self.held


def single(record_id: str, arrival_index: int) -> SingleRecordSet:
    return SingleRecordSet(set_id=record_id, arrival_index=arrival_index, record=inference(record_id))


def named(unit_id: str, arrival_index: int, group_key: int | None = None) -> RecordSetUnit:
    """A unit of one inference record whose ID is ``unit_id``."""
    return ManySetUnit(
        unit_id=unit_id, arrival_index=arrival_index, group_key=group_key, held=(single(unit_id, arrival_index),)
    )


class ForeignUnit(SelectionPolicy):
    def select(self, candidates: Sequence[RecordSetUnit], max_unit_count: int) -> tuple[RecordSetUnit, ...]:
        return (named("not-held", 1),)


def test_a_record_set_holds_one_or_more_records_and_every_record_they_reference() -> None:
    assert RecordsSet(set_id="r", arrival_index=1, held=(inference("i"), report("r", "i"))).record_ids() == ("i", "r")
    with pytest.raises(ValueError, match="one or more records"):
        RecordsSet(set_id="empty", arrival_index=1, held=())
    with pytest.raises(ValueError, match=r"does not hold the referenced records \['i'\]"):
        RecordsSet(set_id="r", arrival_index=1, held=(report("r", "i"),))


def test_a_record_set_unit_holds_one_or_more_record_sets_and_consumes_all_their_records() -> None:
    unit = ManySetUnit(unit_id="u", arrival_index=1, held=(single("a", 1), single("b", 2)))
    assert unit.set_ids() == ("a", "b")
    assert unit.consumed_ids() == frozenset({"a", "b"})
    record_set = single("c", 3)
    assert SingleSetUnit.of(record_set) == SingleSetUnit(unit_id="c", arrival_index=3, record_set=record_set)
    with pytest.raises(ValueError, match="one or more record sets"):
        ManySetUnit(unit_id="empty", arrival_index=1, held=())


def test_a_stage_puts_a_record_in_one_record_set_only() -> None:
    stage = InferenceRecord2Record()
    stage.form(RecordsSet(set_id="first", arrival_index=1, held=(inference("i"), report("r1", "i"))))
    # A retry of a set takes its records again.
    stage.form(RecordsSet(set_id="first", arrival_index=1, held=(inference("i"), report("r1", "i"))))
    with pytest.raises(ValueError, match=r"records \['i'\] are in another record set"):
        stage.form(RecordsSet(set_id="second", arrival_index=2, held=(inference("i"), report("r2", "i"))))
    stage.release_records(frozenset({"i", "r1"}))
    stage.form(RecordsSet(set_id="second", arrival_index=2, held=(inference("i"), report("r2", "i"))))


def test_a_ready_result_holds_one_or_more_tasks_and_a_skipped_result_none() -> None:
    task = TaskItem(Path("task"))
    assert TaskResult.ready((task,)).tasks == (task,)
    for status, tasks in ((TaskStatus.READY, ()), (TaskStatus.SKIPPED, (task,))):
        with pytest.raises(ValueError, match="READY result holds one or more tasks"):
            TaskResult(status, tasks)


def test_arrival_order_takes_the_oldest_units_up_to_the_maximum_count() -> None:
    candidates = (named("c", 3), named("a", 1), named("b", 2))
    assert [selected.unit_id for selected in ArrivalOrder().select(candidates, 2)] == ["a", "b"]


def test_group_key_order_takes_ungrouped_units_before_groups_in_key_order() -> None:
    candidates = (named("late-group", 1, group_key=2), named("single", 3), named("early-group", 2, group_key=1))
    selected = GroupKeyOrder().select(candidates, 3)
    assert [chosen.unit_id for chosen in selected] == ["single", "early-group", "late-group"]


def test_a_reservation_does_not_change_until_it_is_consumed() -> None:
    pool = RecordSetUnitPool()
    for name in ("a", "b", "c"):
        pool.put(named(name, pool.next_arrival_index()))
    reserved = pool.reserve(pool.ordered_units()[:2])
    pool.remove("a")
    pool.put(named("d", pool.next_arrival_index()))
    assert pool.reserved_units() == reserved
    # The consumed batch includes a reserved unit that was removed after the reservation.
    assert [consumed.unit_id for consumed in pool.consume_reserved()] == ["a", "b"]
    assert [remaining.unit_id for remaining in pool.units()] == ["c", "d"]
    assert pool.reserved_units() == ()


def test_a_reservation_refuses_a_repeated_unit_and_units_the_pool_does_not_hold() -> None:
    pool = RecordSetUnitPool()
    held = named("held", pool.next_arrival_index())
    pool.put(held)
    with pytest.raises(ValueError, match="more than one time"):
        pool.reserve((held, held))
    with pytest.raises(ValueError, match="units that the pool holds"):
        pool.reserve((named("not-held", 2),))
    assert pool.reserved_units() == ()


def test_the_pool_refuses_a_selection_policy_that_returns_units_it_does_not_hold() -> None:
    pool = RecordSetUnitPool(ForeignUnit())
    pool.put(named("held", pool.next_arrival_index()))
    with pytest.raises(ValueError, match="does not hold"):
        pool.ordered_units()


def test_staleness_check_measures_lag_in_versions_the_scenario_served() -> None:
    units = tuple(
        named(name, index)
        for index, name in enumerate(
            ("current", "one-behind", "two-behind", "never-served", "unknown", "mixed", "rolled-out"), start=1
        )
    )
    producing_runtime_load_ids_by_set = {
        "current": frozenset({"e:5"}),
        "one-behind": frozenset({"e:4"}),
        "two-behind": frozenset({"e:3"}),
        "never-served": frozenset({"e:6"}),
        "unknown": frozenset({None}),
        "mixed": frozenset({"e:5", "e:3"}),
        "rolled-out": frozenset(),
    }
    # The engine sequence skips e:2: lag counts the scenario's own versions, not sequence numbers.
    served = ("e:1", "e:3", "e:4", "e:5")
    stale = {"two-behind", "never-served", "unknown", "mixed"}
    assert {unit.unit_id for unit in stale_units(units, producing_runtime_load_ids_by_set, served, 1)} == stale
    # Window 0 keeps only data of the current version, and data that the algorithm rolls out.
    assert {unit.unit_id for unit in stale_units(units, producing_runtime_load_ids_by_set, served, 0)} == stale | {
        "one-behind"
    }
    # An unknown serving version checks nothing.
    assert stale_units(units, producing_runtime_load_ids_by_set, (), 0) == ()


def test_producing_runtime_load_ids_come_from_every_sample_and_token_span() -> None:
    spans = (
        RuntimeLoadSpan(start=0, end=1, runtime_load_id="e:1"),
        RuntimeLoadSpan(start=1, end=2, runtime_load_id="e:2"),
    )
    tasks = (
        policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0, runtime_load_id="e:3"),
        policy_trajectory("b", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, runtime_load_spans=spans),
        policy_trajectory("c", (1, 2), (1,), (-0.1,), 1.0),
        TaskItem(Path("task")),
    )
    assert producing_runtime_load_ids(tasks) == frozenset({"e:1", "e:2", "e:3", None})


def test_the_stage_drops_held_units_behind_the_serving_version_but_not_a_reserved_batch() -> None:
    stage = InferenceRecord2Record()
    for record_id, version in (("old", "e:1"), ("new", "e:2")):
        stage.settle(single(record_id, stage.pool.next_arrival_index()), TaskStatus.READY, frozenset({version}))
    stage.observe_serving_runtime_load_id("e:1")
    stage.observe_serving_runtime_load_id("e:2")
    stage.pool.reserve(stage.pool.ordered_units())
    stage.drop_stale_units(0)
    assert len(stage.pool) == 2

    stage.pool.release_reservation()
    stage.drop_stale_units(0)
    assert [held.unit_id for held in stage.pool.units()] == ["new"]
    assert stage.take_ended() == frozenset({"old"})
    assert "old" in stage.releasable_record_ids()
    assert set(stage.producing_runtime_load_ids) == {"new"}


def test_a_serving_version_of_another_incarnation_restarts_the_served_versions() -> None:
    stage = InferenceRecord2Record()
    for version in ("e:1", "e:2", "e:2", "f:0"):
        stage.observe_serving_runtime_load_id(version)
    assert stage.served_runtime_load_ids == ["f:0"]
