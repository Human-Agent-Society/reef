"""Record set pool contracts: selection order, reservations, and the staleness check."""

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
    Pool,
    RecordSet,
    RecordSetPool,
    SelectionPolicy,
    SingleRecordSet,
    TaskSet,
    TaskStatus,
)
from reef.record2record.staleness import producing_runtime_load_ids, stale_sets


@dataclass(frozen=True, kw_only=True)
class NamedSet(RecordSet):
    """A set that holds only its name."""

    def record_ids(self) -> tuple[str, ...]:
        return (str(self.set_id),)


def named(set_id: str, arrival_index: int, group_key: int | None = None) -> NamedSet:
    return NamedSet(set_id=set_id, arrival_index=arrival_index, group_key=group_key)


def inference(record_id: str) -> AgentRecord:
    return AgentRecord.create(scenario="s", request_type=RequestType.INFERENCE, payload={}, agent_record_id=record_id)


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


def test_staleness_check_measures_lag_in_versions_the_scenario_served() -> None:
    record_sets = tuple(
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
    assert {
        stale_set.set_id for stale_set in stale_sets(record_sets, producing_runtime_load_ids_by_set, served, 1)
    } == stale
    # Window 0 keeps only data of the current version, and data that the algorithm rolls out.
    assert {
        stale_set.set_id for stale_set in stale_sets(record_sets, producing_runtime_load_ids_by_set, served, 0)
    } == stale | {"one-behind"}
    # An unknown serving version checks nothing.
    assert stale_sets(record_sets, producing_runtime_load_ids_by_set, (), 0) == ()


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


def test_the_stage_drops_held_sets_behind_the_serving_version_but_not_a_reserved_batch() -> None:
    stage = InferenceRecord2Record()
    for record_id, version in (("old", "e:1"), ("new", "e:2")):
        stage.settle(
            SingleRecordSet(
                set_id=record_id, arrival_index=stage.pool.next_arrival_index(), record=inference(record_id)
            ),
            TaskStatus.READY,
            frozenset({version}),
        )
    stage.observe_serving_runtime_load_id("e:1")
    stage.observe_serving_runtime_load_id("e:2")
    stage.pool.reserve(2)
    stage.drop_stale_sets(0)
    assert len(stage.pool) == 2

    stage.pool.release_reservation()
    stage.drop_stale_sets(0)
    assert [held.set_id for held in stage.pool.sets()] == ["new"]
    assert stage.take_ended() == frozenset({"old"})
    assert "old" in stage.releasable_record_ids()
    assert set(stage.producing_runtime_load_ids) == {"new"}


def test_a_serving_version_of_another_incarnation_restarts_the_served_versions() -> None:
    stage = InferenceRecord2Record()
    for version in ("e:1", "e:2", "e:2", "f:0"):
        stage.observe_serving_runtime_load_id(version)
    assert stage.served_runtime_load_ids == ["f:0"]


def test_a_pool_of_saved_tasks_reserves_them_in_arrival_order() -> None:
    pool: Pool[TaskSet] = Pool()
    for name in ("first", "second", "third"):
        pool.put(TaskSet(set_id=name, arrival_index=pool.next_arrival_index(), item=TaskItem(Path(name))))
    assert [task_set.set_id for task_set in pool.reserve(2)] == ["first", "second"]
    assert [task_set.item.task_path.name for task_set in pool.consume_reserved()] == ["first", "second"]
    assert [task_set.set_id for task_set in pool.sets()] == ["third"]
