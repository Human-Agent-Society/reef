"""Experience buffer contracts: selection order, reservations, eligibility checks, and the processor hooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from reef.core import AgentRecord, RequestType
from reef.train.experience import (
    ArrivalOrder,
    EligibilityCheck,
    ExperienceBuffer,
    ExperienceUnit,
    GroupKeyOrder,
    IneligibleUnit,
    NewestVersionCheck,
    SelectionPolicy,
)
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor
from reef.train.types import ProcessorContext, TaskItem, TrainDataItem, TrainingBatch


def unit(unit_id: str, arrival_index: int, group_key: int | None = None) -> ExperienceUnit:
    return ExperienceUnit(unit_id=unit_id, arrival_index=arrival_index, group_key=group_key)


class NewestFirst(SelectionPolicy):
    def select(self, candidates: Sequence[ExperienceUnit], max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        return tuple(sorted(candidates, key=lambda candidate: -candidate.arrival_index)[:max_unit_count])


class ForeignUnit(SelectionPolicy):
    def select(self, candidates: Sequence[ExperienceUnit], max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        return (unit("not-held", 1),)


def test_arrival_order_takes_the_oldest_units_up_to_the_maximum_count() -> None:
    candidates = (unit("c", 3), unit("a", 1), unit("b", 2))
    assert [selected.unit_id for selected in ArrivalOrder().select(candidates, 2)] == ["a", "b"]


def test_group_key_order_takes_ungrouped_units_before_groups_in_key_order() -> None:
    candidates = (unit("late-group", 1, group_key=2), unit("single", 3), unit("early-group", 2, group_key=1))
    selected = GroupKeyOrder().select(candidates, 3)
    assert [chosen.unit_id for chosen in selected] == ["single", "early-group", "late-group"]


def test_a_reservation_does_not_change_until_it_is_consumed() -> None:
    buffer = ExperienceBuffer()
    for name in ("a", "b", "c"):
        buffer.put(unit(name, buffer.next_arrival_index()))
    reserved = buffer.reserve(2)
    buffer.remove("a")
    buffer.put(unit("d", buffer.next_arrival_index()))
    assert buffer.reserved_units() == reserved
    # The consumed batch includes a reserved unit that was removed after the reservation.
    assert [consumed.unit_id for consumed in buffer.consume_reserved()] == ["a", "b"]
    assert [remaining.unit_id for remaining in buffer.units()] == ["c", "d"]
    assert buffer.reserved_units() == ()


def test_a_reservation_refuses_a_negative_count_and_units_the_buffer_does_not_hold() -> None:
    with pytest.raises(ValueError, match="max_unit_count must not be negative"):
        ExperienceBuffer().reserve(-1)
    buffer = ExperienceBuffer(ForeignUnit())
    buffer.put(unit("held", buffer.next_arrival_index()))
    with pytest.raises(ValueError, match="does not hold"):
        buffer.reserve(1)


def test_a_processor_changes_its_batch_order_through_selection_policy() -> None:
    class NewestFirstProcessor(ReportedFeedbackProcessor):
        def selection_policy(self) -> SelectionPolicy:
            return NewestFirst()

        def make_sample(self, context: ReportContext) -> TrainDataItem:
            return TaskItem(Path(context.report.agent_record_id))

        def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
            return TrainingBatch(f"batch:{batch_number}", items)

    processor = NewestFirstProcessor(ProcessorContext("math", {"batch_size": 2}))
    for index in (1, 2, 3):
        processor.ingest(
            AgentRecord.create(
                scenario="math", request_type=RequestType.INFERENCE, payload={}, agent_record_id=f"i{index}"
            )
        )
        processor.ingest(
            AgentRecord.create(
                scenario="math",
                request_type=RequestType.REPORT,
                payload={"score": 1.0, "references": [f"i{index}"]},
                agent_record_id=f"r{index}",
            )
        )
    batch = processor.build_batch()
    assert [str(item.task_path) for item in batch.items] == ["r3", "r2"]
    assert processor.acknowledge(batch.batch_id) == {"r3", "r2", "i3", "i2"}
    assert processor.status()["ready_units"] == 1


def test_newest_version_check_keeps_the_version_of_the_last_source_record() -> None:
    # The unit that arrived last in the buffer is not the newest: its source record arrived first.
    late_judgment = ExperienceUnit(unit_id="late", arrival_index=3, runtime_load_id="v1", source_index=1)
    newest = ExperienceUnit(unit_id="newest", arrival_index=2, runtime_load_id="v2", source_index=3)
    same_version = ExperienceUnit(unit_id="same", arrival_index=1, runtime_load_id="v2", source_index=2)
    dropped = NewestVersionCheck().ineligible_units((late_judgment, newest, same_version))
    assert [(result.unit.unit_id, result.reason) for result in dropped] == [("late", "older_runtime_load_id")]


def test_drop_ineligible_removes_and_returns_the_units_that_fail_each_check() -> None:
    buffer = ExperienceBuffer(checks=(NewestVersionCheck(),))
    for name, version in (("old", "v1"), ("new", "v2")):
        buffer.put(ExperienceUnit(unit_id=name, arrival_index=buffer.next_arrival_index(), runtime_load_id=version))
    assert [result.unit.unit_id for result in buffer.drop_ineligible()] == ["old"]
    assert [held.unit_id for held in buffer.units()] == ["new"]

    class ForeignCheck(EligibilityCheck):
        def ineligible_units(self, units: Sequence[ExperienceUnit]) -> tuple[IneligibleUnit, ...]:
            return (IneligibleUnit(unit("not-held", 1), "foreign"),)

    with pytest.raises(ValueError, match="does not hold"):
        ExperienceBuffer(checks=(ForeignCheck(),)).drop_ineligible()
