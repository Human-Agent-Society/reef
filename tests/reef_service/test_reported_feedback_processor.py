"""Reported feedback contracts: existing references, sample assembly, and consumption."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace
from pathlib import Path

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor
from reef.train.types import ProcessorContext, TaskItem, TrainDataItem, TrainingBatch


def inference(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math", request_type=RequestType.INFERENCE, payload={}, agent_record_id=record_id
    )


def report(record_id: str, *references: str, score: float = 1.0) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": list(references)},
        agent_record_id=record_id,
    )


class SampleProcessor(ReportedFeedbackProcessor):
    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, batch_size: int = 1) -> None:
        super().__init__(ProcessorContext("math", {"batch_size": batch_size}))
        self.assembled: list[str] = []
        self.fail_assembly = False

    def make_sample(self, context: ReportContext) -> TrainDataItem:
        if self.fail_assembly:
            raise ValueError("broken training data")
        self.assembled.append(context.report.agent_record_id)
        return TaskItem(Path(context.report.agent_record_id))

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"batch:{batch_number}", items)


class GroupProcessor(SampleProcessor):
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__(batch_size)
        self.discard = False

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload.get("metadata", {})
        return metadata.get("group", "g"), metadata.get("slot")

    def decide_group(self, key: object, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if len(items) < 2:
            return GroupDecision.INCOMPLETE
        return GroupDecision.DISCARD if self.discard else GroupDecision.READY


def test_a_report_shaped_for_another_component_is_released_not_raised() -> None:
    """Ingress of a scenario with several components admits what any accepts; this trainer keeps only its own."""
    from recipes.tttd.report import TTTDGroupedRolloutReport

    class GridProcessor(SampleProcessor):
        def __init__(self) -> None:
            ReportedFeedbackProcessor.__init__(
                self, ProcessorContext("math", {"batch_size": 1}, report_type=TTTDGroupedRolloutReport)
            )
            self.assembled = []
            self.fail_assembly = False

    processor = GridProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(inference("i2"))
    processor.ingest(report("r1", "i1"))  # the harness component's plain scored report
    assert processor.assembled == []
    assert "r1" in processor.retention_decision().releasable_agent_record_ids
    grid = AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={
            "score": 1.0,
            "references": ["i2"],
            "metadata": {
                "algorithm": "tttd",
                "step": 0,
                "group": 0,
                "rollout": 0,
                "groups_per_step": 1,
                "rollouts_per_group": 2,
            },
        },
        agent_record_id="r2",
    )
    processor.ingest(grid)
    assert processor.assembled == ["r2"]
    # A report of its own shape naming an unknown inference still raises, as it did.
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.ingest(
            AgentRecord.create(
                scenario="math",
                request_type=RequestType.REPORT,
                payload={**grid.payload, "references": ["i9"]},
                agent_record_id="r3",
            )
        )


def test_processor_requires_sample_assembly_instead_of_judge() -> None:
    with pytest.raises(TypeError, match="abstract"):
        ReportedFeedbackProcessor(ProcessorContext("math"))
    assert not hasattr(ReportedFeedbackProcessor, "judge")


def test_operational_backlog_excludes_reserved_reports() -> None:
    processor = SampleProcessor(batch_size=1)
    for index in (1, 2):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
    assert processor.operational_metrics()["unreserved_reports"] == 2
    batch = processor.build_batch()
    sample = processor.operational_metrics()
    assert sample["unreserved_reports"] == 1
    assert sample["reserved_reports"] == 1
    assert sample["oldest_report_wait_seconds"] >= 0
    processor.acknowledge(batch.batch_id)
    assert processor.operational_metrics()["unreserved_reports"] == 1
    assert processor.operational_metrics()["reserved_reports"] == 0


def test_inferences_wait_for_reports_and_batch_counts_completed_samples() -> None:
    processor = SampleProcessor(batch_size=2)
    for record_id in ("i1", "i2", "i3"):
        processor.ingest(inference(record_id))
    assert not processor.ready()
    processor.ingest(report("r1", "i1"))
    assert not processor.ready()
    processor.ingest(report("r2", "i2"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1", "r2")
    assert "i3" in processor.retention_decision().protected_agent_record_ids


def test_report_before_inference_fails_without_queuing_it() -> None:
    processor = SampleProcessor()
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.ingest(report("r1", "i1"))
    processor.ingest(inference("i1"))
    assert not processor.ready()
    assert processor.assembled == []
    processor.ingest(report("r1", "i1"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1",)


@pytest.mark.parametrize("references", [(), ("i1", "i1"), ("i1", "missing")])
def test_invalid_references_do_not_create_a_sample(references: tuple[str, ...]) -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    with pytest.raises(ReportValidationError):
        processor.ingest(report("r1", *references))
    assert not processor.ready()
    assert processor.retention_decision().protected_agent_record_ids == {"i1"}


@pytest.mark.parametrize("request_type", [RequestType.INFERENCE, RequestType.REPORT])
def test_processor_rejects_cross_scenario_records(request_type: RequestType) -> None:
    processor = SampleProcessor()
    record = inference("i1") if request_type is RequestType.INFERENCE else report("r1", "i1")
    with pytest.raises(ReportValidationError, match="scenario"):
        processor.ingest(replace(record, scenario="other"))


@pytest.mark.parametrize("eligible", [True, False, None])
def test_reports_cannot_set_training_eligibility(eligible: object) -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    record = report("r1", "i1")
    record = replace(record, payload={**record.payload, "metadata": {"training": {"eligible": eligible}}})
    with pytest.raises(ReportValidationError, match="eligible"):
        processor.ingest(record)
    assert not processor.ready()


@pytest.mark.parametrize("score", [0.0, -1.0, 1.0])
def test_valid_feedback_is_assembled_without_score_filtering(score: float) -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1", score=score))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1",)


def test_assembly_failure_protects_inputs_and_does_not_acknowledge_a_retry() -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.fail_assembly = True
    for _ in range(2):
        with pytest.raises(ValueError, match="broken training data"):
            processor.ingest(report("r1", "i1"))
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r1"}
    assert not processor.ready()
    processor.fail_assembly = False
    processor.ingest(report("r1", "i1"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1",)


def test_duplicate_reports_and_batch_polls_do_not_assemble_twice() -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    processor.ingest(report("r1", "i1"))
    batch = processor.build_batch()
    assert processor.build_batch() is batch
    assert processor.assembled == ["r1"]
    processor.release_batch(batch.batch_id)
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r1"}
    batch = processor.build_batch()
    assert processor.acknowledge(batch.batch_id) == {"i1", "r1"}
    processor.ingest(report("late", "i1"))
    assert not processor.ready()
    assert processor.assembled == ["r1"]


def test_live_report_keeps_a_shared_source_protected_until_its_consumption() -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    processor.ingest(report("r2", "i1"))
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r2"}
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == {"i1", "r1", "r2"}
    processor.compaction_applied(frozenset({"i1", "r1", "r2"}))
    assert processor.retention_decision().protected_agent_record_ids == set()
    assert processor.retention_decision().releasable_agent_record_ids == set()


def test_group_waits_for_complete_samples() -> None:
    processor = GroupProcessor()
    for index in (1, 2):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
        assert processor.ready() == (index == 2)
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1", "r2")


def test_group_discard_releases_all_members_and_refuses_later_members() -> None:
    processor = GroupProcessor()
    processor.discard = True
    for index in (1, 2, 3):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == {"i1", "i2", "i3", "r1", "r2", "r3"}


def test_slot_retry_preserves_first_report() -> None:
    processor = GroupProcessor()
    for record_id, slot in (("first", "a"), ("retry", "a"), ("second", "b")):
        processor.ingest(inference(record_id))
        record = report(f"r-{record_id}", record_id)
        processor.ingest(replace(record, payload={**record.payload, "metadata": {"slot": slot}}))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r-first", "r-second")
    assert {"r-retry", "retry"} <= processor.retention_decision().releasable_agent_record_ids


def test_ordered_groups_can_coexist_with_singleton_samples() -> None:
    class MixedProcessor(SampleProcessor):
        ordered_groups = True

        def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
            record_id = context.report.agent_record_id
            return (record_id if record_id.startswith("g-") else None), None

        def decide_group(self, key: object, items: tuple[TrainDataItem, ...]) -> GroupDecision:
            return GroupDecision.READY

    processor = MixedProcessor(batch_size=3)
    for record_id in ("g-z", "g-a", "single"):
        processor.ingest(inference(f"source-{record_id}"))
        processor.ingest(report(record_id, f"source-{record_id}"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("single", "g-a", "g-z")


def test_group_arrivals_after_reservation_are_left_for_the_next_batch() -> None:
    processor = GroupProcessor()
    for index in (1, 2):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
    first = processor.build_batch()
    processor.ingest(inference("i3"))
    processor.ingest(report("r3", "i3"))
    assert processor.build_batch() is first
    assert processor.acknowledge(first.batch_id) == {"i1", "i2", "r1", "r2"}
    assert processor.retention_decision().protected_agent_record_ids == {"i3", "r3"}
    assert not processor.ready()
    processor.ingest(inference("i4"))
    processor.ingest(report("r4", "i4"))
    second = processor.build_batch()
    assert [str(item.task_path) for item in second.items] == ["r3", "r4"]
    assert processor.acknowledge(second.batch_id) == {"i3", "i4", "r3", "r4"}


def test_recipe_receives_items_and_omitting_one_does_not_lose_consumption() -> None:
    class SelectedProcessor(GroupProcessor):
        def decide_group(self, key, items):
            assert all(isinstance(item, TaskItem) for item in items)
            return super().decide_group(key, items)

        def make_batch(self, items, batch_number):
            assert [item.source_agent_record_ids for item in items] == [("i1", "r1"), ("i2", "r2")]
            return super().make_batch(items[:1], batch_number)

    processor = SelectedProcessor()
    for index in (1, 2):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
    batch = processor.build_batch()
    assert [str(item.task_path) for item in batch.items] == ["r1"]
    assert processor.acknowledge(batch.batch_id) == {"i1", "i2", "r1", "r2"}
    assert processor.retention_decision().protected_agent_record_ids == set()
    assert not processor.ready()


def test_untyped_assembly_output_fails_and_keeps_report_retryable() -> None:
    class BrokenProcessor(SampleProcessor):
        def make_sample(self, context):
            return object() if self.fail_assembly else super().make_sample(context)

    processor = BrokenProcessor()
    processor.ingest(inference("i1"))
    processor.fail_assembly = True
    with pytest.raises(TypeError, match="TrainDataItem"):
        processor.ingest(report("r1", "i1"))
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r1"}
    processor.fail_assembly = False
    processor.ingest(report("r1", "i1"))
    assert processor.acknowledge(processor.build_batch().batch_id) == {"i1", "r1"}
