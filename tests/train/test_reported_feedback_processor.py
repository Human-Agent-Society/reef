"""Reported feedback contracts: existing references, sample assembly, and consumption."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace
from pathlib import Path

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError
from reef.pipeline_context import PipelineContext
from reef.record2record import GroupDecision, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2task import ReportTasks
from reef.task2batch import Task2Batch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch


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


def exclusive_stage(
    context: PipelineContext, grouping: ReportGrouping | None = None, *, ordered_groups: bool = False
) -> ReportRecord2Record:
    """A report stage in which a terminal report owns its sources."""
    return ReportRecord2Record(
        context,
        ReportGrouping() if grouping is None else grouping,
        exclusive_sources=True,
        ordered_groups=ordered_groups,
    )


class SampleTasks(ReportTasks):
    """Each report becomes a task named after it. ``fail_assembly`` breaks the training contract."""

    def __init__(self) -> None:
        super().__init__()
        self.assembled: list[str] = []
        self.fail_assembly = False

    def make_sample(self, context: ReportContext) -> TrainDataItem:
        if self.fail_assembly:
            raise ValueError("broken training data")
        self.assembled.append(context.report.agent_record_id)
        return TaskItem(Path(context.report.agent_record_id))


class SampleBatches(Task2Batch):
    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


class SamplePipeline(TrainingPipeline):
    """A reported-feedback pipeline over an exclusive report stage, with ``SampleTasks`` and ``batches``."""

    def __init__(
        self,
        batch_size: int = 1,
        *,
        grouping: ReportGrouping | None = None,
        ordered_groups: bool = False,
        report_type: type[ReportBase] | None = None,
        tasks: SampleTasks | None = None,
        batches: type[SampleBatches] = SampleBatches,
    ) -> None:
        context = PipelineContext("math", report_type)
        self.sample_tasks = SampleTasks() if tasks is None else tasks
        super().__init__(
            context,
            record2record=exclusive_stage(context, grouping, ordered_groups=ordered_groups),
            record2task=self.sample_tasks,
            task2batch=batches(context, batch_size),
        )

    @property
    def assembled(self) -> list[str]:
        return self.sample_tasks.assembled

    @property
    def fail_assembly(self) -> bool:
        return self.sample_tasks.fail_assembly

    @fail_assembly.setter
    def fail_assembly(self, fail: bool) -> None:
        self.sample_tasks.fail_assembly = fail


class PairGrouping(ReportGrouping):
    """Groups by ``metadata.group`` (default ``g``) with slots from ``metadata.slot``; ready at two members."""

    def __init__(self) -> None:
        self.discard = False

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload.get("metadata", {})
        return metadata.get("group", "g"), metadata.get("slot")

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < 2:
            return GroupDecision.INCOMPLETE
        return GroupDecision.DISCARD if self.discard else GroupDecision.READY


class GroupPipeline(SamplePipeline):
    def __init__(self, batch_size: int = 1, *, batches: type[SampleBatches] = SampleBatches) -> None:
        super().__init__(batch_size, grouping=PairGrouping(), batches=batches)

    @property
    def grouping_policy(self) -> PairGrouping:
        stage = self.record2record
        assert isinstance(stage, ReportRecord2Record)
        grouping = stage.grouping
        assert isinstance(grouping, PairGrouping)
        return grouping


def test_a_report_shaped_for_another_component_is_released_not_raised() -> None:
    """Ingress of a scenario with several components admits what any accepts; this trainer keeps only its own."""
    from recipes.tttd.report import TTTDGroupedRolloutReport
    from reef.core.reports import ScoredRolloutReport
    from reef.recipe.composite import any_component_report

    processor = SamplePipeline(report_type=TTTDGroupedRolloutReport)
    processor.push(inference("i1"))
    processor.push(inference("i2"))
    # Alone, the processor refuses a report of another shape, as ingress would have.
    with pytest.raises(ReportValidationError):
        processor.push(report("r0", "i1"))
    # Told what the scenario's ingress admits, it releases such a report as another component's.
    processor.admit_reports_of(any_component_report((TTTDGroupedRolloutReport, ScoredRolloutReport)))
    processor.push(report("r1", "i1"))  # the harness component's plain scored report
    assert processor.assembled == []
    assert "r1" in processor.releasable_record_ids()
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
    processor.push(grid)
    assert processor.assembled == ["r2"]
    # A report of its own shape naming an unknown inference still raises, as it did.
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.push(
            AgentRecord.create(
                scenario="math",
                request_type=RequestType.REPORT,
                payload={**grid.payload, "references": ["i9"]},
                agent_record_id="r3",
            )
        )


def test_processor_requires_sample_assembly_instead_of_judge() -> None:
    with pytest.raises(TypeError, match="abstract"):
        ReportTasks()
    assert not hasattr(ReportTasks, "judge")


def test_operational_backlog_excludes_reserved_reports() -> None:
    processor = SamplePipeline(batch_size=1)
    for index in (1, 2):
        processor.push(inference(f"i{index}"))
        processor.push(report(f"r{index}", f"i{index}"))
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
    processor = SamplePipeline(batch_size=2)
    for record_id in ("i1", "i2", "i3"):
        processor.push(inference(record_id))
    assert not processor.ready()
    processor.push(report("r1", "i1"))
    assert not processor.ready()
    processor.push(report("r2", "i2"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1", "r2")
    assert "i3" not in processor.releasable_record_ids()


def test_report_before_inference_fails_without_queuing_it() -> None:
    processor = SamplePipeline()
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.push(report("r1", "i1"))
    processor.push(inference("i1"))
    assert not processor.ready()
    assert processor.assembled == []
    processor.push(report("r1", "i1"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1",)


@pytest.mark.parametrize("references", [(), ("i1", "i1"), ("i1", "missing")])
def test_invalid_references_do_not_create_a_sample(references: tuple[str, ...]) -> None:
    processor = SamplePipeline()
    processor.push(inference("i1"))
    with pytest.raises(ReportValidationError):
        processor.push(report("r1", *references))
    assert not processor.ready()
    assert processor.releasable_record_ids().isdisjoint({"i1"})


@pytest.mark.parametrize("request_type", [RequestType.INFERENCE, RequestType.REPORT])
def test_processor_rejects_cross_scenario_records(request_type: RequestType) -> None:
    processor = SamplePipeline()
    record = inference("i1") if request_type is RequestType.INFERENCE else report("r1", "i1")
    with pytest.raises(ReportValidationError, match="scenario"):
        processor.push(replace(record, scenario="other"))


@pytest.mark.parametrize("eligible", [True, False, None])
def test_reports_cannot_set_training_eligibility(eligible: object) -> None:
    processor = SamplePipeline()
    processor.push(inference("i1"))
    record = report("r1", "i1")
    record = replace(record, payload={**record.payload, "metadata": {"training": {"eligible": eligible}}})
    with pytest.raises(ReportValidationError, match="eligible"):
        processor.push(record)
    assert not processor.ready()


@pytest.mark.parametrize("score", [0.0, -1.0, 1.0])
def test_valid_feedback_is_assembled_without_score_filtering(score: float) -> None:
    processor = SamplePipeline()
    processor.push(inference("i1"))
    processor.push(report("r1", "i1", score=score))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1",)


def test_assembly_failure_protects_inputs_and_does_not_acknowledge_a_retry() -> None:
    processor = SamplePipeline()
    processor.push(inference("i1"))
    processor.fail_assembly = True
    for _ in range(2):
        with pytest.raises(ValueError, match="broken training data"):
            processor.push(report("r1", "i1"))
    assert processor.releasable_record_ids().isdisjoint({"i1", "r1"})
    assert not processor.ready()
    processor.fail_assembly = False
    processor.push(report("r1", "i1"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1",)


def test_duplicate_reports_and_batch_polls_do_not_assemble_twice() -> None:
    processor = SamplePipeline()
    processor.push(inference("i1"))
    processor.push(report("r1", "i1"))
    processor.push(report("r1", "i1"))
    batch = processor.build_batch()
    assert processor.build_batch() is batch
    assert processor.assembled == ["r1"]
    processor.release_batch(batch.batch_id)
    assert processor.releasable_record_ids().isdisjoint({"i1", "r1"})
    batch = processor.build_batch()
    assert processor.acknowledge(batch.batch_id) == {"i1", "r1"}
    processor.push(report("late", "i1"))
    assert not processor.ready()
    assert processor.assembled == ["r1"]


def test_a_second_report_on_the_inference_of_a_live_report_ends_without_training() -> None:
    processor = SamplePipeline()
    processor.push(inference("i1"))
    processor.push(report("r1", "i1"))
    processor.push(report("r2", "i1"))
    # r2 ends at once. The live r1 keeps i1.
    assert processor.releasable_record_ids() == {"r2"}
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.assembled == ["r1"]
    assert not processor.ready()
    assert processor.releasable_record_ids() == {"i1", "r1", "r2"}
    processor.release_records(frozenset({"i1", "r1", "r2"}))
    assert processor.releasable_record_ids() == set()


def test_group_waits_for_complete_samples() -> None:
    processor = GroupPipeline()
    for index in (1, 2):
        processor.push(inference(f"i{index}"))
        processor.push(report(f"r{index}", f"i{index}"))
        assert processor.ready() == (index == 2)
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r1", "r2")


def test_group_discard_releases_all_members_and_refuses_later_members() -> None:
    processor = GroupPipeline()
    processor.grouping_policy.discard = True
    for index in (1, 2, 3):
        processor.push(inference(f"i{index}"))
        processor.push(report(f"r{index}", f"i{index}"))
    assert not processor.ready()
    assert processor.releasable_record_ids() == {"i1", "i2", "i3", "r1", "r2", "r3"}


def test_slot_retry_preserves_first_report() -> None:
    processor = GroupPipeline()
    for record_id, slot in (("first", "a"), ("retry", "a"), ("second", "b")):
        processor.push(inference(record_id))
        record = report(f"r-{record_id}", record_id)
        processor.push(replace(record, payload={**record.payload, "metadata": {"slot": slot}}))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("r-first", "r-second")
    assert {"r-retry", "retry"} <= processor.releasable_record_ids()


def test_ordered_groups_can_coexist_with_singleton_samples() -> None:
    class PrefixGrouping(ReportGrouping):
        def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
            record_id = context.report.agent_record_id
            return (record_id if record_id.startswith("g-") else None), None

        def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
            return GroupDecision.READY

    processor = SamplePipeline(batch_size=3, grouping=PrefixGrouping(), ordered_groups=True)
    for record_id in ("g-z", "g-a", "single"):
        processor.push(inference(f"source-{record_id}"))
        processor.push(report(record_id, f"source-{record_id}"))
    assert tuple(str(item.task_path) for item in processor.build_batch().items) == ("single", "g-a", "g-z")


def test_group_arrivals_after_reservation_are_left_for_the_next_batch() -> None:
    processor = GroupPipeline()
    for index in (1, 2):
        processor.push(inference(f"i{index}"))
        processor.push(report(f"r{index}", f"i{index}"))
    first = processor.build_batch()
    processor.push(inference("i3"))
    processor.push(report("r3", "i3"))
    assert processor.build_batch() is first
    assert processor.acknowledge(first.batch_id) == {"i1", "i2", "r1", "r2"}
    assert processor.releasable_record_ids().isdisjoint({"i3", "r3"})
    assert not processor.ready()
    processor.push(inference("i4"))
    processor.push(report("r4", "i4"))
    second = processor.build_batch()
    assert [str(item.task_path) for item in second.items] == ["r3", "r4"]
    assert processor.acknowledge(second.batch_id) == {"i3", "i4", "r3", "r4"}


def test_recipe_receives_items_and_omitting_one_does_not_lose_consumption() -> None:
    class SelectedBatches(SampleBatches):
        def make_batch(self, tasks, batch_id):
            assert [task.source_agent_record_ids for task in tasks] == [("i1", "r1"), ("i2", "r2")]
            return super().make_batch(tasks[:1], batch_id)

    processor = GroupPipeline(batches=SelectedBatches)
    for index in (1, 2):
        processor.push(inference(f"i{index}"))
        processor.push(report(f"r{index}", f"i{index}"))
    batch = processor.build_batch()
    assert [str(item.task_path) for item in batch.items] == ["r1"]
    assert processor.acknowledge(batch.batch_id) == {"i1", "i2", "r1", "r2"}
    assert not processor.ready()


def test_untyped_assembly_output_fails_and_keeps_report_retryable() -> None:
    class BrokenTasks(SampleTasks):
        def make_sample(self, context):
            return object() if self.fail_assembly else super().make_sample(context)

    processor = SamplePipeline(tasks=BrokenTasks())
    processor.push(inference("i1"))
    processor.fail_assembly = True
    with pytest.raises(TypeError, match="TrainDataItem"):
        processor.push(report("r1", "i1"))
    assert processor.releasable_record_ids().isdisjoint({"i1", "r1"})
    processor.fail_assembly = False
    processor.push(report("r1", "i1"))
    assert processor.acknowledge(processor.build_batch().batch_id) == {"i1", "r1"}
