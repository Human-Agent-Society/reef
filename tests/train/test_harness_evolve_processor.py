"""The trace pipeline: the trace batching half of harness evolution.

The pipeline pairs reports with their inference records and batches every
valid scored trace; the evolution backend (``CordisBackend``) consumes the
batch. The stages are built from a ``pipeline`` section, as a deployment
configures them, rather than through any Recipe wrapper.
"""

from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.core.trajectories import recorded_payload, recorded_payloads, source_record_id
from reef.pipeline_context import PipelineContext
from reef.recipe.cordis import CordisRecipe
from reef.record2record import InferenceRecord2Record, ReportGrouping, ReportRecord2Record
from reef.train.cordis_backend.processor import InferenceSamples, TraceJobData, TraceSamples
from reef.train.pipeline import PipelineConfig, TrainingPipeline
from reef.train.types import JobData


def _inference(agent_record_id: str, payload: dict) -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
    )


def _report(agent_record_id: str, score: object, references: list[str], feedback: object = None) -> AgentRecord:
    payload: dict = {"score": score, "references": references}
    if feedback is not None:
        payload["feedback"] = feedback
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload=payload,
        agent_record_id=agent_record_id,
    )


#: The ``pipeline`` section that evolves on recorded inference traffic alone, without reports.
INFERENCE_STAGES = {
    "record2record": {"module": "reef.record2record:InferenceRecord2Record"},
    "record2sample": {"module": "reef.train.cordis_backend.processor:InferenceSamples"},
}


def harness_stages(*, inference: bool = False, buffer_size: int | None = None) -> PipelineConfig:
    """The harness evolution stages as a deployment's ``pipeline`` section configures them."""
    section: dict = dict(INFERENCE_STAGES) if inference else {}
    if buffer_size is not None:
        section["sample2job"] = {"config": {"buffer_size": buffer_size}}
    return PipelineConfig.from_setting(section, spec=CordisRecipe.pipeline_spec(), environ={})


def reports_pipeline(context: PipelineContext, buffer_size: int | None = None) -> TrainingPipeline:
    """Harness traces from reported scores: the default stages, built directly."""
    return TrainingPipeline(
        context,
        record2record=ReportRecord2Record(context, ReportGrouping()),
        record2sample=TraceSamples(),
        sample2job=TraceJobData(context, buffer_size, None),
    )


def records_pipeline(context: PipelineContext, buffer_size: int | None = None) -> TrainingPipeline:
    """Harness traces from recorded inference traffic, unscored: the inference stages, built directly."""
    return TrainingPipeline(
        context,
        record2record=InferenceRecord2Record(),
        record2sample=InferenceSamples(),
        sample2job=TraceJobData(context, buffer_size, None),
    )


def _processor() -> TrainingPipeline:
    return harness_stages().build(PipelineContext("s"))


def test_the_default_stages_train_reports_and_the_inference_stages_train_records() -> None:
    reported = harness_stages(buffer_size=6).build(PipelineContext("s", default_job_samples=2))
    assert (type(reported.record2record), type(reported.record2sample), type(reported.sample2job)) == (
        ReportRecord2Record,
        TraceSamples,
        TraceJobData,
    )
    assert reported.buffer_size == 6
    recorded = harness_stages(inference=True).build(PipelineContext("s", default_job_samples=2))
    assert (type(recorded.record2record), type(recorded.record2sample), type(recorded.sample2job)) == (
        InferenceRecord2Record,
        InferenceSamples,
        TraceJobData,
    )
    assert recorded.buffer_size == 8


def test_trace_processor_batches_a_failed_trace() -> None:
    processor = _processor()
    payload = {"messages": [{"role": "system", "content": "skill"}, {"role": "user", "content": "q"}]}
    processor.push(_inference("inf-1", payload))
    processor.push(_report("rep-1", 0.0, ["inf-1"]))
    assert processor.ready(1)
    batch = processor.build_job_data(1)
    assert isinstance(batch, JobData)
    assert len(batch.samples) == 1
    sample = batch.samples[0]
    assert source_record_id(sample) == "inf-1"
    assert sample.metadata.get("reward") == 0.0
    assert recorded_payload(sample)["messages"][0]["content"] == "skill"
    processor.acknowledge(batch.data_id)
    retention = processor.releasable_record_ids()
    assert "rep-1" in retention


def test_trace_processor_batches_successful_reports() -> None:
    processor = _processor()
    processor.push(_inference("inf-1", {"messages": []}))
    processor.push(_report("rep-1", 1.0, ["inf-1"]))
    assert processor.build_job_data(1).samples[0].metadata.get("reward") == 1.0


def test_trace_processor_carries_the_task_a_report_names() -> None:
    processor = _processor()
    processor.push(_inference("inf-1", {"messages": []}))
    report = _report("rep-1", 1.0, ["inf-1"])
    task = {"name": "hello-file", "path": "/tasks/hello-file", "digest": "ab" * 32}
    named = AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload={**report.payload, "metadata": {"task": task}},
        agent_record_id="rep-1",
    )
    processor.push(named)
    sample = processor.build_job_data(1).samples[0]
    assert sample.metadata["task"] == task and sample.metadata["reward"] == 1.0


def test_trace_processor_batches_a_multi_reference_report_as_one_trajectory() -> None:
    """A report over a whole run becomes one sample: the trajectory holds
    every referenced payload in reference order, the payload is the last
    exchange, and the feedback rides along verbatim."""
    processor = _processor()
    first = {"messages": [{"role": "user", "content": "first"}]}
    second = {"messages": [{"role": "user", "content": "second"}]}
    processor.push(_inference("inf-1", first))
    processor.push(_inference("inf-2", second))
    processor.push(_report("rep-1", 0.0, ["inf-1", "inf-2"], feedback="wrong file"))

    assert processor.ready(1)
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert source_record_id(sample) == "inf-2"
    assert recorded_payload(sample) == second
    assert recorded_payloads(sample) == (first, second)
    assert sample.metadata.get("feedback") == "wrong file"
    assert sample.metadata.get("reward") == 0.0


def test_trace_processor_keeps_single_reference_trajectory_and_carries_feedback() -> None:
    processor = _processor()
    payload = {"messages": [{"role": "user", "content": "q"}]}
    processor.push(_inference("inf-1", payload))
    processor.push(_report("rep-1", 0.0, ["inf-1"], feedback={"reason": "timeout"}))

    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert recorded_payload(sample) == payload
    assert recorded_payloads(sample) == (payload,)
    assert sample.metadata.get("feedback") == {"reason": "timeout"}


def test_trace_processor_rejects_a_report_without_references() -> None:
    processor = _processor()
    with pytest.raises(ReportValidationError, match="references"):
        processor.push(_report("rep-1", 0.0, []))
    assert not processor.ready(1)


def test_trace_processor_has_no_judge_hook() -> None:
    for stage in (TrainingPipeline, TraceSamples, TraceJobData):
        assert not hasattr(stage, "judge")


def test_trace_processor_has_no_training_preparation_hook() -> None:
    for stage in (TrainingPipeline, TraceSamples, TraceJobData):
        assert not hasattr(stage, "prepare_training_step")
        assert not hasattr(stage, "prepare")


def test_trace_processor_refuses_a_non_finite_score() -> None:
    for bad in (float("inf"), float("-inf"), float("nan")):
        processor = _processor()
        processor.push(_inference("inf-1", {"messages": []}))
        with pytest.raises(ReportValidationError, match="finite"):
            processor.push(_report("rep-1", bad, ["inf-1"]))
        assert not processor.ready(1)


def _record_processor() -> TrainingPipeline:
    return harness_stages(inference=True).build(PipelineContext("s"))


def test_record_driven_processor_batches_every_n_inferences_unscored() -> None:
    """The report-free policy: recorded traffic alone forms the batch, in
    arrival order, with score None on every sample."""
    processor = _record_processor()
    first = {"messages": [{"role": "user", "content": "first"}]}
    second = {"messages": [{"role": "user", "content": "second"}]}
    processor.push(_inference("inf-1", first))
    assert not processor.ready(2)
    processor.push(_inference("inf-2", second))
    assert processor.ready(2)

    batch = processor.build_job_data(2)
    assert isinstance(batch, JobData)
    assert [source_record_id(s) for s in batch.samples] == ["inf-1", "inf-2"]
    assert [recorded_payload(s) for s in batch.samples] == [first, second]
    assert {s.metadata.get("reward") for s in batch.samples} == {None}

    retention = processor.releasable_record_ids()
    assert retention.isdisjoint(frozenset({"inf-1", "inf-2"}))

    consumed = processor.acknowledge(batch.data_id)
    assert consumed == frozenset({"inf-1", "inf-2"})
    retention = processor.releasable_record_ids()
    assert retention == frozenset({"inf-1", "inf-2"})
    assert not processor.ready(2)


def test_record_driven_processor_releases_reports_untouched() -> None:
    processor = _record_processor()
    processor.push(_report("rep-1", 0.0, ["inf-0"], feedback="ignored"))
    assert not processor.ready(1)
    retention = processor.releasable_record_ids()
    assert retention == frozenset({"rep-1"})

    processor.push(_inference("inf-1", {"messages": []}))
    batch = processor.build_job_data(1)
    (sample,) = batch.samples
    assert sample.metadata.get("reward") is None
    assert sample.metadata.get("feedback") is None


def test_record_driven_processor_overflow_stays_pending_for_the_next_batch() -> None:
    processor = _record_processor()
    for index in range(3):
        processor.push(_inference(f"inf-{index}", {"messages": []}))
    batch = processor.build_job_data(2)
    assert [source_record_id(s) for s in batch.samples] == ["inf-0", "inf-1"]
    processor.acknowledge(batch.data_id)
    retention = processor.releasable_record_ids()
    assert retention.isdisjoint(frozenset({"inf-2"}))
    assert not processor.ready(2)
