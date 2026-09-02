"""CordisProcessor: the failed-trace batching half of harness evolution.

The processor pairs reports with their inference records and batches the
traces whose scores fall inside the configured window; the evolution backend
(``CordisBackend``) consumes the batch. Exercised directly rather
than through any Recipe wrapper.
"""

from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.train.cordis_backend.processor import CordisProcessor, RecordDrivenTraceProcessor
from reef.train.types import ProcessorContext, TraceBatch


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


def _processor(config: dict | None = None) -> CordisProcessor:
    return CordisProcessor(ProcessorContext("s", config or {}))


def test_trace_processor_batches_a_failed_trace() -> None:
    processor = _processor({"max_score": 0.0})
    payload = {"messages": [{"role": "system", "content": "skill"}, {"role": "user", "content": "q"}]}
    processor.ingest(_inference("inf-1", payload))
    processor.ingest(_report("rep-1", 0.0, ["inf-1"]))
    assert processor.ready()
    batch = processor.build_batch()
    assert isinstance(batch, TraceBatch)
    assert len(batch.samples) == 1
    sample = batch.samples[0]
    assert sample.source_agent_record_id == "inf-1"
    assert sample.score == 0.0
    assert sample.payload["messages"][0]["content"] == "skill"
    processor.acknowledge(batch.batch_id)
    retention = processor.retention_decision()
    assert "rep-1" in retention.releasable_agent_record_ids


def test_trace_processor_ignores_reports_outside_the_score_window() -> None:
    processor = _processor({"max_score": 0.0})
    processor.ingest(_inference("inf-1", {"messages": []}))
    processor.ingest(_report("rep-1", 1.0, ["inf-1"]))
    assert not processor.ready()
    retention = processor.retention_decision()
    assert "rep-1" in retention.releasable_agent_record_ids


def test_trace_processor_batches_a_multi_reference_report_as_one_trajectory() -> None:
    """A report over a whole run becomes one sample: the trajectory holds
    every referenced payload in reference order, the payload is the last
    exchange, and the feedback rides along verbatim."""
    processor = _processor({"max_score": 0.0})
    first = {"messages": [{"role": "user", "content": "first"}]}
    second = {"messages": [{"role": "user", "content": "second"}]}
    processor.ingest(_inference("inf-1", first))
    processor.ingest(_inference("inf-2", second))
    processor.ingest(_report("rep-1", 0.0, ["inf-1", "inf-2"], feedback="wrong file"))

    assert processor.ready()
    batch = processor.build_batch()
    (sample,) = batch.samples
    assert sample.source_agent_record_id == "inf-2"
    assert sample.payload == second
    assert sample.trajectory == (first, second)
    assert sample.feedback == "wrong file"
    assert sample.score == 0.0


def test_trace_processor_keeps_single_reference_samples_flat_and_carries_feedback() -> None:
    processor = _processor({"max_score": 0.0})
    payload = {"messages": [{"role": "user", "content": "q"}]}
    processor.ingest(_inference("inf-1", payload))
    processor.ingest(_report("rep-1", 0.0, ["inf-1"], feedback={"reason": "timeout"}))

    batch = processor.build_batch()
    (sample,) = batch.samples
    assert sample.payload == payload
    assert sample.trajectory == ()
    assert sample.feedback == {"reason": "timeout"}


def test_trace_processor_never_trains_a_report_without_references() -> None:
    processor = _processor({"max_score": 0.0})
    processor.ingest(_report("rep-1", 0.0, []))
    assert not processor.ready()
    retention = processor.retention_decision()
    assert "rep-1" in retention.releasable_agent_record_ids


def test_trace_processor_rejects_an_inverted_window() -> None:
    with pytest.raises(ValueError):
        _processor({"min_score": 1.0, "max_score": 0.0})


def test_trace_processor_validates_assembly_config_fields_at_construction() -> None:
    # The trace-batch spec never assembles a policy sample, but the assembly
    # settings are part of every reported-feedback processor's config surface: a bad value
    # fails at construction instead of passing silently.
    with pytest.raises(ValueError, match="realign_threshold"):
        _processor({"realign_threshold": -1})
    with pytest.raises(ValueError, match="accept_multi_turn_policy_samples"):
        _processor({"accept_multi_turn_policy_samples": "yes"})


def test_trace_processor_has_no_training_preparation_hook() -> None:
    assert not hasattr(CordisProcessor, "prepare_training_step")
    assert not hasattr(CordisProcessor, "prepare")


def test_trace_processor_refuses_a_non_finite_score() -> None:
    # An open window still admits no score that cannot be compared: the
    # layer's invariant is that NaN and inf never train, and the default
    # window is [-inf, inf], where a chained comparison alone would pass inf.
    for bad in (float("inf"), float("-inf"), float("nan")):
        processor = _processor()
        processor.ingest(_inference("inf-1", {"messages": []}))
        processor.ingest(_report("rep-1", bad, ["inf-1"]))
        assert not processor.ready()
        assert "rep-1" in processor.retention_decision().releasable_agent_record_ids


def _record_processor(batch_size: int = 2) -> RecordDrivenTraceProcessor:
    return RecordDrivenTraceProcessor(ProcessorContext("s", {"batch_size": batch_size}))


def test_record_driven_processor_batches_every_n_inferences_unscored() -> None:
    """The report-free policy: recorded traffic alone forms the batch, in
    arrival order, with score None on every sample."""
    processor = _record_processor(batch_size=2)
    first = {"messages": [{"role": "user", "content": "first"}]}
    second = {"messages": [{"role": "user", "content": "second"}]}
    processor.ingest(_inference("inf-1", first))
    assert not processor.ready()
    processor.ingest(_inference("inf-2", second))
    assert processor.ready()

    batch = processor.build_batch()
    assert isinstance(batch, TraceBatch)
    assert [s.source_agent_record_id for s in batch.samples] == ["inf-1", "inf-2"]
    assert [s.payload for s in batch.samples] == [first, second]
    assert {s.score for s in batch.samples} == {None}

    retention = processor.retention_decision()
    assert retention.protected_agent_record_ids == frozenset({"inf-1", "inf-2"})

    consumed = processor.acknowledge(batch.batch_id)
    assert consumed == frozenset({"inf-1", "inf-2"})
    retention = processor.retention_decision()
    assert retention.releasable_agent_record_ids == frozenset({"inf-1", "inf-2"})
    assert not processor.ready()


def test_record_driven_processor_releases_reports_untouched() -> None:
    processor = _record_processor(batch_size=1)
    processor.ingest(_report("rep-1", 0.0, ["inf-0"], feedback="ignored"))
    assert not processor.ready()
    retention = processor.retention_decision()
    assert retention.releasable_agent_record_ids == frozenset({"rep-1"})

    processor.ingest(_inference("inf-1", {"messages": []}))
    batch = processor.build_batch()
    (sample,) = batch.samples
    assert sample.score is None
    assert sample.feedback is None


def test_record_driven_processor_overflow_stays_pending_for_the_next_batch() -> None:
    processor = _record_processor(batch_size=2)
    for index in range(3):
        processor.ingest(_inference(f"inf-{index}", {"messages": []}))
    batch = processor.build_batch()
    assert [s.source_agent_record_id for s in batch.samples] == ["inf-0", "inf-1"]
    processor.acknowledge(batch.batch_id)
    retention = processor.retention_decision()
    assert retention.protected_agent_record_ids == frozenset({"inf-2"})
    assert not processor.ready()
