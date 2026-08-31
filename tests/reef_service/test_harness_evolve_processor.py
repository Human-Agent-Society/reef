"""HarnessEvolveProcessor: the failed-trace batching half of harness evolution.

The processor pairs reports with their inference records and batches the
traces whose scores fall inside the configured window; the evolution backend
(``HarnessEvolveBackend``) consumes the batch. Exercised directly rather
than through any Recipe wrapper.
"""

from __future__ import annotations

import pytest

from recipes.harness_evolve.processor import HarnessEvolveProcessor
from reef.core import AgentRecord, RequestType
from reef.train.types import ProcessorContext, TraceBatch


def _inference(agent_record_id: str, payload: dict) -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
    )


def _report(agent_record_id: str, score: object, references: list[str]) -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": references},
        agent_record_id=agent_record_id,
    )


def _processor(config: dict | None = None) -> HarnessEvolveProcessor:
    return HarnessEvolveProcessor(ProcessorContext("s", config or {}))


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


def test_trace_processor_rejects_multi_reference_reports_without_selecting_the_first() -> None:
    processor = _processor({"max_score": 0.0})
    processor.ingest(_inference("inf-1", {"messages": [{"role": "user", "content": "first"}]}))
    processor.ingest(_inference("inf-2", {"messages": [{"role": "user", "content": "second"}]}))
    processor.ingest(_report("rep-1", 0.0, ["inf-1", "inf-2"]))

    assert not processor.ready()
    retention = processor.retention_decision()
    assert retention.protected_agent_record_ids == frozenset()
    assert retention.releasable_agent_record_ids == frozenset({"inf-1", "inf-2", "rep-1"})


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
    assert not hasattr(HarnessEvolveProcessor, "prepare_training_step")
    assert not hasattr(HarnessEvolveProcessor, "prepare")


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
