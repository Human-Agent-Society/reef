"""The default record-to-sample stage (RFC #1054): each trajectory becomes one ATIF trajectory sample."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.conversations import conversation

from reef.core import AgentRecord, RequestType
from reef.record2sample import ReportContext, ReportSamples, SkippedReport, TrajectorySamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.train import PipelineContext
from reef.train.types import TrajectorySample


def test_a_trajectory_becomes_one_atif_sample_of_its_calls(monkeypatch) -> None:
    # Each call ends with a user turn, and its reply is the assistant turn that the next call resends.
    monkeypatch.setattr("reef.record2sample.trajectories.time.monotonic", lambda: 0.0)
    stage = TrajectorySamples.from_config({}, context=PipelineContext(scenario="math"), environ={})
    for turns in (1, 3):
        stage.push(conversation(f"call-{turns}", turns))
    stage.advance(1000.0)
    (unit,) = stage.take_ready_units()
    ((set_id, (sample,)),) = stage.unit_samples(unit)
    assert set_id == ("trajectory", "call-3")
    assert sample.source_agent_record_ids == ("call-1", "call-3")
    steps = sample.trajectory["steps"]
    assert [step["source"] for step in steps] == ["system", "user", "agent", "user", "agent"]
    assert not any(step.get("is_copied_context") for step in steps)
    assert sample.trajectory["agent"]["tool_definitions"][0]["function"]["name"] == "read_file"
    assert sample.metadata["reward"] is None


def test_a_scored_trajectory_carries_the_report_and_its_score() -> None:
    stage = ReportSamples(PipelineContext(scenario="math"))
    for turns in (1, 2):
        stage.push(conversation(f"call-{turns}", turns))
    stage.push(
        AgentRecord.create(
            agent_record_id="r1",
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 0.75, "feedback": "good", "references": ["call-2"]},
        )
    )
    (unit,) = stage.take_ready_units()
    ((_, (sample,)),) = stage.unit_samples(unit)
    assert sample.source_agent_record_ids == ("call-1", "call-2", "r1")
    assert sample.metadata["reward"] == 0.75 and sample.metadata["feedback"] == "good"


@pytest.mark.parametrize("multi_turn", [False, True])
def test_a_policy_report_trains_its_last_call_or_its_whole_trajectory(multi_turn: bool) -> None:
    policy_tensors = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    first_call = conversation("call-1", 1)
    first = replace(first_call, payload={**first_call.payload, **policy_tensors})
    recorded = conversation("call-3", 3)
    last = replace(recorded, payload={**recorded.payload, **policy_tensors, "artifact_id": "a1"})
    scored = AgentRecord.create(
        agent_record_id="r1", scenario="math", request_type=RequestType.REPORT, payload={"references": ["call-3"]}
    )
    assembly = SampleAssembly(SampleAssemblySettings(accept_multi_turn_policy_samples=multi_turn))
    sample = assembly.build(ReportContext(scored, 1.0, (first, last)), 1.0)
    assert isinstance(sample, TrajectorySample)
    agent_steps = [step for step in sample.trajectory["steps"] if step["source"] == "agent"]
    # The last call alone resends the first reply as context; the whole trajectory trains both replies.
    assert [step.get("is_copied_context", False) for step in agent_steps] == [not multi_turn, False]
    assert sample.source_agent_record_ids == (("call-1", "call-3") if multi_turn else ("call-3",))
    assert sample.metadata["reward"] == 1.0 and sample.metadata["report_agent_record_id"] == "r1"
    assert sample.artifact_id == "a1"
    assert "tokens" not in sample.training


@pytest.mark.parametrize(
    "mask_fields",
    (
        pytest.param({}, id="missing"),
        pytest.param({"loss_mask": None}, id="none"),
        pytest.param({"loss_mask": []}, id="empty"),
        pytest.param({"loss_mask": [0, 0]}, id="all-zero"),
    ),
)
def test_a_policy_report_without_trainable_tokens_is_skipped(mask_fields: dict[str, object]) -> None:
    record = conversation("call-1", 1)
    record = replace(
        record,
        payload={**record.payload, "tokens": [1, 2], "rollout_log_probs": [-0.2], **mask_fields},
    )
    scored = AgentRecord.create(
        agent_record_id="r1", scenario="math", request_type=RequestType.REPORT, payload={"references": ["call-1"]}
    )

    sample = SampleAssembly(SampleAssemblySettings()).build(ReportContext(scored, 1.0, (record,)), 1.0)

    assert isinstance(sample, SkippedReport)
    assert "call-1" in sample.reason


@pytest.mark.parametrize(
    ("loss_mask", "message"),
    (
        pytest.param(1, "must be a sequence", id="non-sequence"),
        pytest.param("invalid", "must be a sequence", id="string"),
        pytest.param([None], "only 0 or 1", id="invalid-value"),
        pytest.param([0, 2], "only 0 or 1", id="out-of-range"),
    ),
)
def test_a_policy_report_rejects_a_malformed_loss_mask(loss_mask: object, message: str) -> None:
    record = conversation("call-1", 1)
    record = replace(record, payload={**record.payload, "loss_mask": loss_mask})
    scored = AgentRecord.create(
        agent_record_id="r1", scenario="math", request_type=RequestType.REPORT, payload={"references": ["call-1"]}
    )

    with pytest.raises(ValueError, match=message):
        SampleAssembly(SampleAssemblySettings()).build(ReportContext(scored, 1.0, (record,)), 1.0)


def test_a_policy_report_of_several_trajectories_requires_multi_turn_samples() -> None:
    first, second = conversation("call-a", 1), conversation("call-b", 1)
    policy_tensors = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    first = replace(first, payload={**first.payload, **policy_tensors})
    second = replace(second, payload={**second.payload, **policy_tensors})
    scored = AgentRecord.create(
        agent_record_id="r1",
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"references": ["call-a", "call-b"]},
    )
    assembly = SampleAssembly(SampleAssemblySettings())
    with pytest.raises(ValueError, match="several trajectories"):
        assembly.build(ReportContext(scored, 1.0, (first, second)), 1.0)
