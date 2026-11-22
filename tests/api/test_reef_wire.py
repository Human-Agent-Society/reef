from __future__ import annotations

import pytest

from reef.api.wire import (
    HeaderError,
    ReportPayload,
    RequestHeaders,
    contract_with_older_keys,
    parse_request_headers,
    status_with_older_keys,
)
from reef.core import AgentRecord, RequestType


@pytest.mark.unit
def test_parse_request_headers_requires_scenario() -> None:
    with pytest.raises(HeaderError, match="x-reef-scenario"):
        parse_request_headers({}, RequestType.INFERENCE)


@pytest.mark.unit
def test_parse_request_headers_is_case_insensitive_and_preserves_scenario() -> None:
    parsed = parse_request_headers(
        {
            "X-reef-Scenario": "code-repair",
            "X-reef-Release-Id": "checkpoint-v42",
        },
        RequestType.REPORT,
    )

    # A request's x-reef-release-id names nothing that Reef reads: only an inference answer carries it.
    assert parsed == RequestHeaders(scenario="code-repair", request_type=RequestType.REPORT)


@pytest.mark.unit
def test_agent_record_preserves_native_inference_payload() -> None:
    payload = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}

    item = AgentRecord.create(
        scenario="chat",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id="req-1",
        created_at=12.5,
    )

    assert item.payload == payload
    assert "scenario" not in item.payload
    assert "type" not in item.payload


@pytest.mark.unit
def test_report_payload_round_trips_references() -> None:
    report = ReportPayload.from_dict({"score": 0.75, "feedback": "good", "references": ["req-1"]})

    assert report.to_dict() == {
        "score": 0.75,
        "feedback": "good",
        "references": ["req-1"],
    }


@pytest.mark.unit
def test_payload_references_must_be_strings() -> None:
    with pytest.raises(ValueError, match="references"):
        ReportPayload.from_dict({"score": 1.0, "references": [3]})


@pytest.mark.unit
def test_report_without_score_is_valid_and_omits_score_from_dict() -> None:
    report = ReportPayload.from_dict({"feedback": "no numeric grade here", "references": ["req-1"]})

    assert report.score is None
    assert report.to_dict() == {
        "feedback": "no numeric grade here",
        "references": ["req-1"],
    }


@pytest.mark.unit
def test_report_feedback_may_be_a_structured_object_and_round_trips() -> None:
    feedback = {"rubric": {"correctness": 1, "style": 0.5}}

    report = ReportPayload.from_dict({"feedback": feedback, "references": ["req-1"]})

    assert report.feedback == feedback
    assert report.to_dict() == {
        "feedback": feedback,
        "references": ["req-1"],
    }


@pytest.mark.unit
def test_report_score_rejects_non_numeric_value() -> None:
    with pytest.raises(ValueError, match="score must be a number"):
        ReportPayload.from_dict({"score": "high", "feedback": "text"})


@pytest.mark.unit
def test_report_feedback_rejects_invalid_types() -> None:
    with pytest.raises(ValueError, match="feedback must be a string or an object"):
        ReportPayload.from_dict({"feedback": ["not", "valid"]})

    with pytest.raises(ValueError, match="feedback must be a string or an object"):
        ReportPayload.from_dict({"feedback": 42})


@pytest.mark.unit
def test_the_status_answer_also_holds_each_pipeline_object_under_its_older_key() -> None:
    status = {
        "error": None,
        "scenarios": {
            "one": {"pipeline": {"ready_samples": 2}, "components": {"weights": {"pipeline": {"ready_samples": 1}}}},
            "none": {"job_data_ready": False},
        },
    }
    answer = status_with_older_keys(status)
    block = answer["scenarios"]["one"]
    assert block["pipeline"] == block["processor"] == {"ready_samples": 2}
    component = block["components"]["weights"]
    assert component["pipeline"] == component["processor"] == {"ready_samples": 1}
    assert answer["scenarios"]["none"] == {"job_data_ready": False}
    assert answer["error"] is None
    # The value that control built holds the one key only.
    assert "processor" not in status["scenarios"]["one"]
    assert "processor" not in status["scenarios"]["one"]["components"]["weights"]


@pytest.mark.unit
def test_the_contract_answer_also_holds_the_stage_name_under_its_older_key() -> None:
    contract = {"scenario": "one", "sample2job": "NoJobData"}
    assert contract_with_older_keys(contract) == {
        "scenario": "one",
        "sample2job": "NoJobData",
        "processor": "NoJobData",
    }
    assert "processor" not in contract
