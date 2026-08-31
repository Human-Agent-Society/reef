from __future__ import annotations

import pytest

from reef.core import AgentRecord, RequestType
from reef.service.wire import HeaderError, ReportPayload, parse_request_headers


@pytest.mark.unit
def test_parse_request_headers_requires_scenario() -> None:
    with pytest.raises(HeaderError, match="x-reef-scenario"):
        parse_request_headers({}, RequestType.INFERENCE)


@pytest.mark.unit
def test_parse_request_headers_is_case_insensitive_and_preserves_scenario() -> None:
    parsed = parse_request_headers(
        {
            "X-reef-Scenario": "code-repair",
            "X-reef-Artifact-Version": "checkpoint-v42",
        },
        RequestType.REPORT,
    )

    assert parsed.scenario == "code-repair"
    assert parsed.artifact_version == "checkpoint-v42"
    assert parsed.request_type is RequestType.REPORT


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
