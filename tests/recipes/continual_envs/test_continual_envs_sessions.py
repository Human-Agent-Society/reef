"""The traffic sessions of a round's records, and the record folders that a generation run reads from them."""

from __future__ import annotations

import json

import pytest
from tests.recipes.continual_envs.support import call, report, session

from recipes.beta.continual_envs.sessions import TrafficSession, record_files, traffic_sessions


def test_each_report_forms_a_session_with_the_calls_up_to_the_one_it_references() -> None:
    extra = call("after-the-end", [])
    records = (*session("a", 0.0), *session("b", 1.0, turns=3)[:-1], extra, session("b", 1.0, turns=3)[-1])
    first, second = traffic_sessions(records)
    assert first.record_ids == ("a-call-0", "a-call-1", "a-report") and first.score == 0.0
    # A call after the referenced one belongs to no session.
    assert second.record_ids == ("b-call-0", "b-call-1", "b-call-2", "b-report") and second.score == 1.0
    # A report that references no call of its group forms no session.
    assert traffic_sessions((call("x", []), report("r", ("elsewhere",), 0.0))) == ()


def test_a_session_file_holds_the_last_call_and_the_score_and_no_report_metadata() -> None:
    (traffic,) = traffic_sessions(session("a", 0.25))
    document = traffic.document()
    assert document == {
        "record_id": "a-report",
        "score": 0.25,
        "feedback": "the agent stopped early",
        "calls": 2,
        "request": {"model": "policy", "messages": [{"role": "user", "content": "task a"}] * 2},
        "response": {"choices": [{"message": {"role": "assistant", "content": "reply of a-call-1"}}]},
    }
    assert "hidden" not in json.dumps(document)


def test_the_sessions_fit_in_order_within_the_byte_limit() -> None:
    sessions = traffic_sessions((*session("a", 0.0, turns=1), *session("b", 0.0, turns=40), *session("c", 0.0)))
    size = len(json.dumps(sessions[0].document(), ensure_ascii=False).encode())
    files, left_out = record_files(sessions, max_bytes=3 * size)
    assert sorted(files) == ["a-report/session.json", "c-report/session.json"] and left_out == ("b-report",)
    assert json.loads(files["c-report/session.json"])["record_id"] == "c-report"


def test_a_session_must_end_with_the_call_that_its_report_references() -> None:
    calls = session("a", 0.0)[:2]
    with pytest.raises(ValueError, match="reference its last call"):
        TrafficSession(report("r", ("a-call-0",), 0.0), calls)
    with pytest.raises(ValueError, match="at least one inference record"):
        TrafficSession(report("r", ("a-call-1",), 0.0), ())
