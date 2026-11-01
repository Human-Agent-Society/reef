"""The trajectory rule (RFC #1054): keys compare shape, tools and request prefixes; the index links as a store does."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.conversations import conversation, turn

from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectory_index import CallKey, TrajectoryIndex, call_key, sha256_text, trajectory_calls
from reef.storage.sqlite import SQLiteRecordStore

SYSTEM = {"role": "system", "content": "be brief"}
USER = {"role": "user", "content": "海 🐟"}
TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {}}}]


def test_prefix_hashes_are_equal_exactly_as_long_as_the_shape_the_tools_and_the_messages_are() -> None:
    first = call_key({"messages": [SYSTEM, USER], "tools": TOOLS})
    second = call_key({"messages": [SYSTEM, USER, SYSTEM], "tools": TOOLS, "temperature": 1})
    others = [
        call_key({"messages": [USER, USER], "tools": TOOLS}),
        call_key({"messages": [SYSTEM, USER], "tools": [{"type": "function", "function": {"name": "u"}}]}),
        call_key({"messages": [SYSTEM, USER]}),
        call_key({"input": [SYSTEM, USER], "tools": TOOLS}),
    ]
    assert first is not None and second is not None
    assert second.prefix_sha256[:2] == first.prefix_sha256
    assert second.prefix_sha256[2] != first.prefix_sha256[1]
    for other in others:
        assert other is not None and set(other.prefix_sha256).isdisjoint(first.prefix_sha256)
    anthropic = call_key({"system": "Be precise", "messages": [USER]})
    assert anthropic is not None and anthropic.request == ("Be precise", USER)
    assert call_key({"prompt": "hi"}) is None
    empty = call_key({"messages": []})
    assert empty is not None and empty.list_sha256 == empty.trajectory_sha256


def test_a_key_checks_its_hashes() -> None:
    key = call_key({"messages": [SYSTEM, USER]})
    assert key is not None
    with pytest.raises(ValueError, match="chained SHA-256"):
        replace(key, prefix_sha256=tuple(reversed(key.prefix_sha256)))
    with pytest.raises(ValueError, match="chained SHA-256"):
        replace(key, trajectory_sha256=sha256_text("other"))
    with pytest.raises(ValueError, match="one prefix hash"):
        replace(key, request=(USER,))
    with pytest.raises(ValueError, match="a hash and at least one message"):
        CallKey(key.trajectory_sha256, key.request, key.prefix_sha256, 1, None)
    with pytest.raises(TypeError, match="JSON-serializable"):
        call_key({"messages": [object()]})


def test_a_retry_takes_the_place_of_the_call_with_the_same_request() -> None:
    assert trajectory_calls([("a", 2), ("b", 3), ("b-retry", 3), ("c", 5)]) == ("a", "b-retry", "c")
    assert trajectory_calls([("a", 2)]) == ("a",)
    assert trajectory_calls([]) == ()


def calls() -> list[AgentRecord]:
    """Calls of two conversations, with a retry, a continued retry, a tool change and records without messages."""
    first = conversation("first", 1)
    retry = replace(first, agent_record_id="retry", payload={**first.payload, "response": {"output": [turn(9)]}})
    later = conversation("later", 1)
    later.payload["messages"] += [turn(9), turn(2)]
    plain = AgentRecord.create(
        agent_record_id="plain", scenario="math", request_type=RequestType.INFERENCE, payload={"prompt": "hi"}
    )
    return [
        *(conversation(f"call-{turns}", turns, filler="!") for turns in (1, 2, 3)),
        first,
        retry,
        later,
        conversation("retooled", 2, tools=("bash",)),
        plain,
    ]


def test_the_index_links_the_calls_as_the_record_store_does(tmp_path) -> None:
    index = TrajectoryIndex()
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        for call in calls():
            index.add(call)
            records.append(call)
        for call in calls():
            if call.agent_record_id == "plain":
                continue
            assert index.trajectory(call.agent_record_id) == records.trajectory("math", call.agent_record_id)
    assert index.trajectory("later") == ("retry", "later")
    assert index.trajectory("call-3") == ("call-1", "call-2", "call-3")
    assert index.trajectory("plain") == ("plain",)
    assert index.trajectory("missing") == ()
    assert index.continued("call-2") and index.continued("retry")
    assert not index.continued("call-3") and not index.continued("first")


def test_forgetting_a_call_keeps_it_while_a_later_call_links_to_it() -> None:
    index = TrajectoryIndex()
    for call in calls():
        index.add(call)
    index.forget(["call-1", "call-3", "plain"])
    assert index.trajectory("call-2") == ("call-1", "call-2")
    assert index.trajectory("call-3") == ()
    assert index.trajectory("plain") == ()
