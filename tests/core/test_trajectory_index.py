"""The trajectory rule (RFC #1054): keys compare shape, tools and request prefixes; the index links as a store does."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import pytest
from tests.support.conversations import conversation, routed_call, turn

from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import make_trajectory
from reef.core.trajectory_index import (
    CallKey,
    TrajectoryIndex,
    call_key,
    conversations,
    sha256_text,
    shadow_request_id,
    trajectory_calls,
)
from reef.storage.record_items import ItemRecordStore, record_item
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.tasks import LocalTaskFileStore

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
    assert anthropic is not None and anthropic.request == (
        [{"source": "system", "message": "Be precise"}],
        [{"source": "user", "message": "海 🐟"}],
    ), "each request value is in its ATIF form"
    assert call_key({"prompt": "hi"}) is None
    empty = call_key({"messages": []})
    assert empty is not None and empty.list_sha256 == empty.trajectory_sha256


def test_calls_link_when_a_client_spells_a_resent_message_another_way() -> None:
    """Claude Code sends the last message as text parts with cache_control, and the same message as a string when it
    resends it (RFC #1300)."""
    cached = {"type": "ephemeral"}
    task = {"role": "user", "content": [{"type": "text", "text": "Judge the run.", "cache_control": cached}]}
    reply = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/a"}}],
    }
    result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "READY", "cache_control": cached}],
    }
    first = {"system": "Judge.", "messages": [task], "response": {"type": "message", **reply}}
    resent = {"role": "user", "content": "Judge the run."}
    answer = {"role": "assistant", "content": [{"type": "text", "text": "yes"}]}
    second = {"system": "Judge.", "messages": [resent, reply, result], "response": {"type": "message", **answer}}
    records = [
        AgentRecord(name, "s", RequestType.INFERENCE, payload, float(index))
        for index, (name, payload) in enumerate([("first", first), ("second", second)])
    ]
    first_key, second_key = call_key(first), call_key(second)
    assert first_key is not None and second_key is not None
    assert second_key.prefix_sha256[:2] == first_key.prefix_sha256
    assert second_key.continues(first_key.entry_count, first_key.reply_count, first_key.reply_sha256)
    (linked,) = conversations(records)
    assert [record.agent_record_id for record in linked] == ["first", "second"]
    steps = make_trajectory(linked).trajectory["steps"]
    assert [step["source"] for step in steps] == ["system", "user", "agent", "agent"], "no resent step repeats"


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


def stored_trajectory(data: SQLiteDataStore, record_id: str) -> tuple[str, ...]:
    """The record IDs of the trajectory that the record item of ``record_id`` ends, from the data store's links."""
    path: list[tuple[str, int]] = []
    item_id: str | None = f"math/{record_id}"
    while item_id is not None:
        link = data.record_links("math", [item_id])[item_id]
        path.append((item_id.removeprefix("math/"), link.entry_count))
        item_id = link.parent
    return trajectory_calls(path[::-1])


def test_the_index_links_the_calls_as_the_data_store_does(tmp_path) -> None:
    index = TrajectoryIndex()
    with closing(LocalTaskFileStore(None)) as task_files, closing(SQLiteDataStore(None, task_files)) as data:
        for call in calls():
            index.add(call)
            data.create("math", [record_item(call)])
        for call in calls():
            if call.agent_record_id == "plain":
                continue
            assert index.trajectory(call.agent_record_id) == stored_trajectory(data, call.agent_record_id)
    assert index.trajectory("later") == ("retry", "later")
    assert index.trajectory("call-3") == ("call-1", "call-2", "call-3")
    assert index.trajectory("plain") == ("plain",)
    assert index.trajectory("missing") == ()
    assert index.continued("call-2") and index.continued("retry")
    assert not index.continued("call-3") and not index.continued("first")


@pytest.mark.parametrize("shadow_first", [False, True])
def test_a_shadow_call_with_the_primarys_messages_has_a_trajectory_of_its_own(tmp_path, shadow_first: bool) -> None:
    primary = routed_call(conversation("primary", 1), "serve", "primary-request")
    shadow = routed_call(conversation("shadow", 1), "shadow", "shadow-request", reply=9)
    following = routed_call(conversation("next", 3), "serve", "next-request")
    # ATIF normalizes the re-sent text parts, including cache metadata, without removing the shadow boundary.
    for record in (shadow, following):
        record.payload["messages"] = [
            {
                **message,
                "content": [{"type": "text", "text": message["content"], "cache_control": {"type": "ephemeral"}}],
            }
            for message in record.payload["messages"]
        ]
    primary_key, shadow_key = call_key(primary.payload), call_key(shadow.payload)
    assert primary_key is not None and shadow_key is not None
    # Primary calls retain the upstream key, while a shadow gets a separate namespace.
    assert primary_key == call_key(conversation("plain", 1).payload)
    assert shadow_key.request == primary_key.request
    assert shadow_request_id(shadow.payload) == "shadow-request" and shadow_request_id(primary.payload) is None
    assert set(shadow_key.prefix_sha256).isdisjoint(primary_key.prefix_sha256)
    index = TrajectoryIndex()
    expected = {"primary": ("primary",), "shadow": ("shadow",), "next": ("primary", "next")}
    with closing(LocalTaskFileStore(tmp_path / "task-files")) as task_files:
        with closing(SQLiteDataStore(tmp_path / "data", task_files)) as data:
            records = ItemRecordStore(data, "acme", "math", data.scenario_generation("acme", "math"))
            for record in (shadow, primary) if shadow_first else (primary, shadow):
                index.add(record)
                records.append(record)
            # Neither call retries the other, whichever finished first.
            assert index.is_end("primary") and index.is_end("shadow")
            index.add(following)
            records.append(following)
            for record_id, trajectory in expected.items():
                assert index.trajectory(record_id) == records.trajectory("math", record_id) == trajectory
            assert index.is_end("shadow") and not index.is_end("primary")
            stored = records.replay("math")
        # Reopening uses the actual item-backed record store and its persisted trajectory links.
        with closing(SQLiteDataStore(tmp_path / "data", task_files)) as reopened:
            records = ItemRecordStore(reopened, "acme", "math", reopened.scenario_generation("acme", "math"))
            assert records.replay("math") == stored
            for record_id, trajectory in expected.items():
                assert records.trajectory("math", record_id) == trajectory
    assert {record.agent_record_id: record.payload for record in stored}["shadow"] == shadow.payload
    rebuilt = TrajectoryIndex()
    for record in stored:
        rebuilt.add(record)
    assert {record_id: rebuilt.trajectory(record_id) for record_id in expected} == expected


def test_forgetting_a_call_keeps_it_while_a_later_call_links_to_it() -> None:
    index = TrajectoryIndex()
    for call in calls():
        index.add(call)
    index.forget(["call-1", "call-3", "plain"])
    assert index.trajectory("call-2") == ("call-1", "call-2")
    assert index.trajectory("call-3") == ()
    assert index.trajectory("plain") == ()
