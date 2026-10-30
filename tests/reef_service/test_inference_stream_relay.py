"""The inference stream relay: what a Codex client reads when OpenRouter searches the web.

OpenRouter streams its web search server tool as an output item of type
``openrouter:web_search``; Codex only parses ``web_search_call`` and drops an
item of an unknown type, so the search never reaches the TUI, the session
rollout, or the next turn's replay. These tests pin the relay's contract: a
codex-bound Responses stream carries ``web_search_call`` items on the three
event points Codex reads, any other deployment's stream passes through
untouched, and Reef's own record keeps the raw upstream item.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact.artifact import Artifact
from reef.core import RequestType
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.runtime.interfaces import FixedEngineClients, InferenceHandler, InferenceStream
from reef.service.inference_service import InferenceReplica, ServedModelsCache, add_inference_routes
from reef.service.streaming import frame_for_codex
from reef.storage.control_lease import MemoryControlLease
from reef.storage.served_models import MemoryServedModels
from reef.storage.sqlite import SQLiteScenarioStorage

#: One OpenRouter web search item, in the shape the OpenRouter SDK documents:
#: the action already matches OpenAI's ``web_search_call.action``.
WEB_SEARCH_ITEM = {
    "id": "ws_1",
    "type": "openrouter:web_search",
    "status": "completed",
    "action": {
        "type": "search",
        "query": "reef openrouter codex",
        "sources": [{"type": "url", "url": "https://example.com/source"}],
    },
}

MESSAGE_ITEM = {
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "The answer is 7."}],
}


def _frame(payload: dict[str, Any]) -> str:
    """One SSE frame for the payload, in the shape OpenRouter sends."""
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"


#: One Responses turn that searches the web once and then answers.
OPENROUTER_SSE = "".join(
    _frame(payload)
    for payload in (
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**WEB_SEARCH_ITEM, "status": "in_progress", "action": None},
        },
        {"type": "response.output_text.delta", "delta": "The answer"},
        {"type": "response.output_item.done", "output_index": 0, "item": WEB_SEARCH_ITEM},
        {"type": "response.output_item.done", "output_index": 1, "item": MESSAGE_ITEM},
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "output": [WEB_SEARCH_ITEM, MESSAGE_ITEM]},
        },
    )
)


def _frames(body: str) -> list[dict[str, Any]]:
    """The JSON payload of each SSE frame in the body, in order."""
    parsed: list[dict[str, Any]] = []
    for frame in body.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                value = json.loads(line[len("data: ") :])
                assert isinstance(value, dict)
                parsed.append(value)
    return parsed


def test_frame_for_codex_renames_the_search_item_on_all_three_event_points() -> None:
    added = frame_for_codex(
        _frame({"type": "response.output_item.added", "output_index": 0, "item": WEB_SEARCH_ITEM}).encode()
    )
    done = frame_for_codex(
        _frame({"type": "response.output_item.done", "output_index": 0, "item": WEB_SEARCH_ITEM}).encode()
    )
    completed = frame_for_codex(
        _frame(
            {"type": "response.completed", "response": {"id": "resp_1", "output": [WEB_SEARCH_ITEM, MESSAGE_ITEM]}}
        ).encode()
    )

    # Only the item type changes; the action, the id, and the status pass through.
    added_payload = json.loads(added.decode("utf-8").split("data: ", 1)[1])
    assert added_payload["item"] == {**WEB_SEARCH_ITEM, "type": "web_search_call"}
    assert json.loads(done.decode("utf-8").split("data: ", 1)[1])["item"] == {
        **WEB_SEARCH_ITEM,
        "type": "web_search_call",
    }
    completed_payload = json.loads(completed.decode("utf-8").split("data: ", 1)[1])
    assert completed_payload["response"]["output"] == [{**WEB_SEARCH_ITEM, "type": "web_search_call"}, MESSAGE_ITEM]

    # The event name and the frame's shape survive, so receipts keep working.
    for frame in (added, done, completed):
        text = frame.decode("utf-8")
        assert "openrouter:web_search" not in text
        assert text.count("data: ") == 1
        assert text.endswith("\n\n")
        assert text.startswith("event: ")


def test_frame_for_codex_leaves_every_other_frame_untouched() -> None:
    delta = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    terminal = b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r1"}}\n\n'
    message = b'data: {"type":"response.output_item.done","item":{"type":"message","content":[]}}\n\n'
    broken = b"data: {broken json openrouter:web_search}\n\n"
    assert frame_for_codex(delta) == delta
    assert frame_for_codex(terminal) == terminal
    assert frame_for_codex(message) == message
    assert frame_for_codex(broken) == broken


class ScriptedStreamHandler(InferenceHandler):
    """An engine handler that answers every stream with ``OPENROUTER_SSE``."""

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the relay tests only stream")

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        async def chunks() -> AsyncIterator[bytes]:
            for frame in OPENROUTER_SSE.split("\n\n"):
                if frame:
                    yield f"{frame}\n\n".encode()

        return InferenceStream(status=200, headers={"content-type": "text/event-stream"}, chunks=chunks())


def _relay_client(tmp_path: Path, *, for_codex: bool) -> tuple[TestClient, SQLiteScenarioStorage]:
    storage = SQLiteScenarioStorage(tmp_path / "records")
    table = MemoryServedModels(MemoryControlLease())
    number = table.control_lease.acquire("control-1").number
    table.write(
        ScenarioModel("reef-codex", ACTIVE),
        ServedArtifact("release-1", "http://engine.invalid", "anthropic/claude"),
        control_lease_number=number,
    )
    replica = InferenceReplica(
        ServedModelsCache(table), storage, FixedEngineClients(ScriptedStreamHandler()), for_codex=for_codex
    )
    app = web.Application()
    add_inference_routes(app, replica)
    return TestClient(TestServer(app)), storage


def recorded_responses(storage: SQLiteScenarioStorage) -> list[Mapping[str, Any]]:
    store = storage.open("reef-codex")
    try:
        return [
            item.payload["response"]
            for item in store.records.replay("reef-codex")
            if item.request_type is RequestType.INFERENCE
        ]
    finally:
        store.close()


def _post_codex_stream(client: TestClient) -> Any:
    return client.post(
        "/v1/responses",
        json={"stream": True, "model": "anthropic/claude", "input": []},
        headers={"x-reef-scenario": "reef-codex"},
    )


@pytest.mark.unit
def test_codex_deployment_reads_web_search_call_and_reef_keeps_the_raw_item(tmp_path: Path) -> None:
    async def run() -> None:
        client, storage = _relay_client(tmp_path, for_codex=True)
        async with client:
            response = await _post_codex_stream(client)
            assert response.status == 200
            body = await response.text()

        payloads = _frames(body)
        # No frame of the client stream still names OpenRouter's item type.
        assert "openrouter:web_search" not in body
        # The search item reads as web_search_call on each of the three event points Codex reads.
        added = [payload["item"] for payload in payloads if payload.get("type") == "response.output_item.added"]
        done = [payload["item"] for payload in payloads if payload.get("type") == "response.output_item.done"]
        assert [item["type"] for item in added + done] == ["web_search_call", "web_search_call", "message"]
        assert done[0] == {**WEB_SEARCH_ITEM, "type": "web_search_call"}
        completed = next(payload for payload in payloads if payload.get("type") == "response.completed")
        assert completed["response"]["output"][0] == {**WEB_SEARCH_ITEM, "type": "web_search_call"}
        # The receipt rides the terminal event as before.
        assert completed["reef"]["agent_record_id"]

        # Reef's own record keeps the raw upstream item, never the rewrite.
        (recorded,) = recorded_responses(storage)
        assert "openrouter:web_search" in recorded["body"]
        assert "web_search_call" not in recorded["body"]

    asyncio.run(run())


@pytest.mark.unit
def test_non_codex_deployment_reads_the_raw_item(tmp_path: Path) -> None:
    async def run() -> None:
        client, storage = _relay_client(tmp_path, for_codex=False)
        async with client:
            response = await _post_codex_stream(client)
            assert response.status == 200
            body = await response.text()

        assert "web_search_call" not in body
        completed = next(payload for payload in _frames(body) if payload.get("type") == "response.completed")
        assert completed["response"]["output"][0] == WEB_SEARCH_ITEM
        # The receipt rides the terminal event here too.
        assert completed["reef"]["agent_record_id"]
        # Reef's record and the client both see the raw stream.
        (recorded,) = recorded_responses(storage)
        assert "openrouter:web_search" in recorded["body"]

    asyncio.run(run())
