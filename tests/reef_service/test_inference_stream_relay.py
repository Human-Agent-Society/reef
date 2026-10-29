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
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.core import AgentRecord, RequestType
from reef.runtime.interfaces import InferenceStream
from reef.service.routes.inference import register_inference_routes
from reef.service.routing import RoutingDecision
from reef.service.streaming import frame_for_codex

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


class _RecordingRequestService:
    """The parts of ``RequestService`` the inference relay touches, over a scripted upstream stream.

    ``dispatcher`` exposes only ``recipe.harness_adapter``, and ``routing`` is None, so no request is
    routed. ``start_stream`` appends ``(record, evaluated, decision, artifact_id)`` to ``started`` and returns a new
    ``upstream`` stream of the ``OPENROUTER_SSE`` frames, with a pending value that holds ``record`` and a
    new ``reef-codex`` inference record. ``record_stream`` appends the relayed response to ``recorded`` and
    returns that record, ``persist`` awaits its write, and ``harness_head`` returns None. It writes no
    storage and holds no connection.
    """

    def __init__(self, harness_adapter: str | None) -> None:
        self.recorded: list[Mapping[str, Any]] = []
        #: What each stream was asked for: recording, evaluated component, routing decision, and served artifact.
        self.started: list[tuple[bool, str | None, RoutingDecision | None, str | None]] = []
        self.dispatcher = SimpleNamespace(recipe=SimpleNamespace(harness_adapter=harness_adapter))
        self.routing = None

    async def persist(self, write: Awaitable[AgentRecord]) -> AgentRecord:
        return await write

    async def upstream(self) -> InferenceStream:
        async def chunks() -> AsyncIterator[bytes]:
            for frame in OPENROUTER_SSE.split("\n\n"):
                if frame:
                    yield f"{frame}\n\n".encode()

        return InferenceStream(status=200, headers={"content-type": "text/event-stream"}, chunks=chunks())

    async def start_stream(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        handler: object = None,
        *,
        record: bool = True,
        evaluated: str | None = None,
        decision: RoutingDecision | None = None,
        artifact_id: str | None = None,
    ) -> tuple[InferenceStream, Any]:
        self.started.append((record, evaluated, decision, artifact_id))
        pending = SimpleNamespace(
            item=AgentRecord.create(scenario="reef-codex", request_type=RequestType.INFERENCE, payload=dict(payload)),
            record=record,
        )
        return await self.upstream(), pending

    def harness_head(self, headers: Mapping[str, str]) -> None:
        return None

    def record_stream(self, pending: Any, response: Mapping[str, Any]) -> AgentRecord:
        self.recorded.append(dict(response))
        return pending.item


class HeaderLookupService(_RecordingRequestService):
    """A resident ``pi`` harness whose release-header lookup blocks, or fails, after the stream is open.

    ``upstream`` streams like the base service's but counts each ``close`` of its stream in ``closes``.
    ``harness_head`` sets ``entered``, then raises ``failure`` if one was given, or else waits up to five
    seconds for the test to set ``release``. Both events are ``threading.Event`` values.
    """

    def __init__(self, failure: Exception | None) -> None:
        super().__init__("pi")
        self.failure = failure
        self.closes = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    async def upstream(self) -> InferenceStream:
        stream = await super().upstream()

        async def close() -> None:
            self.closes += 1

        return InferenceStream(status=stream.status, headers=stream.headers, chunks=stream.chunks, close=close)

    def harness_head(self, headers: Mapping[str, str]) -> None:
        self.entered.set()
        if self.failure is not None:
            raise self.failure
        self.release.wait(5)


def _relay_client(harness_adapter: str | None) -> tuple[TestClient, _RecordingRequestService]:
    request_service = _RecordingRequestService(harness_adapter)
    app = web.Application()
    register_inference_routes(app, request_service=request_service, inference_handler=None)
    return TestClient(TestServer(app)), request_service


def _post_codex_stream(client: TestClient) -> Any:
    return client.post(
        "/v1/responses",
        json={"stream": True, "model": "anthropic/claude", "input": []},
        headers={"x-reef-scenario": "reef-codex"},
    )


@pytest.mark.unit
def test_codex_deployment_reads_web_search_call_and_reef_keeps_the_raw_item() -> None:
    async def run() -> None:
        client, request_service = _relay_client("codex")
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
        recorded = request_service.recorded[0]
        assert "openrouter:web_search" in recorded["body"]
        assert "web_search_call" not in recorded["body"]

    asyncio.run(run())


@pytest.mark.unit
def test_non_codex_deployment_reads_the_raw_item() -> None:
    async def run() -> None:
        client, request_service = _relay_client("pi")
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
        assert "openrouter:web_search" in request_service.recorded[0]["body"]

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    [None, RuntimeError("head lookup failed"), ConnectionResetError("provider connection reset")],
    ids=["cancelled", "failed", "reset"],
)
def test_an_open_stream_whose_header_lookup_ends_early_is_closed_and_recorded_once(failure: Exception | None) -> None:
    async def run() -> None:
        request_service = HeaderLookupService(failure)
        handler_failures: list[Exception] = []

        @web.middleware
        async def keep_failure(
            request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
        ) -> web.StreamResponse:
            try:
                return await handler(request)
            except Exception as exc:
                handler_failures.append(exc)
                raise

        app = web.Application(middlewares=[keep_failure])
        register_inference_routes(app, request_service=request_service, inference_handler=None)
        async with TestClient(TestServer(app)) as client:

            async def send() -> int:
                return (await _post_codex_stream(client)).status

            sent = asyncio.create_task(send())
            try:
                if failure is None:
                    assert await asyncio.to_thread(request_service.entered.wait, 5)
                    # Dropping the connection cancels Reef's handler while it waits for the release header.
                    sent.cancel()
                else:
                    assert await sent == 500
                async with asyncio.timeout(5):
                    while not request_service.recorded:
                        await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
            finally:
                request_service.release.set()
                sent.cancel()
                await asyncio.gather(sent, return_exceptions=True)

        assert request_service.closes == 1
        [recorded] = request_service.recorded
        assert recorded["complete"] is False
        if failure is None:
            assert handler_failures == [] and "error" not in recorded
        else:
            # The handler fails with the lookup's own exception, not one raised by the relay.
            assert handler_failures == [failure]
            assert recorded["error"] == f"{type(failure).__name__}: {failure}"

    asyncio.run(run())


@pytest.mark.unit
def test_evaluation_stream_names_its_component_and_carries_no_receipt() -> None:
    async def run() -> None:
        client, request_service = _relay_client("pi")
        async with client:
            response = await client.post(
                "/reef/scenarios/reef-codex/components/harness/evaluation/v1/responses",
                json={"stream": True, "model": "anthropic/claude", "input": []},
            )
            assert response.status == 200
            assert "x-reef-agent-record-id" not in response.headers
            body = await response.text()

        # The stream opens without a record or a routing decision, for the evaluated component only.
        assert request_service.started == [(False, "harness", None, None)]
        completed = next(payload for payload in _frames(body) if payload.get("type") == "response.completed")
        assert "reef" not in completed

    asyncio.run(run())


@pytest.mark.unit
def test_artifact_stream_names_its_artifact_without_reading_the_active_release() -> None:
    async def run() -> None:
        request_service = HeaderLookupService(ValueError("an artifact URL does not read the active release"))
        app = web.Application()
        register_inference_routes(app, request_service=request_service, inference_handler=None)
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/reef/scenarios/reef-codex/artifacts/older-release/v1/responses",
                json={"stream": True, "model": "anthropic/claude", "input": []},
            )
            assert response.status == 200
            assert "x-reef-release-id" not in response.headers
            body = await response.text()

        assert request_service.started == [(True, None, None, "older-release")]
        assert not request_service.entered.is_set()
        completed = next(payload for payload in _frames(body) if payload.get("type") == "response.completed")
        assert completed["reef"]["agent_record_id"]
        assert len(request_service.recorded) == 1 and request_service.recorded[0]["complete"] is True
        assert request_service.closes == 1

    asyncio.run(run())
