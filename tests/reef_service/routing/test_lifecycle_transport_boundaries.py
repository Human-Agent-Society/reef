"""A managed request stays admitted until its response is sent, even after its record is durable.

A local provider returns one large JSON or SSE response, and a raw-socket client reads only the response
headers, so TCP backpressure holds the rest of the response inside Reef. The fake router answers only Reef's
readiness previews: the request pins its target. No Docker or Semantic Router process runs.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
from pathlib import Path

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from tests.reef_service.routing.routing_fakes import (
    OPEN_DEADLINE,
    ROUTER_HASH,
    SELECTION_TOKEN,
    provider_models,
    routing_adapter,
    routing_dispatcher,
    routing_headers,
    running_router,
)

from reef.dispatcher import Dispatcher
from reef.service.app import CONTROL_APPLICATION, create_app
from reef.service.lifecycle import LifecycleSettings
from reef.service.streaming import SSEFrameDecoder, is_terminal_sse_event, receipt_sse_events

pytestmark = pytest.mark.unit
CHAT_PATH = "/v1/chat/completions"
LARGE_PROMPT = "transport-large-response"
CONTROL_HEADERS = {"Authorization": "Bearer control"}


def terminal_frame(padding_bytes: int) -> bytes:
    """One legal terminal event: an SSE comment line does not change its ``data: [DONE]``."""
    return b":" + b"p" * padding_bytes + b"\ndata: [DONE]\n\n"


class LargeProvider:
    """A provider whose answer to ``LARGE_PROMPT`` is ``size_bytes`` long; it counts every completion."""

    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.completions = 0

    async def completion(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.completions += 1
        large = payload.get("messages") == [{"role": "user", "content": LARGE_PROMPT}]
        size = self.size_bytes if large else 8
        identity = {"id": "chatcmpl-transport", "created": 1, "model": payload.get("model")}
        if payload.get("stream") is not True:
            choice = {"index": 0, "message": {"role": "assistant", "content": "x" * size}, "finish_reason": "stop"}
            return web.json_response({**identity, "object": "chat.completion", "choices": [choice]})
        first = {**identity, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "a"}}]}
        body = b"data: " + json.dumps(first).encode() + b"\n\n" + terminal_frame(size)
        return web.Response(body=body, content_type="text/event-stream")

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/models", provider_models)
        app.router.add_post(CHAT_PATH, self.completion)
        return app


class PausedClient:
    """One keep-alive connection that reads only response headers until ``read`` is called."""

    def __init__(self, port: int) -> None:
        self.socket = socket.socket()
        # A small receive window fills quickly; the rest of the response then waits in Reef's send.
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self.socket.settimeout(40)
        self.socket.connect(("127.0.0.1", port))
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
        self.connection.sock = self.socket

    def send(self, content: str, stream: bool) -> http.client.HTTPResponse:
        body = json.dumps({"model": "student", "stream": stream, "messages": [{"role": "user", "content": content}]})
        headers = {**routing_headers(), "Content-Type": "application/json"}
        self.connection.request("POST", CHAT_PATH, body=body, headers=headers)
        return self.connection.getresponse()

    def read(self, response: http.client.HTTPResponse) -> bytes:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        return response.read()

    def close(self) -> None:
        self.connection.close()
        self.socket.close()


def sse_receipt(body: bytes) -> str | None:
    for line in body.split(b"\n"):
        if line.startswith(b"data: {"):
            value = json.loads(line.removeprefix(b"data: "))
            if "reef" in value:
                return value["reef"]["agent_record_id"]
    return None


def test_a_large_terminal_sse_frame_decodes_whole_and_its_receipt_comes_first() -> None:
    size = 128 * 1024
    first = b'data: {"id":"chatcmpl-transport","choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
    body = first + terminal_frame(size)
    decoder = SSEFrameDecoder()
    frames: list[bytes] = []
    for start in range(0, len(body), 7001):
        frames.extend(decoder.feed(body[start : start + 7001]))
    trailing, remainder = decoder.finalize()
    frames.extend(trailing)
    assert not remainder
    assert frames == [first, terminal_frame(size)]
    assert not is_terminal_sse_event(CHAT_PATH, frames[0])
    assert is_terminal_sse_event(CHAT_PATH, frames[1])
    emitted = receipt_sse_events(CHAT_PATH, {}, frames[1], "stored-record")
    # The terminal event is sent unchanged and last, after the frame that carries the receipt.
    assert emitted[-1] == frames[1]
    assert json.loads(emitted[0].removeprefix(b"data: "))["reef"]["agent_record_id"] == "stored-record"


@pytest.mark.parametrize("stream", [False, True])
def test_a_large_response_keeps_its_admission_after_its_record_and_close_rejects_the_next(
    tmp_path: Path, stream: bool
) -> None:
    async def run() -> None:
        size = 16 * 1024 * 1024
        provider = LargeProvider(size)
        async with TestServer(provider.app()) as provider_server:
            adapter = routing_adapter(tmp_path, str(provider_server.make_url("")).rstrip("/"), timeout_s=40)
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    lifecycle = LifecycleSettings(
                        "deployment",
                        "generation",
                        "control",
                        adapter.snapshot.version,
                        selection_token=SELECTION_TOKEN,
                        router_config_hash=ROUTER_HASH,
                    )
                    app = create_app(
                        dispatcher,
                        tokens="gateway-test-token",
                        routing=adapter,
                        preview=router.settings,
                        lifecycle=lifecycle,
                    )
                    await serve_large_response(app, dispatcher, provider, stream)
                    # The pin never asks the router; it saw only the readiness previews.
                    assert router.selections == []
            finally:
                dispatcher.close()

    asyncio.run(run())


async def serve_large_response(
    app: web.Application, dispatcher: Dispatcher, provider: LargeProvider, stream: bool
) -> None:
    # The lifecycle is a separate application: a Unix socket in production, loopback TCP in this test.
    async with (
        TestServer(app) as server,
        TestServer(app[CONTROL_APPLICATION]) as control_server,
        ClientSession() as control,
    ):
        control_url = str(control_server.make_url("/reef/lifecycle"))
        command = {"deployment_id": "deployment", "generation": "generation", "control_revision": 0}

        async def lifecycle_status() -> dict[str, object]:
            return await (await control.get(control_url, headers=CONTROL_HEADERS)).json()

        opened = await control.post(
            control_url,
            headers=CONTROL_HEADERS,
            json={**command, "action": "open", "open_deadline_monotonic": OPEN_DEADLINE},
        )
        assert opened.status == 200, await opened.text()
        client = await asyncio.to_thread(PausedClient, server.port)
        try:
            response = await asyncio.to_thread(client.send, LARGE_PROMPT, stream)
            assert response.status == 200
            async with asyncio.timeout(30):
                while True:
                    records = list(dispatcher.get_or_create_scenario("coding").records.replay("coding"))
                    status = await lifecycle_status()
                    if records and status["active_requests"] == 1 and status["pending_writes"] == 0:
                        break
                    await asyncio.sleep(0.025)
            # The record is durable, yet the request still holds its admission while its response is unsent.
            await asyncio.sleep(0.25)
            status = await lifecycle_status()
            assert (status["active_requests"], status["pending_writes"], len(records)) == (1, 0, 1)
            closed = await control.post(
                control_url, headers=CONTROL_HEADERS, json={**command, "action": "close", "control_revision": 1}
            )
            assert closed.status == 200
            body = await asyncio.to_thread(client.read, response)
            if stream:
                assert body.endswith(terminal_frame(provider.size_bytes))
                assert sse_receipt(body) == records[0].agent_record_id
            else:
                assert json.loads(body)["choices"][0]["message"]["content"] == "x" * provider.size_bytes
                assert response.getheader("x-reef-agent-record-id") == records[0].agent_record_id
            assert (await lifecycle_status())["active_requests"] == 0
            # A closed deployment refuses the next request on the same open connection, before any model call.
            first_socket = client.connection.sock
            refused = await asyncio.to_thread(client.send, "after close", False)
            await asyncio.to_thread(refused.read)
            assert refused.status == 503
            assert client.connection.sock is first_socket
            assert provider.completions == 1
            assert len(list(dispatcher.get_or_create_scenario("coding").records.replay("coding"))) == 1
        finally:
            client.close()
