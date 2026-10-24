"""The HTTP inference adapter: who owns its provider session, and what a provider's reply means."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.artifact.artifact import Artifact
from reef.core.artifact_ref import ArtifactRef
from reef.inference.http import HttpInferenceHandler
from reef.runtime.interfaces import UpstreamStatusError

pytestmark = pytest.mark.unit
ARTIFACT = Artifact(ArtifactRef("test-content", "test-release", None), None)
CHAT_PATH = "/v1/chat/completions"


def created_sessions(monkeypatch: pytest.MonkeyPatch) -> list[aiohttp.ClientSession]:
    """Every provider session the adapter creates from now on."""
    sessions: list[aiohttp.ClientSession] = []
    create_session = aiohttp.ClientSession

    def create(*args: object, **kwargs: object) -> aiohttp.ClientSession:
        session = create_session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", create)
    return sessions


def provider(path: str, answer: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> TestServer:
    app = web.Application()
    app.router.add_post(path, answer)
    return TestServer(app)


def test_cancellation_while_waiting_for_provider_headers_closes_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = created_sessions(monkeypatch)

    async def run() -> None:
        arrived, release = asyncio.Event(), asyncio.Event()

        async def headers_later(request: web.Request) -> web.Response:
            await request.read()
            arrived.set()
            await release.wait()
            return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream")

        async with provider(CHAT_PATH, headers_later) as server:
            handler = HttpInferenceHandler(str(server.make_url("")), timeout_s=5)
            opening = asyncio.create_task(handler.inference_stream(ARTIFACT, CHAT_PATH, {"stream": True}))
            try:
                await asyncio.wait_for(arrived.wait(), 5)
                opening.cancel()
                # The cancellation reaches the caller unchanged.
                with pytest.raises(asyncio.CancelledError):
                    await opening
                [session] = sessions
                assert session.closed
            finally:
                release.set()
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 429])
def test_a_returned_stream_owns_its_session_until_it_is_closed(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    sessions = created_sessions(monkeypatch)

    async def run() -> None:
        async def answer(request: web.Request) -> web.Response:
            return web.Response(body=b"data: [DONE]\n\n", content_type="text/event-stream", status=status)

        async with provider(CHAT_PATH, answer) as server:
            handler = HttpInferenceHandler(str(server.make_url("")), timeout_s=5)
            if status != 200:
                # A provider error status is read, and its session closed, before the error reaches the caller.
                with pytest.raises(UpstreamStatusError) as error:
                    await handler.inference_stream(ARTIFACT, CHAT_PATH, {"stream": True})
                assert error.value.status == status
                [session] = sessions
                assert session.closed
                return
            stream = await handler.inference_stream(ARTIFACT, CHAT_PATH, {"stream": True})
            try:
                [session] = sessions
                assert not session.closed
                assert b"".join([chunk async for chunk in stream.chunks]) == b"data: [DONE]\n\n"
            finally:
                await stream.close()
            assert session.closed

    asyncio.run(run())


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"{not json", "application/json"),
        (b"[]", "application/json"),
        (b"", "application/json"),
        (b'{"text": "\xff"}', "application/json"),
        (b'{"choices": []}', "text/plain"),
    ],
    ids=["malformed", "array", "empty", "undecodable", "content_type"],
)
def test_a_successful_status_with_an_unusable_body_is_a_provider_failure(body: bytes, content_type: str) -> None:
    async def run() -> None:
        async def answer(request: web.Request) -> web.Response:
            return web.Response(body=body, content_type=content_type, charset="utf-8")

        async with provider(CHAT_PATH, answer) as server:
            handler = HttpInferenceHandler(str(server.make_url("")), timeout_s=5)
            with pytest.raises(UpstreamStatusError) as error:
                await handler.inference(ARTIFACT, CHAT_PATH, {"model": "student"})
            # 502: the provider failed on its own account, so the caller's request is not blamed.
            assert error.value.status == 502

    asyncio.run(run())


@pytest.mark.parametrize(("status", "body"), [(200, {"choices": []}), (422, {"error": "max_tokens too large"})])
def test_a_json_object_is_returned_and_a_provider_client_error_keeps_its_status(
    status: int, body: dict[str, object]
) -> None:
    async def run() -> None:
        async def answer(request: web.Request) -> web.Response:
            return web.json_response(body, status=status)

        async with provider(CHAT_PATH, answer) as server:
            handler = HttpInferenceHandler(str(server.make_url("")), timeout_s=5)
            if status == 200:
                assert await handler.inference(ARTIFACT, CHAT_PATH, {"model": "student"}) == body
                return
            with pytest.raises(UpstreamStatusError) as error:
                await handler.inference(ARTIFACT, CHAT_PATH, {"model": "student"})
            assert error.value.status == status
            assert "max_tokens too large" in str(error.value)

    asyncio.run(run())
