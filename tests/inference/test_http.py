"""The HTTP inference adapter: who owns its provider session, and what a provider's reply means."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from email.utils import formatdate

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.core.errors import UpstreamLimit, UpstreamStatusError
from reef.inference.http import HttpInferenceHandler, retry_after_seconds
from reef.inference.interfaces import InferenceArtifact

pytestmark = pytest.mark.unit
ARTIFACT = InferenceArtifact("test-content")
CHAT_PATH = "/v1/chat/completions"
#: A whole epoch second, so that an HTTP date names it exactly.
NOW = 1_800_000_000.0


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
            assert error.value.status == 502 and error.value.is_unusable_answer

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, {"choices": []}),
        # An answer's own error field is no refusal: only the status is.
        (200, {"choices": [], "error": {"code": "payment_required"}}),
        (422, {"error": "max_tokens too large"}),
    ],
)
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


@pytest.mark.unit
@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan")])
def test_an_http_handler_refuses_a_timeout_that_is_not_positive(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        HttpInferenceHandler("https://provider.example", timeout_s=timeout_s)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("3", 3.0),
        (" 0 ", 0.0),
        (formatdate(NOW + 5, usegmt=True), 5.0),
        # An RFC 2822 date with the zone -0000 names no zone; it reads as GMT.
        (formatdate(NOW + 5), 5.0),
        (formatdate(NOW - 60, usegmt=True), 0.0),
        ("1.5", None),
        ("-1", None),
        ("soon", None),
        ("", None),
        (None, None),
        ("9" * 400, None),
        # A year that no date holds is no wait, and does not raise.
        ("Wed, 21 Oct 999999999999999999999 07:28:00 GMT", None),
    ],
)
def test_a_retry_after_header_is_whole_seconds_or_an_http_date(value: str | None, seconds: float | None) -> None:
    assert retry_after_seconds(value, NOW) == seconds


RATE, PAYMENT = UpstreamLimit.RATE, UpstreamLimit.PAYMENT
UNTYPED_402 = b'{"error": {"code": 402, "message": "Insufficient credits"}}'
# A gateway that wraps the provider's refusal: the code is text, and the provider's number is in the message.
WRAPPED = json.dumps(
    {"error": {"message": 'OpenrouterException - {"error":{"code":402}}', "type": None, "param": None, "code": "402"}}
).encode()


def limited(**metadata: object) -> bytes:
    """An OpenRouter 402 whose ``error.metadata`` is ``metadata``."""
    return json.dumps({"error": {"code": 402, "message": "refused", "metadata": metadata}}).encode()


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "stream"])
@pytest.mark.parametrize(
    ("status", "body", "retry_after", "limit", "wait_seconds"),
    [
        (429, b'{"error": {"message": "slow down"}}', "2", RATE, 2.0),
        # A gateway that found no free deployment answers in plain text: the status alone names the limit.
        (429, b"No deployments available for selected model", None, RATE, None),
        (429, b'{"error": {"type": "insufficient_quota", "code": "insufficient_quota"}}', None, PAYMENT, None),
        # Any 402 that no wait marks as temporary requires payment for this request, whatever its body holds.
        (402, UNTYPED_402, None, PAYMENT, None),
        (402, WRAPPED, None, PAYMENT, None),
        (402, b"", None, PAYMENT, None),
        (402, b"\xffPayment Required", None, PAYMENT, None),
        (402, b"[402]", None, PAYMENT, None),
        (402, limited(limit_source=["openrouter_in_flight_budget"]), None, PAYMENT, None),
        # OpenRouter's in-flight limit passes with time, with or without a wait.
        (402, limited(limit_source="openrouter_in_flight_budget"), None, RATE, None),
        (402, limited(limit_source="openrouter_in_flight_budget"), "3", RATE, 3.0),
        # A named limit that the same request cannot pass stays a payment refusal, also with a wait.
        (402, limited(limit_source="openrouter_credits"), "5", PAYMENT, 5.0),
        (402, limited(limit_source="openrouter_key_limit"), "5", PAYMENT, 5.0),
        (402, limited(limit_source="openrouter_credits", reason="weight_exceeds_budget"), "5", PAYMENT, 5.0),
        (402, limited(reason="weight_exceeds_budget"), "5", PAYMENT, 5.0),
        # A wait with no named lasting limit marks a 402 as temporary.
        (402, UNTYPED_402, "4", RATE, 4.0),
        (402, limited(limit_source="another_budget"), "4", RATE, 4.0),
        # A limit_source that is no string names no limit.
        (402, limited(limit_source={"name": "openrouter_credits"}), "4", RATE, 4.0),
        # OpenRouter's 403 for a spending cap is a payment refusal, also with a wait; any other 403 names no limit.
        (403, limited(limit_source="openrouter_guardrail_budget"), None, PAYMENT, None),
        (403, limited(limit_source="openrouter_workspace_budget"), "5", PAYMENT, 5.0),
        (403, limited(limit_source="openrouter_key_limit"), None, PAYMENT, None),
        (403, limited(limit_source="openrouter_credits"), None, None, None),
        (403, b'{"error": {"code": 403, "message": "Forbidden"}}', "5", None, 5.0),
        (401, b'{"error": {"code": 401, "message": "No auth credentials found"}}', None, None, None),
        (503, b"{}", "7", None, 7.0),
    ],
)
def test_a_provider_refusal_keeps_its_status_wait_and_limit_and_is_sent_once(
    streams: bool,
    status: int,
    body: bytes,
    retry_after: str | None,
    limit: UpstreamLimit | None,
    wait_seconds: float | None,
) -> None:
    async def run() -> None:
        requests: list[bytes] = []

        async def answer(request: web.Request) -> web.Response:
            requests.append(await request.read())
            headers = {} if retry_after is None else {"Retry-After": retry_after}
            return web.Response(body=body, status=status, headers=headers, content_type="application/json")

        async with provider(CHAT_PATH, answer) as server:
            handler = HttpInferenceHandler(str(server.make_url("")), timeout_s=5, error_label="inference provider")
            with pytest.raises(UpstreamStatusError) as error:
                if streams:
                    await handler.inference_stream(ARTIFACT, CHAT_PATH, {"stream": True})
                else:
                    await handler.inference(ARTIFACT, CHAT_PATH, {"model": "student"})
        refusal = error.value
        # An error body that is not UTF-8 keeps the status too.
        assert (refusal.status, refusal.limit, refusal.retry_after_seconds) == (status, limit, wait_seconds)
        assert str(refusal).startswith(f"inference provider returned {status}: ")
        assert len(requests) == 1, "the handler sends each request once"

    asyncio.run(run())
