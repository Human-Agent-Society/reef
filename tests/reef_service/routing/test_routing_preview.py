"""Contract tests for the routing preview client.

Every server here is a fake Semantic Router on loopback. These tests check
what the client sends and how it validates replies. They do not verify the
real router's behavior.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import socket
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.service import routing_preview
from reef.service.routing import RoutingAdapter, RoutingDecision, RoutingRequestError, RoutingScenario, RoutingSnapshot
from reef.service.routing_preview import (
    EXPECTED_HASH_HEADER,
    PREVIEW_BODY_LIMIT_BYTES,
    PREVIEW_PATH,
    RETURNED_HASH_HEADER,
    PreviewSettings,
    RoutingPreviewClient,
    preview_body,
)

pytestmark = pytest.mark.unit
ACTIVE = "a" * 64
TOKEN = "selection-token-value"
PROVIDER_SECRET = "provider-secret-value"
CALLER_TOKEN = "caller-bearer-value"
MANIFEST = {
    "schema_version": 2,
    "automatic_model": "auto",
    "deployment": {
        "semantic_router_image": "registry.example/sr@sha256:" + "1" * 64,
    },
    "targets": {
        "chat-student": {"provider_model": "student", "revision": "r1", "url": "http://127.0.0.1:9001/v1"},
        "base-student": {"provider_model": "base-student", "revision": "r1", "url": "http://127.0.0.1:9002/v1"},
        "shared-teacher": {
            "provider_model": "teacher",
            "revision": "r2",
            "url": "http://127.0.0.1:9003/v1",
            "api_key_env": "REEF_PREVIEW_TEST_PROVIDER_KEY",
        },
        "retired": {"provider_model": "old", "revision": "r0", "url": "http://127.0.0.1:9004/v1", "enabled": False},
    },
    "scenarios": {
        "chat-opd": {
            "default_target": "chat-student",
            "targets": ["chat-student", "shared-teacher"],
            "rules": [
                {"name": "teacher_requested", "keywords": ["reef_teacher_required"], "target": "shared-teacher"}
            ],
        },
        "base-opd": {
            "default_target": "base-student",
            "targets": ["base-student", "shared-teacher", "retired"],
            "rules": [{"name": "escalate_base", "keywords": ["reef_escalate_base"], "target": "shared-teacher"}],
        },
    },
}
SNAPSHOT = RoutingSnapshot.from_mapping(MANIFEST)
#: The published policy a request captured: the manifest and the router runtime ``ACTIVE``. No provider is called.
ROUTING = RoutingAdapter(SNAPSHOT, ACTIVE, {})
CHAT, BASE = SNAPSHOT.scenarios["chat-opd"], SNAPSHOT.scenarios["base-opd"]


class Reply(ABC):
    """One scripted way to answer a preview, including delays and failures.

    ``respond(request, data)`` receives the request and its body bytes and returns the response, or delays,
    closes the connection or raises as a subclass chooses. The fake router's server owns the connection.
    """

    @abstractmethod
    async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
        """Return the response for this request."""


def selection_reply(scenario: str, rule: str, target: str, *, method: str = "single") -> dict[str, object]:
    """A 200 preview reply shaped like the pinned EvalResponse for a static decision."""
    signals = None if rule == "default" else {"keywords": [rule]}
    return {
        "config_hash": ACTIVE,
        "original_text": "hello",
        "requested_model": "reef-entrypoint/" + scenario,
        "recipe": scenario,
        "decision_result": {
            "decision_name": rule,
            "algorithm": "static",
            "used_signals": signals,
            "matched_signals": signals,
            "unmatched_signals": None,
        },
        "recommended_models": [target],
        "selected_model": target,
        "selection_status": "selected",
        "selection_method": method,
        "selection_reason": "selected by the live runtime selector",
        "selection_provenance": {
            "mode": "stateless",
            "config_hash": ACTIVE,
            "state_dependent": False,
            "sampled": False,
        },
        "routing_decision": rule,
        "metrics": None,
    }


def json_reply(body: object, *, status: int = 200, hashes: tuple[str, ...] = (ACTIVE,)) -> Reply:
    class ReplyImplementation(Reply):
        """Answers ``body`` as JSON with ``status`` and one returned-hash header per ``hashes`` entry."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            response = web.json_response(body, status=status)
            for value in hashes:
                response.headers.add(RETURNED_HASH_HEADER, value)
            return response

    return ReplyImplementation()


def error_reply(status: int, code: str, message: str = "router detail that must not be relayed") -> Reply:
    error = {"error": {"code": code, "message": message, "timestamp": "2026-10-16T00:00:00Z", "request_id": "r1"}}
    return json_reply(error, status=status, hashes=())


class FakeRouter:
    """A fake Semantic Router preview handler that records each request it receives.

    ``handle`` appends ``(method, path, headers, body, peer address)`` to ``requests`` in arrival order and
    then answers with ``reply``, whatever the path. ``fake_router`` serves it on loopback and closes it.
    """

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.requests: list[tuple[str, str, dict[str, str], bytes, object]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        data = await request.read()
        transport = request.transport
        peer = None if transport is None else transport.get_extra_info("peername")
        self.requests.append((request.method, request.path, dict(request.headers), data, peer))
        return await self.reply.respond(request, data)


@contextlib.asynccontextmanager
async def fake_router(reply: Reply) -> AsyncIterator[tuple[FakeRouter, str]]:
    router = FakeRouter(reply)
    app = web.Application(client_max_size=4 * PREVIEW_BODY_LIMIT_BYTES)
    app.router.add_route("*", "/{tail:.*}", router.handle)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    try:
        yield router, f"http://127.0.0.1:{server.port}"
    finally:
        await server.close()


@contextlib.asynccontextmanager
async def started(base_url: str, *, timeout_seconds: int = 10) -> AsyncIterator[RoutingPreviewClient]:
    client = RoutingPreviewClient(PreviewSettings(base_url, TOKEN, timeout_seconds))
    await client.start()
    try:
        yield client
    finally:
        await client.close()


def request_payload(text: str = "hello", **extra: object) -> dict[str, object]:
    return {"model": "auto", "messages": [{"role": "user", "content": text}], **extra}


async def until_disconnected(request: web.Request) -> None:
    while request.transport is not None and not request.transport.is_closing():
        await asyncio.sleep(0.01)


Recorded = list[tuple[str, str, dict[str, str], bytes, object]]


def select_once(
    reply: Reply, payload: dict[str, object] | None = None, scenario: str = "chat-opd"
) -> tuple[object, Recorded]:
    """Run one selection; returns its result or the RoutingRequestError, and the fake router's requests."""

    async def run() -> tuple[object, Recorded]:
        async with fake_router(reply) as (router, url), started(url) as client:
            try:
                result: object = await client.select(
                    ROUTING, SNAPSHOT.scenarios[scenario], payload or request_payload()
                )
            except RoutingRequestError as error:
                result = error
            return result, router.requests

    return asyncio.run(run())


def test_default_and_keyword_selections_become_routing_decisions_for_the_scenarios_targets() -> None:
    cases = [
        ("chat-opd", "default", "chat-student", "static", "default"),
        ("chat-opd", "teacher_requested", "shared-teacher", "single", "rule"),
        ("base-opd", "default", "base-student", "single", "default"),
        ("base-opd", "escalate_base", "shared-teacher", "static", "rule"),
    ]
    for scenario, rule, target, method, reason in cases:
        result, requests = select_once(
            json_reply(selection_reply(scenario, rule, target, method=method)), None, scenario
        )
        assert result == RoutingDecision(
            SNAPSHOT.targets[target], SNAPSHOT.version, scenario, "auto", reason, rule, ACTIVE
        )
        assert len(requests) == 1


def test_the_request_holds_only_the_entrypoint_messages_and_selection_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REEF_PREVIEW_TEST_PROVIDER_KEY", PROVIDER_SECRET)
    tool_call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "be brief", "name": None},
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "x"}}]},
            {"role": "assistant", "content": None, "tool_calls": [tool_call], "audio": {"id": "a1"}},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "reef_teacher_required", "annotations": [], "reasoning_content": "r"},
        ],
        "temperature": 0.2,
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "authorization": CALLER_TOKEN,
        "user": "caller-identity",
    }
    original = copy.deepcopy(payload)
    result, requests = select_once(
        json_reply(selection_reply("chat-opd", "teacher_requested", "shared-teacher")), payload
    )
    assert isinstance(result, RoutingDecision) and result.rule_name == "teacher_requested"
    assert payload == original
    [recorded] = requests
    method, path, headers, data = recorded[:4]
    assert (method, path) == ("POST", PREVIEW_PATH)
    sent = json.loads(data)
    assert sent == {
        "model": "reef-entrypoint/chat-opd",
        "messages": [
            {"role": "system", "content": "be brief", "name": None},
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "x"}}]},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "reef_teacher_required", "reasoning_content": "r"},
        ],
    }
    assert data == preview_body(CHAT, original)
    assert headers["Authorization"] == "Bearer " + TOKEN
    assert headers[EXPECTED_HASH_HEADER] == ACTIVE
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept-Encoding"] == "identity"
    everything = json.dumps(headers) + data.decode()
    assert PROVIDER_SECRET not in everything and CALLER_TOKEN not in everything
    assert "caller-identity" not in everything and "x-reef" not in everything.lower()


def test_the_projection_copies_messages_instead_of_sharing_them() -> None:
    payload = request_payload()
    projected = routing_preview.preview_messages(payload)
    assert projected == payload["messages"]
    assert projected[0] is not payload["messages"][0]
    projected[0]["content"] = "changed"
    assert payload["messages"][0]["content"] == "hello"


def test_connections_are_reused_across_selections() -> None:
    async def run() -> tuple[list[RoutingDecision], list[object]]:
        reply = json_reply(selection_reply("chat-opd", "default", "chat-student"))
        async with fake_router(reply) as (router, url), started(url) as client:
            results = [await client.select(ROUTING, CHAT, request_payload(str(index))) for index in range(3)]
            return results, [request[4] for request in router.requests]

    results, peers = asyncio.run(run())
    assert [result.rule_name for result in results] == ["default"] * 3
    assert len(peers) == 3 and len(set(peers)) == 1


def mutated(path: tuple[str, ...], value: object) -> dict[str, object]:
    reply = selection_reply("chat-opd", "teacher_requested", "shared-teacher")
    target = reply
    for key in path[:-1]:
        target = target[key]
    if value is MISSING:
        target.pop(path[-1])
    else:
        target[path[-1]] = value
    return reply


MISSING = object()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("config_hash",), "b" * 64),
        (("config_hash",), MISSING),
        (("selection_provenance", "config_hash"), "b" * 64),
        (("selection_provenance", "mode"), "read_only_snapshot"),
        (("selection_provenance", "state_dependent"), True),
        (("selection_provenance", "state_dependent"), MISSING),
        (("selection_provenance", "sampled"), True),
        (("selection_provenance", "sampled"), MISSING),
        (("selection_provenance", "sampled"), 0),
        (("selection_provenance",), MISSING),
        (("selection_status",), "planned_final"),
        (("selection_status",), "fallback"),
        (("selection_method",), "elo"),
        (("selection_method",), MISSING),
        (("selection_method",), ["static"]),
        (("recipe",), "base-opd"),
        (("requested_model",), "reef-entrypoint/base-opd"),
        (("requested_model",), "auto"),
        (("decision_result", "algorithm"), "router_dc"),
        (("decision_result", "plugins"), ["header_mutation"]),
        (("decision_result", "decision_name"), "escalate_base"),
        (("decision_result", "decision_name"), "unknown_rule"),
        (("decision_result", "decision_name"), ""),
        (("decision_result",), MISSING),
        (("routing_decision",), "default"),
        (("routing_decision",), MISSING),
        (("selected_model",), "chat-student"),
        (("selected_model",), "base-student"),
        (("selected_model",), MISSING),
        (("decision_error",), "a signal failed"),
        (("signal_errors",), {"keyword:teacher_requested": "classifier unavailable"}),
        (("signal_error_matches",), {"keyword:teacher_requested": False}),
        (("applied_unknown_policies",), {"keyword:teacher_requested": "treat_as_unmatched"}),
    ],
)
def test_any_foreign_or_incomplete_selection_is_refused(path: tuple[str, ...], value: object) -> None:
    result, requests = select_once(json_reply(mutated(path, value)))
    assert isinstance(result, RoutingRequestError)
    assert (result.status, result.code) == (502, "routing_selection")
    assert len(requests) == 1


def test_empty_diagnostics_are_accepted() -> None:
    reply = selection_reply("chat-opd", "default", "chat-student")
    reply.update(decision_error="", signal_errors={}, signal_error_matches={}, applied_unknown_policies=None)
    selection = routing_preview.checked_selection(ROUTING, CHAT, reply, [ACTIVE])
    assert selection.target.target_id == "chat-student"


@pytest.mark.parametrize("hashes", [(), ("b" * 64,), (ACTIVE, ACTIVE), ("A" * 64,)])
def test_the_returned_runtime_header_must_be_exactly_the_expected_hash(hashes: tuple[str, ...]) -> None:
    reply = json_reply(selection_reply("chat-opd", "default", "chat-student"), hashes=hashes)
    result = select_once(reply)[0]
    assert isinstance(result, RoutingRequestError) and result.code == "routing_selection"


@pytest.mark.parametrize(
    ("rule", "selected"),
    [("default", "retired"), ("escalate_base", "retired"), ("escalate_base", "chat-student"), ("old_rule", "retired")],
)
def test_a_reply_selecting_a_disabled_or_foreign_target_or_an_unknown_rule_is_refused(
    rule: str, selected: str
) -> None:
    """The manifest cannot route there (see test_routing.py); the reply check refuses such a reply on its own."""
    good = selection_reply("base-opd", "default", "base-student")
    assert routing_preview.checked_selection(ROUTING, BASE, good, [ACTIVE]).target.target_id == "base-student"
    reply = selection_reply("base-opd", rule, selected)
    with pytest.raises(RoutingRequestError) as caught:
        routing_preview.checked_selection(ROUTING, BASE, reply, [ACTIVE])
    assert (caught.value.status, caught.value.code) == (502, "routing_selection")


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"not json", "application/json"),
        (b"[]", "application/json"),
        (b'{"selected_model": "a", "selected_model": "b"}', "application/json"),
        (b'{"config_hash": NaN}', "application/json"),
        (b'{"x": 1e999}', "application/json"),
        (json.dumps(selection_reply("chat-opd", "default", "chat-student")).encode(), "text/plain"),
    ],
)
def test_a_reply_that_is_not_one_plain_json_object_is_refused(body: bytes, content_type: str) -> None:
    class ReplyImplementation(Reply):
        """Answers the raw ``body`` with ``content_type`` and the active hash."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            response = web.Response(body=body, content_type=content_type)
            response.headers[RETURNED_HASH_HEADER] = ACTIVE
            return response

    reply = ReplyImplementation()

    result = select_once(reply)[0]
    assert isinstance(result, RoutingRequestError)
    assert (result.status, result.code) == (502, "routing_protocol")


@pytest.mark.parametrize(
    ("status", "code", "expected_status", "expected_code"),
    [
        (412, "CONFIG_HASH_MISMATCH", 503, "routing_runtime_changed"),
        (429, "OVERLOADED", 429, "routing_overloaded"),
        (504, "REQUEST_TIMEOUT", 504, "routing_timeout"),
        (400, "INVALID_INPUT", 400, "routing_input_rejected"),
        (400, "INVALID_ROUTING_MODEL", 502, "routing_protocol"),
        (400, "INVALID_CONFIG_HASH", 502, "routing_protocol"),
        (413, "REQUEST_BODY_TOO_LARGE", 413, "routing_input_too_large"),
        (401, "UNAUTHORIZED", 502, "routing_credentials"),
        (403, "FORBIDDEN", 502, "routing_credentials"),
        (412, "CONFIG_CHANGED", 502, "routing_protocol"),
        (503, "SHUTTING_DOWN", 503, "routing_unavailable"),
        (500, "INTERNAL", 502, "routing_protocol"),
    ],
)
def test_router_errors_map_to_sanitized_reef_errors_after_one_attempt(
    status: int, code: str, expected_status: int, expected_code: str
) -> None:
    result, requests = select_once(error_reply(status, code))
    assert isinstance(result, RoutingRequestError)
    assert (result.status, result.code) == (expected_status, expected_code)
    assert "must not be relayed" not in str(result) and TOKEN not in str(result)
    assert len(requests) == 1


def test_an_error_code_outside_the_routers_format_is_not_repeated() -> None:
    result = select_once(error_reply(429, "<script>" + TOKEN))[0]
    assert isinstance(result, RoutingRequestError) and result.code == "routing_overloaded"
    assert "script" not in str(result) and TOKEN not in str(result)


def test_redirects_are_refused_and_never_followed() -> None:
    class ReplyImplementation(Reply):
        """Redirects the preview path; any other path would receive a valid selection."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            if request.path == PREVIEW_PATH:
                raise web.HTTPTemporaryRedirect("/elsewhere")
            return web.json_response(selection_reply("chat-opd", "default", "chat-student"))

    reply = ReplyImplementation()

    result, requests = select_once(reply)
    assert isinstance(result, RoutingRequestError) and (result.status, result.code) == (502, "routing_protocol")
    assert [request[1] for request in requests] == [PREVIEW_PATH]


def test_a_connection_closed_before_any_reply_is_unavailable_and_not_retried() -> None:
    # As when the router process stops after accepting a preview and before answering it.
    class ReplyImplementation(Reply):
        """Closes the connection before writing any reply."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            if request.transport is not None:
                request.transport.close()
            await asyncio.sleep(0)
            return web.Response()

    reply = ReplyImplementation()

    result, requests = select_once(reply)
    assert isinstance(result, RoutingRequestError) and (result.status, result.code) == (503, "routing_unavailable")
    assert len(requests) == 1


@contextlib.asynccontextmanager
async def raw_router(reply: bytes | None) -> AsyncIterator[tuple[list[bytes], str]]:
    """A loopback listener below HTTP: it reads one request's headers, then writes ``reply`` and closes.

    With ``reply`` None it resets the connection instead, as a crashed router's host does.
    """
    received: list[bytes] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        received.append(await reader.readuntil(b"\r\n\r\n"))
        if reply is None:
            # A zero linger makes close send a TCP reset rather than an orderly shutdown.
            writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        else:
            writer.write(reply)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        yield received, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        await server.wait_closed()


def select_from_raw_router(reply: bytes | None) -> tuple[object, list[bytes]]:
    async def run() -> tuple[object, list[bytes]]:
        async with raw_router(reply) as (received, url), started(url) as client:
            try:
                result: object = await client.select(ROUTING, CHAT, request_payload())
            except RoutingRequestError as error:
                result = error
            return result, received

    return asyncio.run(run())


def test_a_reset_connection_is_unavailable_and_not_retried() -> None:
    result, received = select_from_raw_router(None)
    assert isinstance(result, RoutingRequestError) and (result.status, result.code) == (503, "routing_unavailable")
    assert len(received) == 1


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(b"NOT-HTTP\r\n\r\n", id="malformed_status_line"),
        pytest.param(
            b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{"config_hash":',
            id="body_cut_off_before_its_length",
        ),
    ],
)
def test_a_malformed_or_cut_off_reply_is_a_protocol_failure_not_unavailability(reply: bytes) -> None:
    # Once the router has begun a reply, a broken one is the router's fault, never a sign it is down.
    result, received = select_from_raw_router(reply)
    assert isinstance(result, RoutingRequestError) and (result.status, result.code) == (502, "routing_protocol")
    assert len(received) == 1


def test_an_unreachable_router_is_unavailable() -> None:
    async def run() -> RoutingRequestError:
        async with fake_router(error_reply(500, "INTERNAL")) as served:
            url = served[1]
        # The fake router has stopped; its port no longer accepts connections.
        async with started(url) as client:
            with pytest.raises(RoutingRequestError) as caught:
                await client.select(ROUTING, CHAT, request_payload())
            return caught.value

    error = asyncio.run(run())
    assert (error.status, error.code) == (503, "routing_unavailable")


def test_environment_proxies_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[int, int]:
        proxy_reply = error_reply(500, "INTERNAL")
        router_reply = json_reply(selection_reply("chat-opd", "default", "chat-student"))
        async with fake_router(proxy_reply) as (proxy, proxy_url), fake_router(router_reply) as (router, url):
            for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
                monkeypatch.setenv(name, proxy_url)
            monkeypatch.setenv("no_proxy", "")
            monkeypatch.setenv("NO_PROXY", "")
            async with started(url) as client:
                await client.select(ROUTING, CHAT, request_payload())
            return len(proxy.requests), len(router.requests)

    assert asyncio.run(run()) == (0, 1)


def test_a_router_that_never_answers_times_out_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing_preview, "RESPONSE_WRITE_ALLOWANCE_SECONDS", 0.0)

    class ReplyImplementation(Reply):
        """Never answers: waits until the client disconnects."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            await until_disconnected(request)
            return web.Response()

    reply = ReplyImplementation()

    async def run() -> tuple[RoutingRequestError, int]:
        async with fake_router(reply) as (router, url), started(url, timeout_seconds=1) as client:
            with pytest.raises(RoutingRequestError) as caught:
                await client.select(ROUTING, CHAT, request_payload())
            return caught.value, len(router.requests)

    error, attempts = asyncio.run(run())
    assert (error.status, error.code, attempts) == (504, "routing_timeout", 1)


def test_cancellation_closes_the_connection_and_leaves_the_client_usable() -> None:
    arrived, disconnected = asyncio.Event(), asyncio.Event()
    transport_after_cancel: list[object] = []
    calls: list[int] = []

    class ReplyImplementation(Reply):
        """Holds the first request until it is cancelled; answers every later one with a valid selection."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            calls.append(1)
            if len(calls) > 1:
                return web.json_response(
                    selection_reply("chat-opd", "default", "chat-student"), headers={RETURNED_HASH_HEADER: ACTIVE}
                )
            arrived.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                # TestServer runs with handler_cancellation=True: aiohttp cancels this handler only once the
                # peer's connection is lost, after it has already dropped the transport.
                transport_after_cancel.append(request.transport)
                disconnected.set()
                raise
            return web.Response()

    reply = ReplyImplementation()

    async def run() -> tuple[RoutingDecision, list[object]]:
        async with fake_router(reply) as (router, url), started(url) as client:
            task = asyncio.create_task(client.select(ROUTING, CHAT, request_payload()))
            await asyncio.wait_for(arrived.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # The fake router sees the disconnect while it is still serving, not at shutdown.
            await asyncio.wait_for(disconnected.wait(), 5)
            selection = await client.select(ROUTING, CHAT, request_payload())
            return selection, [request[4] for request in router.requests]

    selection, peers = asyncio.run(run())
    assert transport_after_cancel == [None]
    assert selection.rule_name == "default"
    # The cancelled connection was closed, not returned to the pool: the next selection uses a new one.
    assert len(peers) == 2 and peers[0] != peers[1]


def sized_payload(size_bytes: int, character: str = "x") -> dict[str, object]:
    """A request whose preview body is exactly ``size_bytes`` bytes, measured without the client's encoder."""

    def encoded_size(text: str) -> int:
        body = {"model": "reef-entrypoint/chat-opd", "messages": [{"role": "user", "content": text}]}
        return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())

    text = character * (size_bytes - encoded_size(""))
    assert encoded_size(text) == size_bytes
    return request_payload(text)


def test_a_preview_body_at_the_limit_is_sent_and_one_byte_more_is_refused_locally() -> None:
    class ReplyImplementation(Reply):
        """A valid selection that echoes the last message as ``original_text``."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            reply = selection_reply("chat-opd", "default", "chat-student")
            reply["original_text"] = json.loads(data)["messages"][-1]["content"]
            return web.json_response(reply, headers={RETURNED_HASH_HEADER: ACTIVE})

    echo = ReplyImplementation()

    result, requests = select_once(echo, sized_payload(PREVIEW_BODY_LIMIT_BYTES))
    assert isinstance(result, RoutingDecision) and len(requests[0][3]) == PREVIEW_BODY_LIMIT_BYTES
    result, requests = select_once(echo, sized_payload(PREVIEW_BODY_LIMIT_BYTES + 1))
    assert isinstance(result, RoutingRequestError) and (result.status, result.code) == (413, "routing_input_too_large")
    assert requests == []


def test_the_reply_limit_allows_gos_html_escaping_of_an_echoed_prompt() -> None:
    class ReplyImplementation(Reply):
        """Echoes the last message as ``original_text``, escaping ``<`` as Go's JSON encoder does."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            reply = selection_reply("chat-opd", "default", "chat-student")
            reply["original_text"] = json.loads(data)["messages"][-1]["content"]
            body = json.dumps(reply).replace("<", "\\u003c").encode()
            return web.Response(body=body, content_type="application/json", headers={RETURNED_HASH_HEADER: ACTIVE})

    go_escaped_echo = ReplyImplementation()

    payload = sized_payload(1024 * 1024, "<")
    assert isinstance(select_once(go_escaped_echo, payload)[0], RoutingDecision)


def test_a_reply_beyond_its_limit_is_refused() -> None:
    class ReplyImplementation(Reply):
        """A valid selection whose ``original_text`` alone exceeds the reply limit."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            reply = selection_reply("chat-opd", "default", "chat-student")
            reply["original_text"] = "x" * (2 * 1024 * 1024)
            return web.json_response(reply, headers={RETURNED_HASH_HEADER: ACTIVE})

    oversized = ReplyImplementation()

    result = select_once(oversized)[0]
    assert isinstance(result, RoutingRequestError) and (result.status, result.code) == (502, "routing_protocol")


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "auto"},
        {"model": "auto", "messages": []},
        {"model": "auto", "messages": {"role": "user"}},
        {"model": "auto", "messages": ["hello"]},
        {"model": "auto", "messages": [{"content": "hello"}]},
        {"model": "auto", "messages": [{"role": 1, "content": "hello"}]},
        {"model": "auto", "messages": [{"role": "user", "content": "hello", "name": 7}]},
        {"model": "auto", "messages": [{"role": "tool", "content": "x", "tool_call_id": ["c"]}]},
        {"model": "auto", "messages": [{"role": "assistant", "content": None, "tool_calls": {"id": "c"}}]},
        {"model": "auto", "messages": [{"role": "user", "content": float("nan")}]},
        {"model": "auto", "messages": [{"role": "user", "content": [{1: "text"}]}]},
        {"model": "auto", "messages": [{"role": "user", "content": ("a", "b")}]},
        {"model": "auto", "messages": [{"role": "user", "content": "\ud800"}]},
    ],
)
def test_unrepresentable_input_is_refused_before_the_network(payload: dict[str, object]) -> None:
    result, requests = select_once(json_reply(selection_reply("chat-opd", "default", "chat-student")), payload)
    assert isinstance(result, RoutingRequestError)
    assert (result.status, result.code) == (400, "routing_input_unrepresentable")
    assert requests == []


def test_deeply_nested_content_is_refused_before_the_network() -> None:
    depth = routing_preview.MAX_JSON_DEPTH + 1
    content = json.loads("[" * depth + '"x"' + "]" * depth)
    payload = {"model": "auto", "messages": [{"role": "user", "content": content}]}
    result, requests = select_once(json_reply({}), payload)
    assert isinstance(result, RoutingRequestError) and result.code == "routing_input_unrepresentable"
    assert requests == []


def test_caller_contract_errors_never_reach_the_router() -> None:
    narrowed = RoutingScenario("chat-opd", "chat-student", ("chat-student",))
    unknown = RoutingScenario("missing", "chat-student", ("chat-student",))
    # The expected runtime belongs to the published policy, so a malformed hash never becomes one.
    for runtime_hash in ("A" * 64, "abc"):
        with pytest.raises(ValueError, match="router_config_hash"):
            RoutingAdapter(SNAPSHOT, runtime_hash, {})

    async def run() -> int:
        async with fake_router(json_reply({})) as (router, url), started(url) as client:
            for scenario, payload in (
                (CHAT, {**request_payload(), "model": "shared-teacher"}),
                (CHAT, {**request_payload(), "model": "reef-entrypoint/chat-opd"}),
                (narrowed, request_payload()),
                (unknown, request_payload()),
            ):
                with pytest.raises(ValueError):
                    await client.select(ROUTING, scenario, payload)
            return len(router.requests)

    assert asyncio.run(run()) == 0


def test_readiness_previews_each_scenario_and_accepts_a_rule_selection() -> None:
    class ReplyImplementation(Reply):
        """Selects by rule for the chat Scenario and by default for every other one."""

        async def respond(self, request: web.Request, data: bytes) -> web.StreamResponse:
            scenario = json.loads(data)["model"].removeprefix("reef-entrypoint/")
            # The readiness message happens to match the chat Scenario's rule; any valid selection passes.
            selected = (
                ("teacher_requested", "shared-teacher") if scenario == "chat-opd" else ("default", "base-student")
            )
            return web.json_response(selection_reply(scenario, *selected), headers={RETURNED_HASH_HEADER: ACTIVE})

    reply = ReplyImplementation()

    async def run() -> list[dict[str, object]]:
        async with fake_router(reply) as (router, url), started(url) as client:
            await client.check_ready(ROUTING)
            return [json.loads(request[3]) for request in router.requests]

    sent = asyncio.run(run())
    assert [body["model"] for body in sent] == ["reef-entrypoint/chat-opd", "reef-entrypoint/base-opd"]
    assert all(body["messages"] == [{"role": "user", "content": routing_preview.READINESS_TEXT}] for body in sent)


def test_the_client_must_be_started_once_and_closes_cleanly() -> None:
    with pytest.raises(ValueError):
        RoutingPreviewClient(object())

    async def run() -> None:
        client = RoutingPreviewClient(PreviewSettings("http://127.0.0.1:9", TOKEN))
        with pytest.raises(RuntimeError, match="not started"):
            await client.select(ROUTING, CHAT, request_payload())
        await client.start()
        with pytest.raises(RuntimeError, match="already started"):
            await client.start()
        await client.close()
        await client.close()
        assert client.session is None

    asyncio.run(run())


@pytest.mark.parametrize(
    ("base_url", "token", "timeout"),
    [
        ("https://127.0.0.1:8080", TOKEN, 10),
        ("http://localhost:8080", TOKEN, 10),
        ("http://10.0.0.5:8080", TOKEN, 10),
        ("http://127.0.0.1", TOKEN, 10),
        ("http://127.0.0.1:0", TOKEN, 10),
        ("http://127.0.0.1:8080/", TOKEN, 10),
        ("http://127.0.0.1:8080/api", TOKEN, 10),
        ("http://user:pw@127.0.0.1:8080", TOKEN, 10),
        ("http://127.0.0.1:8080", "", 10),
        ("http://127.0.0.1:8080", "two words", 10),
        ("http://127.0.0.1:8080", ["t", "o", "k"], 10),
        ("http://127.0.0.1:8080", TOKEN, 0),
        ("http://127.0.0.1:8080", TOKEN, 301),
        ("http://127.0.0.1:8080", TOKEN, 1.5),
        ("http://127.0.0.1:8080", TOKEN, True),
    ],
)
def test_settings_accept_only_a_loopback_listener_a_token_and_whole_seconds(
    base_url: str, token: object, timeout: object
) -> None:
    with pytest.raises(ValueError):
        PreviewSettings(base_url, token, timeout)


def test_settings_never_show_the_token_and_wait_out_the_routers_write_allowance() -> None:
    settings = PreviewSettings("http://127.0.0.1:8080", TOKEN, 7)
    assert TOKEN not in repr(settings)
    assert settings.preview_url == "http://127.0.0.1:8080" + PREVIEW_PATH
    assert settings.http_timeout_seconds == 7 + routing_preview.RESPONSE_WRITE_ALLOWANCE_SECONDS == 12
