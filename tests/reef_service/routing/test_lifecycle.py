"""Managed HTTP admission, private control, response EOF, and storage finalization."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponse, ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.routing.routing_fakes import (
    OPEN_DEADLINE,
    ROUTER_HASH,
    SELECTION_TOKEN,
    SERVICE_TOKEN,
    FakeRuntime,
    FakeSemanticRouter,
    provider_models,
    routing_adapter,
    routing_dispatcher,
    routing_headers,
    running_router,
)
from yarl import URL

from reef.core import AgentRecord
from reef.scenario.scenario import Scenario
from reef.service.app import CONTROL_APPLICATION, create_app
from reef.service.auth import create_authentication_middleware
from reef.service.deploy.routing.routing_config import RoutingDeploymentSettings
from reef.service.deploy.routing.routing_control import ReefLifecycleClient, RoutingControlError
from reef.service.deploy.routing.routing_controller import RoutingDeploymentController
from reef.service.deploy.service_config import ServiceConfig
from reef.service.lifecycle import (
    INFERENCE_FINALIZATION,
    REQUEST_ACCEPTANCE,
    LifecycleSettings,
    PendingWrites,
    RequestAcceptance,
    RoutingPublisher,
    WriteAdmission,
    create_acceptance_middleware,
    create_control_app,
    register_admission,
    serve_managed,
)
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.wire import RequestHeaders
from reef.storage.records import AppendResult, RecordConflict
from reef.storage.sqlite import SQLiteRecordStore

pytestmark = pytest.mark.unit

SERVICE_HEADERS = {"Authorization": "Bearer service"}
CONTROL_HEADERS = {"Authorization": "Bearer control"}


class ManagedClients:
    """A public application with admission and its separate control application, each with its own client.

    ``get`` and ``post`` send ``/reef/lifecycle`` to the control application (over loopback TCP here; production
    serves it only on a Unix socket) and every other path to the public application. The public ``app``
    authenticates the ``service`` and ``evaluation`` tokens and admits through ``acceptance``; a test may add
    routes to it before entering. ``async with`` starts both test servers and closes both on exit.
    """

    def __init__(self, acceptance: RequestAcceptance) -> None:
        self.app = web.Application(
            middlewares=[
                create_authentication_middleware("service", evaluation_tokens="evaluation"),
                create_acceptance_middleware(acceptance),
            ]
        )
        register_admission(self.app, acceptance)
        self.public = TestClient(TestServer(self.app))
        self.control = TestClient(TestServer(create_control_app(acceptance)))

    async def __aenter__(self) -> ManagedClients:
        await self.public.start_server()
        await self.control.start_server()
        return self

    async def __aexit__(self, *exception: object) -> None:
        await self.public.close()
        await self.control.close()

    def route(self, path: str | URL) -> TestClient:
        return self.control if path == "/reef/lifecycle" else self.public

    async def get(self, path: str | URL, **options: object) -> ClientResponse:
        return await self.route(path).get(path, **options)

    async def post(self, path: str | URL, **options: object) -> ClientResponse:
        return await self.route(path).post(path, **options)


def client_for(acceptance: RequestAcceptance) -> ManagedClients:
    return ManagedClients(acceptance)


def settings() -> LifecycleSettings:
    return LifecycleSettings("deployment", "generation", "control", "snapshot")


def command(action: str, revision: int = 0, **overrides: object) -> dict[str, object]:
    return {
        "deployment_id": "deployment",
        "generation": "generation",
        "control_revision": revision,
        "action": action,
        **({"open_deadline_monotonic": OPEN_DEADLINE} if action == "open" else {}),
        **overrides,
    }


def test_control_is_exclusive_and_terminal_with_monotonic_revisions() -> None:
    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        client = client_for(acceptance)

        async def ok(request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        client.app.router.add_post("/v1/chat/completions", ok)
        client.app.router.add_post("/reef/report", ok)
        client.app.router.add_get("/healthz", ok)
        async with client:
            for token in (None, "service", "evaluation"):
                headers = {} if token is None else {"Authorization": f"Bearer {token}"}
                response = await client.control.get("/reef/lifecycle", headers=headers)
                assert response.status == 401
            # The public application has no lifecycle route, and the control token is not a service token there.
            for path in ("/reef/lifecycle", "/reef/%6cifecycle"):
                assert (await client.public.get(path, headers=SERVICE_HEADERS)).status == 404
                assert (await client.public.get(path, headers=CONTROL_HEADERS)).status == 401
            for path in ("/v1/chat/completions", "/reef/report"):
                response = await client.post(path, headers=CONTROL_HEADERS, json={})
                assert response.status == 401
            response = await client.get("/healthz")
            assert response.status == 200
            response = await client.post("/v1/chat/completions", headers=SERVICE_HEADERS, json={})
            assert response.status == 503
            for body, status in (
                (command("open"), 200),
                (command("open"), 200),
                (command("close"), 409),
                (command("close", 1), 200),
                (command("open"), 409),
                (command("stop", 2), 200),
                (command("open", 3), 409),
                (command("stop", 2), 200),
            ):
                response = await client.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=body)
                assert response.status == status, body
            response = await client.post("/reef/report", headers=SERVICE_HEADERS, json={})
            assert response.status == 503
            response = await client.get("/reef/lifecycle", headers=CONTROL_HEADERS)
            state = await response.json()
            assert state["stopping"] is True
            assert state["active_requests"] == state["pending_writes"] == 0
            assert "control_token" not in state

    asyncio.run(run())


@pytest.mark.parametrize("action", ["close", "stop"])
def test_an_admitted_upload_counts_toward_the_drain(action: str) -> None:
    """Admission happens when the headers arrive; closing drains the upload, stopping refuses it at its end."""

    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        acceptance.control(command("open"))
        client = client_for(acceptance)
        body_started = asyncio.Event()
        body_finish = asyncio.Event()
        calls = []

        async def upload() -> AsyncIterator[bytes]:
            yield b'{"model":'
            body_started.set()
            await body_finish.wait()
            yield b'"student"}'

        async def inference(request: web.Request) -> web.Response:
            calls.append(await read_object(request))
            return web.json_response({"ok": True})

        client.app.router.add_post("/v1/chat/completions", inference)
        async with client:
            request = asyncio.create_task(client.post("/v1/chat/completions", headers=SERVICE_HEADERS, data=upload()))
            await body_started.wait()
            async with asyncio.timeout(5):
                while not acceptance.active:
                    await asyncio.sleep(0.01)
            response = await client.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command(action, 1))
            assert response.status == 200
            assert (await response.json())["active_requests"] == 1
            body_finish.set()
            assert (await request).status == (200 if action == "close" else 503)
            assert calls == ([{"model": "student"}] if action == "close" else [])
            assert acceptance.status()["active_requests"] == 0

    asyncio.run(run())


def test_encoded_evaluation_path_uses_resolved_route_admission() -> None:
    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        client = client_for(acceptance)
        called = []

        async def evaluation(request: web.Request) -> web.Response:
            called.append(request.rel_url.raw_path)
            return web.json_response({"active": acceptance.status()["active_requests"]})

        client.app.router.add_post("/reef/scenarios/{scenario}/evaluation/v1/{route:.+}", evaluation)
        path = URL("/reef/scenarios/s/%65valuation/v1/chat/completions", encoded=True)
        async with client:
            response = await client.post(path, headers=SERVICE_HEADERS, json={})
            assert response.status == 503
            assert not called
            acceptance.control(command("open"))
            response = await client.post(path, headers=SERVICE_HEADERS, json={})
            assert response.status == 200
            assert (await response.json())["active"] == 1
            assert called == ["/reef/scenarios/s/%65valuation/v1/chat/completions"]

    asyncio.run(run())


def test_unexpected_handler_error_returns_500_instead_of_connection_reset() -> None:
    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        acceptance.control(command("open"))
        client = client_for(acceptance)

        async def fail(request: web.Request) -> web.Response:
            raise RuntimeError("deliberate handler failure")

        client.app.router.add_post("/v1/chat/completions", fail)
        async with client:
            response = await client.post("/v1/chat/completions", headers=SERVICE_HEADERS, json={})
            assert response.status == 500
            assert "Internal Server Error" in await response.text()
            assert "deliberate" not in await response.text()
            assert acceptance.status()["active_requests"] == 0
            state = await client.get("/reef/lifecycle", headers=CONTROL_HEADERS)
            assert state.status == 200

    asyncio.run(run())


def test_unexpected_stream_error_does_not_send_a_successful_terminal_response() -> None:
    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        acceptance.control(command("open"))
        client = client_for(acceptance)

        async def fail(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b"data: partial\n\n")
            raise RuntimeError("deliberate stream failure")

        client.app.router.add_post("/v1/chat/completions", fail)
        async with client:
            response = await client.post("/v1/chat/completions", headers=SERVICE_HEADERS, json={})
            assert response.status == 200
            with pytest.raises(ClientPayloadError):
                await response.read()
            assert acceptance.status()["active_requests"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", [ValueError, RecordConflict, RuntimeError])
def test_a_record_failure_after_stream_headers_never_writes_a_second_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    """Error translation maps the first two to 400 and 409 inside admission; the third is unmapped."""
    appends: list[str] = []

    def fail_append(store: SQLiteRecordStore, item: AgentRecord) -> AppendResult:
        appends.append(item.agent_record_id)
        raise failure("injected record failure")

    async def run() -> None:
        async def completion(request: web.Request) -> web.Response:
            await request.read()
            return web.Response(
                body=b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\ndata: [DONE]\n\n',
                content_type="text/event-stream",
            )

        provider_app = web.Application()
        provider_app.router.add_get("/v1/models", provider_models)
        provider_app.router.add_post("/v1/chat/completions", completion)
        async with TestServer(provider_app) as provider, contextlib.AsyncExitStack() as stack:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")))
            dispatcher = routing_dispatcher(tmp_path)
            stack.callback(dispatcher.close)
            router = await stack.enter_async_context(running_router(adapter.snapshot))
            lifecycle = LifecycleSettings(
                "deployment",
                "generation",
                "control",
                adapter.snapshot.version,
                selection_token=SELECTION_TOKEN,
                router_config_hash=ROUTER_HASH,
            )
            app = create_app(
                dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings, lifecycle=lifecycle
            )
            acceptance = app[REQUEST_ACCEPTANCE]
            acceptance.control(command("open"))
            server = await stack.enter_async_context(TestServer(app))
            monkeypatch.setattr(SQLiteRecordStore, "append_result", fail_append)
            payload = json.dumps({"model": "student", "messages": [{"role": "user", "content": "x"}], "stream": True})
            headers = {
                "Host": f"{server.host}:{server.port}",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
                **routing_headers(),
            }
            reader, writer = await asyncio.open_connection(server.host, server.port)
            try:
                writer.write(
                    (
                        "POST /v1/chat/completions HTTP/1.1\r\n"
                        + "".join(f"{name}: {value}\r\n" for name, value in headers.items())
                        + "\r\n"
                        + payload
                    ).encode()
                )
                await writer.drain()
                # The connection is kept alive, so it ends only because Reef closes it after the failure.
                raw = await asyncio.wait_for(reader.read(), 5)
            finally:
                writer.close()
                await writer.wait_closed()
            async with asyncio.timeout(5):
                while acceptance.active or acceptance.writes.tasks:
                    await asyncio.sleep(0.01)
            # One status line: the started SSE response is cut short, never followed by another response.
            assert re.findall(rb"HTTP/1\.[01] \d{3}", raw) == [b"HTTP/1.1 200"]
            assert b'"content":"first"' in raw
            assert b'"reef"' not in raw and b"[DONE]" not in raw
            # The stream was recorded once, and that write failed; nothing was stored.
            assert len(appends) == 1 and acceptance.writes.failed_count == 1
            assert dispatcher.get_or_create_scenario("coding").records.replay("coding") == ()

    asyncio.run(run())


def test_drain_tracks_preclose_writes_and_late_inference_finalization_but_not_new_reports() -> None:
    async def run() -> None:
        writes = PendingWrites()
        acceptance = RequestAcceptance(settings(), writes)
        acceptance.control(command("open"))
        old_report, inference, new_report = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def blocked(event: asyncio.Event) -> None:
            await event.wait()

        before_close = asyncio.create_task(writes.run(blocked(old_report)))
        await asyncio.sleep(0)
        acceptance.control(command("close", 1))
        context = INFERENCE_FINALIZATION.set(True)
        after_close_inference = asyncio.create_task(writes.run(blocked(inference)))
        INFERENCE_FINALIZATION.reset(context)
        after_close_report = asyncio.create_task(writes.run(blocked(new_report)))
        await asyncio.sleep(0)
        try:
            assert acceptance.status()["pending_writes"] == 3
            assert acceptance.status()["drain_pending_writes"] == 2
            acceptance.control(command("close", 2, cancel_active=True))
            assert acceptance.status()["drain_pending_writes"] == 2
            with pytest.raises(web.HTTPConflict):
                acceptance.control(command("open", 3))
            old_report.set()
            await before_close
            assert acceptance.status()["drain_pending_writes"] == 1
            inference.set()
            await after_close_inference
            assert acceptance.status()["drain_pending_writes"] == 0
            assert acceptance.status()["pending_writes"] == 1
            acceptance.control(command("open", 3))
            assert acceptance.accepting
            acceptance.control(command("stop", 4))
            assert acceptance.status()["drain_pending_writes"] == 1
        finally:
            old_report.set()
            inference.set()
            new_report.set()
            await asyncio.gather(before_close, after_close_inference, after_close_report)

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_root_is_counted_until_response_eof(stream: bool) -> None:
    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        acceptance.control(command("open"))
        client = client_for(acceptance)
        eof_started = asyncio.Event()
        eof_finish = asyncio.Event()

        class DelayedResponse(web.Response):
            """A response whose end of body is held until the test releases ``eof_finish``."""

            async def write_eof(self, data: bytes = b"") -> None:
                eof_started.set()
                await eof_finish.wait()
                await super().write_eof(data)

        async def inference(request: web.Request) -> web.Response:
            response = DelayedResponse(
                body=b"done", content_type="text/event-stream" if stream else "application/json"
            )
            if stream:
                await response.prepare(request)
            return response

        client.app.router.add_post("/v1/chat/completions", inference)
        async with client:
            request = asyncio.create_task(client.post("/v1/chat/completions", headers=SERVICE_HEADERS, json={}))
            await eof_started.wait()
            response = await client.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command("close", 1))
            assert (await response.json())["active_requests"] == 1
            rejected = await client.post("/v1/chat/completions", headers=SERVICE_HEADERS, json={})
            assert rejected.status == 503
            eof_finish.set()
            assert await (await request).read() == b"done"
            response = await client.get("/reef/lifecycle", headers=CONTROL_HEADERS)
            assert (await response.json())["active_requests"] == 0

    asyncio.run(run())


def test_cancelled_record_write_is_retained_and_submitted_once() -> None:
    async def run() -> None:
        writes = PendingWrites()
        started = threading.Event()
        finish = threading.Event()
        calls = []

        def persist() -> str:
            calls.append("write")
            started.set()
            if not finish.wait(timeout=5):
                raise TimeoutError("test write was not released")
            return "record"

        task = asyncio.create_task(writes.run(asyncio.to_thread(persist)))
        await asyncio.to_thread(started.wait, 5)
        try:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(writes.tasks) == 1
            assert not await writes.wait(0.01)
            assert calls == ["write"]
        finally:
            finish.set()
        assert await writes.wait(5)
        assert len(writes.tasks) == 0
        assert writes.failed_count == 0

    asyncio.run(run())


def test_cancelled_scenario_initialization_remains_in_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    finish = threading.Event()
    original = RequestService.inference_scenario

    def initialize(service: RequestService, parsed: RequestHeaders, *, record: bool) -> Scenario:
        started.set()
        if not finish.wait(5):
            raise TimeoutError("test initialization was not released")
        return original(service, parsed, record=record)

    monkeypatch.setattr(RequestService, "inference_scenario", initialize)
    dispatcher = routing_dispatcher(tmp_path)

    async def run() -> None:
        writes = PendingWrites()
        service = RequestService(dispatcher, writes=writes)
        task = asyncio.create_task(service.inference_operations({"x-reef-scenario": "coding"}))
        await asyncio.to_thread(started.wait, 5)
        try:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(writes.tasks) == 1
            assert not await writes.wait(0.01)
        finally:
            finish.set()
        assert await writes.wait(5)
        assert dispatcher.loaded_scenario("coding") is not None

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_forced_close_cancels_active_roots_but_not_finalization() -> None:
    async def run() -> None:
        acceptance = RequestAcceptance(settings(), PendingWrites())
        acceptance.control(command("open"))
        client = client_for(acceptance)
        started = asyncio.Event()
        finalization_started = asyncio.Event()
        finish = asyncio.Event()

        async def persist() -> None:
            finalization_started.set()
            await finish.wait()

        async def inference(request: web.Request) -> web.Response:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await acceptance.writes.run(persist())
            return web.json_response({"ok": True})

        client.app.router.add_post("/v1/chat/completions", inference)
        async with client:
            request = asyncio.create_task(client.post("/v1/chat/completions", headers=SERVICE_HEADERS, json={}))
            await started.wait()
            response = await client.post(
                "/reef/lifecycle", headers=CONTROL_HEADERS, json=command("close", 1, cancel_active=True)
            )
            assert response.status == 200
            await finalization_started.wait()
            response = await client.get("/reef/lifecycle", headers=CONTROL_HEADERS)
            state = await response.json()
            assert state["pending_writes"] == state["drain_pending_writes"] == 1
            finish.set()
            with pytest.raises(ClientConnectionError):
                await request
            assert await acceptance.writes.wait(1)
            assert acceptance.status()["forced_requests"] == 1

    asyncio.run(run())


def test_private_control_settings(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    raw = {
        "deployment_id": "deployment",
        "generation": "generation",
        "snapshot_version": "snapshot",
        "control_token": "control",
    }
    path.write_text(json.dumps(raw))
    path.chmod(0o600)
    assert LifecycleSettings.load(path).control_token == "control"
    assert "control_token" not in repr(LifecycleSettings.load(path))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="group or other"):
        LifecycleSettings.load(path)


@pytest.mark.parametrize(
    "make",
    [
        lambda: LifecycleSettings(1, "generation", "control"),
        lambda: LifecycleSettings("deployment", "generation", "control", None),
        lambda: LifecycleSettings("deployment", "generation", "control", body_timeout_seconds=True),
        lambda: LifecycleSettings("deployment", "generation", "control", finalization_timeout_seconds=math.inf),
        lambda: WriteAdmission(0, False),
        lambda: WriteAdmission(1, "no"),
    ],
)
def test_lifecycle_values_refuse_what_their_definitions_exclude(make: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        make()
    # Before a manifest is named, the snapshot version is not yet known.
    assert LifecycleSettings("deployment", "generation", "control").snapshot_version == ""


@pytest.mark.parametrize("stream", [False, True])
def test_managed_requests_keep_routing_records_and_refuse_record_imports(tmp_path: Path, stream: bool) -> None:
    async def run() -> None:
        async def inference(request: web.Request) -> web.Response:
            if stream:
                return web.Response(
                    text='data: {"id":"chat","choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n',
                    content_type="text/event-stream",
                )
            return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

        provider_app = web.Application()
        provider_app.router.add_get("/v1/models", provider_models)
        provider_app.router.add_post("/v1/chat/completions", inference)
        async with TestServer(provider_app) as provider, contextlib.AsyncExitStack() as stack:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")))
            dispatcher = routing_dispatcher(tmp_path)
            router = await stack.enter_async_context(running_router(adapter.snapshot))
            configured = LifecycleSettings(
                "deployment",
                "generation",
                "control",
                adapter.snapshot.version,
                selection_token=SELECTION_TOKEN,
                router_config_hash=ROUTER_HASH,
            )
            app = create_app(
                dispatcher,
                tokens=SERVICE_TOKEN,
                routing=adapter,
                preview=router.settings,
                lifecycle=configured,
                console_origins=["https://console.example"],
            )
            try:
                async with (
                    TestClient(TestServer(app)) as client,
                    TestClient(TestServer(app[CONTROL_APPLICATION])) as control,
                ):
                    unauthorized = await client.post("/v1/chat/completions", json={})
                    assert unauthorized.status == 401
                    opened = await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command("open"))
                    assert opened.status == 200, await opened.text()
                    headers = routing_headers()
                    response = await client.post(
                        "/v1/chat/completions",
                        headers=headers,
                        json={"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": stream},
                    )
                    assert response.status == 200, await response.text()
                    await response.read()
                    records = list(dispatcher.get_or_create_scenario("coding").records.replay("coding"))
                    assert len(records) == 1
                    assert records[0].payload["metadata"]["routing"]["execution_outcome"] == "completed"
                    await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command("close", 1))
                    # A routed deployment serves no record import; reports stay open while closed.
                    imported = {"agent_record_id": "imported", "request_type": "inference", "payload": {}}
                    assert (await client.post("/reef/records", headers=headers, json=imported)).status == 404
                    receipt = records[0].agent_record_id
                    report = await client.post(
                        "/reef/report", headers=headers, json={"score": 1, "references": [receipt]}
                    )
                    assert report.status == 200, await report.text()
                    state = await (await control.get("/reef/lifecycle", headers=CONTROL_HEADERS)).json()
                    assert state["active_requests"] == state["pending_writes"] == 0
            finally:
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_on_disconnect", [False, True])
def test_disconnected_managed_inference_keeps_exactly_one_record(tmp_path: Path, cancel_on_disconnect: bool) -> None:
    async def run() -> None:
        started = asyncio.Event()
        finish = asyncio.Event()

        async def inference(request: web.Request) -> web.Response:
            started.set()
            await finish.wait()
            return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

        provider_app = web.Application()
        provider_app.router.add_get("/v1/models", provider_models)
        provider_app.router.add_post("/v1/chat/completions", inference)
        async with TestServer(provider_app) as provider, contextlib.AsyncExitStack() as stack:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")))
            dispatcher = routing_dispatcher(tmp_path)
            router = await stack.enter_async_context(running_router(adapter.snapshot))
            app = create_app(
                dispatcher,
                tokens=SERVICE_TOKEN,
                routing=adapter,
                preview=router.settings,
                lifecycle=LifecycleSettings(
                    "deployment",
                    "generation",
                    "control",
                    adapter.snapshot.version,
                    selection_token=SELECTION_TOKEN,
                    router_config_hash=ROUTER_HASH,
                ),
            )
            # TestServer forces handler_cancellation=True. Use the production
            # AppRunner directly so both cancellation policies are exercised.
            runner = web.AppRunner(app, handler_cancellation=cancel_on_disconnect)
            await runner.setup()
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            host, port = listener.getsockname()
            await web.SockSite(runner, listener).start()
            control = await stack.enter_async_context(TestClient(TestServer(app[CONTROL_APPLICATION])))
            try:
                opened = await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command("open"))
                assert opened.status == 200, await opened.text()
                reader, writer = await asyncio.open_connection(host, port)
                payload = json.dumps({"model": "auto", "messages": [{"role": "user", "content": "hello"}]}).encode()
                headers = {
                    "Host": f"{host}:{port}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(payload)),
                    **routing_headers(),
                }
                request = (
                    "POST /v1/chat/completions HTTP/1.1\r\n"
                    + "".join(f"{name}: {value}\r\n" for name, value in headers.items())
                    + "\r\n"
                )
                writer.write(request.encode() + payload)
                await writer.drain()
                await started.wait()
                writer.close()
                await writer.wait_closed()
                # Allow the transport's disconnect callback to run before
                # the provider finishes. Both policies must retain one record.
                await asyncio.sleep(0.03)
                finish.set()
                async with asyncio.timeout(2):
                    while True:
                        records = list(dispatcher.get_or_create_scenario("coding").records.replay("coding"))
                        if records:
                            break
                        await asyncio.sleep(0.01)
                assert len(records) == 1
                expected = "cancelled" if cancel_on_disconnect else "completed"
                assert records[0].payload["metadata"]["routing"]["execution_outcome"] == expected
                await reader.read()
            finally:
                finish.set()
                await runner.cleanup()
                dispatcher.close()

    asyncio.run(run())


def test_managed_server_keeps_control_on_its_socket_and_never_dispatches_for_a_gone_client(tmp_path: Path) -> None:
    """The production server: Unix-only control (0600), public TCP, and real disconnect during selection.

    The fake router holds the preview; the client's TCP connection closes; the router's answer comes later.
    The provider must never be called and no record is kept.
    """
    import signal
    import stat
    import tempfile

    from aiohttp import UnixConnector

    async def run() -> None:
        provider_calls: list[object] = []

        async def inference(request: web.Request) -> web.Response:
            provider_calls.append(await request.json())
            return web.json_response({"choices": [{"message": {"content": "late"}, "finish_reason": "stop"}]})

        provider_app = web.Application()
        provider_app.router.add_get("/v1/models", provider_models)
        provider_app.router.add_post("/v1/chat/completions", inference)
        directory = Path(tempfile.mkdtemp(prefix="reef-ctl-"))
        directory.chmod(0o700)
        control_socket = directory / "control.sock"
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        async with TestServer(provider_app) as provider, contextlib.AsyncExitStack() as stack:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")))
            dispatcher = routing_dispatcher(tmp_path)
            stack.callback(dispatcher.close)
            router = await stack.enter_async_context(running_router(adapter.snapshot))
            lifecycle = LifecycleSettings(
                "deployment",
                "generation",
                "control",
                adapter.snapshot.version,
                selection_token=SELECTION_TOKEN,
                router_config_hash=ROUTER_HASH,
                control_socket=str(control_socket),
            )
            app = create_app(
                dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings, lifecycle=lifecycle
            )
            server = asyncio.create_task(
                serve_managed(
                    app,
                    app[CONTROL_APPLICATION],
                    host="127.0.0.1",
                    port=port,
                    control_socket=control_socket,
                    shutdown_timeout_seconds=1,
                )
            )
            try:
                async with asyncio.timeout(5):
                    while True:
                        try:
                            _, writer = await asyncio.open_connection("127.0.0.1", port)
                        except OSError:
                            await asyncio.sleep(0.02)
                            continue
                        writer.close()
                        break
                assert stat.S_IMODE(os.lstat(control_socket).st_mode) == 0o600
                async with ClientSession(connector=UnixConnector(path=str(control_socket))) as control:
                    opened = await control.post(
                        "http://control/reef/lifecycle", headers=CONTROL_HEADERS, json=command("open")
                    )
                    assert opened.status == 200, await opened.text()
                async with ClientSession() as public:
                    # No lifecycle route on public TCP, whatever the token.
                    for headers in (CONTROL_HEADERS, routing_headers()):
                        response = await public.get(f"http://127.0.0.1:{port}/reef/lifecycle", headers=headers)
                        assert response.status in (401, 404)
                router.hold = asyncio.Event()
                payload = json.dumps({"model": "auto", "messages": [{"role": "user", "content": "hello"}]}).encode()
                headers = {
                    "Host": f"127.0.0.1:{port}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(payload)),
                    **routing_headers(),
                }
                writer = (await asyncio.open_connection("127.0.0.1", port))[1]
                writer.write(
                    (
                        "POST /v1/chat/completions HTTP/1.1\r\n"
                        + "".join(f"{name}: {value}\r\n" for name, value in headers.items())
                        + "\r\n"
                    ).encode()
                    + payload
                )
                await writer.drain()
                await asyncio.wait_for(router.arrived.wait(), 5)
                writer.close()
                await writer.wait_closed()
                # Reef's handler is cancelled, which closes its preview connection; only then does the router answer.
                await asyncio.wait_for(router.disconnected.wait(), 5)
                router.hold.set()
                await asyncio.sleep(0.2)
                assert provider_calls == []
                loaded = dispatcher.loaded_scenario("coding")
                assert loaded is None or loaded.records.replay("coding") == ()
            finally:
                if not server.done():
                    os.kill(os.getpid(), signal.SIGTERM)
                await asyncio.wait_for(server, 10)
        # The server unlinked only its own socket; the private directory is the owner's to remove.
        assert not control_socket.exists() and directory.exists()
        directory.rmdir()

    asyncio.run(run())


@pytest.mark.parametrize("cleanup_mode", ["normal", "blocked", "closed_loop"])
def test_owner_pipe_eof_bounds_child_exit(cleanup_mode: str) -> None:
    owner_read, owner_write = os.pipe()
    code = """
import asyncio
import concurrent.futures
import sys
import threading
from aiohttp import web
from reef.service.lifecycle import LifecycleSettings, PendingWrites, RequestAcceptance, configure_owner_watch
app = web.Application()
state = RequestAcceptance(LifecycleSettings('test', 'generation', 'control', finalization_timeout_seconds=0.1), PendingWrites())
configure_owner_watch(app, state, int(sys.argv[1]))
async def stuck_cleanup(app):
    await asyncio.to_thread(threading.Event().wait)
if sys.argv[2] == 'blocked':
    app.on_cleanup.append(stuck_cleanup)
if sys.argv[2] == 'closed_loop':
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(threading.Event().wait)
web.run_app(app, host='127.0.0.1', port=0, shutdown_timeout=0.1, print=None)
if sys.argv[2] == 'closed_loop':
    print('loop-closed', flush=True)
"""
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(owner_read), cleanup_mode],
            pass_fds=(owner_read,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.close(owner_write)
        owner_write = -1
        try:
            stdout, stderr = child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise
        assert child.returncode in (0, 1), (stdout, stderr)
        if cleanup_mode != "normal":
            assert child.returncode == 1
        if cleanup_mode == "closed_loop":
            assert b"loop-closed" in stdout
        assert not stderr
    finally:
        os.close(owner_read)
        if owner_write >= 0:
            os.close(owner_write)


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_revision": True},
        {"control_revision": -1},
        {"control_revision": "1"},
        {"generation": "other"},
        {"deployment_id": "other"},
        {"action": "restart"},
        {"cancel_active": "yes"},
        {"cancel_active": True},
        {"extra": 1},
        {"open_deadline_monotonic": None},
        {"open_deadline_monotonic": True},
        {"open_deadline_monotonic": "1"},
        {"open_deadline_monotonic": 0},
        {"open_deadline_monotonic": float("nan")},
        {"open_deadline_monotonic": float("inf")},
    ],
)
def test_invalid_control_commands_do_not_change_acceptance(overrides: dict[str, object]) -> None:
    acceptance = RequestAcceptance(settings(), PendingWrites())
    with pytest.raises(web.HTTPError):
        acceptance.control({**command("open"), **overrides})
    assert acceptance.control_revision == -1
    assert not acceptance.accepting


def test_an_open_past_its_deadline_never_opens_but_an_applied_open_still_answers() -> None:
    acceptance = RequestAcceptance(settings(), PendingWrites())
    with pytest.raises(web.HTTPServiceUnavailable):
        acceptance.control(command("open", open_deadline_monotonic=time.monotonic() - 1))
    # An expired open changes nothing; the revision stays free.
    assert (acceptance.control_revision, acceptance.accepting) == (-1, False)
    applied = command("open", 1, open_deadline_monotonic=time.monotonic() + 0.05)
    assert acceptance.control(applied)["accepting"] is True
    time.sleep(0.1)
    # The owner's retry after the deadline reads the open it already applied.
    assert acceptance.control(applied)["control_revision"] == 1
    with pytest.raises(web.HTTPConflict):
        acceptance.control({**applied, "open_deadline_monotonic": time.monotonic() + 60})
    # Only an open carries a deadline.
    with pytest.raises(web.HTTPBadRequest):
        acceptance.control(command("close", 2, open_deadline_monotonic=time.monotonic() + 60))
    assert (acceptance.control_revision, acceptance.accepting) == (1, True)


class ScriptedReadiness(RoutingPublisher):
    """Readiness that passes after ``seconds``, waiting on the event loop or, with ``blocking``, holding it.

    Only the readiness part of a publisher: nothing is published (``current`` is None) and ``publish`` is unsupported.
    """

    def __init__(self, seconds: float, *, blocking: bool = False) -> None:
        self.seconds = seconds
        self.blocking = blocking
        self.checks = 0
        self.started = asyncio.Event()

    def current(self) -> None:
        return None

    def publish(self, manifest_path: str, snapshot_version: str, router_config_hash: str) -> None:
        raise NotImplementedError

    async def check_ready(self) -> None:
        self.checks += 1
        self.started.set()
        if self.blocking:
            # The event loop cannot run the open's deadline timer while this thread sleeps.
            time.sleep(self.seconds)
        else:
            await asyncio.sleep(self.seconds)


def routed_acceptance(readiness: ScriptedReadiness) -> RequestAcceptance:
    acceptance = RequestAcceptance(
        LifecycleSettings("deployment", "generation", "control", "snapshot", router_config_hash=ROUTER_HASH),
        PendingWrites(),
    )
    acceptance.publisher = readiness
    return acceptance


def test_readiness_that_holds_the_event_loop_past_the_deadline_cannot_open() -> None:
    async def run() -> None:
        readiness = ScriptedReadiness(1.0, blocking=True)
        acceptance = routed_acceptance(readiness)
        async with TestClient(TestServer(create_control_app(acceptance))) as control:
            late = command("open", open_deadline_monotonic=time.monotonic() + 0.5)
            response = await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=late)
            assert response.status == 503, await response.text()
        assert readiness.checks == 1
        assert (acceptance.control_revision, acceptance.accepting) == (-1, False)

    asyncio.run(run())


@pytest.mark.parametrize("interruption", ["stop", "owner_lost"])
def test_a_stop_waits_for_a_running_open_and_owner_loss_refuses_it(interruption: str) -> None:
    async def run() -> None:
        readiness = ScriptedReadiness(0.3)
        acceptance = routed_acceptance(readiness)
        async with TestClient(TestServer(create_control_app(acceptance))) as control:

            async def lifecycle(body: dict[str, object]) -> int:
                return (await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=body)).status

            opening = asyncio.create_task(lifecycle(command("open")))
            await asyncio.wait_for(readiness.started.wait(), 5)
            if interruption == "stop":
                # Commands are serialized: the stop runs once the open finishes, and then it wins.
                assert await lifecycle(command("stop", 1)) == 200
                assert await opening == 200
            else:
                acceptance.owner_lost()
                assert await opening == 409
        assert acceptance.stopping and not acceptance.accepting
        assert readiness.checks == 1

    asyncio.run(run())


@dataclass
class SocketChild:
    """A running routed child: its control socket, its acceptance and the fake router it routes through."""

    socket_path: Path
    acceptance: RequestAcceptance
    router: FakeSemanticRouter
    #: The provider of every model-list check the child started, and every lifecycle POST it received.
    checks: list[str]
    posts: list[str]


@contextlib.asynccontextmanager
async def child_on_socket(tmp_path: Path, delay_seconds: float, timeout_s: float) -> AsyncIterator[SocketChild]:
    """A routed child whose private control listens on a Unix socket without handler cancellation, as in production.

    Its two targets have separate provider endpoints; each answers a model-list check after ``delay_seconds``.
    """
    checks: list[str] = []
    posts: list[str] = []

    def provider(name: str) -> web.Application:
        async def models(request: web.Request) -> web.Response:
            checks.append(name)
            await asyncio.sleep(delay_seconds)
            return await provider_models(request)

        app = web.Application()
        app.router.add_get("/v1/models", models)
        return app

    @web.middleware
    async def count_posts(
        request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        if request.method == "POST":
            posts.append(request.path)
        return await handler(request)

    # A short directory, since a Unix socket path is limited in length; its exit also removes the socket.
    with tempfile.TemporaryDirectory(prefix="reef-ctl-") as directory:
        async with (
            TestServer(provider("student")) as student,
            TestServer(provider("teacher")) as teacher,
            contextlib.AsyncExitStack() as stack,
        ):
            adapter = routing_adapter(
                tmp_path, str(student.make_url("")), teacher_url=str(teacher.make_url("")), timeout_s=timeout_s
            )
            dispatcher = routing_dispatcher(tmp_path)
            stack.callback(dispatcher.close)
            router = await stack.enter_async_context(running_router(adapter.snapshot, timeout_seconds=1))
            lifecycle = LifecycleSettings(
                "deployment",
                "generation",
                "control",
                adapter.snapshot.version,
                selection_token=SELECTION_TOKEN,
                router_config_hash=ROUTER_HASH,
            )
            app = create_app(
                dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings, lifecycle=lifecycle
            )
            # The public application's startup starts the preview client that readiness uses.
            await stack.enter_async_context(TestServer(app))
            control_app = app[CONTROL_APPLICATION]
            control_app.middlewares.append(count_posts)
            runner = web.AppRunner(control_app)
            await runner.setup()
            stack.push_async_callback(runner.cleanup)
            socket_path = Path(directory) / "control.sock"
            await web.UnixSite(runner, str(socket_path)).start()
            yield SocketChild(socket_path, app[REQUEST_ACCEPTANCE], router, checks, posts)


def test_an_open_with_two_slow_healthy_providers_opens_on_its_first_attempt(tmp_path: Path) -> None:
    """The reviewed case: sequential provider checks outlast the old budget of only Scenario previews (11 s here)."""
    policy = RoutingDeploymentSettings(deployment_id="unit-routing", selection_timeout_seconds=1)
    controller = RoutingDeploymentController(
        ServiceConfig(recipe="base", routing_deployment=policy, inference_timeout_s=6),
        FakeRuntime([]),
        registry_root=tmp_path / "registry",
    )

    async def run() -> None:
        async with child_on_socket(tmp_path, delay_seconds=5.6, timeout_s=6) as child:
            allowance = controller.open_timeout_seconds(child.router.snapshot)
            client = ReefLifecycleClient(
                child.socket_path, "deployment", "generation", "control", open_timeout_seconds=allowance
            )
            started = time.monotonic()
            status = await asyncio.to_thread(client.request, "open")
            elapsed = time.monotonic() - started
            assert (status["accepting"], status["control_revision"]) == (True, 1)
            # One attempt and one check per provider: a retry that read an applied open would hide a short budget.
            assert child.posts == ["/reef/lifecycle"]
            assert child.checks == ["student", "teacher"]
            assert len(child.router.previews) == 1
            assert 11 < elapsed < allowance

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["provider_check", "preview"])
def test_an_open_whose_deadline_passes_during_readiness_stays_closed(tmp_path: Path, phase: str) -> None:
    async def run() -> None:
        async with child_on_socket(
            tmp_path, delay_seconds=0.8 if phase == "provider_check" else 0, timeout_s=6
        ) as child:
            if phase == "preview":
                child.router.hold = asyncio.Event()
            # Deliberately shorter than readiness: the owner stops waiting, and the child must not open later.
            client = ReefLifecycleClient(
                child.socket_path, "deployment", "generation", "control", open_timeout_seconds=0.4
            )
            with pytest.raises(RoutingControlError) as error:
                await asyncio.to_thread(client.request, "open")
            # The child refused the expired open, whether in the reply to the first attempt or to a retry.
            assert error.value.status_code == 503
            if phase == "preview":
                # The deadline cancelled the held preview and closed its connection.
                await asyncio.wait_for(child.router.disconnected.wait(), 5)
            await asyncio.sleep(1.0)
            status = child.acceptance.status()
            assert (status["accepting"], status["control_revision"]) == (False, -1)
            # Readiness ran once, for the first attempt only.
            assert child.checks == (["student"] if phase == "provider_check" else ["student", "teacher"])
            assert len(child.router.previews) == (0 if phase == "provider_check" else 1)

    asyncio.run(run())
