"""Clients call Reef directly; automatic requests select through one routing preview.

The router here is ``FakeSemanticRouter``, a fake preview listener on
loopback, and the provider is a deterministic local aiohttp app. No Docker or
Semantic Router process runs. These tests check what Reef sends, when it calls
each side, and what it records; they say nothing about the real router's
answers.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient, TestServer
from multidict import CIMultiDict
from tests.reef_service.routing.routing_fakes import (
    OPEN_DEADLINE,
    ROUTER_HASH,
    SELECTION_TOKEN,
    SERVICE_TOKEN,
    FakeSemanticRouter,
    routing_dispatcher,
    routing_headers,
    running_router,
)

from reef.dispatcher import Dispatcher
from reef.service import routing_preview
from reef.service.app import CONTROL_APPLICATION, MAX_REQUEST_BYTES, create_app
from reef.service.assembly import build_app
from reef.service.deploy.service_config import ServiceConfig
from reef.service.errors import translate_error
from reef.service.lifecycle import REQUEST_ACCEPTANCE, LifecycleSettings
from reef.service.routing import RoutingAdapter, RoutingRequestError, RoutingSnapshot
from reef.service.routing_preview import EXPECTED_HASH_HEADER, PREVIEW_BODY_LIMIT_BYTES

pytestmark = pytest.mark.unit
NEW_ROUTER_HASH = "2" * 64
CONTROL_HEADERS = {"Authorization": "Bearer control"}
PROVIDER_MODELS = {"student": "student-v1", "teacher": "teacher-v2", "math-student": "math-v1", "retired": "old-v0"}


def write_manifest(path: Path, provider_url: str, *, keyword: str = "reef_teacher_required") -> Path:
    """Two Scenarios sharing a teacher; the coding Scenario also lists a disabled target."""
    targets = {
        target_id: {"provider_model": model, "revision": "r1", "url": provider_url, "enabled": target_id != "retired"}
        for target_id, model in PROVIDER_MODELS.items()
    }
    manifest = {
        "schema_version": 2,
        "automatic_model": "auto",
        "targets": targets,
        "scenarios": {
            "coding": {
                "default_target": "student",
                "targets": ["student", "teacher", "retired"],
                "rules": [{"name": "teacher_requested", "keywords": [keyword], "target": "teacher"}],
            },
            "math": {"default_target": "math-student", "targets": ["math-student", "teacher"]},
        },
        "deployment": {
            "semantic_router_image": "semantic-router@sha256:" + "a" * 64,
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@dataclass
class Provider:
    """A deterministic CPU provider that records every request body it receives."""

    received: list[dict[str, object]] = field(default_factory=list)

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": model} for model in PROVIDER_MODELS.values()]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received.append(payload)
        if payload.get("stream") is True:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"id":"chat-1","choices":[{"index":0,"delta":{"content":"a"}}]}\n\n')
            await response.write(b"data: [DONE]\n\n")
            return response
        return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})


@dataclass(frozen=True)
class RoutedService:
    """A routed Reef app with its fake router, provider and dispatcher."""

    client: TestClient
    #: The private control application, here over loopback TCP for the test; production serves it on a socket.
    control: TestClient | None
    router: FakeSemanticRouter
    provider: Provider
    provider_url: str
    adapter: RoutingAdapter
    dispatcher: Dispatcher

    def records(self, scenario: str = "coding") -> list[Mapping[str, object]]:
        loaded = self.dispatcher.loaded_scenario(scenario)
        return [] if loaded is None else [record.payload for record in loaded.records.replay(scenario)]

    async def chat(self, body: object, **headers: str) -> tuple[int, str, Mapping[str, str]]:
        response = await self.client.post("/v1/chat/completions", headers={**routing_headers(), **headers}, json=body)
        return response.status, await response.text(), response.headers

    async def lifecycle(self, body: Mapping[str, object]) -> ClientResponse:
        if self.control is None:
            raise RuntimeError("this routed service is not managed")
        return await self.control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=body)


@contextlib.asynccontextmanager
async def routed_service(
    tmp_path: Path,
    *,
    managed: bool = False,
    keyword: str = "reef_teacher_required",
    timeout_seconds: int = 5,
    reply_changes: Mapping[str, object] | None = None,
) -> AsyncIterator[RoutedService]:
    """Start a provider, a fake router and a routed Reef; ``managed`` adds the private lifecycle control."""
    provider = Provider()
    # Pinned requests may exceed the router's 10 MiB preview limit; the provider accepts them like Reef does.
    provider_app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    provider_app.router.add_get("/v1/models", provider.models)
    provider_app.router.add_post("/v1/chat/completions", provider.chat)
    async with TestServer(provider_app) as provider_server:
        url = str(provider_server.make_url("")).rstrip("/")
        path = write_manifest(tmp_path / "routing.json", url, keyword=keyword)
        adapter = RoutingAdapter.from_manifest(path, router_config_hash=ROUTER_HASH, environ={})
        dispatcher = routing_dispatcher(tmp_path)
        lifecycle = None
        if managed:
            lifecycle = LifecycleSettings(
                "deployment",
                "generation",
                "control",
                adapter.snapshot.version,
                body_timeout_seconds=5,
                selection_token=SELECTION_TOKEN,
                router_config_hash=ROUTER_HASH,
            )
        try:
            async with running_router(adapter.snapshot, timeout_seconds=timeout_seconds) as router:
                router.reply_changes = dict(reply_changes or {})
                app = create_app(
                    dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings, lifecycle=lifecycle
                )
                async with TestClient(TestServer(app)) as client, contextlib.AsyncExitStack() as stack:
                    control = None
                    if managed:
                        control = await stack.enter_async_context(TestClient(TestServer(app[CONTROL_APPLICATION])))
                        # Opening previews every Scenario under the published runtime first.
                        opened = await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command("open"))
                        assert opened.status == 200, await opened.text()
                    yield RoutedService(client, control, router, provider, url, adapter, dispatcher)
        finally:
            dispatcher.close()


def command(action: str, revision: int = 0, **fields: object) -> dict[str, object]:
    return {
        "deployment_id": "deployment",
        "generation": "generation",
        "control_revision": revision,
        "action": action,
        **({"open_deadline_monotonic": OPEN_DEADLINE} if action == "open" else {}),
        **fields,
    }


def user_message(text: str) -> list[dict[str, object]]:
    return [{"role": "user", "content": text}]


REQUEST: dict[str, object] = {
    "model": "auto",
    "messages": [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}}],
        },
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "user", "content": "please answer"},
    ],
    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    "temperature": 0.3,
    "max_tokens": 7,
    "metadata": {"caller": "kept", "routing": {"target_id": "forged"}},
}


@pytest.mark.parametrize(
    ("model", "text", "target", "reason", "rule", "previews"),
    [
        ("auto", "please answer", "student", "default", "default", 1),
        ("auto", "reef_teacher_required", "teacher", "rule", "teacher_requested", 1),
        ("teacher", "reef_teacher_required", "teacher", "explicit_pin", None, 0),
    ],
)
def test_auto_previews_once_and_a_pin_never_asks_the_router(
    tmp_path: Path, model: str, text: str, target: str, reason: str, rule: str | None, previews: int
) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            body = json.loads(json.dumps({**REQUEST, "model": model}))
            body["messages"][-1]["content"] = text
            status, response_text, headers = await service.chat(body)
            assert status == 200, response_text
            # The provider gets the client's body unchanged except for its own model name.
            assert service.provider.received == [{**body, "model": PROVIDER_MODELS[target]}]
            assert len(service.router.selections) == previews
            if previews:
                [(router_headers, sent)] = service.router.selections
                assert router_headers["Authorization"] == f"Bearer {SELECTION_TOKEN}"
                assert router_headers[EXPECTED_HASH_HEADER] == ROUTER_HASH
                assert sent == {"model": "reef-entrypoint/coding", "messages": body["messages"]}
                assert SERVICE_TOKEN not in json.dumps(router_headers)
            [record] = service.records()
            assert record["model"] == PROVIDER_MODELS[target]
            assert record["metadata"]["caller"] == "kept"
            routing = record["metadata"]["routing"]
            assert (routing["target_id"], routing["reason"], routing["rule_name"]) == (target, reason, rule)
            assert routing["original_selector"] == model
            assert routing["snapshot_version"] == service.adapter.snapshot.version
            assert routing["router_config_hash"] == (ROUTER_HASH if previews else None)
            report = await service.client.post(
                "/reef/report",
                headers=routing_headers(),
                json={"score": 1, "references": [headers["x-reef-agent-record-id"]]},
            )
            assert report.status == 200, await report.text()

    asyncio.run(run())


def test_a_stream_shares_the_selection_and_its_receipt_links_feedback(tmp_path: Path) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            body = {**REQUEST, "stream": True, "messages": user_message("reef_teacher_required")}
            status, text, _ = await service.chat(body)
            assert status == 200
            receipt = next(
                json.loads(line.removeprefix("data: "))["reef"]["agent_record_id"]
                for line in text.splitlines()
                if line.startswith("data: {") and '"reef"' in line
            )
            assert text.endswith("data: [DONE]\n\n")
            assert service.provider.received == [{**body, "model": "teacher-v2"}]
            assert len(service.router.selections) == 1
            [record] = service.records()
            assert record["metadata"]["routing"]["router_config_hash"] == ROUTER_HASH
            report = await service.client.post(
                "/reef/report", headers=routing_headers(), json={"score": 0, "references": [receipt]}
            )
            assert report.status == 200, await report.text()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("headers", "body", "status", "code"),
    [
        ([("x-reef-routing-version", "forged")], None, 400, "reserved_header"),
        ([("x-selected-model", "teacher")], None, 400, "reserved_header"),
        ([("x-reef-sr-decision", "teacher_requested")], None, 400, "reserved_header"),
        ([("x-reef-scenario", "math")], None, 400, "duplicate_header"),
        ([("x-reef-release-id", "release-1")], None, 400, "release_pin"),
        ([("x-reef-scenario", None), ("x-reef-scenario", "unknown")], None, 403, "unknown_scenario"),
        ([("x-reef-scenario", None)], None, 400, "missing_scenario"),
        ([], b'{"model":"math-student","messages":[{"role":"user","content":"x"}]}', 403, "target_not_allowed"),
        ([], b'{"model":"retired","messages":[{"role":"user","content":"x"}]}', 503, "target_unavailable"),
        ([], b'{"model":"student-v1","messages":[{"role":"user","content":"x"}]}', 400, "unknown_target"),
        ([], b'{"model":"reef-entrypoint/coding","messages":[]}', 400, "unknown_target"),
        ([], b'{"model":"auto","model":"teacher","messages":[]}', 400, "repeated keys"),
        ([], b'{"model":"auto","messages":[],"temperature":NaN}', 400, "non-finite"),
        ([], b'{"model":"auto","messages":[{"role":"user","content":"x"}],"x":1e999}', 400, "non-finite"),
        ([], b'["model"]', 400, "one JSON object"),
    ],
)
def test_forged_conflicting_or_cross_scenario_requests_never_reach_a_model(
    tmp_path: Path, headers: list[tuple[str, str | None]], body: bytes | None, status: int, code: str
) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            sent = CIMultiDict(routing_headers())
            sent["Content-Type"] = "application/json"
            for name, value in headers:
                if value is None:
                    sent.popall(name)
                else:
                    sent.add(name, value)
            data = body or b'{"model":"auto","messages":[{"role":"user","content":"x"}]}'
            response = await service.client.post("/v1/chat/completions", headers=sent, data=data)
            assert response.status == status
            assert code in await response.text()
            assert service.provider.received == [] and service.router.selections == []
            assert "x-reef-agent-record-id" not in response.headers
            assert service.dispatcher.list_scenarios() == ()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/responses"),
        ("POST", "/v1/messages"),
        ("POST", "/reef/records"),
        ("POST", "/reef/scenarios/coding/evaluation/v1/chat/completions"),
        ("GET", "/reef/lifecycle"),
        ("POST", "/reef/lifecycle"),
    ],
)
def test_a_routed_deployment_serves_only_chat_reports_and_health(tmp_path: Path, method: str, path: str) -> None:
    async def run() -> None:
        async with routed_service(tmp_path, managed=True) as service:
            response = await service.client.request(method, path, headers=routing_headers(), json={"model": "auto"})
            assert response.status == 404
            # The lifecycle stays private even with its own token; on public TCP that token opens nothing.
            control = await service.client.request(method, path, headers=CONTROL_HEADERS, json={})
            assert control.status == 401
            assert service.provider.received == [] and service.router.selections == []
            assert service.dispatcher.list_scenarios() == ()
            assert (await service.client.get("/healthz")).status == 200

    asyncio.run(run())


@pytest.mark.parametrize(
    ("failure", "changes", "status", "code"),
    [
        ((429, "OVERLOADED"), {}, 429, "routing_overloaded"),
        ((504, "REQUEST_TIMEOUT"), {}, 504, "routing_timeout"),
        ((400, "INVALID_INPUT"), {}, 400, "routing_input_rejected"),
        ((503, "SHUTTING_DOWN"), {}, 503, "routing_unavailable"),
        (None, {"selected_model": "math-student"}, 502, "routing_selection"),
        (None, {"selection_status": "fallback"}, 502, "routing_selection"),
        (None, {"signal_errors": {"keyword:teacher_requested": "failed"}}, 502, "routing_selection"),
    ],
)
def test_router_failures_have_reef_statuses_and_create_no_record(
    tmp_path: Path, failure: tuple[int, str] | None, changes: dict[str, object], status: int, code: str
) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            service.router.failure, service.router.reply_changes = failure, changes
            response_status, text, _ = await service.chat({"model": "auto", "messages": user_message("x")})
            assert response_status == status
            assert text.startswith(code) and "fake router error" not in text
            assert len(service.router.selections) == 1
            assert service.provider.received == [] and service.records() == []

    asyncio.run(run())


def test_a_stale_router_runtime_is_refused_without_selecting_again(tmp_path: Path) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            service.router.runtime_hash = NEW_ROUTER_HASH
            status, text, _ = await service.chat({"model": "auto", "messages": user_message("x")})
            assert status == 503 and text.startswith("routing_runtime_changed")
            assert len(service.router.selections) == 1
            assert service.provider.received == [] and service.records() == []

    asyncio.run(run())


def test_an_oversized_preview_is_refused_locally_while_the_same_body_can_pin(tmp_path: Path) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            large = {"model": "auto", "messages": user_message("x" * PREVIEW_BODY_LIMIT_BYTES)}
            status, text, _ = await service.chat(large)
            assert status == 413 and text.startswith("routing_input_too_large")
            assert service.router.selections == [] and service.provider.received == []
            assert service.records() == []
            status, text, _ = await service.chat({**large, "model": "student"})
            assert status == 200, text
            assert len(service.provider.received) == 1 and service.router.selections == []

    asyncio.run(run())


def test_a_preview_timeout_never_dispatches_late(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing_preview, "RESPONSE_WRITE_ALLOWANCE_SECONDS", 0.0)

    async def run() -> None:
        async with routed_service(tmp_path, timeout_seconds=1) as service:
            service.router.hold = asyncio.Event()
            status, text, _ = await service.chat({"model": "auto", "messages": user_message("x")})
            assert status == 504 and text.startswith("routing_timeout")
            await asyncio.wait_for(service.router.disconnected.wait(), 5)
            service.router.hold.set()
            await asyncio.sleep(0.05)
            assert len(service.router.selections) == 1
            assert service.provider.received == [] and service.records() == []

    asyncio.run(run())


def test_a_cancelled_request_closes_its_preview_and_never_dispatches(tmp_path: Path) -> None:
    async def run() -> None:
        async with routed_service(tmp_path) as service:
            service.router.hold = asyncio.Event()
            request = asyncio.create_task(service.chat({"model": "auto", "messages": user_message("x")}))
            await asyncio.wait_for(service.router.arrived.wait(), 5)
            # Dropping the client connection cancels Reef's handler (TestServer cancels on disconnect).
            request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request
            await asyncio.wait_for(service.router.disconnected.wait(), 5)
            service.router.hold.set()
            await asyncio.sleep(0.05)
            assert service.provider.received == [] and service.records() == []

    asyncio.run(run())


async def slow_upload(started: asyncio.Event, finish: asyncio.Event, body: bytes) -> AsyncIterator[bytes]:
    yield body[:10]
    started.set()
    await finish.wait()
    yield body[10:]


async def start_slow_chat(service: RoutedService, body: bytes) -> tuple[asyncio.Task[ClientResponse], asyncio.Event]:
    """A chat request that Reef has admitted while its upload stops after the first bytes until ``finish`` is set.

    The client having sent bytes does not mean Reef has read the headers, so this waits for Reef's own
    admission, which fixes the request's routing policy.
    """
    started, finish = asyncio.Event(), asyncio.Event()
    request = asyncio.create_task(
        service.client.post(
            "/v1/chat/completions",
            headers={**routing_headers(), "Content-Type": "application/json"},
            data=slow_upload(started, finish, body),
        )
    )
    acceptance = service.client.app[REQUEST_ACCEPTANCE]
    async with asyncio.timeout(5):
        await started.wait()
        while not acceptance.active:
            await asyncio.sleep(0.01)
    return request, finish


def test_an_admitted_upload_keeps_its_policy_and_holds_activation(tmp_path: Path) -> None:
    async def run() -> None:
        async with routed_service(tmp_path, managed=True) as service:
            new_path = write_manifest(tmp_path / "new.json", service.provider_url, keyword="escalate")
            body = json.dumps({"model": "auto", "messages": user_message("escalate")}).encode()
            request, finish = await start_slow_chat(service, body)
            closed = await service.lifecycle(command("close", 1))
            assert (await closed.json())["active_requests"] == 1
            activation = {
                "manifest_path": str(new_path),
                "snapshot_version": RoutingSnapshot.load(new_path).version,
                "router_config_hash": NEW_ROUTER_HASH,
                "expected_snapshot_version": service.adapter.snapshot.version,
            }
            refused = await service.lifecycle(command("activate", 2, **activation))
            assert refused.status == 409
            finish.set()
            response = await request
            assert response.status == 200, await response.text()
            [(router_headers, sent)] = service.router.selections
            assert router_headers[EXPECTED_HASH_HEADER] == ROUTER_HASH
            assert sent["messages"] == user_message("escalate")
            # The admitted request kept the old policy, which has no "escalate" rule.
            [record] = service.records()
            routing = record["metadata"]["routing"]
            assert (routing["target_id"], routing["snapshot_version"]) == ("student", service.adapter.snapshot.version)
            activated = await service.lifecycle(command("activate", 3, **activation))
            assert activated.status == 200, await activated.text()
            state = await activated.json()
            assert state["snapshot_version"] == activation["snapshot_version"]
            assert state["router_config_hash"] == NEW_ROUTER_HASH and state["active_requests"] == 0
            # Reopening previews every Scenario under the new runtime; the router still runs the old one.
            assert (await service.lifecycle(command("open", 4))).status == 503
            service.router.snapshot = RoutingSnapshot.load(new_path)
            service.router.runtime_hash = NEW_ROUTER_HASH
            reopened = await service.lifecycle(command("open", 5))
            assert reopened.status == 200, await reopened.text()
            assert (await reopened.json())["accepting"] is True

    asyncio.run(run())


def test_a_managed_child_starts_closed_and_opens_only_after_a_verified_publication(tmp_path: Path) -> None:
    async def run() -> None:
        provider = Provider()
        provider_app = web.Application()
        provider_app.router.add_get("/v1/models", provider.models)
        provider_app.router.add_post("/v1/chat/completions", provider.chat)
        async with TestServer(provider_app) as provider_server:
            url = str(provider_server.make_url("")).rstrip("/")
            path = write_manifest(tmp_path / "routing.json", url)
            adapter = RoutingAdapter.from_manifest(path, router_config_hash=None, environ={})
            dispatcher = routing_dispatcher(tmp_path)
            lifecycle = LifecycleSettings(
                "deployment", "generation", "control", adapter.snapshot.version, selection_token=SELECTION_TOKEN
            )
            try:
                # The router is not running yet when the child starts: nothing is checked or previewed.
                async with running_router(adapter.snapshot) as router:
                    app = create_app(
                        dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings, lifecycle=lifecycle
                    )
                    async with (
                        TestClient(TestServer(app)) as client,
                        TestClient(TestServer(app[CONTROL_APPLICATION])) as control,
                    ):
                        assert router.previews == []
                        response = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={"model": "auto", "messages": user_message("x")},
                        )
                        assert response.status == 503
                        refused = await control.post("/reef/lifecycle", headers=CONTROL_HEADERS, json=command("open"))
                        assert refused.status == 409 and router.previews == []
                        publication = {
                            "manifest_path": str(path),
                            "snapshot_version": adapter.snapshot.version,
                            "router_config_hash": ROUTER_HASH,
                            "expected_snapshot_version": adapter.snapshot.version,
                        }
                        published = await control.post(
                            "/reef/lifecycle", headers=CONTROL_HEADERS, json=command("activate", 1, **publication)
                        )
                        assert (await published.json())["router_config_hash"] == ROUTER_HASH
                        opened = await control.post(
                            "/reef/lifecycle", headers=CONTROL_HEADERS, json=command("open", 2)
                        )
                        assert opened.status == 200, await opened.text()
                        assert len(router.previews) == 2  # one readiness preview per Scenario
                        status, _, _ = await RoutedService(
                            client, control, router, provider, url, adapter, dispatcher
                        ).chat({"model": "auto", "messages": user_message("x")})
                        assert status == 200
                        assert router.selections[0][0][EXPECTED_HASH_HEADER] == ROUTER_HASH
                        unauthorized = await control.get("/reef/lifecycle", headers=routing_headers())
                        assert unauthorized.status == 401
            finally:
                dispatcher.close()

    asyncio.run(run())


def test_a_policy_published_during_upload_does_not_reach_the_admitted_request(tmp_path: Path) -> None:
    async def run() -> None:
        async with routed_service(tmp_path, managed=True) as service:
            body = json.dumps({"model": "auto", "messages": user_message("x")}).encode()
            request, finish = await start_slow_chat(service, body)
            # Publish directly, bypassing the drain check, to show what the request captured at admission.
            publisher = service.client.app[REQUEST_ACCEPTANCE].publisher
            assert publisher is not None
            publisher.publish(str(tmp_path / "routing.json"), service.adapter.snapshot.version, NEW_ROUTER_HASH)
            service.router.runtime_hash = NEW_ROUTER_HASH
            finish.set()
            response = await request
            # The admitted request asked for the runtime it captured, which the router no longer runs.
            assert response.status == 503
            assert (await response.text()).startswith("routing_runtime_changed")
            assert service.router.selections[0][0][EXPECTED_HASH_HEADER] == ROUTER_HASH
            assert service.provider.received == []
            status, text, _ = await service.chat(json.loads(body))
            assert status == 200, text
            assert service.router.selections[1][0][EXPECTED_HASH_HEADER] == NEW_ROUTER_HASH
            [record] = service.records()
            assert record["metadata"]["routing"]["router_config_hash"] == NEW_ROUTER_HASH

    asyncio.run(run())


@pytest.mark.parametrize("keyword", ["reef_teacher_required", "readiness"])
def test_startup_accepts_any_valid_readiness_selection(tmp_path: Path, keyword: str) -> None:
    """The readiness message may match a rule; any selection valid for the Scenario passes."""

    async def run() -> None:
        async with routed_service(tmp_path, keyword=keyword) as service:
            assert len(service.router.previews) == 2  # one per Scenario
            assert service.router.selections == []
            assert (await service.client.get("/healthz")).status == 200

    asyncio.run(run())


def test_startup_fails_when_a_readiness_preview_is_unusable(tmp_path: Path) -> None:
    async def run() -> None:
        with pytest.raises(RoutingRequestError, match="recipe"):
            async with routed_service(tmp_path, reply_changes={"recipe": "another"}):
                pass

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 403, 409, 413, 429, 502, 503, 504])
def test_every_routing_failure_status_has_an_http_answer(status: int) -> None:
    translated = translate_error(RoutingRequestError("message", code="routing_code", status=status))
    assert translated is not None and translated.status == status
    assert translated.text == "routing_code: message"


def control_file(tmp_path: Path, **fields: object) -> Path:
    path = tmp_path / f"control-{len(list(tmp_path.glob('control-*')))}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    identity = {"deployment_id": "unit", "generation": "g", "control_token": "control", "snapshot_version": "s"}
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({**identity, **fields}, stream)
    return path


def test_the_control_file_carries_the_selection_credential_and_router_runtime(tmp_path: Path) -> None:
    settings = LifecycleSettings.load(
        control_file(tmp_path, selection_token="selection-secret", router_config_hash=ROUTER_HASH)
    )
    assert (settings.selection_token, settings.router_config_hash) == ("selection-secret", ROUTER_HASH)
    assert "selection-secret" not in repr(settings)
    # An empty credential, a malformed hash and an unknown field are refused rather than ignored.
    for fields in ({"selection_token": ""}, {"router_config_hash": "A" * 64}, {"unknown_token": "value"}):
        with pytest.raises(ValueError):
            LifecycleSettings.load(control_file(tmp_path, **fields))


def test_routing_runs_only_as_a_managed_deployment(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "routing.json", "http://127.0.0.1:9")
    settings = ServiceConfig(recipe="recipe", routing_manifest=str(path), tokens=("service",))
    with pytest.raises(ValueError, match="managed routing deployment"):
        build_app(settings, environ={})
    # A child without its private control socket cannot be managed.
    without_control = LifecycleSettings("deployment", "generation", "control", "s", selection_token=SELECTION_TOKEN)
    with pytest.raises(ValueError, match="managed routing deployment"):
        build_app(settings, environ={}, lifecycle=without_control)
    managed = LifecycleSettings(
        "deployment", "generation", "control", "s", selection_token=SELECTION_TOKEN, control_socket="/tmp/reef.sock"
    )
    with pytest.raises(ValueError, match="sr_management_port"):
        build_app(
            ServiceConfig(recipe="recipe", routing_manifest=str(path), tokens=("service",), port=50053),
            environ={},
            lifecycle=managed,
        )


def test_the_tutorial_cpu_provider_serves_auto_pin_json_sse_and_feedback(tmp_path: Path) -> None:
    """The provider the tutorial starts passes Reef's provider check and answers both response modes."""
    script = Path(__file__).resolve().parents[3] / "tutorials/semantic-router/cpu_provider.py"
    spec = importlib.util.spec_from_file_location("reef_tutorial_cpu_provider", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("the tutorial CPU provider is missing")
    cpu_provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpu_provider)

    async def run() -> None:
        provider_app = cpu_provider.chat_completion_app(tuple(PROVIDER_MODELS.values()))
        async with TestServer(provider_app) as provider_server:
            path = write_manifest(tmp_path / "routing.json", str(provider_server.make_url("")).rstrip("/"))
            adapter = RoutingAdapter.from_manifest(path, router_config_hash=ROUTER_HASH, environ={})
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        receipts = []
                        for model, text, answered_by in (
                            ("auto", "hello", "student-v1"),
                            ("auto", "reef_teacher_required hello", "teacher-v2"),
                            ("teacher", "hello", "teacher-v2"),
                        ):
                            body = {"model": model, "messages": user_message(text)}
                            response = await client.post("/v1/chat/completions", headers=routing_headers(), json=body)
                            assert response.status == 200, await response.text()
                            answer = await response.json()
                            assert answer["choices"][0]["message"]["content"] == f"{answered_by} answered."
                            receipts.append(response.headers["x-reef-agent-record-id"])
                        body = {"model": "auto", "stream": True, "messages": user_message("reef_teacher_required")}
                        response = await client.post("/v1/chat/completions", headers=routing_headers(), json=body)
                        events = await response.text()
                        assert response.status == 200
                        assert "teacher-v2 answered." in events and events.endswith("data: [DONE]\n\n")
                        receipt_events = [
                            json.loads(line.removeprefix("data: "))
                            for line in events.splitlines()
                            if line.startswith("data: {") and '"reef"' in line
                        ]
                        receipts.append(receipt_events[0]["reef"]["agent_record_id"])
                        report = await client.post(
                            "/reef/report", headers=routing_headers(), json={"score": 1, "references": receipts}
                        )
                        assert report.status == 200, await report.text()
                    assert len(router.selections) == 3
            finally:
                dispatcher.close()

    asyncio.run(run())
