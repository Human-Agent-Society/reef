"""Clients call Reef directly; automatic requests select through one routing preview.

The router here is ``FakeSemanticRouter``, a fake preview listener on
loopback, and the provider is a deterministic local aiohttp app. No Docker or
Semantic Router process runs. These tests check what Reef sends, when it calls
each side, and what it records; they say nothing about the real router's
answers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import threading
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

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

from reef.core.records_types import ROLLOUT_HEADER, AgentRecord, RequestType
from reef.dispatcher import Dispatcher, ingest
from reef.scenario.scenario import Scenario
from reef.service import routing_preview
from reef.service.app import CONTROL_APPLICATION, REQUEST_SERVICE_KEY, create_app
from reef.service.assembly import build_app
from reef.service.deploy.service_config import ServiceConfig
from reef.service.errors import translate_error
from reef.service.lifecycle import INFERENCE_FINALIZATION, REQUEST_ACCEPTANCE, LifecycleSettings
from reef.service.routed_inference import DispatcherShadowWriter
from reef.service.routed_shadow import ShadowLimits, ShadowTasks
from reef.service.routing import RoutingAdapter, RoutingRequestError, RoutingSnapshot, check_policy_change
from reef.service.routing_preview import EXPECTED_HASH_HEADER, PREVIEW_BODY_LIMIT_BYTES
from reef.service.web import MAX_REQUEST_BYTES

pytestmark = pytest.mark.unit
NEW_ROUTER_HASH = "2" * 64
CONTROL_HEADERS = {"Authorization": "Bearer control"}
PROVIDER_MODELS = {"student": "student-v1", "teacher": "teacher-v2", "math-student": "math-v1", "retired": "old-v0"}


def write_manifest(
    path: Path,
    provider_url: str,
    *,
    keyword: str = "reef_teacher_required",
    domain: bool = False,
    default: str = "student",
    shadow: str | None = None,
    target_urls: Mapping[str, str] | None = None,
) -> Path:
    """Two Scenarios sharing a teacher; the coding Scenario also lists a disabled target.

    With ``domain``, each Scenario also has a ``math_request`` domain rule for the teacher, after any keyword
    rule. Its classifier names a directory that does not exist: serving requests never reads it. ``shadow``
    names the coding Scenario's shadow target; ``target_urls`` gives some targets their own endpoint, and the
    shadow-only target ``observer`` is in the catalog exactly when it names it.
    """
    urls = dict(target_urls or {})
    models = {**PROVIDER_MODELS, **({"observer": "observer-v1"} if "observer" in urls else {})}
    targets = {
        target_id: {
            "provider_model": model,
            "revision": "r1",
            "url": urls.get(target_id, provider_url),
            "enabled": target_id != "retired",
        }
        for target_id, model in models.items()
    }
    domain_rules = [{"name": "math_request", "domains": ["math"], "target": "teacher"}] if domain else []
    deployment: dict[str, object] = {"semantic_router_image": "semantic-router@sha256:" + "a" * 64}
    if domain:
        deployment["domain_classifier"] = {
            "model_directory": "/nonexistent/reef-domain-classifier",
            "threshold": 0.5,
            "files": {
                name: "sha256:" + digit * 64
                for name, digit in zip(
                    ("config.json", "tokenizer.json", "model.safetensors", "category_mapping.json"),
                    "1234",
                    strict=True,
                )
            },
        }
    manifest = {
        "schema_version": 2,
        "automatic_model": "auto",
        "targets": targets,
        "scenarios": {
            "coding": {
                "default_target": default,
                "targets": ["student", "teacher", "retired"],
                "rules": [{"name": "teacher_requested", "keywords": [keyword], "target": "teacher"}, *domain_rules],
                **({"shadow_target": shadow} if shadow is not None else {}),
            },
            "math": {"default_target": "math-student", "targets": ["math-student", "teacher"], "rules": domain_rules},
        },
        "deployment": deployment,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@dataclass
class Provider:
    """A deterministic CPU provider for every target: its request handlers, not a server.

    ``models`` lists every ``PROVIDER_MODELS`` value. ``chat`` appends each request body to ``received``, in
    arrival order, and answers whatever model it names: with ``"stream": true`` one SSE delta ``a`` and
    ``[DONE]``, otherwise one ``answer`` message. The aiohttp server that mounts these handlers owns the
    connections.
    """

    received: list[dict[str, object]] = field(default_factory=list)

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": model} for model in PROVIDER_MODELS.values()]})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received.append(payload)
        if payload.get("stream") is True:
            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "x-reef-agent-record-id": "provider-record",
                    "x-reef-artifact-id": "provider-artifact",
                    "x-reef-release-id": "provider-release",
                    "x-reef-artifact-version": "provider-version",
                }
            )
            await response.prepare(request)
            await response.write(b'data: {"id":"chat-1","choices":[{"index":0,"delta":{"content":"a"}}]}\n\n')
            await response.write(b"data: [DONE]\n\n")
            return response
        return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})


@dataclass(frozen=True)
class RoutedService:
    """A running routed Reef app and what ``routed_service`` assembled it from.

    ``client`` reaches the public app and ``control`` its private control app, or is None when unmanaged.
    ``router`` is the fake preview listener; ``provider`` serves every target at ``provider_url``.
    ``adapter`` is the adapter the app started with; a later publication may replace the app's.
    ``dispatcher`` stores the records. ``routed_service`` starts and closes all of them; this value owns
    none. ``records(scenario)`` replays that Scenario's stored payloads, or is empty if it is not loaded;
    ``chat`` posts one chat completion with the routing headers and returns its status, text and headers;
    ``lifecycle`` posts one control command and raises ``RuntimeError`` when unmanaged.
    """

    client: TestClient
    #: The private control application, here over loopback TCP for the test; production serves it on a socket.
    control: TestClient | None
    router: FakeSemanticRouter
    provider: Provider
    provider_url: str
    adapter: RoutingAdapter
    dispatcher: Dispatcher

    def records(self, scenario: str = "coding") -> list[Mapping[str, object]]:
        loaded = self.dispatcher.registry.get_loaded(scenario)
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
    domain: bool = False,
    default: str = "student",
    provider_chat: web.Handler | None = None,
    shadow: str | None = None,
    target_urls: Mapping[str, str] | None = None,
    shadow_limits: ShadowLimits | None = None,
    close_dispatcher: bool = False,
) -> AsyncIterator[RoutedService]:
    """Start a provider, a fake router and a routed Reef; ``managed`` adds the private lifecycle control.

    ``shadow`` and ``target_urls`` go to ``write_manifest``; ``shadow_limits`` and ``close_dispatcher`` to
    ``create_app``. The dispatcher is closed at the end either way; closing it again does nothing.
    """
    provider = Provider()
    # Pinned requests may exceed the router's 10 MiB preview limit; the provider accepts them like Reef does.
    provider_app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    provider_app.router.add_get("/v1/models", provider.models)
    provider_app.router.add_post("/v1/chat/completions", provider.chat if provider_chat is None else provider_chat)
    async with TestServer(provider_app) as provider_server:
        url = str(provider_server.make_url("")).rstrip("/")
        path = write_manifest(
            tmp_path / "routing.json",
            url,
            keyword=keyword,
            domain=domain,
            default=default,
            shadow=shadow,
            target_urls=target_urls,
        )
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
                    dispatcher,
                    tokens=SERVICE_TOKEN,
                    routing=adapter,
                    preview=router.settings,
                    lifecycle=lifecycle,
                    shadow_limits=shadow_limits,
                    close_dispatcher=close_dispatcher,
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
            status, text, headers = await service.chat(body)
            assert status == 200
            for header in (
                "x-reef-agent-record-id",
                "x-reef-artifact-id",
                "x-reef-release-id",
                "x-reef-artifact-version",
            ):
                assert header not in headers
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


def test_domain_rule_selections_are_served_and_recorded_per_scenario_like_keyword_rules(tmp_path: Path) -> None:
    def receipt_of(text: str, headers: Mapping[str, str], stream: bool) -> str:
        if not stream:
            return headers["x-reef-agent-record-id"]
        assert text.endswith("data: [DONE]\n\n")
        return next(
            json.loads(line.removeprefix("data: "))["reef"]["agent_record_id"]
            for line in text.splitlines()
            if line.startswith("data: {") and '"reef"' in line
        )

    async def run() -> None:
        async with routed_service(tmp_path, managed=True, domain=True) as service:
            # (Scenario, prediction, text, stream) -> (target, reason, rule)
            cases = [
                ("coding", "math", "solve x squared equals four", False, ("teacher", "rule", "math_request")),
                ("coding", "math", "solve x squared equals four", True, ("teacher", "rule", "math_request")),
                # An earlier keyword rule outranks a later domain rule that also matches.
                ("coding", "math", "reef_teacher_required solve it", False, ("teacher", "rule", "teacher_requested")),
                ("coding", "biology", "explain photosynthesis", False, ("student", "default", "default")),
                ("math", "math", "solve x squared equals four", True, ("teacher", "rule", "math_request")),
                ("math", "biology", "explain photosynthesis", False, ("math-student", "default", "default")),
            ]
            receipts: dict[str, list[str]] = {"coding": [], "math": []}
            for scenario, prediction, text, stream, _ in cases:
                service.router.predicted_domain = prediction
                body = {"model": "auto", "stream": stream, "messages": user_message(text)}
                status, response_text, headers = await service.chat(body, **{"x-reef-scenario": scenario})
                assert status == 200, response_text
                receipts[scenario].append(receipt_of(response_text, headers, stream))
            # A pin still bypasses the router, whatever the classifier would predict.
            previews = len(service.router.selections)
            pin = {"model": "teacher", "messages": user_message("solve")}
            pinned = await service.chat(pin, **{"x-reef-scenario": "math"})
            assert pinned[0] == 200 and len(service.router.selections) == previews
            for scenario in ("coding", "math"):
                expected = [outcome for name, *_, outcome in cases if name == scenario]
                routing = [record["metadata"]["routing"] for record in service.records(scenario)]
                selected = routing[: len(expected)]
                assert [(item["target_id"], item["reason"], item["rule_name"]) for item in selected] == expected
                assert all(item["router_config_hash"] == ROUTER_HASH for item in selected)
                assert all(item["policy_id"] == "static" for item in routing)
                report = await service.client.post(
                    "/reef/report",
                    headers={**routing_headers(), "x-reef-scenario": scenario},
                    json={"score": 1, "references": receipts[scenario]},
                )
                assert report.status == 200, await report.text()
            foreign = await service.client.post(
                "/reef/report",
                headers={**routing_headers(), "x-reef-scenario": "math"},
                json={"score": 1, "references": receipts["coding"][:1]},
            )
            assert foreign.status == 400

    asyncio.run(run())


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("rollout", ["1", "TrUe"])
def test_routed_rollout_calls_are_refused_before_selection(tmp_path: Path, managed: bool, rollout: str) -> None:
    async def run() -> None:
        async with routed_service(tmp_path, managed=managed) as service:
            status, text, headers = await service.chat(REQUEST, **{ROLLOUT_HEADER: rollout})
            assert status == 400 and "unsupported_rollout" in text
            assert service.router.selections == [] and service.provider.received == []
            assert "x-reef-agent-record-id" not in headers
            assert service.dispatcher.list_scenarios() == ()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("headers", "body", "status", "code"),
    [
        ([("x-reef-routing-version", "forged")], None, 400, "reserved_header"),
        ([("x-selected-model", "teacher")], None, 400, "reserved_header"),
        ([("x-reef-sr-decision", "teacher_requested")], None, 400, "reserved_header"),
        ([("x-reef-scenario", "math")], None, 400, "duplicate_header"),
        ([("x-reef-release-id", "release-1")], None, 400, "release_pin"),
        ([("x-reef-artifact-id", "artifact-1")], None, 400, "release_pin"),
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
        ("POST", "/reef/scenarios/coding/evaluations"),
        ("GET", "/reef/scenarios/coding/evaluations"),
        ("GET", "/reef/scenarios/coding/evaluations/id"),
        ("POST", "/reef/scenarios/coding/artifacts"),
        ("POST", "/reef/scenarios/coding/artifacts/release-1/v1/chat/completions"),
        ("POST", "/reef/scenarios/coding/update"),
        ("GET", "/reef/scenarios/coding/samples"),
        ("GET", "/reef/scenarios/coding/buffer"),
        ("POST", "/reef/scenarios/coding/buffer/insert"),
        ("POST", "/reef/scenarios/coding/buffer/remove"),
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
        async with routed_service(tmp_path, managed=True) as service:
            service.router.hold = asyncio.Event()
            request = asyncio.create_task(service.chat({"model": "auto", "messages": user_message("x")}))
            await asyncio.wait_for(service.router.arrived.wait(), 5)
            closed = await service.lifecycle(command("close_policy", 1))
            assert (await closed.json())["pending_selections"] == 1
            # Dropping the client connection cancels Reef's handler (TestServer cancels on disconnect).
            request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request
            await asyncio.wait_for(service.router.disconnected.wait(), 5)
            service.router.hold.set()
            assert service.client.app[REQUEST_ACCEPTANCE].status()["pending_selections"] == 0
            assert service.provider.received == [] and service.records() == []

    asyncio.run(run())


def test_a_failed_selection_releases_the_policy_before_its_error_response_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        sending, finish = asyncio.Event(), asyncio.Event()
        write_eof = web.Response.write_eof

        async def hold_error_response(response: web.Response, data: bytes = b"") -> None:
            if response.status == 400 and response.text and response.text.startswith("unknown_target:"):
                sending.set()
                await finish.wait()
            await write_eof(response, data)

        monkeypatch.setattr(web.Response, "write_eof", hold_error_response)
        async with routed_service(tmp_path, managed=True) as service, asyncio.timeout(5):
            invalid = asyncio.create_task(service.chat({"model": "unknown", "messages": user_message("hello")}))
            try:
                await sending.wait()
                closed = await service.lifecycle(command("close_policy", 1))
                state = await closed.json()
                assert state["active_requests"] == 1 and state["pending_selections"] == 0
            finally:
                finish.set()
                status, text, _ = await invalid
            assert status == 400 and text.startswith("unknown_target:")
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


@pytest.mark.parametrize("action", ["close", "close_policy"])
def test_an_admitted_upload_keeps_its_policy_and_holds_activation(tmp_path: Path, action: str) -> None:
    async def run() -> None:
        async with routed_service(tmp_path, managed=True) as service:
            new_path = write_manifest(tmp_path / "new.json", service.provider_url, keyword="escalate")
            body = json.dumps({"model": "auto", "messages": user_message("escalate")}).encode()
            request, finish = await start_slow_chat(service, body)
            closed = await service.lifecycle(command(action, 1))
            state = await closed.json()
            assert state["active_requests"] == state["pending_selections"] == 1
            scope = {"policy_close_revision": 1} if action == "close_policy" else {}
            activation = {
                "manifest_path": str(new_path),
                "snapshot_version": RoutingSnapshot.load(new_path).version,
                "router_config_hash": NEW_ROUTER_HASH,
                "expected_snapshot_version": service.adapter.snapshot.version,
            }
            refused = await service.lifecycle(command("activate", 2, **activation, **scope))
            assert refused.status == 409
            assert not service.router.selections
            service.router.hold = asyncio.Event()
            finish.set()
            await asyncio.wait_for(service.router.arrived.wait(), 5)
            refused = await service.lifecycle(command("activate", 2, **activation, **scope))
            assert refused.status == 409
            service.router.hold.set()
            response = await request
            assert response.status == 200, await response.text()
            [(router_headers, sent)] = service.router.selections
            assert router_headers[EXPECTED_HASH_HEADER] == ROUTER_HASH
            assert sent["messages"] == user_message("escalate")
            # The admitted request kept the old policy, which has no "escalate" rule.
            [record] = service.records()
            routing = record["metadata"]["routing"]
            assert (routing["target_id"], routing["snapshot_version"]) == ("student", service.adapter.snapshot.version)
            activated = await service.lifecycle(command("activate", 3, **activation, **scope))
            assert activated.status == 200, await activated.text()
            state = await activated.json()
            assert state["snapshot_version"] == activation["snapshot_version"]
            assert state["router_config_hash"] == NEW_ROUTER_HASH and state["active_requests"] == 0
            # Reopening previews every Scenario under the new runtime; the router still runs the old one.
            assert (await service.lifecycle(command("open", 4, **scope))).status == 503
            service.router.snapshot = RoutingSnapshot.load(new_path)
            service.router.runtime_hash = NEW_ROUTER_HASH
            reopened = await service.lifecycle(command("open", 5, **scope))
            assert reopened.status == 200, await reopened.text()
            assert (await reopened.json())["accepting"] is True

    asyncio.run(run())


async def switch_policy_during_generation(service: RoutedService, manifest: Path) -> str:
    """Close, publish and reopen while exactly one request has bound its V1 handler."""
    closed = await service.lifecycle(command("close_policy", 1))
    assert closed.status == 200, await closed.text()
    state = await closed.json()
    assert not state["accepting"] and state["active_requests"] == 1 and state["pending_selections"] == 0
    status, text, _ = await service.chat({"model": "auto", "messages": user_message("not admitted")})
    assert status == 503 and "not accepting requests" in text
    service.router.snapshot = RoutingSnapshot.load(manifest)
    service.router.runtime_hash = NEW_ROUTER_HASH
    activated = await service.lifecycle(
        command(
            "activate",
            2,
            policy_close_revision=1,
            manifest_path=str(manifest),
            snapshot_version=service.router.snapshot.version,
            router_config_hash=NEW_ROUTER_HASH,
            expected_snapshot_version=service.adapter.snapshot.version,
        )
    )
    assert activated.status == 200, await activated.text()
    opened = await service.lifecycle(command("open", 3, policy_close_revision=1))
    assert opened.status == 200, await opened.text()
    state = await opened.json()
    assert state["accepting"] and state["active_requests"] == 1
    return service.router.snapshot.version


def test_a_new_student_receipt_precedes_the_old_teacher_streams_last_token(tmp_path: Path) -> None:
    async def run() -> None:
        finish = asyncio.Event()
        first = b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'
        last = b'data: {"choices":[{"index":0,"delta":{"content":"last"}}]}\n\n'

        async def chat(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            if not payload.get("stream"):
                assert payload["model"] == PROVIDER_MODELS["student"]
                return web.json_response(
                    {"choices": [{"message": {"content": "student answer"}, "finish_reason": "stop"}]}
                )
            assert payload["model"] == PROVIDER_MODELS["teacher"]
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(first)
            await finish.wait()
            await response.write(last + b"data: [DONE]\n\n")
            return response

        async with routed_service(tmp_path, managed=True, default="teacher", provider_chat=chat) as service:
            payload = {"model": "auto", "messages": user_message("hello")}
            async with asyncio.timeout(5):
                old = await service.client.post(
                    "/v1/chat/completions", headers=routing_headers(), json={**payload, "stream": True}
                )
                try:
                    assert old.status == 200 and await old.content.readuntil(b"\n\n") == first
                    path = write_manifest(tmp_path / "new.json", service.provider_url, default="student")
                    version = await switch_policy_during_generation(service, path)
                    status, text, headers = await service.chat(payload)
                    assert status == 200 and "student answer" in text
                    student_receipt = headers["x-reef-agent-record-id"]
                finally:
                    finish.set()
                assert await old.content.readuntil(b"\n\n") == last
                frame = await old.content.readuntil(b"\n\n")
                teacher_receipt = json.loads(frame.removeprefix(b"data: "))["reef"]["agent_record_id"]
                assert await old.read() == b"data: [DONE]\n\n"
            records = service.dispatcher.get_or_create_scenario("coding").records.replay("coding")
            expected = [
                (student_receipt, "student", version, NEW_ROUTER_HASH),
                (teacher_receipt, "teacher", service.adapter.snapshot.version, ROUTER_HASH),
            ]
            for record, (receipt, target, snapshot, runtime_hash) in zip(records, expected, strict=True):
                detail = record.payload["metadata"]["routing"]
                assert record.agent_record_id == receipt
                assert (detail["target_id"], detail["provider_model"]) == (target, PROVIDER_MODELS[target])
                assert (detail["snapshot_version"], detail["router_config_hash"]) == (snapshot, runtime_hash)

    asyncio.run(run())


def test_an_old_json_request_retries_the_old_endpoint_after_policy_activation(tmp_path: Path) -> None:
    async def run() -> None:
        started, abort = asyncio.Event(), asyncio.Event()
        old_bodies = []

        async def old_provider(request: web.Request) -> web.Response:
            old_bodies.append(await request.json())
            if len(old_bodies) == 1:
                started.set()
                await abort.wait()
                return web.json_response({"choices": [{"finish_reason": "abort"}]})
            return web.json_response({"choices": [{"message": {"content": "old endpoint"}, "finish_reason": "stop"}]})

        endpoint_b = Provider()
        new_app = web.Application()
        new_app.router.add_get("/v1/models", endpoint_b.models)
        new_app.router.add_post("/v1/chat/completions", endpoint_b.chat)
        async with (
            TestServer(new_app) as new_server,
            routed_service(tmp_path, managed=True, default="teacher", provider_chat=old_provider) as service,
            asyncio.timeout(5),
        ):
            old_body = {"model": "auto", "messages": user_message("old request")}
            new_body = {"model": "auto", "messages": user_message("new request")}
            old_request = asyncio.create_task(service.chat(old_body))
            try:
                await started.wait()
                path = write_manifest(
                    tmp_path / "new.json", str(new_server.make_url("")).rstrip("/"), default="teacher"
                )
                version = await switch_policy_during_generation(service, path)
                status, text, new_headers = await service.chat(new_body)
                assert status == 200 and "answer" in text
            finally:
                abort.set()
                old_status, old_text, old_headers = await old_request
            assert old_status == 200 and "old endpoint" in old_text
            assert old_bodies == [{**old_body, "model": PROVIDER_MODELS["teacher"]}] * 2
            assert endpoint_b.received == [{**new_body, "model": PROVIDER_MODELS["teacher"]}]
            assert len(service.router.selections) == 2
            fresh, aborted, retried = service.dispatcher.get_or_create_scenario("coding").records.replay("coding")
            assert fresh.agent_record_id == new_headers["x-reef-agent-record-id"]
            assert retried.agent_record_id == old_headers["x-reef-agent-record-id"]
            for record, snapshot, runtime_hash in (
                (fresh, version, NEW_ROUTER_HASH),
                (aborted, service.adapter.snapshot.version, ROUTER_HASH),
                (retried, service.adapter.snapshot.version, ROUTER_HASH),
            ):
                detail = record.payload["metadata"]["routing"]
                assert (detail["target_id"], detail["snapshot_version"], detail["router_config_hash"]) == (
                    "teacher",
                    snapshot,
                    runtime_hash,
                )
            assert retried.payload["metadata"]["routing"]["prior_attempt_ids"] == [aborted.agent_record_id]

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


@pytest.fixture
def domain_smoke() -> ModuleType:
    script = Path(__file__).resolve().parents[3] / "tutorials/semantic-router/domain_smoke.py"
    spec = importlib.util.spec_from_file_location("reef_tutorial_domain_smoke", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("the tutorial domain smoke is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pid", [-1, 0, 1])
def test_domain_smoke_rejects_group_or_system_pids_before_doing_work(domain_smoke: ModuleType, pid: int) -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        domain_smoke.check(argparse.Namespace(reef_pid=pid))


def test_domain_smoke_does_not_forward_credentials_on_redirect(domain_smoke: ModuleType) -> None:
    async def run() -> None:
        forwarded: list[str] = []

        async def destination(request: web.Request) -> web.Response:
            forwarded.append(request.headers.get("Authorization", ""))
            return web.Response(text="unexpected")

        target = web.Application()
        target.router.add_route("*", "/", destination)
        async with TestServer(target) as other:

            async def redirect(request: web.Request) -> web.Response:
                raise web.HTTPFound(str(other.make_url("/")))

            app = web.Application()
            app.router.add_post("/v1/chat/completions", redirect)
            async with TestServer(app) as server:
                client = domain_smoke.SmokeClient(str(server.make_url("")), "test-token")
                status, _, _ = await asyncio.to_thread(client.post, "/v1/chat/completions", "code-opd", {})
                assert status == 302 and forwarded == []

    asyncio.run(run())


def test_the_tutorial_domain_smoke_writes_manifests_that_differ_only_in_a_reloadable_rule(
    tmp_path: Path, domain_smoke: ModuleType
) -> None:
    domain_smoke.write_manifests(tmp_path / "vela-domain", tmp_path / "manifests")
    initial = RoutingSnapshot.load(tmp_path / "manifests" / "manifest.yaml")
    updated = RoutingSnapshot.load(tmp_path / "manifests" / "manifest.updated.yaml")
    check_policy_change(initial, updated)
    assert initial.domain_classifier is not None and initial.domain_classifier == updated.domain_classifier
    assert initial.domain_classifier.files == domain_smoke.VELA_DOMAIN_FILES
    assert initial.domain_classifier.model_directory == str(tmp_path / "vela-domain")
    targets = {
        (name, snapshot.scenarios["code-opd"].target_for_rule("math_request"))
        for name, snapshot in (("initial", initial), ("updated", updated))
    }
    assert targets == {("initial", "code-teacher"), ("updated", "code-student")}
    assert initial.scenarios["math-opd"] == updated.scenarios["math-opd"]
    assert initial.scenarios["math-opd"].target_for_rule("math_request") == "math-teacher"


def test_the_tutorial_fake_provider_serves_auto_pin_json_sse_and_feedback(tmp_path: Path) -> None:
    """The provider the tutorial starts passes Reef's provider check and answers both response modes."""
    script = Path(__file__).resolve().parents[3] / "tutorials/semantic-router/fake_provider.py"
    spec = importlib.util.spec_from_file_location("reef_tutorial_fake_provider", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("the tutorial fake provider is missing")
    fake_provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake_provider)

    async def run() -> None:
        provider_app = fake_provider.chat_completion_app(tuple(PROVIDER_MODELS.values()))
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


# -- Shadow targets ------------------------------------------------------------------------------------------------


@dataclass
class EchoProvider:
    """A provider whose every answer names the model that answered it.

    ``received`` lists each request body in arrival order. A call to a model in ``held`` is appended to
    ``waiting`` and then waits until ``release`` is set. ``aborts`` gives, per model, how many of its calls
    first answer with an engine abort. ``replies`` replaces the answer of a model with its handler.
    """

    received: list[dict[str, object]] = field(default_factory=list)
    held: set[str] = field(default_factory=set)
    waiting: list[str] = field(default_factory=list)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    aborts: dict[str, int] = field(default_factory=dict)
    replies: dict[str, web.Handler] = field(default_factory=dict)

    async def chat(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received.append(payload)
        model = str(payload["model"])
        if model in self.held:
            self.waiting.append(model)
            await self.release.wait()
        if model in self.replies:
            return await self.replies[model](request)
        if self.aborts.get(model, 0):
            self.aborts[model] -= 1
            return web.json_response({"choices": [{"finish_reason": "abort"}]})
        text = f"{model} answered"
        if payload.get("stream") is True:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            chunk = {"id": "chat-1", "choices": [{"index": 0, "delta": {"content": text}}]}
            await response.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            return response
        return web.json_response({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    async def wait_for(self, count: int) -> None:
        """Return once ``count`` held calls arrived."""
        async with asyncio.timeout(5):
            while len(self.waiting) < count:
                await asyncio.sleep(0.01)


def shadow_tasks(service: RoutedService) -> ShadowTasks:
    return service.client.app[REQUEST_SERVICE_KEY].routed.shadows


async def shadows_finished(service: RoutedService) -> ShadowTasks:
    """Wait until every shadow task of ``service`` has ended and been released."""
    shadows = shadow_tasks(service)
    async with asyncio.timeout(5):
        while shadows.tasks:
            await asyncio.wait(set(shadows.tasks))
            await asyncio.sleep(0)
    return shadows


def records_by_purpose(service: RoutedService) -> tuple[list[AgentRecord], list[AgentRecord]]:
    """The coding Scenario's serve records and shadow records, each in arrival order."""
    records = service.dispatcher.get_or_create_scenario("coding").records.replay("coding")
    serve = [record for record in records if record.payload["metadata"]["routing"]["purpose"] == "serve"]
    shadow = [record for record in records if record.payload["metadata"]["routing"]["purpose"] == "shadow"]
    assert len(serve) + len(shadow) == len(records)
    return serve, shadow


def stream_receipt(text: str) -> str:
    return next(
        json.loads(line.removeprefix("data: "))["reef"]["agent_record_id"]
        for line in text.splitlines()
        if line.startswith("data: {") and '"reef"' in line
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("model", "primary", "shadowed"),
    [("auto", "teacher", True), ("teacher", "teacher", True), ("student", "student", False)],
)
def test_a_shadow_target_also_answers_but_only_the_primary_reaches_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, model: str, primary: str, shadowed: bool
) -> None:
    async def run() -> None:
        provider = EchoProvider()
        async with routed_service(
            tmp_path, default="teacher", shadow="student", provider_chat=provider.chat
        ) as service:
            append_record = ingest.append_record
            # Each record's purpose and whether its write belongs to inference finalization (read in its thread).
            finalizing: list[tuple[str, bool]] = []

            def observed_write(scenario: Scenario, item: AgentRecord) -> tuple[AgentRecord, bool]:
                if item.request_type is RequestType.INFERENCE:
                    finalizing.append((item.payload["metadata"]["routing"]["purpose"], INFERENCE_FINALIZATION.get()))
                return append_record(scenario, item)

            monkeypatch.setattr(ingest, "append_record", observed_write)
            body: dict[str, object] = {
                "model": model,
                "messages": user_message("hello"),
                "temperature": 0.25,
                "metadata": {"caller": "kept"},
            }
            if stream:
                body.update(stream=True, stream_options={"include_usage": True})
            status, text, headers = await service.chat(body)
            assert status == 200, text
            await shadows_finished(service)
            assert f"{PROVIDER_MODELS[primary]} answered" in text
            if primary != "student":
                assert "student-v1 answered" not in text
            receipt = stream_receipt(text) if stream else headers["x-reef-agent-record-id"]
            # The shadow target is the manifest's, never a selection: one preview for auto, none for a pin.
            assert len(service.router.selections) == (1 if model == "auto" else 0)
            [served], shadows = records_by_purpose(service)
            assert served.agent_record_id == receipt
            served_routing = served.payload["metadata"]["routing"]
            assert (served_routing["target_id"], served_routing["client_visible"]) == (primary, True)
            assert "primary_request_id" not in served_routing
            primary_body = {**body, "model": PROVIDER_MODELS[primary]}
            if not shadowed:
                # A shadow target that is the request's primary target is called once.
                assert provider.received == [primary_body] and shadows == []
                return
            shadow_body = {key: value for key, value in body.items() if key != "stream_options"}
            shadow_body.update(model="student-v1", **({"stream": False} if stream else {}))
            assert sorted(provider.received, key=lambda sent: str(sent["model"])) == [shadow_body, primary_body]
            # This app is unmanaged, so its request has no finalization scope; the detached shadow sets its own.
            assert sorted(finalizing) == [("serve", False), ("shadow", True)]
            [shadow] = shadows
            routing = shadow.payload["metadata"]["routing"]
            assert routing["purpose"] == "shadow" and routing["client_visible"] is False
            assert (routing["execution_outcome"], routing["delivery_outcome"]) == ("completed", "not_started")
            assert (routing["target_id"], routing["provider_model"], routing["reason"]) == (
                "student",
                "student-v1",
                "shadow_target",
            )
            assert (routing["original_selector"], routing["rule_name"]) == (model, None)
            assert routing["router_config_hash"] == (ROUTER_HASH if model == "auto" else None)
            assert (
                routing["snapshot_version"] == served_routing["snapshot_version"] == service.adapter.snapshot.version
            )
            assert routing["primary_request_id"] == served_routing["request_id"] != routing["request_id"]
            assert shadow.agent_record_id != served.agent_record_id
            assert shadow.payload["model"] == "student-v1" and shadow.payload["metadata"]["caller"] == "kept"
            assert "student-v1 answered" in json.dumps(shadow.payload["response"])
            report = await service.client.post(
                "/reef/report", headers=routing_headers(), json={"score": 1, "references": [receipt]}
            )
            assert report.status == 200, await report.text()

    asyncio.run(run())


def test_a_held_shadow_never_delays_its_primary_and_a_full_slot_skips_the_next_shadow(tmp_path: Path) -> None:
    async def run() -> None:
        provider = EchoProvider(held={"student-v1"}, aborts={"teacher-v2": 1})
        async with routed_service(
            tmp_path,
            default="teacher",
            shadow="student",
            provider_chat=provider.chat,
            shadow_limits=ShadowLimits(max_concurrency=1),
        ) as service:
            # The primary's abort retry stays inside its own request; the shadow starts once, outside it.
            status, text, _ = await service.chat({"model": "auto", "messages": user_message("first")})
            assert status == 200 and "teacher-v2 answered" in text
            await provider.wait_for(1)
            shadows = shadow_tasks(service)
            assert len(shadows.tasks) == 1
            assert [
                record.payload["metadata"]["routing"]["execution_outcome"] for record in records_by_purpose(service)[0]
            ] == ["aborted", "completed"]
            # No queue: the only slot is busy, so the next request skips its shadow and is served at once.
            status, text, _ = await service.chat({"model": "auto", "messages": user_message("second")})
            assert status == 200 and "teacher-v2 answered" in text
            assert shadows.skipped == {"saturated": 1, "closed": 0}
            provider.release.set()
            await shadows_finished(service)
            served, [shadow] = records_by_purpose(service)
            assert [sent["model"] for sent in provider.received].count("student-v1") == 1
            assert [sent["model"] for sent in provider.received].count("teacher-v2") == 3
            first_request = served[0].payload["metadata"]["routing"]["request_id"]
            assert served[1].payload["metadata"]["routing"]["request_id"] == first_request
            assert shadow.payload["metadata"]["routing"]["primary_request_id"] == first_request
            assert shadow.payload["messages"] == user_message("first")

    asyncio.run(run())


async def oversized(request: web.Request) -> web.Response:
    return web.json_response({"choices": [{"message": {"content": "x" * 4096}}]})


async def oversized_stream(request: web.Request) -> web.StreamResponse:
    # Chunked, with no Content-Length: the limit is found while reading.
    response = web.StreamResponse(headers={"Content-Type": "application/json"})
    await response.prepare(request)
    for _ in range(8):
        await response.write(b" " * 512)
    return response


async def oversized_error(request: web.Request) -> web.Response:
    return web.Response(status=500, text="e" * 4096)


async def provider_error(request: web.Request) -> web.Response:
    return web.Response(status=500, text="shadow provider failed")


@pytest.mark.parametrize(
    ("case", "outcome", "error_type"),
    [
        ("error", "error", "UpstreamStatusError"),
        ("oversized", "error", "ResponseTooLarge"),
        ("oversized_stream", "error", "ResponseTooLarge"),
        ("oversized_error", "error", "ResponseTooLarge"),
        ("timeout", "timeout", "TimeoutError"),
        ("refused", "error", "ClientConnectorError"),
        ("write_failure", None, None),
    ],
)
def test_a_failing_shadow_keeps_its_own_record_and_never_touches_the_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, outcome: str | None, error_type: str | None
) -> None:
    async def run() -> None:
        handlers = {
            "error": provider_error,
            "oversized": oversized,
            "oversized_stream": oversized_stream,
            "oversized_error": oversized_error,
        }
        provider = EchoProvider(held={"student-v1"} if case == "timeout" else set())
        if case in handlers:
            provider.replies["student-v1"] = handlers[case]
        # Port 9 (discard) refuses connections; the observer is shadow-only, so readiness never probes it.
        shadow_target = "observer" if case == "refused" else "student"
        async with routed_service(
            tmp_path,
            default="teacher",
            shadow=shadow_target,
            provider_chat=provider.chat,
            target_urls={"observer": "http://127.0.0.1:9"} if case == "refused" else None,
            shadow_limits=ShadowLimits(timeout_seconds=0.2, max_response_bytes=1024),
        ) as service:

            def failing_shadow_write(writer: DispatcherShadowWriter, item: AgentRecord) -> AgentRecord:
                raise OSError("record store refused the shadow record")

            if case == "write_failure":
                monkeypatch.setattr(DispatcherShadowWriter, "accept_record", failing_shadow_write)
            try:
                status, text, headers = await service.chat({"model": "auto", "messages": user_message("hello")})
                assert status == 200 and "teacher-v2 answered" in text
                shadows = await shadows_finished(service)
            finally:
                provider.release.set()
            [served], recorded = records_by_purpose(service)
            assert served.agent_record_id == headers["x-reef-agent-record-id"]
            assert served.payload["metadata"]["routing"]["execution_outcome"] == "completed"
            if case == "write_failure":
                assert recorded == [] and shadows.failed == 1
                return
            [shadow] = recorded
            routing = shadow.payload["metadata"]["routing"]
            assert (routing["execution_outcome"], shadow.payload["response"]) == (outcome, {"error_type": error_type})
            assert routing["target_id"] == shadow_target and routing["client_visible"] is False
            assert shadows.failed == 0 and shadows.cancelled == 0

    asyncio.run(run())


def test_an_accepted_shadow_outlives_its_failed_or_disconnected_primary(tmp_path: Path) -> None:
    async def run() -> None:
        provider = EchoProvider(held={"student-v1", "teacher-v2"})
        provider.replies["teacher-v2"] = provider_error
        async with routed_service(
            tmp_path, default="teacher", shadow="student", provider_chat=provider.chat
        ) as service:
            # A primary that fails at its provider: its error reaches the client; the shadow still answers.
            failed = asyncio.create_task(service.chat({"model": "auto", "messages": user_message("fails")}))
            await provider.wait_for(2)
            provider.release.set()
            status, _, _ = await failed
            assert status >= 500
            await shadows_finished(service)
            # A client that disconnects while both calls run: the primary is cancelled, the shadow is not.
            del provider.replies["teacher-v2"]
            provider.release.clear()
            provider.waiting.clear()
            disconnected = asyncio.create_task(service.chat({"model": "auto", "messages": user_message("gone")}))
            await provider.wait_for(2)
            disconnected.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnected
            assert len(shadow_tasks(service).tasks) == 1
            provider.release.set()
            await shadows_finished(service)
            served, shadows = records_by_purpose(service)
            assert [record.payload["metadata"]["routing"]["execution_outcome"] for record in shadows] == [
                "completed",
                "completed",
            ]
            assert served[0].payload["metadata"]["routing"]["execution_outcome"] == "error"
            assert [record.payload["messages"] for record in shadows] == [user_message("fails"), user_message("gone")]

    asyncio.run(run())


def test_old_primary_and_shadow_outputs_keep_their_policy_while_a_shadow_only_update_serves_new_requests(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        old_finish = asyncio.Event()
        first = b'data: {"choices":[{"index":0,"delta":{"content":"first"}}]}\n\n'

        async def teacher_stream(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(first)
            await old_finish.wait()
            await response.write(b'data: {"choices":[{"index":0,"delta":{"content":"last"}}]}\n\ndata: [DONE]\n\n')
            return response

        provider = EchoProvider(held={"student-v1"}, replies={"teacher-v2": teacher_stream})
        # Endpoint B: where V2 moves the student and the math student. It serves every model.
        moved = EchoProvider()
        moved_app = web.Application()
        moved_app.router.add_get("/v1/models", Provider().models)
        moved_app.router.add_post("/v1/chat/completions", moved.chat)
        async with (
            TestServer(moved_app) as moved_server,
            routed_service(
                tmp_path, managed=True, default="teacher", shadow="student", provider_chat=provider.chat
            ) as service,
        ):
            moved_url = str(moved_server.make_url("")).rstrip("/")
            old = await service.client.post(
                "/v1/chat/completions",
                headers=routing_headers(),
                json={"model": "auto", "stream": True, "messages": user_message("old")},
            )
            try:
                assert old.status == 200 and await old.content.readuntil(b"\n\n") == first
                await provider.wait_for(1)
                closed = await service.lifecycle(command("close_policy", 1))
                state = await closed.json()
                # Both old calls are bound: the policy close waits for neither the stream nor the shadow.
                assert (state["active_requests"], state["pending_selections"], state["pending_shadows"]) == (1, 0, 1)
                # Only the shadow target and two endpoints change, so the Semantic Router document and runtime
                # stay the same (test_routing_deployment checks the rendered document).
                path = write_manifest(
                    tmp_path / "shadow-only.json",
                    service.provider_url,
                    default="teacher",
                    shadow="math-student",
                    target_urls={"student": moved_url, "math-student": moved_url},
                )
                version = RoutingSnapshot.load(path).version
                activation = {
                    "manifest_path": str(path),
                    "snapshot_version": version,
                    "router_config_hash": ROUTER_HASH,
                    "expected_snapshot_version": service.adapter.snapshot.version,
                }
                activated = await service.lifecycle(command("activate", 2, policy_close_revision=1, **activation))
                assert activated.status == 200, await activated.text()
                service.router.snapshot = RoutingSnapshot.load(path)
                opened = await service.lifecycle(command("open", 3, policy_close_revision=1))
                assert opened.status == 200, await opened.text()
                provider.replies.pop("teacher-v2")
                status, text, headers = await service.chat({"model": "auto", "messages": user_message("new")})
                assert status == 200 and "teacher-v2 answered" in text
                new_receipt = headers["x-reef-agent-record-id"]
            finally:
                old_finish.set()
                provider.release.set()
            rest = await old.read()
            assert b"last" in rest and rest.endswith(b"data: [DONE]\n\n")
            await shadows_finished(service)
            served, shadows = records_by_purpose(service)
            # Each old branch keeps the policy it bound; each new branch has the new one.
            seen = set()
            for record in [*served, *shadows]:
                routing = record.payload["metadata"]["routing"]
                sent = record.payload["messages"][-1]["content"]
                seen.add((sent, routing["purpose"], routing["target_id"]))
                expected = service.adapter.snapshot.version if sent == "old" else version
                assert (routing["snapshot_version"], routing["router_config_hash"]) == (expected, ROUTER_HASH)
            assert seen == {
                ("old", "serve", "teacher"),
                ("old", "shadow", "student"),
                ("new", "serve", "teacher"),
                ("new", "shadow", "math-student"),
            }
            new_served = next(record for record in served if record.agent_record_id == new_receipt)
            new_shadow = next(record for record in shadows if record.payload["messages"] == user_message("new"))
            assert new_shadow.payload["metadata"]["routing"]["primary_request_id"] == (
                new_served.payload["metadata"]["routing"]["request_id"]
            )
            # Each call reached the endpoint of the policy it bound: the old student shadow stayed on endpoint A
            # although V2 moved the student to B, and the new math-student shadow went to B.
            calls = {
                endpoint: sorted((str(sent["model"]), sent["messages"][-1]["content"]) for sent in bodies)
                for endpoint, bodies in (("A", provider.received), ("B", moved.received))
            }
            assert calls == {
                "A": [("student-v1", "old"), ("teacher-v2", "new"), ("teacher-v2", "old")],
                "B": [("math-v1", "new")],
            }

    asyncio.run(run())


def test_a_full_close_drains_shadows_a_forced_close_cancels_them_and_shutdown_records_them(tmp_path: Path) -> None:
    async def run() -> None:
        provider = EchoProvider(held={"student-v1"})
        async with routed_service(
            tmp_path, managed=True, default="teacher", shadow="student", provider_chat=provider.chat
        ) as service:
            status, _, _ = await service.chat({"model": "auto", "messages": user_message("first")})
            assert status == 200
            await provider.wait_for(1)
            closed = await service.lifecycle(command("close", 1))
            state = await closed.json()
            assert (state["active_requests"], state["pending_shadows"]) == (0, 1)
            activation = {
                "manifest_path": str(tmp_path / "routing.json"),
                "snapshot_version": service.adapter.snapshot.version,
                "router_config_hash": ROUTER_HASH,
                "expected_snapshot_version": service.adapter.snapshot.version,
            }
            # A full close drains shadow calls too: no publication while one runs.
            refused = await service.lifecycle(command("activate", 2, **activation))
            assert refused.status == 409 and "drained" in await refused.text()
            forced = await service.lifecycle(command("close", 3, cancel_active=True))
            assert forced.status == 200
            await shadows_finished(service)
            assert (await (await service.lifecycle(command("close", 3, cancel_active=True))).json())[
                "pending_shadows"
            ] == 0
            _, [cancelled] = records_by_purpose(service)
            assert cancelled.payload["metadata"]["routing"]["execution_outcome"] == "cancelled"
            # Shutdown cancels a running shadow and stores its record before the record store closes.
            reopened = await service.lifecycle(command("open", 4))
            assert reopened.status == 200, await reopened.text()
            status, _, _ = await service.chat({"model": "auto", "messages": user_message("second")})
            assert status == 200
            await provider.wait_for(2)
            shadows = shadow_tasks(service)
            assert len(shadows.tasks) == 1
            await service.client.close()
            assert shadows.closed and not shadows.tasks
            _, recorded = records_by_purpose(service)
            assert [record.payload["metadata"]["routing"]["execution_outcome"] for record in recorded] == [
                "cancelled",
                "cancelled",
            ]
            assert recorded[1].payload["messages"] == user_message("second")
            provider.release.set()

    asyncio.run(run())


def blocked_shadow_writes(
    service: RoutedService, monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> tuple[threading.Event, threading.Event]:
    """Make the record store hold each shadow record until the returned ``release`` is set.

    ``entered`` is set when a shadow record write starts; ``events`` gets ``shadow written`` after it is stored.
    Other records are stored at once. The write waits at most five seconds for ``release``.
    """
    accept_record = DispatcherShadowWriter.accept_record
    entered, release = threading.Event(), threading.Event()

    def blocked(writer: DispatcherShadowWriter, item: AgentRecord) -> AgentRecord:
        entered.set()
        if not release.wait(5):
            raise TimeoutError("the test never released the shadow record write")
        stored = accept_record(writer, item)
        events.append("shadow written")
        return stored

    monkeypatch.setattr(DispatcherShadowWriter, "accept_record", blocked)
    return entered, release


def test_an_unmanaged_app_closes_its_store_only_after_a_cancelled_shadows_record_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        provider = EchoProvider()
        async with routed_service(
            tmp_path, default="teacher", shadow="student", provider_chat=provider.chat, close_dispatcher=True
        ) as service:
            events: list[str] = []
            entered, release = blocked_shadow_writes(service, monkeypatch, events)
            close = service.dispatcher.close

            def closing_store() -> None:
                events.append("store closed")
                close()

            monkeypatch.setattr(service.dispatcher, "close", closing_store)
            status, _, _ = await service.chat({"model": "auto", "messages": user_message("hello")})
            assert status == 200
            # The shadow has its answer and is writing its record when the app shuts down.
            assert await asyncio.to_thread(entered.wait, 5)
            shutdown = asyncio.create_task(service.client.close())
            try:
                # Shutdown cancels the shadow task during that write; the write stays tracked, not orphaned.
                shadows = shadow_tasks(service)
                async with asyncio.timeout(5):
                    while shadows.tasks:
                        await asyncio.sleep(0.01)
                done, _ = await asyncio.wait({shutdown}, timeout=0.2)
                assert not done and events == []
            finally:
                release.set()
            await shutdown
            assert events[:2] == ["shadow written", "store closed"]

    asyncio.run(run())


def test_a_second_cancellation_never_interrupts_a_shadows_cancelled_record_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        provider = EchoProvider(held={"student-v1"})
        async with routed_service(
            tmp_path, managed=True, default="teacher", shadow="student", provider_chat=provider.chat
        ) as service:
            entered, release = blocked_shadow_writes(service, monkeypatch, [])
            try:
                status, _, _ = await service.chat({"model": "auto", "messages": user_message("hello")})
                assert status == 200
                await provider.wait_for(1)
                # A forced drain cancels the shadow's provider call; the write of its cancelled record then holds.
                forced = await service.lifecycle(command("close", 1, cancel_active=True))
                assert forced.status == 200
                assert await asyncio.to_thread(entered.wait, 5)
                shadows = shadow_tasks(service)
                [task] = shadows.tasks
                # A later forced close, a direct cancel and the shutdown close each find it already cancelling.
                again = await service.lifecycle(command("close", 2, cancel_active=True))
                assert (await again.json())["pending_shadows"] == 1
                shadows.cancel()
                assert not await shadows.close(0.05)
                assert not task.done() and shadows.tasks == {task}
            finally:
                release.set()
            await shadows_finished(service)
            _, [record] = records_by_purpose(service)
            assert record.payload["metadata"]["routing"]["execution_outcome"] == "cancelled"
            assert (shadows.cancelled, shadows.failed) == (1, 0)
            provider.release.set()

    asyncio.run(run())


def test_an_offline_shadow_only_target_never_blocks_opening_or_serving(tmp_path: Path) -> None:
    async def run() -> None:
        provider = EchoProvider()
        async with routed_service(
            tmp_path,
            managed=True,
            default="teacher",
            shadow="observer",
            provider_chat=provider.chat,
            target_urls={"observer": "http://127.0.0.1:9"},
        ) as service:
            # routed_service opened the child: readiness probed the primary targets, not the observer.
            state = await (await service.lifecycle(command("open"))).json()
            assert state["accepting"] is True
            status, text, _ = await service.chat({"model": "auto", "messages": user_message("hello")})
            assert status == 200 and "teacher-v2 answered" in text
            await shadows_finished(service)
            _, [shadow] = records_by_purpose(service)
            assert shadow.payload["metadata"]["routing"]["execution_outcome"] == "error"
        # A target that may serve as a primary keeps its check: an offline teacher fails startup.
        (tmp_path / "primary").mkdir()
        with pytest.raises(RoutingRequestError, match="readiness"):
            async with routed_service(
                tmp_path / "primary",
                default="teacher",
                shadow="student",
                target_urls={"teacher": "http://127.0.0.1:9"},
            ):
                pass

    asyncio.run(run())


def test_a_late_shadow_never_enters_a_recreated_scenario(tmp_path: Path) -> None:
    async def run() -> None:
        provider = EchoProvider(held={"student-v1"})
        async with routed_service(
            tmp_path, default="teacher", shadow="student", provider_chat=provider.chat
        ) as service:
            try:
                status, _, _ = await service.chat({"model": "auto", "messages": user_message("old")})
                assert status == 200
                await provider.wait_for(1)
                old = service.dispatcher.registry.get_loaded("coding")
                service.dispatcher.delete_scenario("coding")
                replacement = service.dispatcher.get_or_create_scenario("coding")
                assert replacement is not None and replacement is not old
                provider.release.set()
                shadows = await shadows_finished(service)
                assert shadows.failed == 1
                assert service.records() == []
                # The replacement still serves new requests and receives only its own primary/shadow pair.
                assert (await service.chat({"model": "auto", "messages": user_message("new")}))[0] == 200
                await shadows_finished(service)
                assert len(service.records()) == 2
                assert all(record["messages"] == user_message("new") for record in service.records())
            finally:
                provider.release.set()

    asyncio.run(run())
