"""Routed inference preserves provider requests, attempt identities, and receipts.

Clients call Reef directly. The router is ``routing_fakes.FakeSemanticRouter``,
a fake routing preview listener on loopback, not the Semantic Router. The
providers are local aiohttp apps.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import ClientConnectionError, ClientError, ClientPayloadError, ServerTimeoutError, web
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.routing.routing_fakes import (
    ROUTER_HASH,
    SELECTION_TOKEN,
    SERVICE_TOKEN,
    provider_models,
    routing_adapter,
    routing_dispatcher,
    routing_headers,
    running_router,
)

from reef.artifact import InMemoryRepositoryBackend
from reef.core.records_types import AgentRecord
from reef.dispatcher import Dispatcher
from reef.inference.http import HttpInferenceHandler
from reef.recipe import Recipe
from reef.service.app import InferenceRetryPolicy, create_app
from reef.service.routing_preview import PreviewSettings
from reef.storage.records import AppendResult
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage


def provider_server(*routes: tuple[str, str, web.Handler]) -> TestServer:
    """A local provider that lists the routed models and answers the given routes."""
    app = web.Application()
    app.router.add_get("/v1/models", provider_models)
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    return TestServer(app)


@pytest.mark.unit
def test_routed_abort_retry_records_each_attempt_without_selecting_again(tmp_path: Path) -> None:
    async def run() -> None:
        received = []

        async def upstream(request: web.Request) -> web.Response:
            received.append(await request.json())
            if len(received) == 1:
                return web.json_response({"choices": [{"finish_reason": "abort"}], "training": {"response_ids": [10]}})
            return web.json_response({"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"))
            dispatcher = routing_dispatcher(tmp_path)
            retry = InferenceRetryPolicy(initial_s=0.001, max_s=0.002, timeout_s=2)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(
                        dispatcher,
                        tokens=SERVICE_TOKEN,
                        routing=adapter,
                        preview=router.settings,
                        inference_retry_policy=retry,
                    )
                    async with TestClient(TestServer(app)) as client:
                        payload = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
                        response = await client.post("/v1/chat/completions", headers=routing_headers(), json=payload)
                        assert response.status == 200, await response.text()
                        provider_payload = {**payload, "model": "student-v1"}
                        assert received == [provider_payload, provider_payload]
                        assert len(router.selections) == 1
                        aborted, succeeded = dispatcher.get_or_create_scenario("coding").records.replay("coding")
                        first = aborted.payload["metadata"]["routing"]
                        second = succeeded.payload["metadata"]["routing"]
                        assert first["execution_outcome"] == "aborted"
                        assert first["delivery_outcome"] == "not_started"
                        assert "training" not in aborted.payload["response"]
                        assert first["request_id"] == second["request_id"]
                        assert second["attempt_index"] == first["attempt_index"] + 1
                        assert first["target_id"] == second["target_id"] == "student"
                        assert first["router_config_hash"] == second["router_config_hash"] == ROUTER_HASH
                        assert second["prior_attempt_ids"] == [aborted.agent_record_id]
                        assert response.headers["x-reef-agent-record-id"] == succeeded.agent_record_id
            finally:
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure", "status"),
    [
        ("status", 502),
        ("timeout", 503),
        ("non_sse", 502),
        ("malformed_json", 502),
        ("non_object", 502),
        ("provider_client_error", 400),
    ],
)
def test_routed_upstream_failure_is_diagnostic_and_has_no_success_receipt(
    tmp_path: Path, failure: str, status: int
) -> None:
    async def run() -> None:
        finish = asyncio.Event()

        async def upstream(request: web.Request) -> web.Response:
            if failure == "timeout":
                await finish.wait()
            if failure == "non_sse":
                return web.json_response({"choices": [{"message": {"content": "not a stream"}}]})
            if failure == "malformed_json":
                return web.Response(body=b"{not json", content_type="application/json")
            if failure == "non_object":
                return web.json_response([{"message": {"content": "a list"}}])
            if failure == "provider_client_error":
                return web.json_response({"error": "max_tokens too large"}, status=400)
            return web.json_response({"error": "temporarily unavailable"}, status=503)

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"), timeout_s=0.2)
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        response = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={
                                "model": "auto",
                                "messages": [{"role": "user", "content": "x"}],
                                "stream": failure == "non_sse",
                            },
                        )
                        assert response.status == status
                        assert "x-reef-agent-record-id" not in response.headers
                        assert len(router.selections) == 1
                        scenario = dispatcher.get_or_create_scenario("coding")
                        [record] = scenario.records.replay("coding")
                        assert "training" not in record.payload
                        if failure != "timeout":
                            # A provider's error status and an unusable successful reply fail the same way.
                            assert record.payload["response"] == {"error_type": "UpstreamStatusError"}
                        routing = record.payload["metadata"]["routing"]
                        assert routing["execution_outcome"] == ("timeout" if failure == "timeout" else "error")
                        assert routing["delivery_outcome"] == "not_started"
                        assert routing["target_id"] == "student"
                        assert scenario.operations.snapshot()["serve/request/active"] == 0
            finally:
                finish.set()
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("error_type", "status", "outcome"),
    # ServerTimeoutError is both a TimeoutError and a ClientError; the timeout wins.
    [(ServerTimeoutError, 503, "timeout"), (ClientConnectionError, 502, "error")],
)
def test_routed_transport_failure_before_any_output_keeps_one_diagnostic_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    error_type: type[ClientError],
    status: int,
    outcome: str,
) -> None:
    calls: list[dict[str, object]] = []

    async def fail(handler: HttpInferenceHandler, artifact: object, path: str, payload: dict[str, object]) -> None:
        calls.append(payload)
        raise error_type("provider transport failed")

    monkeypatch.setattr(HttpInferenceHandler, "inference_stream" if streaming else "inference", fail)

    async def run() -> None:
        async with provider_server() as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"))
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        body = {"model": "auto", "messages": [{"role": "user", "content": "x"}], "stream": streaming}
                        response = await client.post("/v1/chat/completions", headers=routing_headers(), json=body)
                        assert response.status == status
                        assert "x-reef-agent-record-id" not in response.headers
                        assert b'"reef"' not in await response.read()
                        # One selection and one provider attempt: a transport failure is not retried.
                        assert len(router.selections) == 1 and len(calls) == 1
                        scenario = dispatcher.get_or_create_scenario("coding")
                        [record] = scenario.records.replay("coding")
                        assert record.payload["response"] == {"error_type": error_type.__name__}
                        routing = record.payload["metadata"]["routing"]
                        assert (routing["execution_outcome"], routing["delivery_outcome"]) == (outcome, "not_started")
                        assert routing["client_visible"] is False and routing["attempt_index"] == 1
                        assert scenario.operations.snapshot()["serve/request/active"] == 0
            finally:
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
def test_routed_stream_flushes_then_records_before_issuing_receipt(tmp_path: Path) -> None:
    async def run() -> None:
        first_chunk = b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
        finish = asyncio.Event()

        async def upstream(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(first_chunk)
            await finish.wait()
            await response.write(b"data: [DONE]\n\n")
            return response

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"))
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        response = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={"model": "auto", "messages": [{"role": "user", "content": "x"}], "stream": True},
                        )
                        assert response.status == 200
                        assert "x-reef-agent-record-id" not in response.headers
                        assert await asyncio.wait_for(response.content.readexactly(len(first_chunk)), 1) == first_chunk
                        scenario = dispatcher.get_or_create_scenario("coding")
                        assert scenario.records.replay("coding") == ()
                        finish.set()
                        receipt_frame = await asyncio.wait_for(response.content.readuntil(b"\n\n"), 1)
                        receipt = json.loads(receipt_frame.removeprefix(b"data: "))["reef"]["agent_record_id"]
                        [record] = scenario.records.replay("coding")
                        assert receipt == record.agent_record_id
                        assert await response.read() == b"data: [DONE]\n\n"
                        assert record.payload["response"]["complete"] is True
                        routing = record.payload["metadata"]["routing"]
                        assert routing["target_id"] == "student"
                        assert routing["execution_outcome"] == "completed"
                        assert routing["delivery_outcome"] == "unconfirmed"
                        assert len(router.selections) == 1
            finally:
                finish.set()
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
def test_routed_stream_disconnect_keeps_an_incomplete_attempt(tmp_path: Path) -> None:
    async def run() -> None:
        first_chunk = b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
        continue_generation = asyncio.Event()

        async def upstream(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(first_chunk)
            await continue_generation.wait()
            try:
                await response.write(first_chunk)
                await response.write(b"data: [DONE]\n\n")
            except ConnectionResetError:
                pass
            return response

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"))
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        response = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={"model": "student", "messages": [{"role": "user", "content": "x"}], "stream": True},
                        )
                        assert await asyncio.wait_for(response.content.readexactly(len(first_chunk)), 1) == first_chunk
                        response.close()
                        await asyncio.sleep(0.01)
                        continue_generation.set()
                        scenario = dispatcher.get_or_create_scenario("coding")
                        async with asyncio.timeout(2):
                            while not scenario.records.replay("coding"):
                                await asyncio.sleep(0.01)
                        [record] = scenario.records.replay("coding")
                        assert record.payload["response"]["complete"] is False
                        assert "training" not in record.payload
                        routing = record.payload["metadata"]["routing"]
                        assert routing["target_id"] == "student"
                        assert routing["delivery_outcome"] == "incomplete"
                        assert routing["execution_outcome"] == "unknown"
            finally:
                continue_generation.set()
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ending", "outcome", "delivery"),
    [
        ("abort", "aborted", "unconfirmed"),
        ("stop", "completed", "unconfirmed"),
        ("missing_terminal", "unknown", "incomplete"),
        ("abort_without_terminal", "aborted", "incomplete"),
    ],
)
def test_a_routed_stream_records_an_explicit_abort_apart_from_its_delivery(
    tmp_path: Path, ending: str, outcome: str, delivery: str
) -> None:
    content = b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
    finish = "abort" if ending.startswith("abort") else "stop"
    last = f'data: {{"choices":[{{"index":0,"delta":{{}},"finish_reason":"{finish}"}}]}}\n\n'.encode()
    terminal = ending in ("abort", "stop")
    provider_body = content + last + (b"data: [DONE]\n\n" if terminal else b"")

    async def run() -> None:
        calls = []

        async def upstream(request: web.Request) -> web.Response:
            calls.append(await request.json())
            return web.Response(body=provider_body, content_type="text/event-stream")

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"))
            dispatcher = routing_dispatcher(tmp_path)
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        response = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={"model": "auto", "messages": [{"role": "user", "content": "x"}], "stream": True},
                        )
                        assert response.status == 200
                        body = await response.read()
                        # Provider frames reach the client unchanged, and a stream is never sent twice.
                        assert body.startswith(content + last)
                        assert len(calls) == 1 and len(router.selections) == 1
                        [record] = dispatcher.get_or_create_scenario("coding").records.replay("coding")
                        assert record.payload["response"]["complete"] is terminal
                        routing = record.payload["metadata"]["routing"]
                        assert (routing["execution_outcome"], routing["delivery_outcome"]) == (outcome, delivery)
                        assert routing["client_visible"] is True
                        if terminal:
                            # The receipt still refers to the stored record, aborted or not.
                            receipt = next(
                                json.loads(line.removeprefix(b"data: "))["reef"]["agent_record_id"]
                                for line in body.split(b"\n")
                                if line.startswith(b"data: {") and b'"reef"' in line
                            )
                            assert receipt == record.agent_record_id
                            assert body.endswith(b"data: [DONE]\n\n")
                        else:
                            assert b'"reef"' not in body
            finally:
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
def test_routing_disabled_preserves_direct_inference(tmp_path: Path) -> None:
    async def run() -> None:
        received = []

        async def upstream(request: web.Request) -> web.Response:
            received.append(await request.json())
            return web.json_response({"choices": [{"message": {"content": "direct"}}]})

        provider_app = web.Application()
        provider_app.router.add_post("/v1/chat/completions", upstream)
        async with TestServer(provider_app) as provider:
            dispatcher = routing_dispatcher(tmp_path)
            handler = HttpInferenceHandler(str(provider.make_url("")).rstrip("/"))
            payload = {"model": "unregistered-direct-model", "messages": []}
            try:
                async with TestClient(TestServer(create_app(dispatcher, inference_handler=handler))) as client:
                    response = await client.post(
                        "/v1/chat/completions", headers={"x-reef-scenario": "legacy"}, json=payload
                    )
                    assert response.status == 200
                    assert received == [payload]
                    [record] = dispatcher.get_or_create_scenario("legacy").records.replay("legacy")
                    assert response.headers["x-reef-agent-record-id"] == record.agent_record_id
                    assert "routing" not in record.payload.get("metadata", {})
            finally:
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
def test_direct_evaluation_keeps_no_record_or_receipt(tmp_path: Path, streaming: bool) -> None:
    async def run() -> None:
        received: list[dict[str, object]] = []

        async def upstream(request: web.Request) -> web.Response:
            received.append(await request.json())
            if streaming:
                return web.Response(
                    body=b'data: {"choices":[{"delta":{"content":"evaluated"}}]}\n\ndata: [DONE]\n\n',
                    content_type="text/event-stream",
                )
            return web.json_response({"choices": [{"message": {"content": "evaluated"}}]})

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            dispatcher = routing_dispatcher(tmp_path)
            scenario = dispatcher.get_or_create_scenario("legacy")
            handler = HttpInferenceHandler(str(provider.make_url("")).rstrip("/"))
            payload = {"model": "direct-model", "messages": [], "stream": streaming}
            try:
                async with TestClient(TestServer(create_app(dispatcher, inference_handler=handler))) as client:
                    response = await client.post("/reef/scenarios/legacy/evaluation/v1/chat/completions", json=payload)
                    assert response.status == 200
                    assert "x-reef-agent-record-id" not in response.headers
                    assert "x-reef-release-id" not in response.headers
                    body = await response.text()
                    assert "evaluated" in body and '"reef"' not in body
                    assert received == [payload]
                    assert scenario.records.count("legacy") == 0
            finally:
                dispatcher.close()

    asyncio.run(run())


@pytest.mark.unit
def test_routed_service_requires_authentication_and_its_preview_settings(tmp_path: Path) -> None:
    adapter = routing_adapter(tmp_path, "http://127.0.0.1:1")
    preview = PreviewSettings("http://127.0.0.1:9", SELECTION_TOKEN)
    dispatcher = routing_dispatcher(tmp_path)
    try:
        with pytest.raises(ValueError, match="token"):
            create_app(dispatcher, routing=adapter, preview=preview)
        with pytest.raises(ValueError, match="preview settings"):
            create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter)
        with pytest.raises(ValueError, match="routing manifest"):
            create_app(dispatcher, tokens=SERVICE_TOKEN, preview=preview)
    finally:
        dispatcher.close()


@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
def test_persistence_failure_has_no_receipt_and_releases_the_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    append_result = SQLiteRecordStore.append_result
    failures_remaining = 1

    def fail_first_append(store: SQLiteRecordStore, item: AgentRecord) -> AppendResult:
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise OSError("test record storage is full")
        return append_result(store, item)

    monkeypatch.setattr(SQLiteRecordStore, "append_result", fail_first_append)

    async def run() -> None:
        first_chunk = b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
        finish = asyncio.Event()

        async def upstream(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            if payload.get("stream") is True:
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                await response.write(first_chunk)
                await finish.wait()
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

        async with provider_server(("POST", "/v1/chat/completions", upstream)) as provider:
            adapter = routing_adapter(tmp_path, str(provider.make_url("")).rstrip("/"))
            dispatcher = routing_dispatcher(tmp_path)
            messages = [{"role": "user", "content": "x"}]
            try:
                async with running_router(adapter.snapshot) as router:
                    app = create_app(dispatcher, tokens=SERVICE_TOKEN, routing=adapter, preview=router.settings)
                    async with TestClient(TestServer(app)) as client:
                        response = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={"model": "auto", "messages": messages, "stream": streaming},
                        )
                        assert "x-reef-agent-record-id" not in response.headers
                        if streaming:
                            assert response.status == 200
                            assert (
                                await asyncio.wait_for(response.content.readexactly(len(first_chunk)), 1)
                                == first_chunk
                            )
                            finish.set()
                            trailing = bytearray()
                            try:
                                async for chunk in response.content.iter_any():
                                    trailing.extend(chunk)
                            except ClientPayloadError:
                                pass
                            assert b"[DONE]" not in trailing
                            assert b'"reef"' not in trailing
                        else:
                            assert response.status == 500
                        scenario = dispatcher.get_or_create_scenario("coding")
                        assert scenario.records.replay("coding") == ()
                        assert scenario.operations.snapshot()["serve/request/active"] == 0

                        recovered = await client.post(
                            "/v1/chat/completions",
                            headers=routing_headers(),
                            json={"model": "auto", "messages": messages},
                        )
                        assert recovered.status == 200
                        [record] = scenario.records.replay("coding")
                        assert recovered.headers["x-reef-agent-record-id"] == record.agent_record_id
                        assert scenario.operations.snapshot()["serve/request/active"] == 0
            finally:
                finish.set()
                dispatcher.close()

    asyncio.run(run())


class SpecializedRecipe(Recipe):
    """An extension whose processors are outside record-only routing's contract."""


@pytest.mark.unit
def test_routing_rejects_specialized_recipes(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        SpecializedRecipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(),
    )
    try:
        adapter = routing_adapter(tmp_path, "http://127.0.0.1:1")
        with pytest.raises(ValueError, match="record-only"):
            create_app(
                dispatcher,
                tokens=SERVICE_TOKEN,
                routing=adapter,
                preview=PreviewSettings("http://127.0.0.1:9", SELECTION_TOKEN),
            )
        assert dispatcher.list_scenarios() == ()
    finally:
        dispatcher.close()
