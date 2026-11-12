"""External routing: ``reef serve --component inference`` beside a router that another system runs.

The router is ``routing_fakes.FakeSemanticRouter``, not the Semantic Router. The provider is a local aiohttp app.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.routing.routing_fakes import (
    ROUTER_HASH,
    RUNTIME_HASH,
    SELECTION_TOKEN,
    SERVICE_TOKEN,
    FakeSemanticRouter,
    provider_models,
    routing_headers,
    routing_manifest,
    running_router,
)

from reef.core.errors import DeployConfigError
from reef.core.records_types import AgentRecord
from reef.core.snapshot import SnapshotRef
from reef.service import routed_replica
from reef.service.components import build_component_app
from reef.service.deploy import orchestrator
from reef.service.deploy.routing.cli import main as routing_cli
from reef.service.deploy.routing.renderer import MANAGEMENT_TOKEN_ENV, SELECTION_TOKEN_ENV
from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.deploy.service_config import ServiceConfig
from reef.service.routed_inference import RoutedScenario, ShadowRecordWriter
from reef.service.routed_shadow import ShadowTasks
from reef.service.routing import RoutingSnapshot
from reef.storage.commits import CommitRecord
from reef.storage.data import ScenarioArchived
from reef.storage.scenario import ScenarioSettings
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.tasks import LocalTaskFileStore

ENVIRONMENT = {MANAGEMENT_TOKEN_ENV: "management-test-token", SELECTION_TOKEN_ENV: SELECTION_TOKEN}
CHAT = {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}
MEMORY_BACKEND = "tests.support.artifact_backend_plugin:MemoryArtifactBackend"


def external_settings(tmp_path: Path, manifest: Path | None, port: int = 50053) -> ServiceConfig:
    return ServiceConfig(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        artifact_backend={"module": MEMORY_BACKEND, "config": {"label": "memory"}},
        tokens=(SERVICE_TOKEN,),
        routing_manifest=None if manifest is None else str(manifest),
        routing_deployment=RoutingDeploymentSettings(lifecycle="external", sr_management_port=port),
    )


def stored_records(tmp_path: Path, scenario: str = "coding", *, payload_only: bool = True) -> list[dict[str, object]]:
    """The inference records that the replica stored in its DataStore: their payloads, or full record content when
    ``payload_only`` is false, including the record IDs returned in client receipts."""
    task_files = LocalTaskFileStore(tmp_path / "unused-task-files")
    with closing(task_files), closing(SQLiteDataStore(tmp_path / "records" / "data", task_files)) as data:
        items = data.query(scenario, kind="record", scenario=scenario, origin=scenario, request_type="inference")
        return [dict(item.content["payload"] if payload_only else item.content) for item in items]


async def provider_chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    if body.get("stream") is True:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"id":"chat-1","choices":[{"index":0,"delta":{"content":"a"}}]}\n\n')
        if body.get("stop") == "hold":
            await asyncio.Event().wait()
        await response.write(b"data: [DONE]\n\n")
        return response
    return web.json_response({"id": "chat-1", "model": body["model"], "choices": []})


class ShadowProvider:
    """A provider that can hold or fail the teacher shadow, and optionally waits for both models to start before
    answering either. This makes concurrent execution observable without a timing threshold."""

    def __init__(self, *, failure: str | None = None, require_both: bool = True) -> None:
        self.failure = failure
        self.require_both = require_both
        self.received: list[dict[str, object]] = []
        self.started: set[str] = set()
        self.both_started = asyncio.Event()
        self.shadow_started = asyncio.Event()
        self.release_shadow = asyncio.Event()

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.received.append(body)
        model = body["model"]
        self.started.add(model)
        if model == "teacher-v2":
            self.shadow_started.set()
        if self.started == {"student-v1", "teacher-v2"}:
            self.both_started.set()
        if self.require_both:
            await asyncio.wait_for(self.both_started.wait(), timeout=5)
        if model == "teacher-v2":
            if self.failure == "error":
                return web.Response(status=500, text="shadow failed")
            if self.failure == "hold":
                await self.release_shadow.wait()
        text = f"{model} answered"
        if body.get("stream") is True:
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            chunk = {"id": "primary", "choices": [{"index": 0, "delta": {"content": text}}]}
            await response.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            return response
        return web.json_response({"model": model, "choices": [{"message": {"role": "assistant", "content": text}}]})


@asynccontextmanager
async def replica(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shadow_target: str | None = None,
    shadow_provider: ShadowProvider | None = None,
    shadow_timeout_seconds: float = 300,
) -> AsyncIterator[tuple[TestClient, FakeSemanticRouter]]:
    """A running inference replica beside a fake router and one provider, with an optional shadow target."""
    monkeypatch.setattr(routed_replica, "ROUTER_CHECK_INTERVAL_SECONDS", 0.01)
    provider = web.Application()
    provider.router.add_get("/v1/models", provider_models)
    provider.router.add_post(
        "/v1/chat/completions", provider_chat if shadow_provider is None else shadow_provider.chat
    )
    async with TestServer(provider) as provider_server:
        manifest = routing_manifest(str(provider_server.make_url("")).rstrip("/"))
        if shadow_target is not None:
            manifest["scenarios"]["coding"]["shadow_target"] = shadow_target
        path = tmp_path / "manifest.yaml"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        async with running_router(RoutingSnapshot.load(path)) as router:
            assert router.settings is not None
            port = urlsplit(router.settings.base_url).port
            assert port is not None
            settings = external_settings(tmp_path, path, port)
            settings = replace(
                settings,
                routing_deployment=replace(settings.routing_deployment, shadow_timeout_seconds=shadow_timeout_seconds),
            )
            app = build_component_app("inference", settings, environ=ENVIRONMENT)
            try:
                async with TestClient(TestServer(app)) as client:
                    yield client, router
            finally:
                if shadow_provider is not None:
                    shadow_provider.release_shadow.set()


async def wait_for_ready_status(client: TestClient, status: int) -> None:
    async with asyncio.timeout(5):
        while (await client.get("/readyz")).status != status:
            await asyncio.sleep(0.01)


@pytest.mark.unit
def test_a_manifest_scenario_is_routed_and_recorded_and_others_keep_the_ordinary_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        async with replica(tmp_path, monkeypatch) as (client, router):
            await wait_for_ready_status(client, 200)

            answer = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
            assert answer.status == 200
            assert (await answer.json())["model"] == "student-v1"
            assert answer.headers["x-reef-agent-record-id"]
            stream = await client.post(
                "/v1/chat/completions",
                headers=routing_headers(),
                json={**CHAT, "messages": [{"role": "user", "content": "reef_teacher_required"}], "stream": True},
            )
            assert stream.status == 200
            assert b"[DONE]" in await stream.read()
            assert len(router.selections) == 2

            other = await client.post("/v1/chat/completions", headers=routing_headers("math"), json=CHAT)
            assert other.status == 404
            assert len(router.selections) == 2

    asyncio.run(run())
    routing = [record["metadata"]["routing"] for record in stored_records(tmp_path)]  # type: ignore[index]
    assert [(entry["target_id"], entry["reason"]) for entry in routing] == [
        ("student", "default"),
        ("teacher", "rule"),
    ]
    assert {entry["router_config_hash"] for entry in routing} == {ROUTER_HASH}


@pytest.mark.unit
def test_readiness_follows_the_router_and_needs_no_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with replica(tmp_path, monkeypatch) as (client, router):
            await wait_for_ready_status(client, 200)

            router.is_ready = False
            await wait_for_ready_status(client, 503)
            router.is_ready = True
            await wait_for_ready_status(client, 200)

            router.activation_status = "pending"
            await wait_for_ready_status(client, 503)
            router.activation_status = "active"
            await wait_for_ready_status(client, 200)

            # A restarted router can report another hash: the replica checks it before requests use it.
            router.runtime_hash = RUNTIME_HASH
            router.failure = (503, "NOT_READY")
            await wait_for_ready_status(client, 503)
            router.failure = None
            await wait_for_ready_status(client, 200)
            answer = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
            assert answer.status == 200

    asyncio.run(run())
    assert stored_records(tmp_path)[0]["metadata"]["routing"]["router_config_hash"] == RUNTIME_HASH  # type: ignore[index]


@pytest.mark.unit
def test_a_manifest_scenario_keeps_the_routing_request_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with replica(tmp_path, monkeypatch) as (client, _):
            await wait_for_ready_status(client, 200)
            assert (await client.post("/v1/chat/completions", json=CHAT)).status == 401
            forged = await client.post(
                "/v1/chat/completions", headers={**routing_headers(), "x-selected-model": "teacher"}, json=CHAT
            )
            assert forged.status == 400
            unknown = await client.post(
                "/v1/chat/completions", headers=routing_headers(), json={**CHAT, "model": "other"}
            )
            assert unknown.status == 400

    asyncio.run(run())


@pytest.mark.unit
def test_stopping_the_replica_stores_the_record_of_an_open_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        async with replica(tmp_path, monkeypatch) as (client, _):
            await wait_for_ready_status(client, 200)
            stream = await client.post(
                "/v1/chat/completions", headers=routing_headers(), json={**CHAT, "stream": True, "stop": "hold"}
            )
            await stream.content.readany()

    asyncio.run(run())
    routing = stored_records(tmp_path)[0]["metadata"]["routing"]  # type: ignore[index]
    assert routing["delivery_outcome"] == "incomplete"


async def wait_for_record_count(tmp_path: Path, count: int) -> list[dict[str, object]]:
    """Read the replica's durable record items once all expected branches have finalized."""
    async with asyncio.timeout(5):
        while len(records := stored_records(tmp_path, payload_only=False)) < count:
            await asyncio.sleep(0.01)
    assert len(records) == count
    return records


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_replica_primary_and_shadow_run_concurrently_and_store_separate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    async def run() -> None:
        provider = ShadowProvider()
        async with replica(tmp_path, monkeypatch, shadow_target="teacher", shadow_provider=provider) as (
            client,
            router,
        ):
            await wait_for_ready_status(client, 200)
            response = await client.post(
                "/v1/chat/completions", headers=routing_headers(), json={**CHAT, "stream": stream}
            )
            assert response.status == 200
            text = await response.text()
            assert "student-v1 answered" in text and "teacher-v2 answered" not in text
            assert provider.both_started.is_set(), "both providers must start before either can answer"
            assert len(router.selections) == 1, "only the primary uses SR preview"
            bodies = {body["model"]: body for body in provider.received}
            assert len(provider.received) == 2 and bodies["teacher-v2"]["stream"] is False
            assert bodies["student-v1"]["messages"] == bodies["teacher-v2"]["messages"] == CHAT["messages"]
            if stream:
                assert "[DONE]" in text
                events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
                receipt = next(event["reef"]["agent_record_id"] for event in events if "reef" in event)
            else:
                receipt = response.headers["x-reef-agent-record-id"]
            items = await wait_for_record_count(tmp_path, 2)
            primary, shadow = sorted(items, key=lambda item: item["payload"]["metadata"]["routing"]["purpose"])
            primary_route = primary["payload"]["metadata"]["routing"]
            shadow_route = shadow["payload"]["metadata"]["routing"]
            assert primary["agent_record_id"] == receipt != shadow["agent_record_id"]
            assert shadow_route["primary_request_id"] == primary_route["request_id"] != shadow_route["request_id"]
            assert (primary_route["target_id"], shadow_route["target_id"]) == ("student", "teacher")
            assert shadow_route["execution_outcome"] == "completed" and shadow_route["client_visible"] is False
            assert shadow_route["delivery_outcome"] == "not_started" and "training" not in shadow["payload"]
            assert shadow["payload"]["response"]["choices"][0]["message"]["content"] == "teacher-v2 answered"
            assert primary_route["snapshot_version"] == shadow_route["snapshot_version"]

    asyncio.run(run())
    assert len(stored_records(tmp_path)) == 2, "records survive replica shutdown and reopening"


@pytest.mark.unit
def test_replica_deduplicates_a_shadow_that_is_the_primary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        provider = ShadowProvider(require_both=False)
        async with replica(tmp_path, monkeypatch, shadow_target="student", shadow_provider=provider) as (client, _):
            await wait_for_ready_status(client, 200)
            response = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
            assert response.status == 200 and (await response.json())["model"] == "student-v1"
        assert len(provider.received) == 1

    asyncio.run(run())
    (record,) = stored_records(tmp_path)
    assert record["metadata"]["routing"]["purpose"] == "serve"


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_replica_shadow_failure_keeps_the_primary_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    async def run() -> None:
        provider = ShadowProvider(failure="hold" if failure == "timeout" else "error")
        async with replica(
            tmp_path,
            monkeypatch,
            shadow_target="teacher",
            shadow_provider=provider,
            shadow_timeout_seconds=0.2,
        ) as (client, _):
            await wait_for_ready_status(client, 200)
            response = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
            assert response.status == 200 and (await response.json())["model"] == "student-v1"
            items = await wait_for_record_count(tmp_path, 2)
            primary, shadow = sorted(items, key=lambda item: item["payload"]["metadata"]["routing"]["purpose"])
            assert primary["agent_record_id"] == response.headers["x-reef-agent-record-id"]
            assert primary["payload"]["metadata"]["routing"]["execution_outcome"] == "completed"
            assert shadow["payload"]["metadata"]["routing"]["execution_outcome"] == failure
            assert shadow["payload"]["response"] == {
                "error_type": "TimeoutError" if failure == "timeout" else "UpstreamStatusError"
            }
            assert len(provider.received) == 2, "the shadow is not retried"

    asyncio.run(run())


@pytest.mark.unit
def test_replica_shutdown_saves_active_shadow_cancellation_before_closing_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        provider = ShadowProvider(failure="hold")
        async with asyncio.timeout(5):
            async with replica(tmp_path, monkeypatch, shadow_target="teacher", shadow_provider=provider) as (
                client,
                _,
            ):
                await wait_for_ready_status(client, 200)
                response = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
                assert response.status == 200 and (await response.json())["model"] == "student-v1"
                assert provider.shadow_started.is_set() and not provider.release_shadow.is_set()
                (record,) = stored_records(tmp_path)
                assert record["metadata"]["routing"]["purpose"] == "serve", "the shadow is still in its provider"

    asyncio.run(run())
    primary, shadow = sorted(stored_records(tmp_path), key=lambda record: record["metadata"]["routing"]["purpose"])
    assert primary["metadata"]["routing"]["execution_outcome"] == "completed"
    assert shadow["metadata"]["routing"]["execution_outcome"] == "cancelled"
    assert shadow["response"] == {"error_type": "CancelledError"}


@pytest.mark.unit
def test_a_replica_refuses_settings_that_cannot_route(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(json.dumps(routing_manifest("http://127.0.0.1:1")), encoding="utf-8")
    settings = external_settings(tmp_path, path)
    with pytest.raises(ValueError, match=MANAGEMENT_TOKEN_ENV):
        build_component_app("inference", settings, environ={})
    with pytest.raises(ValueError, match="service token"):
        build_component_app("inference", replace(settings, tokens=()), environ=ENVIRONMENT)


@pytest.mark.unit
def test_external_routing_is_only_for_the_inference_component(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(json.dumps(routing_manifest("http://127.0.0.1:1")), encoding="utf-8")
    settings = external_settings(tmp_path, path)

    async def run() -> None:
        # The services of a deployment share one config, so the other services ignore the manifest.
        async with TestClient(TestServer(build_component_app("control", settings, environ={}))) as client:
            assert (await client.get("/healthz")).status == 200
            assert (await client.get("/readyz")).status == 404

    asyncio.run(run())
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "reef": {
                    "routing_manifest": str(path),
                    "routing_deployment": {"lifecycle": "external"},
                },
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:1", "upstream_model": "student"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DeployConfigError, match="--component inference"):
        orchestrator.main(["--config", str(config)])


@pytest.mark.unit
def test_render_writes_a_router_config_bound_to_the_replica_loopback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(json.dumps(routing_manifest("http://127.0.0.1:1")), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "reef": {
                    "routing_manifest": str(manifest),
                    "routing_deployment": {"lifecycle": "external", "sr_management_port": 50100},
                },
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:1", "upstream_model": "student"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "router.yaml"
    routing_cli(["render", "--config", str(config), "--output", str(output)])

    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    management = yaml.safe_load(output.read_text(encoding="utf-8"))["global"]["services"]["management_api"]
    assert (management["bind_address"], management["port"], management["remote_exposure"]) == (
        "127.0.0.1",
        50100,
        False,
    )
    assert "test-token" not in output.read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_replica_never_writes_a_late_shadow_into_a_recreated_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        provider = ShadowProvider(failure="hold")
        captured: list[routed_replica.ReplicaRecords] = []
        rejected, closed = threading.Event(), threading.Event()
        open_shadow = routed_replica.ReplicaRecords.open_shadow_scenario
        append = routed_replica.ReplicaShadowWriter.accept_record
        close_shadow = routed_replica.ReplicaShadowWriter.close

        def capture(records: routed_replica.ReplicaRecords, name: str) -> tuple[RoutedScenario, ShadowRecordWriter]:
            captured.append(records)
            return open_shadow(records, name)

        def write(writer: routed_replica.ReplicaShadowWriter, item: AgentRecord) -> AgentRecord:
            try:
                return append(writer, item)
            except ScenarioArchived:
                rejected.set()
                raise

        def close(writer: routed_replica.ReplicaShadowWriter) -> None:
            close_shadow(writer)
            closed.set()

        monkeypatch.setattr(routed_replica.ReplicaRecords, "open_shadow_scenario", capture)
        monkeypatch.setattr(routed_replica.ReplicaShadowWriter, "accept_record", write)
        monkeypatch.setattr(routed_replica.ReplicaShadowWriter, "close", close)
        async with replica(tmp_path, monkeypatch, shadow_target="teacher", shadow_provider=provider) as (client, _):
            await wait_for_ready_status(client, 200)
            response = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
            assert response.status == 200
            await asyncio.wait_for(provider.shadow_started.wait(), 5)
            records = captured[0]
            # Exercise the stale-generation guard defensively. Production archive requires all Scenario sessions closed.
            records.storage.archive("coding")
            settings = ScenarioSettings(training_mode="manual")
            committed = CommitRecord(
                scenario="coding",
                step=1,
                snapshot=SnapshotRef("release:1", "base"),
                checkpoint=True,
                algorithm_state={"replacement": True},
                high_water_sequence=0,
                high_water_offset=0,
                consumed_ids=frozenset(),
                recorded_at=1001.0,
            )
            with closing(records.storage.open("coding")) as replacement:
                assert list(replacement.records.replay("coding")) == []
                records.storage.write_settings("coding", settings)
                replacement.commit_step(expected_step=0, commit=committed)
                provider.release_shadow.set()
                assert await asyncio.to_thread(rejected.wait, 5), "the old session must reject a late shadow"
                assert await asyncio.to_thread(closed.wait, 5), "the replacement must stay open through the old close"
            with closing(records.storage.open("coding")) as reopened:
                assert reopened.history() == (committed,)
                assert records.storage.read_settings("coding") == settings
            assert stored_records(tmp_path) == []
            # The replacement accepts only new requests, with normal primary and shadow records.
            response = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
            assert response.status == 200
            items = await wait_for_record_count(tmp_path, 2)
            assert {item["payload"]["metadata"]["routing"]["purpose"] for item in items} == {"serve", "shadow"}
        assert len(stored_records(tmp_path)) == 2

    asyncio.run(run())


@pytest.mark.unit
def test_cancelling_a_shadow_while_its_session_opens_still_closes_the_acquired_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        provider = ShadowProvider(require_both=False)
        opened, release, closed = threading.Event(), threading.Event(), threading.Event()
        cancelled = asyncio.Event()
        open_shadow = routed_replica.ReplicaRecords.open_shadow_scenario
        close_writer = routed_replica.ReplicaShadowWriter.close
        cancel_shadows = ShadowTasks.cancel
        closes: list[str] = []

        def hold_open(records: routed_replica.ReplicaRecords, name: str) -> tuple[RoutedScenario, ShadowRecordWriter]:
            bound = open_shadow(records, name)
            opened.set()
            if not release.wait(5):
                bound[1].close()
                raise TimeoutError("the test never released session acquisition")
            return bound

        def close(writer: routed_replica.ReplicaShadowWriter) -> None:
            close_writer(writer)
            closes.append(writer.name)
            closed.set()

        def cancel(tasks: ShadowTasks) -> None:
            cancel_shadows(tasks)
            cancelled.set()

        monkeypatch.setattr(routed_replica.ReplicaRecords, "open_shadow_scenario", hold_open)
        monkeypatch.setattr(routed_replica.ReplicaShadowWriter, "close", close)
        monkeypatch.setattr(ShadowTasks, "cancel", cancel)
        async with replica(tmp_path, monkeypatch, shadow_target="teacher", shadow_provider=provider) as (client, _):
            try:
                await wait_for_ready_status(client, 200)
                response = await client.post("/v1/chat/completions", headers=routing_headers(), json=CHAT)
                assert response.status == 200
                assert await asyncio.to_thread(opened.wait, 5)
                shutdown = asyncio.create_task(client.close())
                await asyncio.wait_for(cancelled.wait(), 5)
                assert not closed.is_set(), "the worker still owns the session being acquired"
                release.set()
                await asyncio.wait_for(shutdown, 5)
                assert closed.is_set() and closes == ["coding"]
                assert provider.started == {"student-v1"}, "cancelled acquisition must not dispatch a shadow"
            finally:
                release.set()
        assert len(stored_records(tmp_path)) == 1

    asyncio.run(run())
