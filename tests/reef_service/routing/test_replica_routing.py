"""External routing: ``reef serve --component inference`` beside a router that another system runs.

The router is ``routing_fakes.FakeSemanticRouter``, not the Semantic Router. The provider is a local aiohttp app.
"""

from __future__ import annotations

import asyncio
import json
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
from reef.service import routed_replica
from reef.service.components import build_component_app
from reef.service.deploy import orchestrator
from reef.service.deploy.routing.cli import main as routing_cli
from reef.service.deploy.routing.renderer import MANAGEMENT_TOKEN_ENV, SELECTION_TOKEN_ENV
from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.deploy.service_config import ServiceConfig
from reef.service.routing import RoutingSnapshot
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


def stored_records(tmp_path: Path, scenario: str = "coding") -> list[dict[str, object]]:
    """The payloads of the INFERENCE records that the replica stored for ``scenario``: its record items, in the data
    store that assembly opens in the record directory."""
    task_files = LocalTaskFileStore(tmp_path / "unused-task-files")
    with closing(task_files), closing(SQLiteDataStore(tmp_path / "records" / "data", task_files)) as data:
        items = data.query(scenario, kind="record", scenario=scenario, origin=scenario, request_type="inference")
        return [dict(item.content["payload"]) for item in items]


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


@asynccontextmanager
async def replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[TestClient, FakeSemanticRouter]]:
    """A running inference replica beside a fake router and one provider."""
    monkeypatch.setattr(routed_replica, "ROUTER_CHECK_INTERVAL_SECONDS", 0.01)
    provider = web.Application()
    provider.router.add_get("/v1/models", provider_models)
    provider.router.add_post("/v1/chat/completions", provider_chat)
    async with TestServer(provider) as provider_server:
        manifest = routing_manifest(str(provider_server.make_url("")).rstrip("/"))
        path = tmp_path / "manifest.yaml"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        async with running_router(RoutingSnapshot.load(path)) as router:
            assert router.settings is not None
            port = urlsplit(router.settings.base_url).port
            assert port is not None
            app = build_component_app("inference", external_settings(tmp_path, path, port), environ=ENVIRONMENT)
            async with TestClient(TestServer(app)) as client:
                yield client, router


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
