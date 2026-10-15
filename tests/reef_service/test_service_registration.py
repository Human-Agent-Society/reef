"""Services join after scheduler startup and refresh their registration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer, unused_port

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.service.app import create_app
from reef.service.assembly import build_app
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.service.scheduling import ServiceSchedulerClient, ServiceSchedulingConfig
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.sqlite import SQLiteScenarioStorage


@pytest.fixture
def dispatcher(tmp_path: Path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        require_scenario_admission=True,
        scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
    )
    try:
        yield dispatcher
    finally:
        dispatcher.close()


def connection(scheduler_url: str, service_port: int) -> ServiceSchedulingConfig:
    return ServiceSchedulingConfig(
        scheduler_url,
        7,
        "development",
        "harness",
        2,
        "test-control",
        scheduler_token="test-scheduler",
        service_url=f"http://127.0.0.1:{service_port}",
        config_revision="revision-1",
        registration_interval_seconds=0.02,
    )


@pytest.mark.parametrize("initial_status", [408, 429, 503])
def test_registration_retries_and_refreshes_after_service_can_accept_control(
    tmp_path: Path, dispatcher: Dispatcher, initial_status: int
):
    async def run() -> None:
        requests: list[dict[str, object]] = []
        registered = asyncio.Event()

        async def register(request: web.Request) -> web.Response:
            assert request.headers["Authorization"] == "Bearer test-scheduler"
            assert "x-reef-tenant" not in request.headers
            payload = await request.json()
            requests.append(payload)
            if len(requests) == 1:
                return web.json_response({}, status=initial_status)
            async with (
                ClientSession() as client,
                client.post(
                    f"{payload['service_url']}/reef/scheduler/control",
                    json={"operation": "describe"},
                    headers={"Authorization": f"Bearer {payload['control_token']}"},
                ) as response,
            ):
                assert response.status == 200
                assert (await response.json())["service_id"] == 7
            registered.set()
            return web.json_response({"service_id": 7, "available": True})

        scheduler_app = web.Application()
        scheduler_app.router.add_post("/reef/services/register", register)
        async with TestServer(scheduler_app) as scheduler:
            port = unused_port()
            settings = connection(str(scheduler.make_url("/")), port)
            async with TestServer(
                create_app(dispatcher, scheduling=settings, staged_model_config_directory=tmp_path / "models"),
                port=port,
            ):
                await asyncio.wait_for(registered.wait(), timeout=3)
                assert requests[-1] == {
                    "service_id": 7,
                    "service_name": "development",
                    "service_url": settings.service_url,
                    "recipe": "harness",
                    "standard_execution_units": 2,
                    "config_revision": "revision-1",
                    "control_token": "test-control",
                }
                registered.clear()
                await asyncio.wait_for(registered.wait(), timeout=3)
                assert len(requests) >= 3

    asyncio.run(run())


def test_service_registration_recovers_when_scheduler_starts_later(tmp_path: Path, dispatcher: Dispatcher):
    async def run() -> None:
        registered = asyncio.Event()

        async def register(request: web.Request) -> web.Response:
            registered.set()
            return web.json_response({"service_id": 7, "available": True})

        scheduler_port, service_port = unused_port(), unused_port()
        settings = connection(f"http://127.0.0.1:{scheduler_port}", service_port)
        async with TestServer(
            create_app(dispatcher, scheduling=settings, staged_model_config_directory=tmp_path / "models"),
            port=service_port,
        ):
            await asyncio.sleep(0.05)
            scheduler_app = web.Application()
            scheduler_app.router.add_post("/reef/services/register", register)
            async with TestServer(scheduler_app, port=scheduler_port):
                await asyncio.wait_for(registered.wait(), timeout=3)

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 401, 409])
def test_registration_stops_on_permanent_rejection(tmp_path: Path, dispatcher: Dispatcher, caplog, status: int):
    async def run() -> None:
        request_count = 0

        async def reject(request: web.Request) -> web.Response:
            nonlocal request_count
            request_count += 1
            return web.json_response({}, status=status)

        scheduler_app = web.Application()
        scheduler_app.router.add_post("/reef/services/register", reject)
        async with TestServer(scheduler_app) as scheduler:
            port = unused_port()
            settings = connection(str(scheduler.make_url("/")), port)
            async with TestServer(
                create_app(dispatcher, scheduling=settings, staged_model_config_directory=tmp_path / "models"),
                port=port,
            ):
                async with asyncio.timeout(3):
                    while "registration stopped" not in caplog.text:
                        await asyncio.sleep(0.01)
                await asyncio.sleep(0.06)
                assert request_count == 1

    with caplog.at_level(logging.ERROR):
        asyncio.run(run())
    assert f"HTTP {status}" in caplog.text
    assert "test-control" not in caplog.text
    assert "test-scheduler" not in caplog.text


def test_service_shutdown_waits_for_inflight_registration(tmp_path: Path, dispatcher: Dispatcher):
    async def run() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def register(request: web.Request) -> web.Response:
            entered.set()
            await release.wait()
            return web.json_response({"service_id": 7, "available": True})

        scheduler_app = web.Application()
        scheduler_app.router.add_post("/reef/services/register", register)
        async with TestServer(scheduler_app) as scheduler:
            port = unused_port()
            service = TestServer(
                create_app(
                    dispatcher,
                    scheduling=connection(str(scheduler.make_url("/")), port),
                    staged_model_config_directory=tmp_path / "models",
                ),
                port=port,
            )
            await service.start_server()
            closing: asyncio.Task[None] | None = None
            try:
                await asyncio.wait_for(entered.wait(), timeout=3)
                closing = asyncio.create_task(service.close())
                await asyncio.sleep(0.02)
                assert not closing.done()
            finally:
                release.set()
                if closing is None:
                    await service.close()
                else:
                    await asyncio.wait_for(closing, timeout=3)

    asyncio.run(run())


def test_registration_config_parses_advertised_address_and_revision():
    settings = service_config_from_mapping(
        {
            "schema-version": 2,
            "recipe": {"implementation": "recipe"},
            "scheduling": {
                "scheduler_url": "http://scheduler:8902",
                "service_id": 7,
                "service_name": "development",
                "service_url": "https://service.example",
                "config_revision": "revision-1",
                "control_token": "test-control",
            },
        }
    )
    assert settings.scheduler_service_url == "https://service.example"
    assert settings.scheduler_config_revision == "revision-1"
    with pytest.raises(ValueError, match="HTTP"):
        replace(settings, scheduler_service_url="file:///tmp/service")
    with pytest.raises(ValueError, match="scheduler_url"):
        ServiceConfig(recipe="recipe", scheduler_service_url="http://service:8901")


@pytest.mark.parametrize(
    ("service_url", "served_url", "expected"),
    [
        ("https://service.example", "http://evaluation:8901", "https://service.example"),
        (None, "http://evaluation:8901", "http://evaluation:8901"),
        (None, None, "http://127.0.0.1:8901"),
    ],
)
def test_assembly_always_supplies_registration_address(
    dispatcher: Dispatcher, service_url: str | None, served_url: str | None, expected: str
):
    settings = ServiceConfig(
        recipe="recipe",
        host="0.0.0.0",
        port=8901,
        served_url=served_url,
        scheduler_url="http://scheduler:8902",
        scheduler_service_id=7,
        scheduler_service_name="development",
        scheduler_service_url=service_url,
        scheduler_config_revision="revision-1",
        scheduler_control_token="test-control",
    )
    with (
        patch("reef.service.assembly.build_dispatcher", return_value=dispatcher),
        patch.object(Recipe, "scenario_execution_units", return_value=2),
        patch("reef.service.assembly.create_app") as create,
    ):
        build_app(settings)
    scheduling = create.call_args.kwargs["scheduling"]
    assert scheduling.service_url == expected
    assert scheduling.config_revision == "revision-1"


def test_manual_service_connection_does_not_register_without_an_advertised_address():
    settings = replace(connection("http://scheduler:8902", 8901), service_url=None)
    with pytest.raises(ValueError, match="service_url"):
        ServiceSchedulerClient(settings).register()
