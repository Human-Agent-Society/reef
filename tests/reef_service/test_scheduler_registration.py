"""Runtime service registration and shutdown-only directory snapshots."""

from __future__ import annotations

import asyncio
import sqlite3
from threading import Event
from unittest.mock import patch

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.scenario.scheduling import SchedulerNotAcceptingRequests, ServiceRegistration
from reef.service.scheduler import build_scheduler, create_scheduler_app
from reef.service.scheduler_config import parse_scheduler_config
from reef.service.scheduling import SchedulerHTTPClient


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 8)
    return {
        "database": str(tmp_path / "scheduler.sqlite"),
        "max_logical_thread_percentage": 50,
        "tenant_quotas": {1: 4},
        "idle_ttl_seconds": None,
    }


@pytest.mark.parametrize("services", [None, []])
def test_first_boot_needs_no_services_or_recipe_costs(settings, services):
    if services is not None:
        settings["services"] = services
    controller = build_scheduler(settings)
    try:
        assert controller.scheduler.service_registrations() == ()
        assert controller.scheduler.recipe_execution_units == {}
        assert controller.scheduler.tenant_allocations() == ()
        controller.reconcile()
    finally:
        controller.close()


def test_shutdown_snapshot_round_trip_preserves_policy_and_queue(settings, tmp_path, monkeypatch):
    settings["token_env"] = "REEF_SNAPSHOT_TEST_TOKEN"
    monkeypatch.setenv("REEF_SNAPSHOT_TEST_TOKEN", "scheduler-test-value")
    path = tmp_path / "scheduler.yaml"
    original = yaml.safe_dump(settings)
    path.write_text(original)
    controller = build_scheduler(settings, config_path=path)
    registration = ServiceRegistration(1, "local", "recipe", "http://127.0.0.1:8901", 2)
    controller.scheduler.register_service(registration)
    controller.scheduler.set_service_availability(1, True)
    admission = controller.scheduler.request_scenario_admission(1, "local", "recipe", "work")
    controller.controls[1] = SchedulerHTTPClient(registration.service_url, token="control-test-value")
    assert path.read_text() == original
    controller.close()
    saved = yaml.safe_load(path.read_text())
    assert saved["services"] == [
        {
            "service_id": 1,
            "service_name": "local",
            "recipe": "recipe",
            "service_url": registration.service_url,
            "standard_execution_units": 2,
            "config_revision": "",
        }
    ]
    assert saved["recipe_execution_units"] == {"recipe": 2}
    assert saved["tenant_quotas"] == {1: 4}
    assert saved["max_logical_thread_percentage"] == 50
    assert "global_execution_units" not in saved
    assert saved["token_env"] == "REEF_SNAPSHOT_TEST_TOKEN"
    assert "control-test-value" not in path.read_text()
    assert "scheduler-test-value" not in path.read_text()
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 16)
    recovered = build_scheduler(saved, config_path=path)
    try:
        assert recovered.scheduler.config.global_execution_units == 8
        recovered.reconcile()
        assert recovered.scheduler.service_registrations() == (registration,)
        assert not recovered.scheduler.availability[1]
        assert recovered.controls == {}
        assert recovered.scheduler.get_scenario_admission(admission.scenario_id, 1).state == "queued"
        assert recovered.scheduler.tenant_allocations()[0].allocated_execution_units == 4
    finally:
        recovered.close()


def test_restart_rejects_host_budget_below_persisted_allocations(settings, monkeypatch):
    settings["max_logical_thread_percentage"] = 100
    settings["tenant_quotas"] = {1: 1, 2: 1, 3: 1}
    controller = build_scheduler(settings)
    registration = ServiceRegistration(1, "local", "recipe", "http://127.0.0.1:8901", 1)
    controller.scheduler.register_service(registration)
    controller.scheduler.set_service_availability(1, True)
    for tenant_id in settings["tenant_quotas"]:
        controller.scheduler.request_scenario_admission(tenant_id, "local", "recipe", f"work-{tenant_id}")
    controller.close()
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 2)
    with pytest.raises(ValueError, match="restored tenant allocations exceed"):
        build_scheduler(settings)
    monkeypatch.setattr("reef.service.scheduler_config.os.cpu_count", lambda: 8)
    recovered = build_scheduler(settings)
    try:
        assert len(recovered.scheduler.tenant_allocations()) == 3
        assert len(recovered.scheduler.list_scenario_admissions()) == 3
    finally:
        recovered.close()


def test_stale_yaml_retains_sqlite_admissions_until_reregistration(settings, tmp_path):
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(settings))
    controller = build_scheduler(settings)
    registration = ServiceRegistration(1, "local", "recipe", "http://127.0.0.1:8901", 2)
    controller.scheduler.register_service(registration)
    controller.scheduler.set_service_availability(1, True)
    queued = controller.scheduler.request_scenario_admission(1, "local", "recipe", "work")
    starting = controller.scheduler.next_scenario_admission(1)
    assert starting is not None
    controller.scheduler.confirm_started(starting.scenario_id, starting.generation)
    controller.close()
    recovered = build_scheduler(yaml.safe_load(path.read_text()), config_path=path)
    try:
        recovered.reconcile()
        assert recovered.scheduler.service_registrations() == ()
        admission = recovered.scheduler.get_scenario_admission(queued.scenario_id, 1)
        assert admission.state == "admitted"
        assert admission.execution_allocated
        assert recovered.status(1)["used"] == 2
        recovered.scheduler.register_service(registration)
        assert not recovered.scheduler.availability[1]
    finally:
        recovered.close()


def test_snapshot_replace_failure_keeps_original_and_closes_database(settings, tmp_path):
    path = tmp_path / "scheduler.yaml"
    original = yaml.safe_dump(settings)
    path.write_text(original)
    controller = build_scheduler(settings, config_path=path)
    with (
        patch("reef.service.scheduler_config.os.replace", side_effect=OSError("snapshot unavailable")),
        pytest.raises(OSError, match="snapshot unavailable"),
    ):
        controller.close()
    assert controller.closed
    assert path.read_text() == original
    assert not list(tmp_path.glob(".scheduler.yaml.*.tmp"))
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        controller.scheduler.store.load()
    controller.close()


@pytest.mark.parametrize("rotate_token", [False, True])
def test_snapshot_keeps_only_current_environment_reference(settings, tmp_path, monkeypatch, rotate_token):
    monkeypatch.setenv("REEF_LEGACY_CONTROL", "control-test-value")
    settings["recipe_execution_units"] = {"recipe": 2}
    settings["services"] = [
        {
            "service_id": 1,
            "service_name": "local",
            "recipe": "recipe",
            "service_url": "http://127.0.0.1:8901",
            "control_token_env": "REEF_LEGACY_CONTROL",
        }
    ]
    path = tmp_path / "scheduler.yaml"
    controller = build_scheduler(settings, config_path=path)
    with patch.object(
        SchedulerHTTPClient,
        "request",
        return_value={"service_id": 1, "service_name": "local", "recipe": "recipe", "standard_execution_units": 2},
    ):
        controller.register_service(
            {
                **settings["services"][0],
                "standard_execution_units": 2,
                "control_token": "control-test-value",
            }
        )
        if rotate_token:
            controller.register_service(
                {
                    **settings["services"][0],
                    "standard_execution_units": 2,
                    "control_token": "rotated-test-value",
                }
            )
    controller.close()
    saved = yaml.safe_load(path.read_text())
    if rotate_token:
        assert "control_token_env" not in saved["services"][0]
    else:
        assert saved["services"][0]["control_token_env"] == "REEF_LEGACY_CONTROL"
    assert "rotated-test-value" not in path.read_text()
    assert "control-test-value" not in path.read_text()
    recovered = build_scheduler(saved)
    try:
        if rotate_token:
            assert recovered.controls == {}
        else:
            assert recovered.controls[1].token == "control-test-value"
    finally:
        recovered.close()


def test_snapshots_reject_raw_control_credentials(settings):
    settings["services"] = [{"control_token": "control-test-value"}]
    with pytest.raises(ValueError, match="not control_token values"):
        parse_scheduler_config(settings)


def test_registration_http_auth_validation_and_idempotency(settings, tmp_path):
    async def run():
        async def describe(request):
            assert request.headers["Authorization"] == "Bearer control-test-value"
            assert (await request.json())["operation"] == "describe"
            return web.json_response(
                {"service_id": 1, "service_name": "local", "recipe": "recipe", "standard_execution_units": 2}
            )

        service_app = web.Application()
        service_app.router.add_post("/reef/scheduler/control", describe)
        path = tmp_path / "scheduler.yaml"
        path.write_text(yaml.safe_dump(settings))
        controller = build_scheduler(settings, config_path=path)
        async with (
            TestClient(TestServer(service_app)) as service,
            TestClient(TestServer(create_scheduler_app(controller, token="scheduler-test-value"))) as client,
        ):
            payload = {
                "service_id": 1,
                "service_name": "local",
                "recipe": "recipe",
                "service_url": str(service.make_url("/")),
                "standard_execution_units": 2,
                "control_token": "control-test-value",
            }
            route = "/reef/services/register"
            denied = await client.post(route, json=payload)
            assert denied.status == 401
            headers = {"Authorization": "Bearer scheduler-test-value"}
            first = await client.post(route, json=payload, headers=headers)
            assert first.status == 200, await first.text()
            registered = await first.json()
            assert registered["available"] is True
            assert "control_token" not in registered
            second = await client.post(route, json=payload, headers=headers)
            assert await second.json() == registered
            conflict = await client.post(route, json={**payload, "service_name": "other"}, headers=headers)
            assert conflict.status == 409
            malformed = await client.post(route, json={**payload, "service_id": True}, headers=headers)
            assert malformed.status == 400
            assert "services" not in yaml.safe_load(path.read_text())
        assert yaml.safe_load(path.read_text())["services"][0]["service_name"] == "local"
        assert "control-test-value" not in path.read_text()

    asyncio.run(run())


def test_cancelled_registration_drains_before_shutdown_snapshot(settings, tmp_path):
    entered, release = Event(), Event()
    path = tmp_path / "scheduler.yaml"
    path.write_text(yaml.safe_dump(settings))
    controller = build_scheduler(settings, config_path=path)
    payload = {
        "service_id": 1,
        "service_name": "local",
        "recipe": "recipe",
        "service_url": "http://127.0.0.1:8901",
        "standard_execution_units": 2,
        "control_token": "control-test-value",
    }

    def describe(client, method, route, *, payload):
        entered.set()
        assert release.wait(5)
        return {"service_id": 1, "service_name": "local", "recipe": "recipe", "standard_execution_units": 2}

    async def run():
        registration = asyncio.create_task(asyncio.to_thread(controller.register_service, payload))
        closing = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            registration.cancel()
            with pytest.raises(asyncio.CancelledError):
                await registration
            closing = asyncio.create_task(asyncio.to_thread(controller.close))
            while not controller.stopping:
                await asyncio.sleep(0)
            assert not closing.done()
            assert "services" not in yaml.safe_load(path.read_text())
            with pytest.raises(SchedulerNotAcceptingRequests):
                controller.register_service(payload)
        finally:
            release.set()
            if closing is not None:
                await asyncio.wait_for(closing, 5)
            else:
                await asyncio.to_thread(controller.close)
        assert yaml.safe_load(path.read_text())["services"][0]["service_id"] == 1

    with patch.object(SchedulerHTTPClient, "request", describe):
        asyncio.run(run())


def test_identical_registration_keeps_admission_available_during_verification(settings):
    entered, release = Event(), Event()
    controller = build_scheduler(settings)
    payload = {
        "service_id": 1,
        "service_name": "local",
        "recipe": "recipe",
        "service_url": "http://127.0.0.1:8901",
        "standard_execution_units": 2,
        "control_token": "control-test-value",
    }
    remote = {"service_id": 1, "service_name": "local", "recipe": "recipe", "standard_execution_units": 2}
    with patch.object(SchedulerHTTPClient, "request", return_value=remote):
        controller.register_service(payload)

    def describe(client, method, route, *, payload):
        entered.set()
        assert release.wait(5)
        return remote

    async def run():
        refresh = asyncio.create_task(asyncio.to_thread(controller.register_service, payload))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            admission = controller.scheduler.request_scenario_admission(1, "local", "recipe", "work")
            assert admission.state == "queued"
        finally:
            release.set()
            await asyncio.wait_for(refresh, 5)
            await asyncio.to_thread(controller.close)

    with patch.object(SchedulerHTTPClient, "request", describe):
        asyncio.run(run())
