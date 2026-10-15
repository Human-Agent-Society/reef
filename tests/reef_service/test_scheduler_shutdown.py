"""Scheduler shutdown drains worker operations after their asyncio callers cancel."""

import asyncio
from collections.abc import Mapping
from threading import Event

import pytest

from reef.scenario.scheduling import (
    MultiTenantScheduler,
    SchedulerConfig,
    SchedulerNotAcceptingRequests,
    ServiceRegistration,
)
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import SchedulerHTTPClient
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore


class BlockingControl(SchedulerHTTPClient):
    def __init__(self, operation: str):
        super().__init__("http://service.invalid")
        self.operation = operation
        self.entered = Event()
        self.release = Event()
        self.operations: list[str] = []

    def request(
        self, method: str, path: str, *, payload: Mapping[str, object] | None = None, tenant_id: int | None = None
    ) -> dict[str, object]:
        assert payload is not None
        operation = str(payload["operation"])
        self.operations.append(operation)
        if operation == "describe":
            return {"service_id": 1, "service_name": "local", "recipe": "recipe", "standard_execution_units": 2}
        if operation == "status":
            return {"state": None}
        assert operation == self.operation
        self.entered.set()
        assert self.release.wait(5), "test did not release the control request"
        state = "deleted" if operation == "cleanup" else "admitted"
        return {
            "scenario_id": payload["scenario_id"],
            "tenant_id": payload["tenant_id"],
            "generation": payload["generation"],
            "state": state,
            "idle_since": None,
            "cleanup_complete": operation == "cleanup",
        }


@pytest.mark.parametrize("origin", ["submit", "cancel", "background"])
def test_shutdown_drains_cancelled_request_and_background_threads(tmp_path, origin):
    async def run():
        database = tmp_path / "scheduler.sqlite"
        store = SQLiteSchedulingStore(database)
        scheduler = MultiTenantScheduler(store, SchedulerConfig(2, {1: 2}, {"recipe": 2}))
        scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
        scheduler.set_service_availability(1, True)
        control = BlockingControl("cleanup" if origin == "cancel" else "start")
        controller = SchedulerController(scheduler, {1: control})
        if origin == "submit":
            operation = asyncio.create_task(
                asyncio.to_thread(
                    controller.submit, {"service_name": "local", "recipe": "recipe", "scenario_name": "work"}, 1
                )
            )
        else:
            admission = scheduler.request_scenario_admission(1, "local", "recipe", "work")
            if origin == "cancel":
                starting = scheduler.next_scenario_admission(1)
                scheduler.confirm_started(admission.scenario_id, starting.generation)
                operation = asyncio.create_task(asyncio.to_thread(controller.cancel, 1, "local", "work"))
            else:
                operation = asyncio.create_task(asyncio.to_thread(controller.reconcile))
        closing = None
        try:
            assert await asyncio.to_thread(control.entered.wait, 5)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
            controller.stop_accepting_requests()
            closing = asyncio.create_task(asyncio.to_thread(controller.close))
            await asyncio.sleep(0)
            assert not closing.done()
            assert store.load().scenarios
            with pytest.raises(SchedulerNotAcceptingRequests):
                controller.submit({"service_name": "local", "recipe": "recipe", "scenario_name": "late"}, 1)
            control.release.set()
            await asyncio.wait_for(closing, 5)
            assert controller.active_operations == {}
            assert control.operations.count("cleanup") == (1 if origin == "cancel" else 0)
            recovered = SQLiteSchedulingStore(database)
            try:
                record = recovered.get_scenario_admission(1)
                assert record.state == ("deleted" if origin == "cancel" else "admitted")
            finally:
                recovered.close()
        finally:
            control.release.set()
            if closing is not None:
                await closing
            else:
                await asyncio.to_thread(controller.close)

    asyncio.run(run())


def test_maintenance_failure_is_visible_and_stops_new_admission(tmp_path, caplog):
    from aiohttp.test_utils import TestClient, TestServer

    from reef.service.scheduler import create_scheduler_app

    class FailingMaintenanceController(SchedulerController):
        def __init__(self, scheduler):
            super().__init__(scheduler, {})
            self.calls = 0

        def reconcile(self):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("synthetic maintenance failure")

    async def run():
        scheduler = MultiTenantScheduler(
            SQLiteSchedulingStore(tmp_path / "scheduler.sqlite"), SchedulerConfig(2, {1: 2}, {"recipe": 2})
        )
        controller = FailingMaintenanceController(scheduler)
        async with TestClient(TestServer(create_scheduler_app(controller))) as client:
            for _ in range(100):
                health = await client.get("/healthz")
                if health.status == 503:
                    break
                await asyncio.sleep(0.01)
            assert health.status == 503
            assert "maintenance failed (RuntimeError)" in (await health.json())["reason"]
            rejected = await client.post(
                "/reef/scenarios",
                json={"tenant_id": 1, "service_name": "local", "recipe": "recipe", "scenario_name": "late"},
            )
            assert rejected.status == 503
            assert controller.stopping
        assert controller.closed

    asyncio.run(run())
    assert "maintenance failed (RuntimeError)" in caplog.text


def test_startup_failure_closes_database_without_rewriting_yaml(tmp_path):
    import sqlite3
    from unittest.mock import patch

    import yaml
    from aiohttp import web

    from reef.service.scheduler import build_scheduler, create_scheduler_app

    settings = {
        "database": str(tmp_path / "scheduler.sqlite"),
        "global_execution_units": 2,
        "tenant_quotas": {1: 2},
    }
    path = tmp_path / "scheduler.yaml"
    original = yaml.safe_dump(settings)
    path.write_text(original)
    controller = build_scheduler(settings, config_path=path)

    async def run():
        runner = web.AppRunner(create_scheduler_app(controller))
        try:
            with (
                patch.object(controller, "reconcile", side_effect=RuntimeError("startup failed")),
                pytest.raises(RuntimeError, match="startup failed"),
            ):
                await runner.setup()
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert controller.closed
    assert path.read_text() == original
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        controller.scheduler.store.load()
