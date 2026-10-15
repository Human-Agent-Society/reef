"""Real HTTP admission across one scheduler and two Dispatcher services."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer, unused_port

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.scenario.scheduling import (
    AdmissionConflict,
    MultiTenantScheduler,
    ScenarioTenantMismatch,
    SchedulerConfig,
    ServiceRegistration,
    UnknownTenant,
)
from reef.service.app import create_app
from reef.service.scheduler import build_scheduler, create_scheduler_app
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import SchedulerHTTPClient, ServiceSchedulingConfig, object_value
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore

SERVICE_TOKEN = "service-test-token"
CONTROL_TOKEN = "control-test-token"
SCHEDULER_TOKEN = "scheduler-test-token"
EVALUATION_TOKEN = "evaluation-test-token"


class EchoInference(InferenceHandler):
    async def inference(self, artifact: Artifact, path: str, payload: dict[str, object]) -> dict[str, object]:
        return {"choices": [{"message": {"role": "assistant", "content": "test response"}}]}


class ScriptedControl(SchedulerHTTPClient):
    """Record controller commands and return a deterministic remote state."""

    def __init__(self, replies: list[tuple[str, dict[str, object]]]) -> None:
        super().__init__("http://service.invalid")
        self.replies = replies
        self.operations: list[str] = []

    def request(
        self, method: str, path: str, *, payload: Mapping[str, object] | None = None, tenant_id: int | None = None
    ) -> dict[str, object]:
        assert payload is not None
        operation = payload["operation"]
        assert isinstance(operation, str)
        expected, response = self.replies.pop(0)
        assert operation == expected
        self.operations.append(operation)
        return response


def tenant_headers(tenant_id: int, *, scheduler: bool = False) -> dict[str, str]:
    token = SCHEDULER_TOKEN if scheduler else SERVICE_TOKEN
    return {"Authorization": f"Bearer {token}", "x-reef-tenant": str(tenant_id)}


def service_dispatcher(directory: Path, *, scheduled: bool = True, recipe: Recipe | None = None) -> Dispatcher:
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir(parents=True)
    return Dispatcher(
        Recipe() if recipe is None else recipe,
        InMemoryRepositoryBackend.factory(bootstrap, root=directory / "repository"),
        scenario_storage=SQLiteScenarioStorage(directory / "record-data"),
        agent_record_dir=directory / "records",
        require_scenario_admission=scheduled,
        scenario_admission_store=(
            JSONScenarioAdmissionStore(directory / "scheduler-assignments") if scheduled else None
        ),
    )


@dataclass
class SchedulerCluster:
    database: Path
    config_path: Path
    controller: SchedulerController
    scheduler_client: TestClient
    services: tuple[TestClient, ...]
    dispatchers: tuple[Dispatcher, ...]

    async def restart_scheduler(self) -> None:
        port = self.scheduler_client.server.port
        await self.scheduler_client.close()
        snapshot = self.config_path.read_text()
        assert CONTROL_TOKEN not in snapshot
        settings = object_value(yaml.safe_load(snapshot))
        with patch("reef.service.scheduler_config.os.cpu_count", return_value=8):
            self.controller = build_scheduler(settings, config_path=self.config_path)
        assert len(self.controller.scheduler.service_registrations()) == len(self.services)
        assert self.controller.controls == {}
        self.scheduler_client = TestClient(
            TestServer(create_scheduler_app(self.controller, token=SCHEDULER_TOKEN), port=port)
        )
        await self.scheduler_client.start_server()
        await self.wait_for_services()

    async def wait_for_services(self) -> None:
        async with asyncio.timeout(5):
            while True:
                response = await self.scheduler_client.get(
                    "/reef/scheduler/status", headers=tenant_headers(1, scheduler=True)
                )
                assert response.status == 200, await response.text()
                status = await response.json()
                services = status["services"]
                if len(services) == len(self.services) and all(service["available"] for service in services):
                    return
                await asyncio.sleep(0.01)


@asynccontextmanager
async def multi_service_cluster(
    directory: Path, *, idle_ttl_seconds: float | None = None, recipe: Recipe | None = None
) -> AsyncIterator[SchedulerCluster]:
    database = directory / "scheduler.sqlite3"
    config_path = directory / "scheduler.yaml"
    settings: dict[str, object] = {
        "database": str(database),
        "max_logical_thread_percentage": 50,
        "tenant_quotas": {1: 2, 2: 2, 3: 2},
        "idle_ttl_seconds": idle_ttl_seconds,
    }
    config_path.write_text(yaml.safe_dump(settings))
    with patch("reef.service.scheduler_config.os.cpu_count", return_value=8):
        controller = build_scheduler(settings, config_path=config_path)
    scheduler_client = TestClient(TestServer(create_scheduler_app(controller, token=SCHEDULER_TOKEN)))
    await scheduler_client.start_server()
    assert controller.scheduler.service_registrations() == ()
    clients: list[TestClient] = []
    dispatchers: list[Dispatcher] = []
    stack = SchedulerCluster(database, config_path, controller, scheduler_client, (), ())
    try:
        for index, name in enumerate(("development", "production")):
            service_id = index + 1
            service_port = unused_port()
            dispatcher = service_dispatcher(directory / name, recipe=recipe)
            dispatchers.append(dispatcher)
            client = TestClient(
                TestServer(
                    create_app(
                        dispatcher,
                        tokens=SERVICE_TOKEN,
                        evaluation_tokens=EVALUATION_TOKEN,
                        inference_handler=EchoInference(),
                        scheduling=ServiceSchedulingConfig(
                            scheduler_url=str(scheduler_client.make_url("/")),
                            service_id=service_id,
                            service_name=name,
                            recipe="recipe",
                            standard_execution_units=2,
                            control_token=CONTROL_TOKEN,
                            scheduler_token=SCHEDULER_TOKEN,
                            service_url=f"http://127.0.0.1:{service_port}",
                            config_revision="test-revision",
                            registration_interval_seconds=0.05,
                        ),
                        staged_model_config_directory=directory / name / "model-settings",
                    ),
                    port=service_port,
                )
            )
            await client.start_server()
            clients.append(client)
        stack.services = tuple(clients)
        stack.dispatchers = tuple(dispatchers)
        await stack.wait_for_services()
        yield stack
    finally:
        await stack.scheduler_client.close()
        for client in clients:
            await client.close()
        for dispatcher in dispatchers:
            await asyncio.to_thread(dispatcher.close)


def test_cross_service_queue_is_metadata_only_and_delete_starts_fifo_head(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            first, second = stack.services
            active = await first.post("/reef/scenarios", json={"name": "active"}, headers=tenant_headers(1))
            assert active.status == 201, await active.text()
            admitted = await active.json()
            waiting = await second.post("/reef/scenarios", json={"name": "waiting"}, headers=tenant_headers(1))
            assert waiting.status == 202, await waiting.text()
            queued = await waiting.json()
            assert queued["state"] == "queued"
            assert "release_id" not in queued
            assert not stack.dispatchers[1].has_loaded("waiting")
            assert not stack.dispatchers[1].has_scenario("waiting")
            assert stack.dispatchers[1].scheduled_scenario_status("waiting") is None
            duplicate = await second.post("/reef/scenarios", json={"name": "waiting"}, headers=tenant_headers(1))
            assert (await duplicate.json())["scenario_id"] == queued["scenario_id"]
            usage = await stack.scheduler_client.get(
                "/reef/scheduler/status", headers=tenant_headers(1, scheduler=True)
            )
            status = await usage.json()
            assert (status["allocated"], status["used"], status["available"]) == (2, 2, 0)
            removed = await first.delete("/reef/scenarios/active", headers=tenant_headers(1))
            assert removed.status == 200, await removed.text()
            assert (await removed.json())["state"] == "deleted"
            assert not stack.dispatchers[0].has_loaded("active")
            assert stack.dispatchers[1].has_loaded("waiting")
            fetched = await stack.scheduler_client.get(
                f"/reef/scenarios/{queued['scenario_id']}", headers=tenant_headers(1, scheduler=True)
            )
            assert (await fetched.json())["state"] == "admitted"
            assert admitted["service_id"] != queued["service_id"]

    asyncio.run(run())


def test_owner_isolation_global_quota_and_private_control_authentication(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            first, second = stack.services
            response = await first.post("/reef/scenarios", json={"name": "shared-name"}, headers=tenant_headers(1))
            alice = await response.json()
            assert response.status == 201
            collision = await first.post("/reef/scenarios", json={"name": "shared-name"}, headers=tenant_headers(2))
            assert collision.status == 403
            wrong_owner = await first.get("/reef/scenarios/shared-name/contract", headers=tenant_headers(2))
            assert wrong_owner.status == 403
            read_owner = await stack.scheduler_client.get(
                f"/reef/scenarios/{alice['scenario_id']}", headers=tenant_headers(2, scheduler=True)
            )
            assert read_owner.status == 403
            deleted = await first.delete("/reef/scenarios/shared-name", headers=tenant_headers(2))
            assert deleted.status == 403
            bob = await second.post("/reef/scenarios", json={"name": "shared-name"}, headers=tenant_headers(2))
            assert bob.status == 201, await bob.text()
            rejected = await first.post("/reef/scenarios", json={"name": "third"}, headers=tenant_headers(3))
            assert rejected.status == 429, await rejected.text()
            visible = await first.get("/reef/scenarios", headers=tenant_headers(2))
            assert (await visible.json())["scenarios"] == []
            missing_tenant = await first.post(
                "/reef/scenarios", json={"name": "missing"}, headers={"Authorization": f"Bearer {SERVICE_TOKEN}"}
            )
            assert missing_tenant.status == 400
            for token in (SERVICE_TOKEN, SCHEDULER_TOKEN):
                denied = await first.post(
                    "/reef/scheduler/control",
                    json={"operation": "describe"},
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert denied.status == 401
            allowed = await first.post(
                "/reef/scheduler/control",
                json={"operation": "describe"},
                headers={"Authorization": f"Bearer {CONTROL_TOKEN}"},
            )
            assert allowed.status == 200

    asyncio.run(run())


def test_scheduler_restart_preserves_queue_and_same_name_recreation(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            first, second = stack.services
            response = await first.post("/reef/scenarios", json={"name": "reused"}, headers=tenant_headers(1))
            original = await response.json()
            queued_response = await second.post("/reef/scenarios", json={"name": "queued"}, headers=tenant_headers(1))
            queued = await queued_response.json()
            assert queued_response.status == 202
            await stack.restart_scheduler()
            status_response = await stack.scheduler_client.get(
                "/reef/scheduler/status", headers=tenant_headers(1, scheduler=True)
            )
            status = await status_response.json()
            assert status["used"] == 2
            assert {value["scenario_id"] for value in status["scenarios"]} == {
                original["scenario_id"],
                queued["scenario_id"],
            }
            removed_queue = await second.delete("/reef/scenarios/queued", headers=tenant_headers(1))
            assert removed_queue.status == 200
            assert not stack.dispatchers[1].has_loaded("queued")
            removed = await first.delete("/reef/scenarios/reused", headers=tenant_headers(1))
            assert removed.status == 200
            recreated_response = await first.post(
                "/reef/scenarios", json={"name": "reused"}, headers=tenant_headers(1)
            )
            assert recreated_response.status == 201, await recreated_response.text()
            recreated = await recreated_response.json()
            assert recreated["scenario_id"] > queued["scenario_id"]
            assert stack.dispatchers[0].scheduled_scenario_status("reused").scenario_id == recreated["scenario_id"]

    asyncio.run(run())


def test_disabled_scheduler_preserves_existing_scenario_http_contract(tmp_path):
    async def run() -> None:
        dispatcher = service_dispatcher(tmp_path, scheduled=False)
        try:
            async with TestClient(TestServer(create_app(dispatcher))) as client:
                created = await client.post("/reef/scenarios", json={"name": "legacy"})
                assert created.status == 201
                payload = await created.json()
                assert payload["scenario"] == "legacy"
                assert payload["release_id"]
                assert "scenario_id" not in payload
                deleted = await client.delete("/reef/scenarios/legacy")
                assert deleted.status == 200
                assert not dispatcher.has_scenario("legacy")
        finally:
            await asyncio.to_thread(dispatcher.close)

    asyncio.run(run())


def test_model_creation_keeps_private_configuration_out_of_scheduler_state(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            first, second = stack.services
            occupied = await first.post("/reef/scenarios", json={"name": "occupied"}, headers=tenant_headers(1))
            assert occupied.status == 201
            model = {
                "url": "http://provider.invalid/v1",
                "model": "test-model",
                "api": "openai",
                "api_key": "synthetic-provider-key-for-test",
            }
            response = await second.post(
                "/reef/scenarios", json={"name": "custom-model", "model": model}, headers=tenant_headers(1)
            )
            assert response.status == 202, await response.text()
            admission = await response.json()
            stored = stack.controller.scheduler.get_scenario_admission(admission["scenario_id"], 1)
            assert stored.model_config_ref is not None and len(stored.model_config_ref) == 64
            assert not stack.dispatchers[1].has_loaded("custom-model")
            assert model["api_key"].encode() not in stack.database.read_bytes()
            private_files = tuple((tmp_path / "production" / "model-settings").glob("*-model.json"))
            assert len(private_files) == 1
            assert private_files[0].stat().st_mode & 0o777 == 0o600
            conflicting = await second.post(
                "/reef/scenarios",
                json={"name": "custom-model", "model": {**model, "model": "changed"}},
                headers=tenant_headers(1),
            )
            assert conflicting.status == 409
            assert tuple((tmp_path / "production" / "model-settings").glob("*-model.json")) == private_files
            released = await first.delete("/reef/scenarios/occupied", headers=tenant_headers(1))
            assert released.status == 200, await released.text()
            scenario = stack.dispatchers[1].get_or_create_scenario("custom-model", allow_implicit_creation=False)
            assert scenario is not None
            assert scenario.model_config.view() == {
                "url": model["url"],
                "model": model["model"],
                "api": "openai",
                "has_api_key": True,
            }

    asyncio.run(run())


def test_implicit_business_and_install_requests_queue_only_metadata(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            first, second = stack.services
            occupied = await first.post("/reef/scenarios", json={"name": "occupied"}, headers=tenant_headers(1))
            assert occupied.status == 201
            marker = "provider-request-content-must-not-enter-admission-storage"
            inference = {"model": "test-model", "messages": [{"role": "user", "content": marker}]}
            record = {"agent_record_id": "imported", "request_type": "inference", "payload": inference}
            requests = (
                ("/v1/chat/completions", "inference", inference),
                ("/reef/report", "report", {"score": 1, "feedback": marker}),
                ("/reef/records", "record", record),
                ("/reef/records/batch", "batch", {"records": [record]}),
            )
            queued_names = []
            for path, name, body in requests:
                response = await second.post(path, json=body, headers={**tenant_headers(1), "x-reef-scenario": name})
                assert response.status == 409, await response.text()
                admission = await response.json()
                assert admission["state"] == "queued"
                assert admission["scenario_name"] == name
                queued_names.append(name)
            for named in (True, False):
                headers = tenant_headers(1)
                if named:
                    headers["x-reef-scenario"] = "install"
                response = await second.get("/reef/harness/install?adapter=pi", headers=headers)
                assert response.status == 409, await response.text()
                admission = await response.json()
                assert admission["state"] == "queued"
                queued_names.append(admission["scenario_name"])
            assert len(set(queued_names)) == 6
            assert all(not stack.dispatchers[1].has_scenario(name) for name in queued_names)
            assert all(stack.dispatchers[1].scheduled_scenario_status(name) is None for name in queued_names)
            assert marker.encode() not in stack.database.read_bytes()
            before = stack.controller.scheduler.snapshot().next_scenario_id
            train = await second.post(
                "/reef/train",
                json={"text": "perform evolution"},
                headers={**tenant_headers(1), "x-reef-scenario": "unknown-train"},
            )
            assert train.status == 404
            assert stack.controller.scheduler.snapshot().next_scenario_id == before

    asyncio.run(run())


def test_only_dedicated_evaluation_tokens_can_omit_tenant_identity(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            service = stack.services[0]
            created = await service.post("/reef/scenarios", json={"name": "evaluation"}, headers=tenant_headers(1))
            assert created.status == 201
            route = "/reef/scenarios/evaluation/evaluation/v1/chat/completions"
            payload = {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}
            missing = await service.post(route, json=payload, headers={"Authorization": f"Bearer {SERVICE_TOKEN}"})
            assert missing.status == 400
            wrong = await service.post(route, json=payload, headers=tenant_headers(2))
            assert wrong.status == 403
            correct = await service.post(route, json=payload, headers=tenant_headers(1))
            assert correct.status == 200, await correct.text()
            internal = await service.post(route, json=payload, headers={"Authorization": f"Bearer {EVALUATION_TOKEN}"})
            assert internal.status == 200, await internal.text()
            internal_wrong_owner = await service.post(
                route,
                json=payload,
                headers={"Authorization": f"Bearer {EVALUATION_TOKEN}", "x-reef-tenant": "2"},
            )
            assert internal_wrong_owner.status == 403

    asyncio.run(run())


def test_invalid_implicit_requests_do_not_allocate_a_new_tenant_or_scenario(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            service = stack.services[0]
            before = stack.controller.scheduler.snapshot()
            valid_record = {
                "agent_record_id": "valid-import",
                "request_type": "inference",
                "payload": {"messages": [{"role": "user", "content": "hello"}]},
            }
            invalid_requests = (
                ("/reef/report", "invalid-report", {"score": "bad"}),
                ("/reef/records", "invalid-record", {"request_type": "inference", "payload": {}}),
                ("/reef/records/batch", "empty-batch", {"records": []}),
                ("/reef/records/batch", "invalid-batch", {"records": [valid_record, {"payload": {}}]}),
            )
            for path, name, body in invalid_requests:
                response = await service.post(path, json=body, headers={**tenant_headers(2), "x-reef-scenario": name})
                assert response.status == 400, await response.text()
                assert stack.controller.scheduler.snapshot() == before
                assert not stack.dispatchers[0].has_scenario(name)
                assert stack.dispatchers[0].scheduled_scenario_status(name) is None
            for path in ("/reef/harness/install?adapter=unknown", "/reef/harness/install"):
                response = await service.get(path, headers=tenant_headers(2))
                assert response.status == 400, await response.text()
                assert stack.controller.scheduler.snapshot() == before
            assert stack.controller.scheduler.tenant_allocations() == ()
            assert stack.dispatchers[0].list_scenarios() == ()

    asyncio.run(run())


def test_controller_rechecks_cleanup_state_and_retains_charge_on_wrong_identity():
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"recipe": 2}))
    scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
    scheduler.set_service_availability(1, True)
    queued = scheduler.request_scenario_admission(1, "local", "recipe", "work")
    starting = scheduler.next_scenario_admission(1)
    pending = scheduler.request_cleanup(queued.scenario_id, 1)
    control = ScriptedControl(
        [
            (
                "cleanup",
                {
                    "scenario_id": queued.scenario_id + 1,
                    "tenant_id": 1,
                    "generation": pending.generation,
                    "cleanup_complete": True,
                },
            )
        ]
    )
    controller = SchedulerController(scheduler, {1: control})
    controller.reconcile_scenario(starting)
    assert scheduler.get_scenario_admission(queued.scenario_id, 1) == pending
    assert controller.status(1)["used"] == 2
    assert "identity mismatch" in controller.errors[1]
    scheduler.close()


def test_controller_resumes_starting_service_assignment_after_restart():
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"recipe": 2}))
    scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
    scheduler.set_service_availability(1, True)
    queued = scheduler.request_scenario_admission(1, "local", "recipe", "work", now_seconds=100)
    starting = scheduler.next_scenario_admission(1, now_seconds=100)
    identity = {"scenario_id": queued.scenario_id, "tenant_id": 1, "generation": starting.generation}
    control = ScriptedControl(
        [
            ("status", {**identity, "state": "starting"}),
            ("start", {**identity, "state": "admitted", "idle_since": 120}),
        ]
    )
    controller = SchedulerController(scheduler, {1: control})
    controller.reconcile_scenario(starting)
    recovered = scheduler.get_scenario_admission(queued.scenario_id, 1)
    assert recovered.state == "admitted"
    assert recovered.idle_since == 120
    assert control.operations == ["status", "start"]
    scheduler.close()


@pytest.mark.parametrize("remote_state", ["admitted", "reclaimed", "reclaim_pending"])
def test_controller_reclaims_only_after_matching_service_confirmation(remote_state):
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"recipe": 2}, idle_ttl_seconds=10)
    )
    scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
    scheduler.set_service_availability(1, True)
    queued = scheduler.request_scenario_admission(1, "local", "recipe", "work", now_seconds=100)
    starting = scheduler.next_scenario_admission(1, now_seconds=100)
    scheduler.confirm_started(starting.scenario_id, starting.generation)
    scheduler.report_scenario_idle(starting.scenario_id, starting.generation, idle_since=100)
    pending = scheduler.request_reclaim(queued.scenario_id, now_seconds=120)
    response = {
        "scenario_id": queued.scenario_id,
        "tenant_id": 1,
        "generation": pending.generation,
        "state": remote_state,
        "reclaim_complete": remote_state == "reclaimed",
        "idle_since": None,
    }
    control = ScriptedControl([("reclaim", response)])
    controller = SchedulerController(scheduler, {1: control})
    controller.reconcile_scenario(pending)
    current = scheduler.get_scenario_admission(queued.scenario_id, 1)
    expected_state = "sleep" if remote_state == "reclaimed" else remote_state
    assert current.state == expected_state
    assert controller.status(1)["used"] == (0 if remote_state == "reclaimed" else 2)
    scheduler.stop_accepting_requests()
    controller.reconcile()
    assert control.operations == ["reclaim"]
    scheduler.close()


@pytest.mark.parametrize(
    ("tenant", "recipe", "scenario", "expected"),
    [
        (1, "other-recipe", "new", AdmissionConflict),
        (1, "recipe", "invalid/name", ValueError),
        (99, "recipe", "new", UnknownTenant),
        (2, "recipe", "owned", ScenarioTenantMismatch),
    ],
)
def test_controller_refuses_invalid_model_requests_before_staging(tenant, recipe, scenario, expected):
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(4, {1: 2, 2: 2}, {"recipe": 2}))
    scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
    scheduler.set_service_availability(1, True)
    scheduler.request_scenario_admission(1, "local", "recipe", "owned")
    control = ScriptedControl([])
    controller = SchedulerController(scheduler, {1: control})
    with pytest.raises(expected):
        controller.submit({"service_name": "local", "recipe": recipe, "scenario_name": scenario, "model": {}}, tenant)
    assert control.operations == []
    scheduler.close()


def test_idle_reclaim_runs_waiter_and_keeps_same_scenario_asleep_over_http(tmp_path):
    async def run():
        async with multi_service_cluster(tmp_path, idle_ttl_seconds=600) as stack:
            first, second = stack.services
            response = await first.post("/reef/scenarios", json={"name": "active"}, headers=tenant_headers(1))
            assert response.status == 201, await response.text()
            original = await response.json()
            record = await first.post(
                "/reef/records",
                json={"agent_record_id": "retained", "request_type": "inference", "payload": {"messages": []}},
                headers={**tenant_headers(1), "x-reef-scenario": "active"},
            )
            assert record.status == 200, await record.text()
            before = stack.dispatchers[0].get_or_create_scenario("active", allow_implicit_creation=False)
            release_id = before.repository.require_current_artifact().release_id
            waiting = await second.post("/reef/scenarios", json={"name": "waiting"}, headers=tenant_headers(1))
            assert waiting.status == 202, await waiting.text()
            scheduler = stack.controller.scheduler
            status = stack.dispatchers[0].scheduled_scenario_status("active")
            assert status.idle_since is not None
            scheduler.report_scenario_idle(status.scenario_id, status.generation, idle_since=status.idle_since)
            pending = scheduler.request_reclaim(status.scenario_id, now_seconds=status.idle_since + 600)
            assert pending.state == "reclaim_pending"
            await asyncio.to_thread(stack.controller.reconcile)
            reclaimed = scheduler.get_scenario_admission(original["scenario_id"], 1)
            assert reclaimed.state == "sleep"
            assert not stack.dispatchers[0].has_loaded("active")
            assert stack.dispatchers[0].has_scenario("active")
            assert stack.dispatchers[1].has_loaded("waiting")
            assert stack.controller.status(1)["used"] == 2
            await stack.restart_scheduler()
            removed = await second.delete("/reef/scenarios/waiting", headers=tenant_headers(1))
            assert removed.status in {200, 202}, await removed.text()
            for _ in range(100):
                await asyncio.to_thread(stack.controller.reconcile)
                resumed = stack.controller.scheduler.get_scenario_admission(original["scenario_id"], 1)
                assert resumed.state == "sleep"
            assert resumed.generation == pending.generation
            assert not stack.dispatchers[0].has_loaded("active")
            assert original["scenario_id"] in stack.controller.scheduler.cold_scenarios_by_id
            assert before.repository.require_current_artifact().release_id == release_id

    asyncio.run(run())


def test_admitted_service_keeps_serving_after_scheduler_shutdown(tmp_path):
    async def run():
        async with multi_service_cluster(tmp_path) as stack:
            service = stack.services[0]
            created = await service.post("/reef/scenarios", json={"name": "active"}, headers=tenant_headers(1))
            assert created.status == 201
            await stack.scheduler_client.close()
            running = await service.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "still running"}]},
                headers={**tenant_headers(1), "x-reef-scenario": "active"},
            )
            assert running.status == 200, await running.text()
            assert stack.dispatchers[0].has_loaded("active")
            rejected = await service.post("/reef/scenarios", json={"name": "late"}, headers=tenant_headers(1))
            assert rejected.status == 503

    asyncio.run(run())


def test_idle_ttl_reports_unsupported_storage_before_accepting_work():
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"recipe": 2}, idle_ttl_seconds=10)
    )
    scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
    description = {"service_id": 1, "service_name": "local", "recipe": "recipe", "standard_execution_units": 2}
    control = ScriptedControl([("describe", {**description, "supports_idle_reclaim": False})])
    controller = SchedulerController(scheduler, {1: control})
    try:
        controller.describe_services()
        service = controller.status(1)["services"][0]
        assert not service["available"]
        assert "durable" in service["reason"]
        assert scheduler.tenant_allocations() == ()
        control.replies.append(("describe", {**description, "supports_idle_reclaim": True}))
        controller.describe_services()
        assert controller.status(1)["services"][0]["available"]
        assert controller.status(1)["services"][0]["reason"] is None
    finally:
        controller.close()


def test_stale_reclaim_confirmation_does_not_release_current_generation():
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"recipe": 2}, idle_ttl_seconds=10)
    )
    scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://service.invalid", 2))
    scheduler.set_service_availability(1, True)
    admission = scheduler.request_scenario_admission(1, "local", "recipe", "work", now_seconds=100)
    starting = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(admission.scenario_id, starting.generation)
    scheduler.report_scenario_idle(admission.scenario_id, starting.generation, idle_since=100)
    pending = scheduler.request_reclaim(admission.scenario_id, now_seconds=120)
    control = ScriptedControl(
        [
            (
                "reclaim",
                {
                    "scenario_id": admission.scenario_id,
                    "tenant_id": 1,
                    "generation": starting.generation,
                    "state": "reclaimed",
                    "reclaim_complete": True,
                },
            )
        ]
    )
    controller = SchedulerController(scheduler, {1: control})
    try:
        controller.reconcile_scenario(pending)
        assert scheduler.get_scenario_admission(admission.scenario_id, 1) == pending
        assert controller.status(1)["used"] == 2
        assert "identity mismatch" in controller.errors[1]
    finally:
        controller.close()


def test_out_of_range_numeric_tenant_is_rejected_before_status_lookup(tmp_path):
    async def run() -> None:
        async with multi_service_cluster(tmp_path) as stack:
            response = await stack.scheduler_client.get(
                "/reef/scheduler/status", headers=tenant_headers(2**63, scheduler=True)
            )
            assert response.status == 400
            assert stack.controller.scheduler.tenant_allocations() == ()

    asyncio.run(run())
