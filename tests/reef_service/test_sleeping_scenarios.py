"""Sleeping scenarios retain work without execution resources and wake only for evolve."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from reef.artifact import InMemoryRepositoryBackend
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentLogger
from reef.recipe import Recipe
from reef.scenario.scheduling import AdmissionConflict, MultiTenantScheduler, SchedulerConfig, ServiceRegistration
from reef.service.scheduler_control import SchedulerController
from reef.service.scheduling import SchedulerUnavailable, ServiceSchedulerClient
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.records import RecordConflict, RecordStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore
from reef.train.cordis_backend.processor import RecordDrivenTraceProcessor
from reef.train.trainer import Trainer

from .test_manual_training import CaptureBackend
from .test_multi_tenant_service import ScriptedControl, multi_service_cluster, tenant_headers


class ManualRecipe(Recipe):
    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, object] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        return Trainer.build(
            scenario,
            records,
            processor_factory=RecordDrivenTraceProcessor,
            candidate_backend=CaptureBackend(),
            algorithm_state=algorithm_state,
            training_mode=self.training_mode,
            experiment_logger=experiment_logger,
        )


@pytest.mark.parametrize("field", ["scenario_id", "tenant_id", "generation", "wake_generation"])
def test_invalid_remote_wake_identity_keeps_scenario_asleep(field: str) -> None:
    with closing(
        MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(1, {1: 1}, {"recipe": 1}, idle_ttl_seconds=1))
    ) as scheduler:
        scheduler.register_service(ServiceRegistration(1, "local", "recipe", "http://localhost", 1))
        scheduler.set_service_availability(1, True)
        first = scheduler.request_scenario_admission(1, "local", "recipe", "work")
        started = scheduler.next_scenario_admission(1)
        scheduler.confirm_started(first.scenario_id, started.generation)
        scheduler.report_scenario_idle(first.scenario_id, started.generation, idle_since=1)
        reclaim = scheduler.request_reclaim(first.scenario_id, now_seconds=3)
        sleeper = scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation)
        remote = {
            "scenario_id": first.scenario_id,
            "tenant_id": 1,
            "generation": reclaim.generation,
            "state": "reclaimed",
            "reclaim_complete": True,
            "wake_generation": reclaim.generation,
            "wake_request_id": "pending",
            field: True,
        }
        controller = SchedulerController(scheduler, {1: ScriptedControl([("status", remote)])})
        with pytest.raises(ValueError, match="positive integer"):
            controller.wake(
                first.scenario_id, 1, {"service_id": 1, "generation": reclaim.generation, "agent_record_id": "pending"}
            )
        assert scheduler.cold_scenarios_by_id == {first.scenario_id: sleeper}
        assert scheduler.used_by_tenant == {} and scheduler.waiting_by_tenant == {}


def test_cold_evolve_queues_once_and_executes_after_quota_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        async with multi_service_cluster(
            tmp_path, idle_ttl_seconds=600, recipe=ManualRecipe(training_mode="manual")
        ) as stack:
            service, other = stack.services
            dispatcher = stack.dispatchers[0]
            created = await service.post("/reef/scenarios", json={"name": "work"}, headers=tenant_headers(1))
            identity = await created.json()
            assert created.status == 201
            before = dispatcher.get_or_create_scenario("work")
            release = before.current_artifact_ref()
            status = dispatcher.scheduled_scenario_status("work")
            assert status is not None and status.idle_since is not None
            scheduler = stack.controller.scheduler
            scheduler.report_scenario_idle(status.scenario_id, status.generation, idle_since=status.idle_since)
            reclaim = scheduler.request_reclaim(status.scenario_id, now_seconds=status.idle_since + 600)
            await asyncio.to_thread(stack.controller.reconcile)
            assert scheduler.get_scenario_admission(status.scenario_id, 1).state == "sleep"
            assert not dispatcher.has_loaded("work")
            for _ in range(3):
                await asyncio.to_thread(stack.controller.reconcile)
            assert status.scenario_id in scheduler.cold_scenarios_by_id

            occupied = await other.post("/reef/scenarios", json={"name": "occupied"}, headers=tenant_headers(1))
            assert occupied.status in {201, 202}, await occupied.text()
            occupied_id = (await occupied.json())["scenario_id"]
            async with asyncio.timeout(5):
                while scheduler.get_scenario_admission(occupied_id, 1).state != "admitted":
                    await asyncio.to_thread(stack.controller.reconcile)
                    await asyncio.sleep(0.01)
            headers = {**tenant_headers(1), "x-reef-scenario": "work"}
            body = {"agent_record_id": "wake-1", "text": "Revise the rule", "session": "test", "release_id": ""}

            def unavailable(self, scenario_id, tenant_id, generation, record_id):
                raise SchedulerUnavailable("scheduler temporarily offline")

            with monkeypatch.context() as patch:
                patch.setattr(ServiceSchedulerClient, "wake", unavailable)
                response = await service.post("/reef/train", json=body, headers=headers)
                assert response.status == 202, await response.text()
                assert scheduler.get_scenario_admission(status.scenario_id, 1).state == "sleep"
                assert len(dispatcher.scenario_admissions.pending_wakes()) == 1
            for _ in range(100):
                if scheduler.get_scenario_admission(status.scenario_id, 1).state == "queued":
                    break
                await asyncio.sleep(0.02)
            assert response.status == 202, await response.text()
            accepted = await response.json()
            assert accepted["agent_record_id"] == "wake-1"
            queued = scheduler.get_scenario_admission(identity["scenario_id"], 1)
            assert queued.state == "queued" and not queued.execution_allocated
            sequence = queued.queue_sequence
            assert not dispatcher.has_loaded("work")
            duplicate = await service.post("/reef/train", json=body, headers=headers)
            assert duplicate.status == 202
            assert scheduler.get_scenario_admission(status.scenario_id, 1).queue_sequence == sequence
            rejected = await service.post("/reef/train", json={**body, "agent_record_id": "wake-2"}, headers=headers)
            assert rejected.status == 409
            forbidden = await service.post("/reef/train", json=body, headers={**headers, "x-reef-tenant": "2"})
            assert forbidden.status == 403
            progress = await service.get("/reef/harness/requests/wake-1/progress", headers=headers)
            assert progress.status == 200, await progress.text()
            assert (await progress.json())["state"] == "queued"
            page = await service.get("/reef/harness/requests/wake-1/page", headers=headers)
            assert page.status == 200, await page.text()
            assert not dispatcher.has_loaded("work")
            await stack.restart_scheduler()
            assert stack.controller.scheduler.get_scenario_admission(status.scenario_id, 1).queue_sequence == sequence
            await other.delete("/reef/scenarios/occupied", headers=tenant_headers(1))
            for _ in range(100):
                await asyncio.to_thread(stack.controller.reconcile)
                current = dispatcher.loaded_scenario("work")
                if current is not None and current.scenario_step == 1:
                    break
                await asyncio.sleep(0.01)
            assert current is not None and current is not before and current.scenario_step == 1
            assert current.current_artifact_ref() == release
            assert current.training_mode == "manual"
            assert sum("wake-1" in commit.consumed_ids for commit in current.store.history()) == 1
            assert dispatcher.scenario_admissions.pending_wakes() == ()
            assert (
                stack.controller.scheduler.get_scenario_admission(status.scenario_id, 1).generation
                > reclaim.generation
            )
            scheduler = stack.controller.scheduler
            for _ in range(100):
                idle = dispatcher.scheduled_scenario_status("work")
                if idle is not None and idle.idle_since is not None:
                    scheduler.report_scenario_idle(idle.scenario_id, idle.generation, idle_since=idle.idle_since)
                    scheduler.request_reclaim(idle.scenario_id, now_seconds=idle.idle_since + 600)
                    await asyncio.to_thread(stack.controller.reconcile)
                    if scheduler.get_scenario_admission(idle.scenario_id, 1).state == "sleep":
                        break
                await asyncio.sleep(0.01)
            assert not dispatcher.has_loaded("work")
            progress = await service.get("/reef/harness/requests/wake-1/progress", headers=headers)
            assert progress.status == 200, await progress.text()
            assert (await progress.json())["settled"]
            page = await service.get("/reef/harness/requests/wake-1/page", headers=headers)
            assert page.status == 200, await page.text()
            duplicate = await service.post("/reef/train", json=body, headers=headers)
            assert duplicate.status in (200, 202), await duplicate.text()
            assert scheduler.get_scenario_admission(status.scenario_id, 1).state == "sleep"
            assert dispatcher.scenario_admissions.pending_wakes() == ()

    asyncio.run(run())


def test_pending_evolve_survives_service_restart_and_delivery_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    factory = InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository")

    def open_dispatcher() -> Dispatcher:
        return Dispatcher(
            ManualRecipe(training_mode="manual"),
            factory,
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            agent_record_dir=tmp_path / "models",
            require_scenario_admission=True,
            scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
        )

    first = open_dispatcher()
    monkeypatch.setattr(first, "_wake_training", lambda current: None)
    first.start_scheduled_scenario(1, 1, "work")
    status = first.scheduled_scenario_status("work")
    assert status is not None and status.idle_since is not None
    assert first.reclaim_scheduled_scenario(
        1, "work", tenant_id=1, generation=2, inactive_before=status.idle_since
    ).reclaim_complete
    item = AgentRecord.create(
        scenario="work",
        request_type=RequestType.TRAIN,
        payload={"text": "Keep this instruction", "session": "test", "release_id": ""},
        agent_record_id="pending",
    )
    first.stage_scheduled_training(item, 1)
    assert not first.has_loaded("work")
    with pytest.raises(RecordConflict):
        first.stage_scheduled_training(replace(item, payload={**item.payload, "text": "Different"}), 1)
    first.close()
    with closing(open_dispatcher()) as restarted:
        pending = restarted.scenario_admissions.pending_wakes()
        assert len(pending) == 1 and pending[0].pending_training.item == item
        with closing(restarted._storage.open("work")) as store:
            store.records.append(item)
        # Simulate a crash after record append, before removing the retained copy.
        restarted.start_scheduled_scenario(1, 1, "work", generation=3)
        current = restarted.get_or_create_scenario("work")
        with closing(restarted._storage.open("work")) as store:
            assert store.records.count("work", request_type=RequestType.TRAIN) == 1
        assert restarted.scenario_admissions.pending_wakes() == ()
        assert current.training_mode == "manual"


def test_cold_request_write_failure_and_explicit_delete_keep_lifecycle_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    with closing(
        Dispatcher(
            ManualRecipe(training_mode="manual"),
            InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            require_scenario_admission=True,
            scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
        )
    ) as dispatcher:
        monkeypatch.setattr(dispatcher, "_wake_training", lambda current: None)
        dispatcher.start_scheduled_scenario(1, 1, "work")
        idle = dispatcher.scheduled_scenario_status("work")
        assert idle is not None and idle.idle_since is not None
        dispatcher.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since)
        item = AgentRecord.create(
            scenario="work",
            request_type=RequestType.TRAIN,
            agent_record_id="pending",
            payload={"text": "Keep this instruction", "session": "test", "release_id": ""},
        )
        storage = dispatcher.scenario_admissions.storage

        def fail_write(name: str, value: Mapping[str, object]) -> None:
            raise OSError("disk unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(storage, "write", fail_write)
            with pytest.raises(OSError, match="disk unavailable"):
                dispatcher.stage_scheduled_training(item, 1)
        assert dispatcher.scenario_admissions.pending_wakes() == ()
        assert dispatcher.stage_scheduled_training(item, 1) == item
        assert not dispatcher.has_loaded("work")
        assert dispatcher.cleanup_scheduled_scenario(1, "work", generation=3, tenant_id=1).cleanup_complete
        assert dispatcher.scenario_admissions.pending_wakes() == ()
        assert dispatcher.scheduled_request_snapshot("work", "pending")[2].state == "deleted"
        assert not list((tmp_path / "records").glob("*.sqlite3"))
        dispatcher.start_scheduled_scenario(2, 1, "work")
        assert dispatcher.scenario_admissions.get_operation("work").pending_training is None


def test_schema_two_migrates_only_reclaimed_waiters_and_rebuilds_cold_dict(tmp_path: Path) -> None:
    database = tmp_path / "scheduler.sqlite3"
    config = SchedulerConfig(1, {1: 1}, {"recipe": 1}, idle_ttl_seconds=1)
    registration = ServiceRegistration(1, "local", "recipe", "http://localhost", 1)
    with closing(SQLiteSchedulingStore(database)) as store:
        scheduler = MultiTenantScheduler(store, config)
        scheduler.register_service(registration)
        scheduler.set_service_availability(1, True)
        first = scheduler.request_scenario_admission(1, "local", "recipe", "first")
        claim = scheduler.next_scenario_admission(1)
        scheduler.confirm_started(first.scenario_id, claim.generation)
        second = scheduler.request_scenario_admission(1, "local", "recipe", "second")
        scheduler.report_scenario_idle(first.scenario_id, claim.generation, idle_since=1)
        reclaim = scheduler.request_reclaim(first.scenario_id, now_seconds=3)
        scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation)
        with store.connection:
            store.connection.execute("UPDATE scheduler_metadata SET schema_version = 2")
            store.connection.execute(
                "UPDATE scenario_admission SET state = 'queued' WHERE scenario_id = ?", (first.scenario_id,)
            )
    with closing(SQLiteSchedulingStore(database)) as store:
        restarted = MultiTenantScheduler(store, config)
        assert set(restarted.cold_scenarios_by_id) == {first.scenario_id}
        assert restarted.waiting_by_tenant == {1: (second.scenario_id,)}
        assert store.connection.execute("SELECT schema_version FROM scheduler_metadata").fetchone()[0] == 3
        with pytest.raises(AdmissionConflict):
            restarted.wake_scenario_admission(first.scenario_id, 1, reclaim.generation - 1)
