"""Queue logs describe committed FIFO changes without exposing creation settings."""

import json
import logging
from contextlib import closing
from pathlib import Path

import pytest

from reef.scenario.scheduling import MultiTenantScheduler, SchedulerConfig, ServiceRegistration
from reef.storage.scheduling import SchedulingStoreConflict
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore


def create_scheduler(database: Path | str = ":memory:") -> MultiTenantScheduler:
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(database),
        SchedulerConfig(6, {1: 2, 2: 4}, {"harness": 2}, idle_ttl_seconds=1),
    )
    for service_id, name in ((11, "development"), (12, "production")):
        scheduler.register_service(ServiceRegistration(service_id, name, "harness", "http://localhost", 2))
        scheduler.set_service_availability(service_id, True)
    return scheduler


def queue_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    events = []
    for record in caplog.records:
        if record.name != "reef.scenario.scheduling" or not record.getMessage().startswith("tenant_queue_changed "):
            continue
        assert record.levelno == logging.INFO
        event = json.loads(record.getMessage().removeprefix("tenant_queue_changed "))
        assert set(event) == {
            "tenant_id",
            "queue_before",
            "queue",
            "allocated_scenario_ids",
            "used_execution_units",
            "allocated_execution_units",
        }
        events.append(event)
    return events


def test_queue_logs_follow_admission_cancellation_and_explicit_wake(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="reef.scenario.scheduling")
    with closing(create_scheduler()) as scheduler:
        first = scheduler.request_scenario_admission(1, "development", "harness", "first")
        starting = scheduler.next_scenario_admission(1)
        assert starting is not None
        scheduler.confirm_started(first.scenario_id, starting.generation)
        second = scheduler.request_scenario_admission(1, "production", "harness", "second")
        cancelled = scheduler.request_scenario_admission(1, "development", "harness", "cancelled")
        scheduler.request_cleanup(cancelled.scenario_id, 1)
        scheduler.report_scenario_idle(first.scenario_id, starting.generation, idle_since=100)
        (reclaim,) = scheduler.reclaim_idle_scenarios(now_seconds=101)
        scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation)
        following = scheduler.next_scenario_admission(1)
        assert following is not None and following.scenario_id == second.scenario_id
        scheduler.wake_scenario_admission(first.scenario_id, 1, reclaim.generation)
        cleanup = scheduler.request_cleanup(second.scenario_id, 1)
        scheduler.confirm_cleaned(second.scenario_id, cleanup.generation)
        resumed = scheduler.next_scenario_admission(1)
        assert resumed is not None and resumed.scenario_id == first.scenario_id

    first_row = {"scenario_id": first.scenario_id, "service_id": 11, "scenario_name": "first"}
    second_row = {"scenario_id": second.scenario_id, "service_id": 12, "scenario_name": "second"}
    cancelled_row = {"scenario_id": cancelled.scenario_id, "service_id": 11, "scenario_name": "cancelled"}
    events = queue_events(caplog)
    assert all(event["tenant_id"] == 1 and event["allocated_execution_units"] == 2 for event in events)
    assert [
        (event["queue_before"], event["queue"], event["allocated_scenario_ids"], event["used_execution_units"])
        for event in events
    ] == [
        ([], [first_row], [], 0),
        ([first.scenario_id], [], [first.scenario_id], 2),
        ([], [second_row], [first.scenario_id], 2),
        ([second.scenario_id], [second_row, cancelled_row], [first.scenario_id], 2),
        ([second.scenario_id, cancelled.scenario_id], [second_row], [first.scenario_id], 2),
        ([second.scenario_id], [], [second.scenario_id], 2),
        ([], [first_row], [second.scenario_id], 2),
        ([first.scenario_id], [], [first.scenario_id], 2),
    ]


def test_queue_logs_separate_tenants_and_omit_private_creation_settings(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="reef.scenario.scheduling")
    with closing(create_scheduler()) as scheduler:
        first = scheduler.request_scenario_admission(
            1, "development", "harness", "shared-name", release_id="private-release", model_config_ref="a" * 64
        )
        second = scheduler.request_scenario_admission(2, "production", "harness", "shared-name")
        scheduler.request_cleanup(first.scenario_id, 1)
        starting = scheduler.next_scenario_admission(2)
        assert starting is not None and starting.scenario_id == second.scenario_id

    assert queue_events(caplog) == [
        {
            "tenant_id": 1,
            "queue_before": [],
            "queue": [{"scenario_id": first.scenario_id, "service_id": 11, "scenario_name": "shared-name"}],
            "allocated_scenario_ids": [],
            "used_execution_units": 0,
            "allocated_execution_units": 2,
        },
        {
            "tenant_id": 2,
            "queue_before": [],
            "queue": [{"scenario_id": second.scenario_id, "service_id": 12, "scenario_name": "shared-name"}],
            "allocated_scenario_ids": [],
            "used_execution_units": 0,
            "allocated_execution_units": 4,
        },
        {
            "tenant_id": 1,
            "queue_before": [first.scenario_id],
            "queue": [],
            "allocated_scenario_ids": [],
            "used_execution_units": 0,
            "allocated_execution_units": 0,
        },
        {
            "tenant_id": 2,
            "queue_before": [second.scenario_id],
            "queue": [],
            "allocated_scenario_ids": [second.scenario_id],
            "used_execution_units": 2,
            "allocated_execution_units": 4,
        },
    ]
    assert "private-release" not in caplog.text
    assert "a" * 64 not in caplog.text


def test_unchanged_queues_do_not_log_reads_retries_or_stale_notifications(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="reef.scenario.scheduling")
    with closing(create_scheduler()) as scheduler:
        first = scheduler.request_scenario_admission(1, "development", "harness", "first")
        starting = scheduler.next_scenario_admission(1)
        assert starting is not None
        scheduler.confirm_started(first.scenario_id, starting.generation)
        second = scheduler.request_scenario_admission(1, "production", "harness", "second")
        caplog.clear()

        scheduler.snapshot()
        scheduler.list_scenario_admissions(1)
        scheduler.get_scenario_admission(second.scenario_id, 1)
        assert scheduler.request_scenario_admission(1, "production", "harness", "second") == second
        assert scheduler.next_scenario_admission(1) is None
        scheduler.set_service_availability(12, False)
        assert scheduler.next_scenario_admission(1) is None
        assert scheduler.next_scenario_admission(1) is None
        scheduler.set_service_availability(12, True)
        scheduler.report_scenario_idle(first.scenario_id, starting.generation, idle_since=100)
        reclaim = scheduler.request_reclaim(first.scenario_id, now_seconds=101)
        scheduler.cancel_reclaim(first.scenario_id, reclaim.generation)
        scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation)
        scheduler.confirm_started(first.scenario_id, starting.generation)
        scheduler.confirm_cleaned(first.scenario_id, starting.generation)
        assert queue_events(caplog) == []


def test_restored_queue_logs_once_and_failed_sqlite_write_does_not_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="reef.scenario.scheduling")
    database = tmp_path / "scheduler.sqlite3"
    with closing(create_scheduler(database)) as original, closing(create_scheduler(database)) as stale:
        active = original.request_scenario_admission(1, "development", "harness", "active")
        starting = original.next_scenario_admission(1)
        assert starting is not None
        original.confirm_started(active.scenario_id, starting.generation)
        waiting = original.request_scenario_admission(1, "production", "harness", "waiting")
        caplog.clear()

        with closing(create_scheduler(database)) as restored:
            assert queue_events(caplog) == [
                {
                    "tenant_id": 1,
                    "queue_before": [],
                    "queue": [{"scenario_id": waiting.scenario_id, "service_id": 12, "scenario_name": "waiting"}],
                    "allocated_scenario_ids": [active.scenario_id],
                    "used_execution_units": 2,
                    "allocated_execution_units": 2,
                }
            ]
            caplog.clear()
            with pytest.raises(SchedulingStoreConflict):
                stale.request_scenario_admission(2, "development", "harness", "not-persisted")
            assert queue_events(caplog) == []
            assert restored.get_scenario_admission(waiting.scenario_id, 1).state == "queued"
            assert restored.list_scenario_admissions(2) == ()
