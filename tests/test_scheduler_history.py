"""Completed lifecycles remain queryable without accumulating in the active working set."""

from dataclasses import replace
from threading import active_count

import pytest

from reef.scenario.scheduling import (
    AdmissionConflict,
    MultiTenantScheduler,
    ScenarioTenantMismatch,
    SchedulerConfig,
    ServiceRegistration,
)
from reef.storage.sqlite_scheduling import SQLiteSchedulingStore


def test_repeated_lifecycles_release_memory_and_recover_history(tmp_path):
    database = tmp_path / "scheduling.sqlite3"
    config = SchedulerConfig(2, {1: 2}, {"harness": 1}, idle_ttl_seconds=1)
    service = ServiceRegistration(11, "development", "harness", "http://localhost:8901", 1)
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    scheduler.register_service(service)
    scheduler.set_service_availability(service.service_id, True)
    threads_before = active_count()
    for cycle in range(300):
        scenario = scheduler.request_scenario_admission(1, "development", "harness", "work", now_seconds=10)
        if cycle % 2 == 0:
            scheduler.request_cleanup(scenario.scenario_id, 1)
        else:
            started = scheduler.next_scenario_admission(1)
            scheduler.confirm_started(started.scenario_id, started.generation)
            scheduler.report_scenario_idle(started.scenario_id, started.generation, idle_since=10)
            reclaim = scheduler.reclaim_idle_scenarios(now_seconds=12)[0]
            scheduler.confirm_reclaimed(reclaim.scenario_id, reclaim.generation)
            cleanup = scheduler.request_cleanup(scenario.scenario_id, 1)
            scheduler.confirm_cleaned(cleanup.scenario_id, cleanup.generation)
        assert scheduler.list_scenario_admissions() == ()
        assert scheduler.tenant_allocations() == ()
        assert scheduler.snapshot().scenarios == ()
    assert active_count() == threads_before
    assert scheduler.get_scenario_admission(1, 1).state == "deleted"
    assert scheduler.find_latest_scenario_admission(11, "work").scenario_id == 300
    scheduler.close()

    recovered = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    recovered.register_service(service)
    recovered.set_service_availability(service.service_id, True)
    assert recovered.snapshot().scenarios == ()
    assert recovered.get_scenario_admission(1, 1).state == "deleted"
    with pytest.raises(ScenarioTenantMismatch):
        recovered.get_scenario_admission(1, 2)
    with pytest.raises(ValueError, match="positive signed 64-bit integer"):
        recovered.get_scenario_admission(2**80, 1)
    replacement = recovered.request_scenario_admission(1, "development", "harness", "work")
    assert replacement.scenario_id == 301
    assert recovered.request_cleanup(1, 1).state == "deleted"
    assert recovered.get_scenario_admission(replacement.scenario_id, 1).state == "queued"
    recovered.close()


@pytest.mark.parametrize("changes", [{"service_url": "http://localhost:8903"}, {"service_name": "renamed"}])
def test_restart_rejects_directory_changes_for_active_lifecycles(tmp_path, changes):
    database = tmp_path / "scheduler.sqlite3"
    config = SchedulerConfig(1, {1: 1}, {"harness": 1})
    service = ServiceRegistration(11, "development", "harness", "http://localhost:8901", 1)
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    scheduler.register_service(service)
    scheduler.set_service_availability(service.service_id, True)
    scheduler.request_scenario_admission(1, "development", "harness", "work")
    scheduler.close()
    recovered = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    with pytest.raises(AdmissionConflict, match="configuration changed"):
        recovered.register_service(replace(service, **changes))
    recovered.close()


def test_restart_retains_sleep_and_only_writes_changed_rows(tmp_path):
    database = tmp_path / "scheduler.sqlite3"
    config = SchedulerConfig(1, {1: 1}, {"harness": 1}, idle_ttl_seconds=1)
    service = ServiceRegistration(11, "local", "harness", "http://localhost", 1)
    store = SQLiteSchedulingStore(database)
    scheduler = MultiTenantScheduler(store, config)
    scheduler.register_service(service)
    scheduler.set_service_availability(11, True)
    first = scheduler.request_scenario_admission(1, "local", "harness", "first")
    starting = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(first.scenario_id, starting.generation)
    second = scheduler.request_scenario_admission(1, "local", "harness", "second")
    scheduler.report_scenario_idle(first.scenario_id, starting.generation, idle_since=10)
    writes_before = store.connection.total_changes
    reclaim = scheduler.request_reclaim(first.scenario_id, now_seconds=12)
    # One metadata update and one changed scenario, not the tenant or waiting scenario.
    assert store.connection.total_changes - writes_before == 2
    scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation)
    scheduler.close()
    recovered = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    recovered.register_service(service)
    recovered.set_service_availability(11, True)
    assert recovered.next_scenario_admission(1).scenario_id == second.scenario_id
    waiting = recovered.get_scenario_admission(first.scenario_id, 1)
    assert waiting.state == "sleep" and waiting.has_started
    assert recovered.cold_scenarios_by_id == {first.scenario_id: waiting}
    cleanup = recovered.request_cleanup(second.scenario_id, 1)
    recovered.confirm_cleaned(second.scenario_id, cleanup.generation)
    assert recovered.next_scenario_admission(1) is None
    awakened = recovered.wake_scenario_admission(first.scenario_id, 1, waiting.generation)
    assert awakened.queue_sequence > second.queue_sequence
    assert recovered.next_scenario_admission(1).scenario_id == first.scenario_id
    recovered.close()


def test_restart_without_directory_or_recipe_costs_preserves_live_admissions(tmp_path):
    database = tmp_path / "scheduler.sqlite3"
    config = SchedulerConfig(2, {1: 2})
    service = ServiceRegistration(11, "development", "harness", "http://localhost:8901", 2)
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    scheduler.register_service(service)
    scheduler.set_service_availability(11, True)
    first = scheduler.request_scenario_admission(1, "development", "harness", "active")
    starting = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(first.scenario_id, starting.generation)
    second = scheduler.request_scenario_admission(1, "development", "harness", "waiting")
    scheduler.close()

    recovered = MultiTenantScheduler(SQLiteSchedulingStore(database), config)
    assert recovered.service_registrations() == ()
    assert recovered.recipe_execution_units == {"harness": 2}
    assert recovered.get_scenario_admission(first.scenario_id, 1).execution_allocated
    assert recovered.tenant_allocations()[0].allocated_execution_units == 2
    assert recovered.next_scenario_admission(1) is None
    assert recovered.get_scenario_admission(second.scenario_id, 1).queue_sequence == second.queue_sequence
    with pytest.raises(AdmissionConflict, match="static requirement"):
        recovered.register_service(replace(service, standard_execution_units=1))
    with pytest.raises(AdmissionConflict, match="configuration changed"):
        recovered.register_service(replace(service, recipe="other"))
    assert recovered.recipe_execution_units == {"harness": 2}
    recovered.register_service(service)
    recovered.set_service_availability(11, True)
    cleanup = recovered.request_cleanup(first.scenario_id, 1)
    recovered.confirm_cleaned(first.scenario_id, cleanup.generation)
    assert recovered.next_scenario_admission(1).scenario_id == second.scenario_id
    recovered.close()

    store = SQLiteSchedulingStore(database)
    with pytest.raises(ValueError, match="restored scenario requirements"):
        MultiTenantScheduler(store, SchedulerConfig(2, {1: 2}, {"harness": 1}))
    store.close()


def create_version_one_database(path, registration):
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE scheduler_metadata (singleton INTEGER PRIMARY KEY, schema_version INTEGER, "
            "version INTEGER, next_scenario_id INTEGER);"
            "INSERT INTO scheduler_metadata VALUES (1,1,9,3);"
            "CREATE TABLE tenant_allocation (tenant_id TEXT PRIMARY KEY, execution_units INTEGER);"
            "INSERT INTO tenant_allocation VALUES ('alice',2);"
            "CREATE TABLE scenario_admission (scenario_id INTEGER PRIMARY KEY, tenant_id TEXT, service_id TEXT, "
            "recipe TEXT, scenario_name TEXT, release_id TEXT, model_config_ref TEXT, execution_units INTEGER, "
            "state TEXT, generation INTEGER, created_at REAL, last_activity_at REAL, service_config_ref TEXT, "
            "failure_reason TEXT, terminal_state TEXT, cleanup_before REAL);"
        )
        for identifier, state in ((1, "admitted"), (2, "queued")):
            connection.execute(
                "INSERT INTO scenario_admission VALUES (?, 'alice', 'service-old', 'harness', ?, 'git-version', "
                "NULL, 2, ?, ?, 10, 12, ?, NULL, 'deleted', NULL)",
                (
                    identifier,
                    f"scenario-{identifier}",
                    state,
                    1 if state == "admitted" else 0,
                    registration.legacy_registration_fingerprint("service-old"),
                ),
            )


def test_version_one_migration_requires_explicit_identity_maps_and_preserves_state(tmp_path):
    import sqlite3

    database = tmp_path / "old.sqlite3"
    registration = ServiceRegistration(11, "local", "harness", "http://localhost", 2, "revision-1")
    create_version_one_database(database, registration)
    with pytest.raises(ValueError, match="tenant_id_map"):
        SQLiteSchedulingStore(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT schema_version FROM scheduler_metadata").fetchone()[0] == 1
    with pytest.raises(ValueError, match="must not merge"):
        SQLiteSchedulingStore(database, tenant_id_map={"alice": 1, "bob": 1}, service_id_map={"service-old": 11})
    fingerprints = {registration.legacy_registration_fingerprint("service-old"): registration.registration_fingerprint}
    store = SQLiteSchedulingStore(
        database,
        tenant_id_map={"alice": 1},
        service_id_map={"service-old": 11},
        registration_fingerprint_map=fingerprints,
    )
    scheduler = MultiTenantScheduler(store, SchedulerConfig(2, {1: 2}, {"harness": 2}))
    scheduler.register_service(registration)
    scheduler.set_service_availability(11, True)
    active = scheduler.get_scenario_admission(1, 1)
    assert active.execution_allocated and active.has_started
    assert active.generation == 1 and active.release_id == "git-version"
    assert active.idle_since is None
    queued = scheduler.get_scenario_admission(2, 1)
    assert not queued.execution_allocated and queued.queue_sequence == 2
    assert scheduler.snapshot().next_scenario_id == 3
    assert scheduler.snapshot().next_queue_sequence == 3
    assert scheduler.next_scenario_admission(1) is None
    scheduler.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"service_url": "http://changed"},
        {"config_revision": "revision-2"},
        {"recipe": "other"},
    ],
)
def test_migration_cannot_disguise_deployment_changes_as_id_conversion(tmp_path, changes):
    import sqlite3

    database = tmp_path / "old.sqlite3"
    registration = ServiceRegistration(11, "local", "harness", "http://localhost", 2, "revision-1")
    create_version_one_database(database, registration)
    changed = replace(registration, **changes)
    with pytest.raises(ValueError, match="unchanged registration"):
        SQLiteSchedulingStore(
            database,
            tenant_id_map={"alice": 1},
            service_id_map={"service-old": 11},
            registration_fingerprint_map={
                changed.legacy_registration_fingerprint("service-old"): changed.registration_fingerprint,
            },
        )
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT tenant_id FROM scenario_admission WHERE scenario_id=1").fetchone()[0] == "alice"
        )
        assert connection.execute("SELECT schema_version FROM scheduler_metadata").fetchone()[0] == 1
