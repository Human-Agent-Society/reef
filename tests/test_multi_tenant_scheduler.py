"""Admission accounting, lifecycle ownership, and durable FIFO behavior."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from reef.scenario.scheduling import (
    MAX_SCENARIO_ID,
    AdmissionConflict,
    GlobalQuotaExceeded,
    MultiTenantScheduler,
    ScenarioRequirementExceedsQuota,
    ScenarioTenantMismatch,
    SchedulerConfig,
    SchedulerNotAcceptingRequests,
    SchedulingError,
    ServiceRegistration,
    ServiceUnavailable,
)
from reef.storage.sqlite_scheduling import SchedulingStoreConflict, SQLiteSchedulingStore


@pytest.fixture
def scheduler():
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(),
        SchedulerConfig(8, {1: 4, 2: 4, 3: 4}, {"harness": 2, "large": 4}),
    )
    scheduler.register_service(ServiceRegistration(11, "development", "harness", "http://localhost:8000", 2))
    scheduler.set_service_availability(11, True)
    scheduler.register_service(ServiceRegistration(12, "production", "harness", "http://localhost:8001", 2))
    scheduler.set_service_availability(12, True)
    scheduler.register_service(ServiceRegistration(13, "larger", "large", "http://localhost:8002", 4))
    scheduler.set_service_availability(13, True)
    yield scheduler
    scheduler.close()


def test_directory_and_invalid_requests_do_not_reserve_tenants(scheduler):
    assert scheduler.tenant_allocations() == ()
    with pytest.raises(AdmissionConflict):
        scheduler.register_service(ServiceRegistration(99, "development", "harness", "http://other", 2))
        scheduler.set_service_availability(99, True)
    with pytest.raises(AdmissionConflict):
        scheduler.register_service(ServiceRegistration(11, "other", "harness", "http://other", 2))
    with pytest.raises(AdmissionConflict):
        scheduler.register_service(ServiceRegistration(99, "other", "harness", "http://other", 3))
    with pytest.raises(ServiceUnavailable):
        scheduler.request_scenario_admission(1, "missing", "harness", "work")
    with pytest.raises(AdmissionConflict):
        scheduler.request_scenario_admission(1, "development", "large", "work")
    assert scheduler.tenant_allocations() == ()
    assert scheduler.snapshot().next_scenario_id == 1


def test_global_quota_is_reserved_lazily_and_released_after_last_cleanup(scheduler):
    first = scheduler.request_scenario_admission(1, "development", "harness", "work")
    scheduler.request_scenario_admission(2, "development", "harness", "other")
    assert sum(tenant.allocated_execution_units for tenant in scheduler.tenant_allocations()) == 8
    with pytest.raises(GlobalQuotaExceeded):
        scheduler.request_scenario_admission(3, "production", "harness", "work")
    starting = scheduler.next_scenario_admission(1)
    cleanup = scheduler.request_cleanup(first.scenario_id, 1)
    assert starting.state == "starting"
    with pytest.raises(GlobalQuotaExceeded):
        scheduler.request_scenario_admission(3, "production", "harness", "work")
    scheduler.confirm_cleaned(cleanup.scenario_id, cleanup.generation)
    scheduler.request_scenario_admission(3, "production", "harness", "work")
    assert {tenant.tenant_id for tenant in scheduler.tenant_allocations()} == {2, 3}


def test_impossible_scenario_is_rejected_before_tenant_allocation():
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(4, {1: 2}, {"large": 4}))
    scheduler.register_service(ServiceRegistration(11, "large", "large", "http://localhost", 4))
    scheduler.set_service_availability(11, True)
    with pytest.raises(ScenarioRequirementExceedsQuota):
        scheduler.request_scenario_admission(1, "large", "large", "work")
    assert scheduler.tenant_allocations() == ()
    scheduler.close()


def test_cross_service_fifo_holds_head_until_full_requirement_fits(scheduler):
    active = scheduler.request_scenario_admission(1, "development", "harness", "active")
    started = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(started.scenario_id, started.generation)
    large = scheduler.request_scenario_admission(1, "larger", "large", "large")
    small = scheduler.request_scenario_admission(1, "production", "harness", "small")
    assert scheduler.next_scenario_admission(1) is None
    cleanup = scheduler.request_cleanup(active.scenario_id, 1)
    assert scheduler.next_scenario_admission(1) is None
    scheduler.confirm_cleaned(cleanup.scenario_id, cleanup.generation)
    assert scheduler.next_scenario_admission(1).scenario_id == large.scenario_id
    assert scheduler.next_scenario_admission(1) is None
    assert scheduler.get_scenario_admission(small.scenario_id, 1).state == "queued"


def test_duplicate_creation_ownership_and_name_reuse(scheduler):
    original = scheduler.request_scenario_admission(1, "development", "harness", "work", release_id="v1")
    assert scheduler.request_scenario_admission(1, "development", "harness", "work", release_id="v1") == original
    with pytest.raises(ScenarioTenantMismatch):
        scheduler.request_scenario_admission(2, "development", "harness", "work", release_id="v1")
    with pytest.raises(ScenarioTenantMismatch):
        scheduler.get_scenario_admission(original.scenario_id, 2)
    with pytest.raises(ScenarioTenantMismatch):
        scheduler.request_cleanup(original.scenario_id, 2)
    with pytest.raises(AdmissionConflict):
        scheduler.request_scenario_admission(1, "development", "harness", "work", release_id="v2")
    elsewhere = scheduler.request_scenario_admission(1, "production", "harness", "work")
    assert elsewhere.scenario_id != original.scenario_id
    assert scheduler.request_cleanup(original.scenario_id, 1).state == "deleted"
    replacement = scheduler.request_scenario_admission(2, "development", "harness", "work")
    assert replacement.scenario_id > original.scenario_id
    scheduler.confirm_started(original.scenario_id, 1)
    assert scheduler.get_scenario_admission(replacement.scenario_id, 2).state == "queued"


def test_service_unavailability_keeps_queue_and_does_not_skip_fifo(scheduler):
    first = scheduler.request_scenario_admission(1, "development", "harness", "first")
    second = scheduler.request_scenario_admission(1, "production", "harness", "second")
    scheduler.set_service_availability(11, False)
    with pytest.raises(ServiceUnavailable):
        scheduler.request_scenario_admission(2, "development", "harness", "new")
    assert scheduler.next_scenario_admission(1) is None
    assert scheduler.get_scenario_admission(first.scenario_id, 1).failure_reason == "selected service is unavailable"
    assert scheduler.get_scenario_admission(second.scenario_id, 1).state == "queued"
    scheduler.set_service_availability(11, True)
    assert scheduler.next_scenario_admission(1).scenario_id == first.scenario_id


def test_concurrent_requests_and_claims_never_overallocate(scheduler):
    with ThreadPoolExecutor(max_workers=12) as workers:
        requests = [
            workers.submit(scheduler.request_scenario_admission, 1, "development", "harness", f"work-{index}")
            for index in range(30)
        ]
        admissions = [request.result() for request in requests]
        claims = [workers.submit(scheduler.next_scenario_admission, 1) for index in range(30)]
        started = [claim.result() for claim in claims]
    assert len({admission.scenario_id for admission in admissions}) == 30
    assert sum(admission.execution_units for admission in started if admission is not None) == 4
    assert len(scheduler.tenant_allocations()) == 1
    assert len(scheduler.pending_scenario_admissions()) == 2


def test_competing_tenants_cannot_overreserve_the_global_budget(scheduler):
    with ThreadPoolExecutor(max_workers=3) as workers:
        requests = [
            workers.submit(scheduler.request_scenario_admission, tenant, "development", "harness", f"tenant-{tenant}")
            for tenant in (1, 2, 3)
        ]
        accepted = 0
        refused = 0
        for request in requests:
            try:
                request.result()
                accepted += 1
            except GlobalQuotaExceeded:
                refused += 1
    assert (accepted, refused) == (2, 1)
    assert sum(tenant.allocated_execution_units for tenant in scheduler.tenant_allocations()) == 8


def test_start_failure_cleanup_and_late_notifications_are_idempotent(scheduler):
    admission = scheduler.request_scenario_admission(1, "larger", "large", "work")
    starting = scheduler.next_scenario_admission(1)
    failed = scheduler.request_cleanup(admission.scenario_id, 1, failed=True, reason="startup failed")
    assert scheduler.confirm_started(starting.scenario_id, starting.generation) == failed
    assert scheduler.confirm_cleaned(failed.scenario_id, starting.generation) == failed
    assert scheduler.tenant_allocations()
    terminal = scheduler.confirm_cleaned(failed.scenario_id, failed.generation)
    assert terminal.state == "failed"
    assert scheduler.confirm_cleaned(failed.scenario_id, failed.generation) == terminal
    assert scheduler.tenant_allocations() == ()


def test_idle_reclaim_sleeps_until_woken_at_tail_and_keeps_tenant_allocation():
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"harness": 2}, idle_ttl_seconds=20)
    )
    scheduler.register_service(ServiceRegistration(11, "local", "harness", "http://localhost", 2))
    scheduler.set_service_availability(11, True)
    first = scheduler.request_scenario_admission(1, "local", "harness", "first", now_seconds=100)
    claim = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(claim.scenario_id, claim.generation)
    second = scheduler.request_scenario_admission(1, "local", "harness", "second", now_seconds=100)
    assert scheduler.reclaim_idle_scenarios(now_seconds=1000) == ()
    scheduler.report_scenario_idle(first.scenario_id, claim.generation, idle_since=100)
    scheduler.get_scenario_admission(first.scenario_id, 1)
    assert scheduler.reclaim_idle_scenarios(now_seconds=119) == ()
    pending = scheduler.reclaim_idle_scenarios(now_seconds=120)[0]
    assert pending.reclaim_before == 100
    assert pending.execution_allocated
    assert scheduler.next_scenario_admission(1) is None
    reclaimed = scheduler.confirm_reclaimed(first.scenario_id, pending.generation)
    assert reclaimed.scenario_id == first.scenario_id
    assert reclaimed.state == "sleep" and not reclaimed.execution_allocated
    assert scheduler.cold_scenarios_by_id == {first.scenario_id: reclaimed}
    assert scheduler.tenant_allocations()[0].allocated_execution_units == 2
    assert scheduler.confirm_reclaimed(first.scenario_id, pending.generation) == reclaimed
    assert scheduler.next_scenario_admission(1).scenario_id == second.scenario_id
    assert scheduler.next_scenario_admission(1) is None
    awakened = scheduler.wake_scenario_admission(first.scenario_id, 1, reclaimed.generation)
    assert awakened.queue_sequence > second.queue_sequence
    assert awakened.state == "queued" and scheduler.cold_scenarios_by_id == {}
    assert scheduler.wake_scenario_admission(first.scenario_id, 1, reclaimed.generation) == awakened
    scheduler.close()


def test_busy_or_fresh_work_cancels_reclaim_and_explicit_delete_wins():
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"harness": 2}, idle_ttl_seconds=1)
    )
    scheduler.register_service(ServiceRegistration(11, "local", "harness", "http://localhost", 2))
    scheduler.set_service_availability(11, True)
    admission = scheduler.request_scenario_admission(1, "local", "harness", "work")
    claim = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(claim.scenario_id, claim.generation)
    scheduler.report_scenario_idle(admission.scenario_id, claim.generation, idle_since=100)
    pending = scheduler.request_reclaim(admission.scenario_id, now_seconds=102)
    resumed = scheduler.cancel_reclaim(admission.scenario_id, pending.generation, idle_since=None)
    assert resumed.state == "admitted" and resumed.execution_allocated
    assert resumed.generation == pending.generation
    assert scheduler.confirm_reclaimed(admission.scenario_id, pending.generation) == resumed
    assert scheduler.report_scenario_idle(admission.scenario_id, claim.generation, idle_since=10) == resumed
    scheduler.report_scenario_idle(admission.scenario_id, resumed.generation, idle_since=103)
    pending = scheduler.request_reclaim(admission.scenario_id, now_seconds=105)
    explicit = scheduler.request_cleanup(admission.scenario_id, 1)
    assert explicit.generation > pending.generation and explicit.reclaim_before is None
    assert scheduler.cancel_reclaim(admission.scenario_id, pending.generation) == explicit
    assert scheduler.confirm_reclaimed(admission.scenario_id, pending.generation) == explicit
    scheduler.close()


def test_deleting_reclaimed_waiter_does_not_recharge_its_execution_units():
    scheduler = MultiTenantScheduler(
        SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"harness": 2}, idle_ttl_seconds=1)
    )
    scheduler.register_service(ServiceRegistration(11, "local", "harness", "http://localhost", 2))
    scheduler.set_service_availability(11, True)
    first = scheduler.request_scenario_admission(1, "local", "harness", "first")
    claim = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(claim.scenario_id, claim.generation)
    scheduler.report_scenario_idle(first.scenario_id, claim.generation, idle_since=100)
    second = scheduler.request_scenario_admission(1, "local", "harness", "second")
    pending = scheduler.request_reclaim(first.scenario_id, now_seconds=102)
    scheduler.confirm_reclaimed(first.scenario_id, pending.generation)
    assert scheduler.next_scenario_admission(1).scenario_id == second.scenario_id
    deletion = scheduler.request_cleanup(first.scenario_id, 1)
    assert deletion.state == "cleanup_pending" and not deletion.execution_allocated
    assert (
        sum(value.execution_units for value in scheduler.list_scenario_admissions() if value.execution_allocated) == 2
    )
    assert scheduler.confirm_cleaned(first.scenario_id, deletion.generation).state == "deleted"
    assert scheduler.get_scenario_admission(second.scenario_id, 1).execution_allocated
    scheduler.close()


def test_idle_ttl_disabled_and_queued_requests_never_expire(scheduler):
    admission = scheduler.request_scenario_admission(1, "development", "harness", "work", now_seconds=0)
    claim = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(claim.scenario_id, claim.generation)
    scheduler.report_scenario_idle(claim.scenario_id, claim.generation, idle_since=0)
    queued = scheduler.request_scenario_admission(1, "development", "harness", "waiting", now_seconds=0)
    assert scheduler.reclaim_idle_scenarios(now_seconds=10**12) == ()
    assert scheduler.get_scenario_admission(queued.scenario_id, 1).state == "queued"
    scheduler.stop_accepting_requests()
    with pytest.raises(SchedulerNotAcceptingRequests):
        scheduler.next_scenario_admission(1)
    with pytest.raises(SchedulerNotAcceptingRequests):
        scheduler.request_scenario_admission(1, "production", "harness", "new")
    cleanup = scheduler.request_cleanup(admission.scenario_id, 1)
    assert scheduler.confirm_cleaned(cleanup.scenario_id, cleanup.generation).state == "deleted"


def test_restart_preserves_queue_owner_config_reference_and_ids(tmp_path):
    path = tmp_path / "scheduler.sqlite3"
    config = SchedulerConfig(2, {1: 2}, {"harness": 2})
    service = ServiceRegistration(11, "local", "harness", "http://localhost", 2, "configuration-v1")
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(path), config)
    scheduler.register_service(service)
    scheduler.set_service_availability(service.service_id, True)
    active = scheduler.request_scenario_admission(1, "local", "harness", "active", model_config_ref="a" * 64)
    claim = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(claim.scenario_id, claim.generation)
    queued = scheduler.request_scenario_admission(1, "local", "harness", "queued")
    scheduler.close()
    recovered = MultiTenantScheduler(SQLiteSchedulingStore(path), config)
    assert recovered.get_scenario_admission(active.scenario_id, 1).model_config_ref == "a" * 64
    with pytest.raises(AdmissionConflict):
        recovered.register_service(replace(service, config_revision="configuration-v2"))
    recovered.register_service(service)
    recovered.set_service_availability(service.service_id, True)
    with pytest.raises(AdmissionConflict):
        recovered.request_scenario_admission(1, "local", "harness", "active", model_config_ref="b" * 64)
    assert recovered.next_scenario_admission(1) is None
    cleanup = recovered.request_cleanup(active.scenario_id, 1)
    recovered.confirm_cleaned(cleanup.scenario_id, cleanup.generation)
    assert recovered.next_scenario_admission(1).scenario_id == queued.scenario_id
    assert recovered.request_scenario_admission(1, "local", "harness", "active").scenario_id > queued.scenario_id
    recovered.close()


def test_sqlite_compare_and_swap_rejects_a_second_authority(tmp_path):
    path = tmp_path / "scheduler.sqlite3"
    first = SQLiteSchedulingStore(path)
    second = SQLiteSchedulingStore(path)
    stale = second.load()
    saved = first.save(replace(first.load(), next_scenario_id=9))
    with pytest.raises(SchedulingStoreConflict):
        second.save(replace(stale, next_scenario_id=2))
    assert second.load() == saved
    first.close()
    second.close()


def test_scenario_ids_never_wrap_or_reuse_at_the_storage_limit():
    store = SQLiteSchedulingStore()
    store.save(replace(store.load(), next_scenario_id=MAX_SCENARIO_ID))
    scheduler = MultiTenantScheduler(store, SchedulerConfig(2, {1: 2}, {"harness": 2}))
    scheduler.register_service(ServiceRegistration(11, "local", "harness", "http://localhost", 2))
    scheduler.set_service_availability(11, True)
    final = scheduler.request_scenario_admission(1, "local", "harness", "last")
    assert final.scenario_id == MAX_SCENARIO_ID
    scheduler.request_cleanup(final.scenario_id, 1)
    with pytest.raises(SchedulingError, match="ID range is exhausted"):
        scheduler.request_scenario_admission(1, "local", "harness", "new")
    assert scheduler.tenant_allocations() == ()
    scheduler.close()


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan"), True])
def test_config_rejects_invalid_ttl(duration):
    with pytest.raises(ValueError, match="TTL"):
        SchedulerConfig(4, {1: 4}, {"harness": 2}, idle_ttl_seconds=duration)


def test_numeric_identifiers_and_unknown_tenant_are_not_silently_coerced(scheduler):
    from reef.scenario.scheduling import UnknownTenant

    for invalid in ("1", True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive signed 64-bit integer"):
            scheduler.request_scenario_admission(invalid, "development", "harness", "work")
        with pytest.raises(ValueError, match="positive signed 64-bit integer"):
            ServiceRegistration(invalid, "other", "harness", "http://localhost", 2)
    with pytest.raises(UnknownTenant):
        scheduler.request_scenario_admission(999, "development", "harness", "work")
    assert scheduler.tenant_allocations() == ()


def test_directory_registration_requires_successful_remote_verification():
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(2, {1: 2}, {"harness": 2}))
    registration = ServiceRegistration(11, "local", "harness", "http://localhost", 2)
    scheduler.register_service(registration)
    with pytest.raises(ServiceUnavailable):
        scheduler.request_scenario_admission(1, "local", "harness", "work")
    scheduler.set_service_availability(11, True)
    admission = scheduler.request_scenario_admission(1, "local", "harness", "work")
    scheduler.register_service(registration)
    assert scheduler.next_scenario_admission(1).scenario_id == admission.scenario_id
    scheduler.close()


def test_empty_directory_learns_recipe_costs_from_registration():
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(4, {1: 4}))
    assert scheduler.service_registrations() == ()
    assert scheduler.tenant_allocations() == ()
    registration = ServiceRegistration(11, "development", "harness", "http://localhost:8901", 2)
    scheduler.register_service(registration)
    scheduler.register_service(registration)
    assert scheduler.service_registrations() == (registration,)
    assert scheduler.recipe_execution_units == {"harness": 2}
    with pytest.raises(ServiceUnavailable):
        scheduler.request_scenario_admission(1, "development", "harness", "work")
    scheduler.set_service_availability(11, True)
    scheduler.request_scenario_admission(1, "development", "harness", "work")
    assert scheduler.next_scenario_admission(1).execution_units == 2
    scheduler.close()


def test_dynamic_registration_conflicts_do_not_publish_new_recipe_costs():
    scheduler = MultiTenantScheduler(SQLiteSchedulingStore(), SchedulerConfig(4, {1: 4}))
    original = ServiceRegistration(11, "development", "harness", "http://localhost:8901", 2)
    scheduler.register_service(original)
    conflicts = (
        ServiceRegistration(12, "development", "other", "http://localhost:8902", 1),
        replace(original, service_name="renamed", recipe="other", standard_execution_units=1),
        ServiceRegistration(12, "production", "harness", "http://localhost:8902", 1),
    )
    for registration in conflicts:
        with pytest.raises(AdmissionConflict):
            scheduler.register_service(registration)
    assert scheduler.service_registrations() == (original,)
    assert scheduler.recipe_execution_units == {"harness": 2}
    scheduler.register_service(ServiceRegistration(12, "production", "other", "http://localhost:8902", 3))
    assert scheduler.recipe_execution_units == {"harness": 2, "other": 3}
    scheduler.stop_accepting_requests()
    with pytest.raises(SchedulerNotAcceptingRequests):
        scheduler.register_service(original)
    scheduler.close()


def test_reclaimed_scenario_can_resume_with_fresh_generation(scheduler):
    scheduler.config = replace(scheduler.config, idle_ttl_seconds=1)
    admission = scheduler.request_scenario_admission(1, "development", "harness", "work", release_id="opaque-version")
    first = scheduler.next_scenario_admission(1)
    scheduler.confirm_started(first.scenario_id, first.generation)
    scheduler.report_scenario_idle(first.scenario_id, first.generation, idle_since=100)
    reclaim = scheduler.request_reclaim(first.scenario_id, now_seconds=102)
    scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation)
    assert scheduler.next_scenario_admission(1) is None
    scheduler.wake_scenario_admission(first.scenario_id, 1, reclaim.generation)
    resumed = scheduler.next_scenario_admission(1)
    assert resumed.scenario_id == admission.scenario_id
    assert resumed.generation > reclaim.generation > first.generation
    assert resumed.release_id == "opaque-version" and resumed.execution_allocated
    assert scheduler.confirm_reclaimed(first.scenario_id, reclaim.generation) == resumed
    assert scheduler.confirm_started(first.scenario_id, first.generation) == resumed
    admitted = scheduler.confirm_started(resumed.scenario_id, resumed.generation)
    assert admitted.state == "admitted" and admitted.idle_since is None
