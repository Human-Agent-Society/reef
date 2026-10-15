"""Coordinate remote scenario lifecycle operations without holding quota locks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from threading import Condition, Lock, get_ident

from reef.scenario.scheduling import RECONCILE_STATES  # Review Comment: What is this for?
from reef.scenario.scheduling import (
    TERMINAL_STATES,
    AdmissionConflict,
    AdmissionNotFound,
    GlobalQuotaExceeded,
    MultiTenantScheduler,
    RequestTenantMismatch,
    ScenarioRequirementExceedsQuota,
    ScenarioTenantMismatch,
    SchedulerNotAcceptingRequests,
    ServiceRegistration,
    ServiceUnavailable,
    UnknownTenant,
    validate_identifier,
)
from reef.service.scheduler_config import save_scheduler_config
from reef.service.scheduling import (
    SchedulerHTTPClient,
    SchedulerResponseError,
    SchedulerUnavailable,
    integer_value,
    text_value,
)
from reef.storage.scheduling import ScenarioAdmission


class SchedulerController:
    """Register services and reconcile their lifecycle with one authoritative scheduler."""

    def __init__(
        self,
        scheduler: MultiTenantScheduler,
        controls: dict[int, SchedulerHTTPClient],
    ) -> None:
        self.scheduler = scheduler
        self.controls = controls
        self.reconcile_lock = Lock()
        self.errors: dict[int, str] = {}
        self.operation_condition = Condition()
        self.active_operations: dict[int, int] = {}
        self.stopping = False
        self.closed = False
        self.maintenance_error: str | None = None

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Keep worker-thread operations alive even when their HTTP task is cancelled."""
        thread_id = get_ident()
        with self.operation_condition:
            if self.stopping and thread_id not in self.active_operations:
                raise SchedulerNotAcceptingRequests("scheduler is shutting down")
            self.active_operations[thread_id] = self.active_operations.get(thread_id, 0) + 1
        try:
            yield
        finally:
            with self.operation_condition:
                remaining = self.active_operations[thread_id] - 1
                if remaining:
                    self.active_operations[thread_id] = remaining
                else:
                    self.active_operations.pop(thread_id)
                self.operation_condition.notify_all()

    def stop_accepting_requests(self) -> None:
        with self.operation_condition:
            self.stopping = True
        self.scheduler.stop_accepting_requests()

    def close(self, *, save_snapshot: bool = True) -> None:
        """Drain control operations and save the directory before closing persistent state."""
        self.stop_accepting_requests()
        with self.operation_condition:
            self.operation_condition.wait_for(lambda: not self.active_operations)
            if not self.closed:
                try:
                    if save_snapshot:
                        save_scheduler_config(self.scheduler)
                finally:
                    try:
                        self.scheduler.close()
                    finally:
                        self.closed = True

    def register_service(self, payload: dict[str, object]) -> dict[str, object]:
        """Install an idempotent registration and verify its control endpoint before admission."""
        revision = payload.get("config_revision", "")
        if not isinstance(revision, str):
            raise ValueError("config_revision must be a string")
        registration = ServiceRegistration(
            service_id=integer_value(payload, "service_id"),
            service_name=text_value(payload, "service_name"),
            recipe=text_value(payload, "recipe"),
            service_url=text_value(payload, "service_url"),
            standard_execution_units=integer_value(payload, "standard_execution_units"),
            config_revision=revision,
        )
        control = SchedulerHTTPClient(registration.service_url, token=text_value(payload, "control_token"))
        with self.operation(), self.reconcile_lock:
            self.scheduler.register_service(registration)
            previous = self.controls.get(registration.service_id)
            credentials_changed = previous is None or previous.token != control.token
            if credentials_changed:
                self.scheduler.set_service_availability(registration.service_id, False)
            self.controls[registration.service_id] = control
            if credentials_changed:
                self.scheduler.config.control_token_envs.pop(registration.service_id, None)
            self.describe_service(registration)
            return {**asdict(registration), "available": self.scheduler.availability[registration.service_id]}

    def control(self, service_id: int, payload: dict[str, object]) -> dict[str, object]:
        """Send control operations to the specific services from the scheduler. (For example chaning)"""
        control = self.controls.get(service_id)
        if control is None:
            raise SchedulerUnavailable("service is awaiting registration to restore its control connection")
        return control.request("POST", "/reef/scheduler/control", payload=payload)

    def describe_services(self) -> None:
        for registration in self.scheduler.service_registrations():
            self.describe_service(registration)

    def describe_service(self, registration: ServiceRegistration) -> None:
        service_id = registration.service_id
        if service_id not in self.controls:
            self.scheduler.set_service_availability(service_id, False)
            self.errors[service_id] = "service is awaiting registration to restore its control connection"
            return
        try:
            remote = self.control(service_id, {"operation": "describe"})
            matches = (
                integer_value(remote, "service_id") == service_id
                and remote.get("service_name") == registration.service_name
                and remote.get("recipe") == registration.recipe
                and integer_value(remote, "standard_execution_units") == registration.standard_execution_units
                and remote.get("config_revision", "") == registration.config_revision
            )
            if self.scheduler.config.idle_ttl_seconds is not None and remote.get("supports_idle_reclaim") is not True:
                self.scheduler.set_service_availability(service_id, False)
                self.errors[service_id] = "idle reclamation requires durable scenario and admission storage"
                return
            self.scheduler.set_service_availability(service_id, matches)
            if matches:
                self.errors.pop(service_id, None)
            else:
                self.errors[service_id] = "service configuration does not match its directory entry"
        except (SchedulerUnavailable, SchedulerResponseError, ValueError):
            self.scheduler.set_service_availability(service_id, False)
            self.errors[service_id] = "service unavailable; existing allocations remain charged"

    def reconcile(self) -> None:
        """Reconcile service and scenario state, reclaim idle resources, and start queued scenarios."""
        if self.scheduler.accepting_requests is False:
            return
        if not self.reconcile_lock.acquire(blocking=False):
            return
        try:
            with self.operation():
                self.describe_services()
                # Resolve uncertain and restored operations before allocating more work.
                for admission in self.scheduler.list_scenario_admissions():
                    if admission.state in RECONCILE_STATES:
                        self.reconcile_scenario(admission)
                self.scheduler.reclaim_idle_scenarios()
                for admission in self.scheduler.pending_scenario_admissions():
                    self.reconcile_scenario(admission)
                # Claim FIFO heads until each tenant runs out of execution quota.
                for tenant in self.scheduler.tenant_allocations():
                    while (claimed := self.scheduler.next_scenario_admission(tenant.tenant_id)) is not None:
                        self.reconcile_scenario(claimed)
        except SchedulerNotAcceptingRequests:
            # Shutdown may stop queue claims after reconciliation has started.
            return
        finally:
            self.reconcile_lock.release()

    def reconcile_scenario(self, admission: ScenarioAdmission) -> None:
        with self.operation():
            admission = self.scheduler.get_scenario_admission(admission.scenario_id, admission.tenant_id)
            if admission.state not in RECONCILE_STATES:
                return
            if not self.scheduler.availability.get(admission.service_id, False):
                return
            payload: dict[str, object] = {
                "scenario_id": admission.scenario_id,
                "scenario_name": admission.scenario_name,
                "tenant_id": admission.tenant_id,
                "generation": admission.generation,
            }
            try:
                # Part 1: pending status
                if admission.state in {"cleanup_pending", "reclaim_pending"}:
                    operation = "reclaim" if admission.state == "reclaim_pending" else "cleanup"
                    remote = self.control(
                        admission.service_id,
                        {**payload, "operation": operation, "inactive_before": admission.reclaim_before},
                    )
                else:
                    # Part 2:
                    remote = self.control(admission.service_id, {**payload, "operation": "status"})
                    if "state" not in remote:
                        raise ValueError("service status requires a lifecycle state")
                    previous_id = remote.get("scenario_id")
                    cleaned_previous = (
                        isinstance(previous_id, int)
                        and not isinstance(previous_id, bool)
                        and previous_id < admission.scenario_id
                        and remote.get("state") in TERMINAL_STATES
                        and remote.get("cleanup_complete") is True
                    )
                    restore_current = (
                        previous_id == admission.scenario_id
                        and remote.get("tenant_id") == admission.tenant_id
                        and remote.get("state") in {"starting", "reclaimed"}
                    )
                    if remote.get("state") is None or cleaned_previous or restore_current:
                        remote = self.control(
                            admission.service_id,
                            {
                                **payload,
                                "operation": "start",
                                "release_id": admission.release_id,
                                "model_config_ref": admission.model_config_ref,
                            },
                        )
                if any(
                    integer_value(remote, field) != payload[field]
                    for field in ("scenario_id", "tenant_id", "generation")
                ):
                    self.errors[admission.service_id] = "service lifecycle identity mismatch; allocation retained"
                    return
                state = remote.get("state")
                if admission.state == "cleanup_pending":
                    if remote.get("cleanup_complete") is True and state in TERMINAL_STATES:
                        self.scheduler.confirm_cleaned(admission.scenario_id, admission.generation)
                    return
                if admission.state == "reclaim_pending":
                    if state == "reclaimed" and remote.get("reclaim_complete") is True:
                        self.scheduler.confirm_reclaimed(admission.scenario_id, admission.generation)
                    elif state == "admitted":
                        self.scheduler.cancel_reclaim(admission.scenario_id, admission.generation)
                    return
                if state == "admitted":
                    idle_since = remote.get("idle_since")
                    if idle_since is not None and (
                        not isinstance(idle_since, int | float) or isinstance(idle_since, bool)
                    ):
                        raise ValueError("service idle_since must be a timestamp or null")
                    if admission.state == "starting":
                        self.scheduler.confirm_started(admission.scenario_id, admission.generation)
                    self.scheduler.report_scenario_idle(
                        admission.scenario_id,
                        admission.generation,
                        idle_since=None if idle_since is None else float(idle_since),
                    )
                elif state in {"failed", "deleted", "cleanup_pending"}:
                    self.scheduler.request_cleanup(
                        admission.scenario_id,
                        admission.tenant_id,
                        failed=True,
                        reason="service scenario startup or lifecycle failed",
                    )
            except (SchedulerUnavailable, SchedulerResponseError, ValueError):
                self.errors[admission.service_id] = "control outcome unknown; allocation retained for reconciliation"

    def submit(self, payload: dict[str, object], tenant_id: int) -> dict[str, object]:
        with self.operation():
            service_name = text_value(payload, "service_name")
            recipe = text_value(payload, "recipe")
            scenario_name = text_value(payload, "scenario_name")
            supplied_tenant = integer_value(payload, "tenant_id") if "tenant_id" in payload else tenant_id
            if supplied_tenant != tenant_id:
                raise RequestTenantMismatch("request tenant differs from its header")
            integer_value({"tenant_id": tenant_id}, "tenant_id")
            validate_identifier(service_name, "service_name")
            validate_identifier(scenario_name, "scenario_name")
            service_id = self.scheduler.service_names.get(service_name)
            if service_id is None:
                raise ServiceUnavailable("unknown service; register it before requesting a scenario")
            release_id = payload.get("release_id")
            if release_id is not None and (not isinstance(release_id, str) or not release_id.strip()):
                raise ValueError("release_id must be a non-empty string")
            model_config_ref = None
            if "model" in payload:
                # Check admission before writing private service configuration. The
                # authoritative request below repeats the quota check atomically.
                if self.scheduler.accepting_requests is False:
                    raise SchedulerNotAcceptingRequests("scheduler has stopped new admission")
                registration = self.scheduler.services[service_id]
                if recipe != registration.recipe:
                    raise AdmissionConflict("requested recipe does not match the service")
                quota = self.scheduler.config.tenant_quotas.get(tenant_id)
                if quota is None:
                    raise UnknownTenant("tenant has no configured quota")
                if registration.standard_execution_units > quota:
                    raise ScenarioRequirementExceedsQuota("scenario requirement exceeds the tenant quota")
                allocations = self.scheduler.tenant_allocations()
                if not any(value.tenant_id == tenant_id for value in allocations) and (
                    sum(value.allocated_execution_units for value in allocations) + quota
                    > self.scheduler.config.global_execution_units
                ):
                    raise GlobalQuotaExceeded("global budget cannot reserve this tenant quota")
                encoded = json.dumps(payload["model"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                model_config_ref = hashlib.sha256(encoded).hexdigest()
                try:
                    existing = self.scheduler.find_latest_scenario_admission(service_id, scenario_name)
                except AdmissionNotFound:
                    existing = None
                if existing is not None and existing.state not in TERMINAL_STATES:
                    if existing.tenant_id != tenant_id:
                        raise ScenarioTenantMismatch("scenario name is owned by another tenant")
                    if existing.release_id != release_id or existing.model_config_ref != model_config_ref:
                        raise AdmissionConflict("scenario creation parameters differ from the original request")
                staged = self.control(service_id, {"operation": "stage_model", "model": payload["model"]})
                if text_value(staged, "model_config_ref") != model_config_ref:
                    raise ValueError("service staged a different model configuration")
            admission = self.scheduler.request_scenario_admission(
                tenant_id,
                service_name,
                recipe,
                scenario_name,
                release_id=release_id,
                model_config_ref=model_config_ref,
            )
            self.reconcile()
            return self.view(self.scheduler.get_scenario_admission(admission.scenario_id, tenant_id))

    def view(self, admission: ScenarioAdmission) -> dict[str, object]:
        registration = self.scheduler.services.get(admission.service_id)
        return {
            "scenario_id": admission.scenario_id,
            "tenant_id": admission.tenant_id,
            "service_id": admission.service_id,
            "service_name": None if registration is None else registration.service_name,
            "service_url": None if registration is None else registration.service_url,
            "recipe": admission.recipe,
            "scenario_name": admission.scenario_name,
            "state": admission.state,
            "generation": admission.generation,
            "execution_units": admission.execution_units,
            "created_at": admission.created_at,
            "idle_since": admission.idle_since,
            "reason": admission.failure_reason or self.errors.get(admission.service_id),
        }

    def wake(self, scenario_id: int, tenant_id: int, payload: dict[str, object]) -> dict[str, object]:
        """Verify that the owning service retained work before requeueing a sleeping scenario."""
        with self.operation():
            admission = self.scheduler.get_scenario_admission(scenario_id, tenant_id)
            service_id = integer_value(payload, "service_id")
            generation = integer_value(payload, "generation")
            record_id = text_value(payload, "agent_record_id")
            if service_id != admission.service_id:
                raise AdmissionConflict("wake service does not own the scenario")
            if admission.state == "sleep":
                remote = self.control(
                    service_id,
                    {
                        "operation": "status",
                        "scenario_id": scenario_id,
                        "scenario_name": admission.scenario_name,
                        "tenant_id": tenant_id,
                        "generation": generation,
                    },
                )
                if (
                    remote.get("state") != "reclaimed"
                    or remote.get("reclaim_complete") is not True
                    or remote.get("wake_request_id") != record_id
                    or any(
                        integer_value(remote, field) != expected
                        for field, expected in (
                            ("scenario_id", scenario_id),
                            ("tenant_id", tenant_id),
                            ("generation", generation),
                            ("wake_generation", generation),
                        )
                    )
                ):
                    raise AdmissionConflict("service has no matching retained evolve instruction")
            self.scheduler.wake_scenario_admission(scenario_id, tenant_id, generation)
            self.reconcile()
            return self.view(self.scheduler.get_scenario_admission(scenario_id, tenant_id))

    def cancel(self, tenant_id: int, service_name: str, scenario_name: str) -> dict[str, object]:
        with self.operation():
            service_id = self.scheduler.service_names.get(service_name)
            if service_id is None:
                raise AdmissionNotFound("unknown scenario")
            admission = self.scheduler.find_latest_scenario_admission(service_id, scenario_name)
            pending = self.scheduler.request_cleanup(admission.scenario_id, tenant_id)
            self.reconcile()
            return self.view(self.scheduler.get_scenario_admission(pending.scenario_id, tenant_id))

    def status(self, tenant_id: int) -> dict[str, object]:
        with self.operation():
            allocations = self.scheduler.tenant_allocations()
            scenarios = self.scheduler.list_scenario_admissions(tenant_id)
            used = sum(value.execution_units for value in scenarios if value.execution_allocated)
            allocated = next(
                (value.allocated_execution_units for value in allocations if value.tenant_id == tenant_id), 0
            )
            return {
                "global_budget": self.scheduler.config.global_execution_units,
                "global_allocated": sum(value.allocated_execution_units for value in allocations),
                "tenant_id": tenant_id,
                "allocated": allocated,
                "used": used,
                "available": allocated - used,
                "scenarios": [self.view(value) for value in scenarios if value.state not in TERMINAL_STATES],
                "services": [
                    {
                        **asdict(value),
                        "available": self.scheduler.availability[value.service_id],
                        "reason": self.errors.get(value.service_id),
                    }
                    for value in self.scheduler.service_registrations()
                ],
            }

    def get_scenario_admission(self, scenario_id: int, tenant_id: int) -> dict[str, object]:
        with self.operation():
            return self.view(self.scheduler.get_scenario_admission(scenario_id, tenant_id))

    def list_scenario_admissions(self, tenant_id: int, service_id: int | None = None) -> dict[str, object]:
        with self.operation():
            return {
                "scenarios": [
                    self.view(value)
                    for value in self.scheduler.list_scenario_admissions(tenant_id)
                    if service_id is None or value.service_id == service_id
                ]
            }
