"""Tenant quotas and FIFO admission; actual scenario operations run in Reef services."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from reef.core.errors import ReefError
from reef.storage.scheduling import (
    ScenarioAdmission,
    SchedulingSnapshot,
    SchedulingStore,
    TenantAllocation,
    TerminalState,
)

logger = logging.getLogger(__name__)

# Sleeping scenarios need service reconciliation only after waking or deletion.
# Cleanup of a sleeper must not allocate execution units again.
RECONCILE_STATES = frozenset({"starting", "admitted", "reclaim_pending", "cleanup_pending"})
TERMINAL_STATES = frozenset({"deleted", "failed"})
# SQLite integers are signed 64-bit; leave room to persist the next unused ID.
MAX_SCENARIO_ID = 2**63 - 2
MAX_IDENTIFIER = 2**63 - 1


class SchedulingError(ReefError):
    """A request cannot be accepted under the configured scheduling contract."""


class AdmissionConflict(SchedulingError):
    """The recipe, creation settings, or service registration conflicts with existing state."""


class AdmissionNotFound(SchedulingError):
    """The requested scenario admission record does not exist."""


class ScenarioTenantMismatch(SchedulingError):
    """The declared tenant does not own this scenario admission."""


class RequestTenantMismatch(SchedulingError):
    """The request body and tenant header identify different tenants."""


class UnknownTenant(SchedulingError):
    """No fixed quota has been configured for the requested tenant."""


class ScenarioRequirementExceedsQuota(SchedulingError):
    """A single scenario can never fit in its tenant's total configured quota."""


class GlobalQuotaExceeded(SchedulingError):
    """The global budget cannot reserve this new tenant's fixed quota."""


class ServiceUnavailable(SchedulingError):
    """The selected service is unknown or temporarily unavailable."""


class SchedulerNotAcceptingRequests(SchedulingError):
    """Graceful shutdown has stopped new admissions and queue starts."""


def validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError(f"{label} must contain 1 to 128 letters, digits, dots, underscores, or hyphens")


def validate_numeric_id(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_IDENTIFIER:
        raise ValueError(f"{label} must be a positive signed 64-bit integer")


def resolve_timestamp(now_seconds: float | None) -> float:
    """Resolve Unix seconds; an explicit timestamp supports deterministic scheduling tests."""
    if now_seconds is None:
        return time.time()
    if (
        isinstance(now_seconds, bool)
        or not isinstance(now_seconds, int | float)
        or not math.isfinite(now_seconds)
        or now_seconds < 0
    ):
        raise ValueError("time must be finite, nonnegative Unix seconds")
    return float(now_seconds)


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler policy and deployment data; service-layer functions perform configuration I/O."""

    global_execution_units: int
    tenant_quotas: Mapping[int, int]
    recipe_execution_units: Mapping[str, int] = field(default_factory=dict)
    idle_ttl_seconds: float | None = None
    database: Path = Path(":memory:")
    services: tuple[ServiceRegistration, ...] = ()
    control_token_envs: dict[int, str] = field(default_factory=dict)
    token_env: str | None = None
    tenant_id_map: dict[str, int] = field(default_factory=dict)
    service_id_map: dict[str, int] = field(default_factory=dict)
    path: Path | None = None
    max_logical_thread_percentage: float = 70.0

    def __post_init__(self) -> None:
        percentage = self.max_logical_thread_percentage
        if (
            isinstance(percentage, bool)
            or not isinstance(percentage, int | float)
            or not math.isfinite(percentage)
            or not 0 < percentage <= 100
        ):
            raise ValueError("max_logical_thread_percentage must be a finite number greater than 0 and at most 100")
        quantities = [self.global_execution_units, *self.tenant_quotas.values(), *self.recipe_execution_units.values()]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in quantities):
            raise ValueError("execution budgets, tenant quotas, and recipe requirements must be positive integers")
        if any(value > self.global_execution_units for value in self.tenant_quotas.values()):
            raise ValueError("a tenant quota cannot exceed the global execution budget")
        for tenant_id in self.tenant_quotas:
            validate_numeric_id(tenant_id, "tenant_id")
        for recipe in self.recipe_execution_units:
            if not isinstance(recipe, str) or not recipe.strip():
                raise ValueError("recipe names must be nonempty strings")
        duration = self.idle_ttl_seconds
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, int | float)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("idle TTL must be positive finite seconds, or None to disable reclamation")
        object.__setattr__(self, "tenant_quotas", MappingProxyType(dict(self.tenant_quotas)))
        object.__setattr__(self, "recipe_execution_units", MappingProxyType(dict(self.recipe_execution_units)))


@dataclass(frozen=True)
class ServiceRegistration:
    service_id: int
    service_name: str
    recipe: str
    service_url: str
    standard_execution_units: int
    config_revision: str = ""

    def __post_init__(self) -> None:
        validate_numeric_id(self.service_id, "service_id")
        validate_identifier(self.service_name, "service_name")
        if not isinstance(self.recipe, str) or not self.recipe.strip():
            raise ValueError("service recipe must be a nonempty string")
        if not isinstance(self.service_url, str) or not self.service_url.strip():
            raise ValueError("service URL must be a nonempty string")
        if (
            isinstance(self.standard_execution_units, bool)
            or not isinstance(self.standard_execution_units, int)
            or self.standard_execution_units < 1
        ):
            raise ValueError("service standard_execution_units must be a positive integer")
        if not isinstance(self.config_revision, str):
            raise ValueError("service config_revision must be a string")

    @property
    def registration_fingerprint(self) -> str:
        """Bind active scenarios to the same registered deployment across restarts."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def legacy_registration_fingerprint(self, legacy_service_id: str) -> str:
        """Verify the prior registration before an explicitly mapped schema migration."""
        value = {
            "service_id": legacy_service_id,
            "service_name": self.service_name,
            "recipe": self.recipe,
            "endpoint": self.service_url,
            "execution_units": self.standard_execution_units,
            "config_ref": self.config_revision,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class MultiTenantScheduler:
    """Reserve tenant quotas and select scenarios for the HTTP controller to start.

    The controller performs service calls outside the scheduler lock. Execution
    remains allocated until the service confirms cleanup or idle reclamation.
    """

    def __init__(self, store: SchedulingStore, config: SchedulerConfig) -> None:
        self.store = store
        self.config = config
        # Public operations call other public operations under one atomic quota
        # transition. Reentrancy allows this without releasing the state lock.
        self.lock = RLock()
        self.state = store.load()
        self.services: dict[int, ServiceRegistration] = {}
        self.service_names: dict[str, int] = {}
        self.availability: dict[int, bool] = {}
        self.recipe_execution_units = dict(config.recipe_execution_units)
        self.accepting_requests = True
        if sum(tenant.allocated_execution_units for tenant in self.state.tenants) > config.global_execution_units:
            raise ValueError("restored tenant allocations exceed the configured global budget")
        for tenant in self.state.tenants:
            if config.tenant_quotas.get(tenant.tenant_id) != tenant.allocated_execution_units:
                raise ValueError("restored tenant quotas differ from configuration")
        for scenario in self.state.scenarios:
            execution_units = self.recipe_execution_units.get(scenario.recipe)
            if execution_units is not None and execution_units != scenario.execution_units:
                raise ValueError("restored scenario requirements differ from configuration")
            self.recipe_execution_units[scenario.recipe] = scenario.execution_units
        self.admissions_by_id: dict[int, ScenarioAdmission] = {}
        self.cold_scenarios_by_id: dict[int, ScenarioAdmission] = {}
        self.admissions_by_name: dict[tuple[int, str], ScenarioAdmission] = {}
        self.waiting_by_tenant: dict[int, tuple[int, ...]] = {}
        self.used_by_tenant: dict[int, int] = {}
        self.refresh_indexes()

    def refresh_indexes(self) -> None:
        """Rebuild read indexes only after a committed state transition, under the lock."""
        previous_queues = self.waiting_by_tenant
        self.admissions_by_id = {value.scenario_id: value for value in self.state.scenarios}
        self.cold_scenarios_by_id = {
            value.scenario_id: value for value in self.state.scenarios if value.state == "sleep"
        }
        self.admissions_by_name = {(value.service_id, value.scenario_name): value for value in self.state.scenarios}
        waiting: dict[int, list[tuple[int, int]]] = {}
        used: dict[int, int] = {}
        for value in self.state.scenarios:
            if value.state == "queued":
                waiting.setdefault(value.tenant_id, []).append((value.queue_sequence, value.scenario_id))
            if value.execution_allocated:
                used[value.tenant_id] = used.get(value.tenant_id, 0) + value.execution_units
        self.waiting_by_tenant = {
            tenant: tuple(identifier for _, identifier in sorted(rows)) for tenant, rows in waiting.items()
        }
        self.used_by_tenant = used
        if logger.isEnabledFor(logging.INFO):
            allocations = {tenant.tenant_id: tenant.allocated_execution_units for tenant in self.state.tenants}
            for tenant_id in sorted(previous_queues.keys() | self.waiting_by_tenant.keys()):
                previous = previous_queues.get(tenant_id, ())
                current = self.waiting_by_tenant.get(tenant_id, ())
                if previous == current:
                    continue
                # Log committed FIFO order, including empty queues, without request bodies or credentials.
                snapshot = {
                    "tenant_id": tenant_id,
                    "queue_before": list(previous),
                    "queue": [
                        {
                            "scenario_id": scenario_id,
                            "service_id": self.admissions_by_id[scenario_id].service_id,
                            "scenario_name": self.admissions_by_id[scenario_id].scenario_name,
                        }
                        for scenario_id in current
                    ],
                    "allocated_scenario_ids": [
                        admission.scenario_id
                        for admission in self.state.scenarios
                        if admission.tenant_id == tenant_id and admission.execution_allocated
                    ],
                    "used_execution_units": used.get(tenant_id, 0),
                    "allocated_execution_units": allocations.get(tenant_id, 0),
                }
                logger.info("tenant_queue_changed %s", json.dumps(snapshot, separators=(",", ":")))

    def register_service(self, registration: ServiceRegistration) -> None:
        with self.lock:
            if not self.accepting_requests:
                raise SchedulerNotAcceptingRequests("scheduler has stopped service registration")
            execution_units = self.recipe_execution_units.get(registration.recipe)
            if execution_units is not None and execution_units != registration.standard_execution_units:
                raise AdmissionConflict("service resources do not match the recipe's static requirement")
            existing_id = self.service_names.get(registration.service_name)
            existing = self.services.get(registration.service_id)
            if existing_id not in (None, registration.service_id) or existing not in (None, registration):
                raise AdmissionConflict("service name and ID must identify one unchanged registration")
            for scenario in self.state.scenarios:
                if scenario.service_id == registration.service_id and (
                    scenario.recipe != registration.recipe
                    or scenario.execution_units != registration.standard_execution_units
                    or scenario.service_config_ref != registration.registration_fingerprint
                ):
                    raise AdmissionConflict("service configuration changed for an existing admission")
            self.recipe_execution_units[registration.recipe] = registration.standard_execution_units
            self.services[registration.service_id] = registration
            self.service_names[registration.service_name] = registration.service_id
            # A directory entry becomes available only after the controller's
            # description request verifies the remote service identity and cost.
            self.availability.setdefault(registration.service_id, False)

    def service_registrations(self) -> tuple[ServiceRegistration, ...]:
        """Return a consistent directory snapshot while registrations may arrive concurrently."""
        with self.lock:
            return tuple(self.services.values())

    def set_service_availability(self, service_id: int, available: bool) -> None:
        if not isinstance(available, bool):
            raise ValueError("service availability must be a boolean")
        with self.lock:
            if service_id not in self.services:
                raise ServiceUnavailable("unknown service ID")
            self.availability[service_id] = available

    def request_scenario_admission(
        self,
        tenant_id: int,
        service_name: str,
        recipe: str,
        scenario_name: str,
        *,
        release_id: str | None = None,
        model_config_ref: str | None = None,
        now_seconds: float | None = None,
    ) -> ScenarioAdmission:
        """Reserve a tenant quota and scenario name, then append its metadata to FIFO."""
        validate_numeric_id(tenant_id, "tenant_id")
        validate_identifier(service_name, "service_name")
        validate_identifier(scenario_name, "scenario_name")
        if release_id is not None and (not isinstance(release_id, str) or not release_id.strip()):
            raise ValueError("release_id must be a nonempty opaque version string when supplied")
        if model_config_ref is not None and (
            not isinstance(model_config_ref, str) or re.fullmatch(r"[a-f0-9]{64}", model_config_ref) is None
        ):
            raise ValueError("model_config_ref must be a SHA-256 reference to service-owned configuration")
        timestamp = resolve_timestamp(now_seconds)
        with self.lock:
            if not self.accepting_requests:
                raise SchedulerNotAcceptingRequests("scheduler has stopped new admission")
            service_id = self.service_names.get(service_name)
            if service_id is None:
                raise ServiceUnavailable("unknown service name")
            service = self.services[service_id]
            if recipe != service.recipe:
                raise AdmissionConflict("requested recipe does not match the service")
            current = self.admissions_by_name.get((service_id, scenario_name))
            if current is not None:
                if current.tenant_id != tenant_id:
                    raise ScenarioTenantMismatch("scenario name is owned by another tenant")
                if current.release_id != release_id or current.model_config_ref != model_config_ref:
                    raise AdmissionConflict("scenario creation parameters differ from the original request")
                return current
            if not self.availability[service_id]:
                raise ServiceUnavailable("selected service is unavailable")
            quota = self.config.tenant_quotas.get(tenant_id)
            if quota is None:
                raise UnknownTenant("tenant has no configured quota")
            if service.standard_execution_units > quota:
                raise ScenarioRequirementExceedsQuota("scenario requirement exceeds the tenant's total quota")
            tenants = self.state.tenants
            if not any(value.tenant_id == tenant_id for value in tenants):
                if (
                    sum(value.allocated_execution_units for value in tenants) + quota
                    > self.config.global_execution_units
                ):
                    raise GlobalQuotaExceeded("global budget cannot reserve this tenant quota")
                tenants = (*tenants, TenantAllocation(tenant_id, quota))
            scenario_id = self.state.next_scenario_id
            # Modulo-based scenario ID reuse can be implemented later if needed.
            # Reuse must wait until cleanup and delayed messages for the previous lifecycle have settled.
            if scenario_id > MAX_SCENARIO_ID:
                raise SchedulingError("scenario ID range is exhausted")
            if self.state.next_queue_sequence > MAX_SCENARIO_ID:
                raise SchedulingError("queue sequence range is exhausted")
            admission = ScenarioAdmission(
                scenario_id=scenario_id,
                tenant_id=tenant_id,
                service_id=service_id,
                recipe=recipe,
                scenario_name=scenario_name,
                release_id=release_id,
                model_config_ref=model_config_ref,
                execution_units=service.standard_execution_units,
                state="queued",
                generation=0,
                created_at=timestamp,
                service_config_ref=service.registration_fingerprint,
                queue_sequence=self.state.next_queue_sequence,
            )
            self.state = self.store.save(
                replace(
                    self.state,
                    next_scenario_id=scenario_id + 1,
                    tenants=tenants,
                    scenarios=(*self.state.scenarios, admission),
                    next_queue_sequence=self.state.next_queue_sequence + 1,
                )
            )
            self.refresh_indexes()
            return admission

    def next_scenario_admission(self, tenant_id: int, *, now_seconds: float | None = None) -> ScenarioAdmission | None:
        """Allocate only the FIFO head when its service and remaining tenant quota allow it."""
        if now_seconds is not None:
            resolve_timestamp(now_seconds)
        with self.lock:
            if not self.accepting_requests:
                raise SchedulerNotAcceptingRequests("scheduler has stopped queue starts")
            waiting_ids = self.waiting_by_tenant.get(tenant_id, ())
            if not waiting_ids:
                return None
            waiting = self.admissions_by_id[waiting_ids[0]]
            if not self.availability.get(waiting.service_id, False):
                if waiting.failure_reason != "selected service is unavailable":
                    self.save_scenario_admission(replace(waiting, failure_reason="selected service is unavailable"))
                return None
            if self.used_by_tenant.get(tenant_id, 0) + waiting.execution_units > self.config.tenant_quotas[tenant_id]:
                return None
            return self.save_scenario_admission(
                replace(
                    waiting,
                    state="starting",
                    generation=waiting.generation + 1,
                    idle_since=None,
                    reclaim_before=None,
                    failure_reason=None,
                    has_started=True,
                    execution_allocated=True,
                )
            )

    def confirm_started(
        self, scenario_id: int, generation: int, *, now_seconds: float | None = None
    ) -> ScenarioAdmission:
        if now_seconds is not None:
            resolve_timestamp(now_seconds)
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            # Ignore acknowledgements superseded by a newer lifecycle operation.
            if scenario.state != "starting" or scenario.generation != generation:
                return scenario
            return self.save_scenario_admission(replace(scenario, state="admitted", idle_since=None))

    def request_cleanup(
        self,
        scenario_id: int,
        tenant_id: int,
        *,
        failed: bool = False,
        reason: str | None = None,
    ) -> ScenarioAdmission:
        """Delete metadata-only requests locally; otherwise await final service cleanup."""
        with self.lock:
            scenario = self.get_scenario_admission(scenario_id, tenant_id)
            if scenario.state in TERMINAL_STATES or scenario.state == "cleanup_pending":
                return scenario
            terminal_state: TerminalState = "failed" if failed else "deleted"
            if scenario.state == "queued" and not scenario.has_started:
                return self.save_scenario_admission(
                    replace(
                        scenario,
                        state=terminal_state,
                        terminal_state=terminal_state,
                        failure_reason=reason,
                    )
                )
            return self.save_scenario_admission(
                replace(
                    scenario,
                    state="cleanup_pending",
                    generation=scenario.generation + 1,
                    terminal_state=terminal_state,
                    failure_reason=reason,
                    reclaim_before=None,
                    idle_since=None,
                )
            )

    def confirm_cleaned(self, scenario_id: int, generation: int) -> ScenarioAdmission:
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            # An old cleanup acknowledgement must not release a newer allocation.
            if scenario.state != "cleanup_pending" or scenario.generation != generation:
                return scenario
            return self.save_scenario_admission(
                replace(
                    scenario,
                    state=scenario.terminal_state,
                    execution_allocated=False,
                    idle_since=None,
                )
            )

    def report_scenario_idle(
        self, scenario_id: int, generation: int, *, idle_since: float | None
    ) -> ScenarioAdmission:
        """Accept service-observed idle state only for the current execution generation."""
        if idle_since is not None:
            idle_since = resolve_timestamp(idle_since)
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            if scenario.state != "admitted" or scenario.generation != generation or scenario.idle_since == idle_since:
                return scenario
            return self.save_scenario_admission(replace(scenario, idle_since=idle_since))

    def request_reclaim(self, scenario_id: int, *, now_seconds: float | None = None) -> ScenarioAdmission:
        """
        The function is initiated by the scheduler to check if the specific scenario has exceeded the idle TTL.
        If so, the scenario state is changed to relaim pending and waiting for the reclaim.
        """
        timestamp = resolve_timestamp(now_seconds)
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            duration = self.config.idle_ttl_seconds
            if (
                scenario.state != "admitted"
                or duration is None
                or scenario.idle_since is None
                or timestamp - scenario.idle_since < duration
            ):
                return scenario
            return self.save_scenario_admission(
                replace(
                    scenario,
                    state="reclaim_pending",
                    generation=scenario.generation + 1,
                    reclaim_before=timestamp - duration,
                )
            )

    def cancel_reclaim(
        self, scenario_id: int, generation: int, *, idle_since: float | None = None
    ) -> ScenarioAdmission:
        """Retain resources when the service's atomic check finds fresh or unfinished work."""
        if idle_since is not None:
            idle_since = resolve_timestamp(idle_since)
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            if scenario.state != "reclaim_pending" or scenario.generation != generation:
                return scenario
            return self.save_scenario_admission(
                replace(
                    scenario,
                    state="admitted",
                    reclaim_before=None,
                    idle_since=idle_since,
                )
            )

    def confirm_reclaimed(self, scenario_id: int, generation: int) -> ScenarioAdmission:
        """Release execution units and retain the logical scenario asleep until new work arrives."""
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            # A delayed reclaim acknowledgement cannot requeue a resumed scenario.
            if scenario.state != "reclaim_pending" or scenario.generation != generation:
                return scenario
            return self.save_scenario_admission(
                replace(
                    scenario,
                    state="sleep",
                    execution_allocated=False,
                    idle_since=None,
                    reclaim_before=None,
                ),
            )

    def wake_scenario_admission(self, scenario_id: int, tenant_id: int, generation: int) -> ScenarioAdmission:
        """Append a sleeping scenario once, after its service has retained a new instruction."""
        validate_numeric_id(generation, "generation")
        with self.lock:
            if not self.accepting_requests:
                raise SchedulerNotAcceptingRequests("scheduler has stopped new admission")
            scenario = self.get_scenario_admission(scenario_id, tenant_id)
            if scenario.state in {"queued", "starting", "admitted"}:
                return scenario
            if scenario.state != "sleep" or scenario.generation != generation:
                raise AdmissionConflict("scenario is not sleeping in the requested generation")
            if self.state.next_queue_sequence > MAX_SCENARIO_ID:
                raise SchedulingError("queue sequence range is exhausted")
            return self.save_scenario_admission(
                replace(scenario, state="queued", queue_sequence=self.state.next_queue_sequence, failure_reason=None),
                next_queue_sequence=self.state.next_queue_sequence + 1,
            )

    def reclaim_idle_scenarios(self, *, now_seconds: float | None = None) -> tuple[ScenarioAdmission, ...]:
        """Request reclamation for service-reported idle scenarios; waiting requests never expire."""
        timestamp = resolve_timestamp(now_seconds)
        with self.lock:
            changed = []
            for scenario in self.state.scenarios:
                if scenario.state == "admitted":
                    reclaimed = self.request_reclaim(scenario.scenario_id, now_seconds=timestamp)
                    if reclaimed.state == "reclaim_pending":
                        changed.append(reclaimed)
            return tuple(changed)

    def get_scenario_admission(self, scenario_id: int, tenant_id: int) -> ScenarioAdmission:
        validate_numeric_id(tenant_id, "tenant_id")
        with self.lock:
            scenario = self.find_scenario_admission(scenario_id)
            if scenario.tenant_id != tenant_id:
                raise ScenarioTenantMismatch("scenario is owned by another tenant")
            return scenario

    def list_scenario_admissions(self, tenant_id: int | None = None) -> tuple[ScenarioAdmission, ...]:
        """List active and queued records; completed history is queried by ID."""
        with self.lock:
            return tuple(value for value in self.state.scenarios if tenant_id is None or value.tenant_id == tenant_id)

    def find_latest_scenario_admission(self, service_id: int, scenario_name: str) -> ScenarioAdmission:
        with self.lock:
            admission = self.admissions_by_name.get((service_id, scenario_name))
            if admission is None:
                admission = self.store.find_latest_scenario_admission(service_id, scenario_name)
            if admission is None:
                raise AdmissionNotFound("unknown scenario name")
            return admission

    def tenant_allocations(self) -> tuple[TenantAllocation, ...]:
        """Return fixed tenant quotas currently reserved from the global budget."""
        with self.lock:
            return self.state.tenants

    def pending_scenario_admissions(self) -> tuple[ScenarioAdmission, ...]:
        """Return records waiting for a start, reclaim, or final cleanup confirmation."""
        with self.lock:
            return tuple(
                value
                for value in self.state.scenarios
                if value.state in {"starting", "reclaim_pending", "cleanup_pending"}
            )

    def snapshot(self) -> SchedulingSnapshot:
        with self.lock:
            return self.state

    def find_scenario_admission(self, scenario_id: int) -> ScenarioAdmission:
        validate_numeric_id(scenario_id, "scenario_id")
        if scenario_id > MAX_SCENARIO_ID:
            raise ValueError("scenario_id is outside the supported positive integer range")
        with self.lock:
            scenario = self.admissions_by_id.get(scenario_id)
            if scenario is not None:
                return scenario
            completed = self.store.get_scenario_admission(scenario_id)
            if completed is not None:
                return completed
            raise AdmissionNotFound("unknown scenario ID")

    def save_scenario_admission(
        self, admission: ScenarioAdmission, *, next_queue_sequence: int | None = None
    ) -> ScenarioAdmission:
        """Commit the transition before publishing its indexes; caller holds the reentrant state lock."""
        scenarios = tuple(
            admission if value.scenario_id == admission.scenario_id else value for value in self.state.scenarios
        )
        active_tenants = {value.tenant_id for value in scenarios if value.state not in TERMINAL_STATES}
        tenants = tuple(value for value in self.state.tenants if value.tenant_id in active_tenants)
        sequence = self.state.next_queue_sequence if next_queue_sequence is None else next_queue_sequence
        self.state = self.store.save(
            replace(self.state, scenarios=scenarios, tenants=tenants, next_queue_sequence=sequence)
        )
        self.refresh_indexes()
        return admission

    def stop_accepting_requests(self) -> None:
        """Stop new admissions during graceful shutdown while retaining cleanup access."""
        with self.lock:
            self.accepting_requests = False

    def close(self) -> None:
        with self.lock:
            self.accepting_requests = False
            self.store.close()
