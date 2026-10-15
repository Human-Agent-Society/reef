"""Durable scheduling records, independent of scenario records and artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from reef.core.errors import ReefError

SchedulerScenarioState = Literal[
    "queued", "starting", "admitted", "reclaim_pending", "sleep", "cleanup_pending", "deleted", "failed"
]
TerminalState = Literal["deleted", "failed"]


class SchedulingStoreConflict(ReefError):
    """Another scheduler changed the persisted state; reload before retrying."""


@dataclass(frozen=True)
class TenantAllocation:
    """A fixed quota reserved from the global budget, distinct from current usage."""

    tenant_id: int
    allocated_execution_units: int


@dataclass(frozen=True)
class ScenarioAdmission:
    """One logical scenario, including repeated resource allocation and idle reclamation."""

    scenario_id: int
    tenant_id: int
    service_id: int
    recipe: str
    scenario_name: str
    release_id: str | None
    model_config_ref: str | None
    execution_units: int
    state: SchedulerScenarioState
    # Each start, cleanup, or reclaim advances this version to reject stale acknowledgements.
    generation: int
    created_at: float
    service_config_ref: str
    queue_sequence: int = 0
    idle_since: float | None = None
    reclaim_before: float | None = None
    has_started: bool = False
    execution_allocated: bool = False
    failure_reason: str | None = None
    terminal_state: TerminalState = "deleted"


@dataclass(frozen=True)
class SchedulingSnapshot:
    version: int = 0
    next_scenario_id: int = 1
    tenants: tuple[TenantAllocation, ...] = ()
    scenarios: tuple[ScenarioAdmission, ...] = ()
    next_queue_sequence: int = 1


class SchedulingStore(ABC):
    """Persist one complete state transition atomically, including ID and queue order."""

    @abstractmethod
    def load(self) -> SchedulingSnapshot:
        """Read a consistent snapshot of active lifecycles."""

    @abstractmethod
    def save(self, snapshot: SchedulingSnapshot) -> SchedulingSnapshot:
        """Persist changes and return the incremented active snapshot.

        Completed lifecycles remain available through point queries without
        retaining historical rows in the scheduler's active working set.
        """

    @abstractmethod
    def get_scenario_admission(self, scenario_id: int) -> ScenarioAdmission | None:
        """Read an active or completed lifecycle by its immutable ID."""

    @abstractmethod
    def find_latest_scenario_admission(self, service_id: int, scenario_name: str) -> ScenarioAdmission | None:
        """Read the newest lifecycle for a service-local name."""

    @abstractmethod
    def close(self) -> None:
        """Release this store's resources."""
