"""Serving requests, the routes that completed requests publish, and the stops of replaced routes.

Control selects artifacts under its control lease. Inference publishes routes and finishes their work under its work lease.
The newest request of a scenario supplies its startup status and stays through queue cleanup.
Direct route writes retain their control lease checks for callers that manage routes without startup work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from reef.core.errors import ReefError
from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact
from reef.storage.work_queue import JsonValue, WorkItem

SERVING_QUEUE = "serving"
SERVING_KIND = "serve"
STOP_SERVING_KIND = "stop_serving"
INFERENCE_OWNER_QUEUE = "inference-owner"


class ControlLeaseConflict(ReefError):
    """A control write carried a ``control_lease_number`` that is not the current one."""


class ServedModels(ABC):
    """Read access to ``served_models``.

    Values: the state is a partial function from ``ScenarioModel`` to ``ServedArtifact``, a finite set of
    ``ScheduledStop`` rows, and ``changed_sequence``, an integer that increases at each change of either.

    Purpose: inference replicas read the served artifact of a request's key (``read``).
    Readers find route changes by ``changed_sequence`` and startup work by ``current_request``.
    """

    @abstractmethod
    def read(self, key: ScenarioModel) -> ServedArtifact | None:
        """The served artifact of ``key``, or None when the scenario serves nothing under that name."""

    @abstractmethod
    def read_scenario(self, scenario: str) -> Mapping[str, ServedArtifact]:
        """Every row of ``scenario``, by name."""

    @abstractmethod
    def changed_sequence(self) -> int:
        """The number of changes so far; it increases at each write, removal and stop change."""

    @abstractmethod
    def current_request(self, scenario: str) -> WorkItem | None:
        """The newest serving request for ``scenario``, or None before its first request.

        An empty payload withdraws the scenario. Otherwise the payload describes its desired artifacts.
        The work item supplies startup status; its result is ready only after its routes were published.
        """


class ServedModelsWriter(ServedModels):
    """Writes of serving work and routes, with the lease of the operation's owner.

    Control selects work with ``request``. Inference completes startup and cleanup with ``complete_request`` and ``complete_stop``.
    Direct operations ``write``, ``remove``, ``schedule_stop`` and ``delete_stop`` check ``control_lease_number``.
    A stale control number raises ``ControlLeaseConflict`` without a change. Reads require no lease.
    """

    @abstractmethod
    def write(self, key: ScenarioModel, served: ServedArtifact, *, control_lease_number: int) -> None:
        """Make ``served`` the row of ``key``."""

    @abstractmethod
    def remove(self, key: ScenarioModel, *, control_lease_number: int) -> ServedArtifact | None:
        """Remove the row of ``key``; return the removed value, or None when there was none."""

    @abstractmethod
    def schedule_stop(self, stop: ScheduledStop, *, control_lease_number: int) -> None:
        """Add ``stop``; a stop of the same served artifact is replaced."""

    @abstractmethod
    def due_stops(self, now: float) -> tuple[ScheduledStop, ...]:
        """The scheduled stops with ``stop_after <= now``, oldest first."""

    @abstractmethod
    def delete_stop(self, served: ServedArtifact, *, control_lease_number: int) -> None:
        """Remove the scheduled stop of ``served``, if there is one."""

    def is_referenced(self, served: ServedArtifact) -> bool:
        """Whether any published route uses ``served``'s engine URL and model name."""
        return any((route.url, route.model_name) == (served.url, served.model_name) for route in self.all_served())

    @abstractmethod
    def all_served(self) -> frozenset[ServedArtifact]:
        """Every served artifact that some row names."""

    @abstractmethod
    def request(
        self,
        scenario: str,
        payload: Mapping[str, JsonValue],
        *,
        control_lease_number: int,
        stop_delay_seconds: float = 0.0,
        retry_failed: bool = False,
        now: float | None = None,
    ) -> WorkItem:
        """Select ``payload`` for ``scenario`` under the control lease and return its serving work.

        An unchanged payload returns the current item, except a failed item when ``retry_failed`` is true.
        A new request cancels earlier queued work and gets a new ID. Held work finishes without publishing.
        All serving work shares one serial key, because runtimes can share an engine across scenarios.
        An empty payload withdraws the routes immediately and records a completed request.
        Unreferenced withdrawn routes stop after the finite, non-negative ``stop_delay_seconds``.
        """

    @abstractmethod
    def complete_request(
        self,
        item: WorkItem,
        worker_id: str,
        models: Mapping[str, ServedArtifact],
        *,
        stop_delay_seconds: float,
        owner: WorkItem | None = None,
        now: float | None = None,
    ) -> WorkItem:
        """Publish ``models`` for ``item.scenario`` and finish its serving work in one transaction.

        The item must be held by ``worker_id`` in the same attempt; otherwise raise ``WorkLeaseLost``.
        When supplied, ``owner`` must also be held in the same attempt.
        Superseded work finishes with an empty result and changes no route. Schedule its unreferenced endpoints for cleanup.
        Each model name is non-empty. Schedule unreferenced replaced routes after ``stop_delay_seconds``.
        The delay must be finite and non-negative.
        The returned finished item's result maps model names to their serialized served artifacts.
        """

    @abstractmethod
    def complete_stop(
        self,
        item: WorkItem,
        worker_id: str,
        stop: ScheduledStop,
        *,
        owner: WorkItem | None = None,
        now: float | None = None,
    ) -> WorkItem:
        """Remove ``stop`` and finish its held cleanup work in one transaction.

        A later stop of the same endpoint stays when its ``stop_after`` differs.
        An expired lease or another attempt raises ``WorkLeaseLost`` without changing anything.
        When supplied, ``owner`` must also be held in the same attempt.
        """

    @abstractmethod
    def recover_completed(
        self,
        owner: WorkItem,
        worker_id: str,
        *,
        stop_delay_seconds: float,
        now: float | None = None,
    ) -> None:
        """Queue completed current selections again after an inference owner takes over.

        ``owner`` is a held item of the inference-owner queue and kind, in the same attempt.
        Remove current selections' old routes and schedule cleanup after the finite, non-negative delay.
        Failed, unfinished, and withdrawn requests keep their state.
        """


__all__ = [
    "INFERENCE_OWNER_QUEUE",
    "SERVING_KIND",
    "SERVING_QUEUE",
    "STOP_SERVING_KIND",
    "ControlLeaseConflict",
    "ServedModels",
    "ServedModelsWriter",
]
