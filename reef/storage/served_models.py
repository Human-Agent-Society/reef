"""The ``served_models`` table: what each scenario serves, and when control stops serving a replaced artifact.

Control is the only writer. Each write carries the ``control_lease_number`` of the control instance that holds
the control lease (``reef.storage.control_lease``); a write with an older number fails, so a control instance
that lost its lease cannot change what is served. Inference replicas only read the table.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping

from reef.core.errors import ReefError
from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact
from reef.storage.control_lease import ControlLease


class ControlLeaseConflict(ReefError):
    """A control write carried a ``control_lease_number`` that is not the current one."""


class ServedModels(ABC):
    """Read access to ``served_models``.

    Values: the state is a partial function from ``ScenarioModel`` to ``ServedArtifact``, a finite set of
    ``ScheduledStop`` rows, and ``changed_sequence``, an integer that increases at each change of either.

    Purpose: inference replicas read the served artifact of a request's key (``read``); readers find changes by
    ``changed_sequence``.
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


class ServedModelsWriter(ServedModels):
    """Write access to ``served_models``, for the control instance that holds the control lease.

    Every operation takes ``control_lease_number`` and raises ``ControlLeaseConflict`` when it is not the number
    that ``current_lease_number`` reports, without a change.

    Purpose: control writes the served artifact of each key at activation and at ``serve`` of an artifact beside
    the active one (``write``, ``remove``); it schedules the stop of a replaced artifact (``schedule_stop``) and
    stops the due ones (``due_stops``, ``delete_stop``).
    """

    @abstractmethod
    def current_lease_number(self) -> int:
        """The ``control_lease_number`` that writes must carry now."""

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
        """Whether any row of the served artifact's scenarios still names ``served``."""
        return served in self.all_served()

    @abstractmethod
    def all_served(self) -> frozenset[ServedArtifact]:
        """Every served artifact that some row names."""


class MemoryServedModels(ServedModelsWriter):
    """``served_models`` in process memory, for the one-process layout and for tests.

    Writes must carry the number of ``control_lease``, as in a coordination database.
    """

    def __init__(self, control_lease: ControlLease) -> None:
        self.lock = threading.Lock()
        self.rows: dict[ScenarioModel, ServedArtifact] = {}
        self.stops: dict[ServedArtifact, ScheduledStop] = {}
        self.sequence = 0
        self.control_lease = control_lease

    def lease_number(self) -> int:
        state = self.control_lease.current()
        return 0 if state is None else state.number

    def check_lease(self, control_lease_number: int) -> None:
        current = self.lease_number()
        if control_lease_number != current:
            raise ControlLeaseConflict(
                f"control lease number {control_lease_number} is not the current number {current}"
            )

    def read(self, key: ScenarioModel) -> ServedArtifact | None:
        with self.lock:
            return self.rows.get(key)

    def read_scenario(self, scenario: str) -> Mapping[str, ServedArtifact]:
        with self.lock:
            return {key.name: served for key, served in self.rows.items() if key.scenario == scenario}

    def changed_sequence(self) -> int:
        with self.lock:
            return self.sequence

    def current_lease_number(self) -> int:
        return self.lease_number()

    def write(self, key: ScenarioModel, served: ServedArtifact, *, control_lease_number: int) -> None:
        with self.lock:
            self.check_lease(control_lease_number)
            self.rows[key] = served
            self.sequence += 1

    def remove(self, key: ScenarioModel, *, control_lease_number: int) -> ServedArtifact | None:
        with self.lock:
            self.check_lease(control_lease_number)
            removed = self.rows.pop(key, None)
            if removed is not None:
                self.sequence += 1
            return removed

    def schedule_stop(self, stop: ScheduledStop, *, control_lease_number: int) -> None:
        with self.lock:
            self.check_lease(control_lease_number)
            self.stops[stop.served] = stop
            self.sequence += 1

    def due_stops(self, now: float) -> tuple[ScheduledStop, ...]:
        with self.lock:
            return tuple(sorted((s for s in self.stops.values() if s.stop_after <= now), key=lambda s: s.stop_after))

    def delete_stop(self, served: ServedArtifact, *, control_lease_number: int) -> None:
        with self.lock:
            self.check_lease(control_lease_number)
            if self.stops.pop(served, None) is not None:
                self.sequence += 1

    def all_served(self) -> frozenset[ServedArtifact]:
        with self.lock:
            return frozenset(self.rows.values())


__all__ = ["ControlLeaseConflict", "MemoryServedModels", "ServedModels", "ServedModelsWriter"]
