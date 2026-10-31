"""The ``served_models`` table: what each scenario serves, and when control stops serving a replaced artifact.

Control is the only writer. Each write carries the ``control_lease_number`` of the control instance that holds
the control lease (``reef.storage.control_lease``); a write with an older number fails, so a control instance
that lost its lease cannot change what is served. Inference replicas only read the table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from reef.core.errors import ReefError
from reef.core.served_models import ScenarioModel, ScheduledStop, ServedArtifact


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

    Every operation that changes the table (``write``, ``remove``, ``schedule_stop``, ``delete_stop``) takes
    ``control_lease_number`` and raises ``ControlLeaseConflict``, without a change, when it is not the number of the
    control lease that holds now (``ControlLease.current``). The reads take none.

    Purpose: control writes the served artifact of each key at activation and at ``serve`` of an artifact beside
    the active one (``write``, ``remove``); it schedules the stop of a replaced artifact (``schedule_stop``) and
    stops the due ones (``due_stops``, ``delete_stop``).
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
        """Whether any row of the served artifact's scenarios still names ``served``."""
        return served in self.all_served()

    @abstractmethod
    def all_served(self) -> frozenset[ServedArtifact]:
        """Every served artifact that some row names."""


__all__ = ["ControlLeaseConflict", "ServedModels", "ServedModelsWriter"]
