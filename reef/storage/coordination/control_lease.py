"""The control lease: which control instance is active, and the number its writes carry.

Control runs as one active instance and, optionally, standby instances. The active instance holds the lease and
extends it. A standby instance takes the lease only after it expires; each take adds one to
``control_lease_number``. Control writes to ``served_models`` carry that number, so an instance that lost the
lease cannot change what is served.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.values import require_text


@dataclass(frozen=True)
class ControlLeaseState:
    """The current holder of the control lease.

    Values: ``instance_id`` is a non-empty string; ``number`` is a positive integer that increases by one at each
    take; ``expires_at`` is a finite time in seconds.

    Purpose: a ``ControlLease`` makes it; control reads ``number`` and puts it on its writes.
    """

    instance_id: str
    number: int
    expires_at: float

    def __post_init__(self) -> None:
        require_text(self.instance_id, "instance_id")
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 1:
            raise ValueError("a control lease number is a positive integer")
        if (
            isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, (int, float))
            or not math.isfinite(self.expires_at)
        ):
            raise ValueError("expires_at is a finite time")


class ControlLease(ABC):
    """The one control lease of a deployment.

    Values: the state is None before any take, else one ``ControlLeaseState``.

    Purpose: each control instance calls ``acquire`` at start-up and on a timer; the active one calls ``extend``, and
    ``release`` when it stops.
    """

    def __init__(self, *, lease_seconds: float = 30.0) -> None:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds is a positive number")
        self.lease_seconds = float(lease_seconds)

    @abstractmethod
    def acquire(self, instance_id: str, *, now: float | None = None) -> ControlLeaseState | None:
        """Take the lease for ``instance_id`` when nobody holds it or the holder's lease expired, adding one to the
        number; when ``instance_id`` already holds it, extend it. Return the new state, or None when another
        instance holds a lease that has not expired."""

    @abstractmethod
    def extend(self, instance_id: str, *, now: float | None = None) -> ControlLeaseState | None:
        """Extend the lease when ``instance_id`` holds it and it has not expired; return the new state, or None when
        the instance lost the lease. The number does not change."""

    @abstractmethod
    def release(self, instance_id: str, *, now: float | None = None) -> bool:
        """End the lease of ``instance_id`` now, so that a standby instance takes it at once; return whether
        ``instance_id`` held it. The number does not change."""

    @abstractmethod
    def current(self) -> ControlLeaseState | None:
        """The current state."""


__all__ = ["ControlLease", "ControlLeaseState"]
