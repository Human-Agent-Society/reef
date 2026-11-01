"""Lifecycle control shared by jobs, local processes and program runs."""

import math
from abc import ABC, abstractmethod


class Execution(ABC):
    """An execution lifetime with an irreversible, fully cleaned terminal state.

    Implementations represent one run or an owner of several runs. Before
    completion, work or owned cleanup can remain pending. Completion means no
    owned work or cleanup remains, and no new work can be admitted. It does
    not imply success. Concrete interfaces expose their own result types.

    request_stop is thread-safe and idempotent. It requests cooperative
    cancellation without waiting for completion. wait_closed is thread-safe;
    it returns True only after completion, or False when its timeout expires.
    A known cleanup failure raises instead of reporting successful closure.
    wait_closed must not block a thread whose progress that wait needs.
    """

    @abstractmethod
    def request_stop(self) -> None:
        """Request cancellation; implementations may give active work a grace period."""

    @abstractmethod
    def wait_closed(self, timeout_seconds: float | None = None) -> bool:
        """Wait for cleanup; None is unbounded, zero polls, finite positive values bound the wait."""


def validate_wait_timeout(timeout_seconds: float | None) -> None:
    """Reject values outside None or finite, nonnegative real seconds; booleans are invalid."""
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        raise ValueError("timeout_seconds must be finite and nonnegative, or None")
