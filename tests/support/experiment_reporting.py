"""Synchronization for tests that hold an experiment provider call open."""

from threading import Event, RLock
from types import TracebackType


class ContendedReportLock:
    """A real reentrant lock that signals when another thread cannot acquire it."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.waiting = Event()

    def __enter__(self) -> "ContendedReportLock":
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            self.lock.acquire()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lock.release()
