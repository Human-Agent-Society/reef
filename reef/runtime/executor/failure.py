"""Process-local, terminal executor failure notification and pending RPC failure."""

from __future__ import annotations

import logging
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import Protocol


@dataclass(frozen=True)
class ExecutorFailure:
    backend: str
    reason: str
    rank: int | None = None


class ExecutorFailedError(RuntimeError):
    def __init__(self, failure: ExecutorFailure):
        self.failure = failure
        super().__init__(f"{failure.backend} executor failed (rank={failure.rank}): {failure.reason}")

    def __reduce__(self):
        return type(self), (self.failure,)


class ExecutorFailureListener(Protocol):
    def on_executor_failure(self, failure: ExecutorFailure) -> None:
        """Observe terminal failure once; do not block the monitor thread."""


class FailureState:
    def __init__(self) -> None:
        self._lock = RLock()
        self._failure: ExecutorFailure | None = None
        self._closed = False
        self._listeners: list[ExecutorFailureListener] = []
        self._pending: set[Future] = set()

    def __getstate__(self):
        # Subscriptions and pending requests belong to the observing process,
        # not to Ray handles serialized into another coordinator.
        with self._lock:
            return self._failure, self._closed

    def __setstate__(self, state):
        self.__init__()
        self._failure, self._closed = state

    @property
    def failure(self) -> ExecutorFailure | None:
        with self._lock:
            return self._failure

    def check(self) -> None:
        with self._lock:
            if self._failure is not None:
                raise ExecutorFailedError(self._failure)
            if self._closed:
                raise RuntimeError("executor is shut down")

    def register(self, listener: ExecutorFailureListener) -> None:
        with self._lock:
            if any(existing is listener for existing in self._listeners):
                return
            self._listeners.append(listener)
            failure = self._failure
        if failure is not None:
            self._notify(listener, failure)

    @staticmethod
    def _notify(listener: ExecutorFailureListener, failure: ExecutorFailure) -> None:
        try:
            listener.on_executor_failure(failure)
        except Exception:
            logging.getLogger(__name__).exception("executor failure listener raised")

    def fail(self, failure: ExecutorFailure) -> bool:
        with self._lock:
            if self._failure is not None or self._closed:
                return False
            self._failure = failure
            for pending in tuple(self._pending):
                self._set_error(pending, ExecutorFailedError(failure))
            self._pending.clear()
            listeners = tuple(self._listeners)
        for listener in listeners:
            self._notify(listener, failure)
        return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for pending in tuple(self._pending):
                self._set_error(pending, RuntimeError("executor is shut down"))
            self._pending.clear()

    @staticmethod
    def _set_error(future: Future, error: BaseException) -> None:
        with suppress(InvalidStateError):
            future.set_exception(error)

    def track(self, source: Future) -> Future:
        target: Future = Future()
        with self._lock:
            self.check()
            self._pending.add(target)

        def complete(future: Future) -> None:
            with self._lock:
                self._pending.discard(target)
                if target.done():
                    return
                try:
                    target.set_result(future.result())
                except BaseException as exc:
                    self._set_error(target, exc)

        source.add_done_callback(complete)
        return target
