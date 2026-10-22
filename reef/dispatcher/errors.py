"""Training errors that the training status and the operational metrics report.

Each error belongs to one scenario and to the thread that recorded it. That
thread is the training thread or the local worker of one component. Only
that thread clears its own error. Thus a local cycle that succeeds does not
hide a failed turn of the training thread.
"""

from __future__ import annotations

from threading import Lock

#: The training thread's name in the error record; local workers are named by their component.
TRAINING_THREAD_SOURCE = "training"


def local_error_source(component: str | None) -> str:
    return "local" if component is None else f"local:{component}"


def error_text(exc: Exception) -> str:
    """The text that the training status shows for ``exc``."""
    return f"{type(exc).__name__}: {exc}"


class TrainingErrors:
    """Keep training errors and failure counts.

    Every method takes ``lock`` and calls no other object while it holds it.
    Thus a caller can hold another lock, such as the coordination lock of the
    local workers, when it calls a method.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        #: The last error for each (scenario, thread that recorded it).
        self.errors: dict[tuple[str, str], str] = {}
        self.failure_counts: dict[str, int] = {}
        #: The last failure to build the training status. It is not specific to one scenario.
        self.status_build_error: str | None = None

    def record(self, scenario: str, error: str, source: str = TRAINING_THREAD_SOURCE) -> None:
        """Keep ``error`` for ``scenario`` under ``source``, and count one failure."""
        with self.lock:
            self.errors[(scenario, source)] = error
            self.failure_counts[scenario] = self.failure_counts.get(scenario, 0) + 1

    def clear(self, scenario: str, source: str = TRAINING_THREAD_SOURCE) -> None:
        """Remove the error that ``source`` recorded for ``scenario``."""
        with self.lock:
            self.errors.pop((scenario, source), None)

    def has_error(self, scenario: str) -> bool:
        with self.lock:
            return any(key[0] == scenario for key in self.errors)

    def failure_count(self, scenario: str) -> int:
        with self.lock:
            return self.failure_counts.get(scenario, 0)

    def forget(self, scenario: str) -> None:
        """Remove the errors and the failure count of a deleted scenario."""
        with self.lock:
            for key in [key for key in self.errors if key[0] == scenario]:
                self.errors.pop(key)
            self.failure_counts.pop(scenario, None)

    def set_status_build_error(self, error: str | None) -> bool:
        """Keep or clear the failure to build the training status, and return whether it changed."""
        with self.lock:
            changed = error != self.status_build_error
            self.status_build_error = error
            return changed

    def snapshot(self) -> tuple[dict[tuple[str, str], str], str | None]:
        """A copy of the recorded errors and the status build error."""
        with self.lock:
            return dict(self.errors), self.status_build_error
