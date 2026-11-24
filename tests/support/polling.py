"""Wait for a condition that another thread or process makes true."""

import time
from collections.abc import Callable


def became_true(
    condition: Callable[[], bool], *, timeout_seconds: float = 10.0, interval_seconds: float = 0.01
) -> bool:
    """Whether ``condition`` returned true within ``timeout_seconds``.

    The call checks ``condition`` at once and then every ``interval_seconds``. It returns at the first true
    result. It returns False when the time is over and the last check was false.
    """
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_seconds)
    return True


def wait_until(
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.01,
    message: str = "the condition did not become true",
) -> None:
    """Return when ``condition`` returns true; raise ``AssertionError`` with ``message`` after ``timeout_seconds``."""
    if not became_true(condition, timeout_seconds=timeout_seconds, interval_seconds=interval_seconds):
        raise AssertionError(f"{message} within {timeout_seconds} s")
