"""Background engine probes with a draining pause barrier."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic
from typing import Protocol

logger = logging.getLogger(__name__)


class EngineHealthTarget(Protocol):
    """One captured engine identity, including any nodes that retire together."""

    def check(self, timeout: float) -> None:
        """Raise on failure and bound the probe by the supplied timeout."""

    def retire(self, timeout: float) -> None:
        """Retire captured handles only; never substitute newer occupants of their slots."""


class EngineHealthChecks(Protocol):
    """Snapshot current targets without changing engine ownership."""

    def targets(self) -> Sequence[EngineHealthTarget]: ...


@dataclass(frozen=True)
class HealthMonitorConfig:
    """Probe timings in seconds, independent of an inference framework."""

    interval: float
    timeout: float
    first_wait: float = 0

    def __post_init__(self) -> None:
        for name, value in (("interval", self.interval), ("timeout", self.timeout), ("first_wait", self.first_wait)):
            if not math.isfinite(value) or value < 0 or (name != "first_wait" and value == 0):
                raise ValueError(
                    f"health monitor {name} must be finite and {'non-negative' if name == 'first_wait' else 'positive'}"
                )


class EngineHealthMonitor:
    """Schedule probes; pause/stop drain any active probe or retirement.

    The owner serializes lifecycle calls. Engine replacement and offload must
    wait for a successful pause. A timeout leaves scheduling disabled and must
    prevent replacement; it never claims the in-flight operation was cancelled.
    Backend targets must bound their check/retire operations. No model framework
    or executor is imported here.
    """

    def __init__(self, checks: EngineHealthChecks, config: HealthMonitorConfig) -> None:
        self._checks = checks
        self._config = config
        self._condition = Condition()
        self._thread: Thread | None = None
        self._enabled = False
        self._stopped = False
        self._active = False
        self._epoch = 0
        self._next_check = 0.0
        self._failure: BaseException | None = None

    def start(self) -> bool:
        """Start once, initially paused; resume explicitly after engine readiness."""
        with self._condition:
            self._raise_failure()
            if self._stopped:
                raise RuntimeError("health monitor is stopped")
            if self._thread is not None:
                return False
            self._thread = Thread(target=self._run, name="reef-engine-health", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                raise
            return True

    def resume(self) -> None:
        with self._condition:
            self._raise_failure()
            if self._thread is None or self._stopped:
                raise RuntimeError("health monitor is not running")
            self._epoch += 1
            self._enabled = True
            self._next_check = monotonic() + self._config.first_wait
            self._condition.notify_all()

    def _drain_deadline(self, timeout: float | None) -> float:
        seconds = 2 * self._config.timeout + 1 if timeout is None else timeout
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("health monitor drain timeout must be finite and non-negative")
        return monotonic() + seconds

    def _disable_and_drain(self, deadline: float) -> None:
        # Called with the condition held; wait releases it so probe completion
        # can observe the epoch change and discard its now-stale failure.
        self._enabled = False
        self._epoch += 1
        self._condition.notify_all()
        while self._active:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("health monitor still has an in-flight probe or retirement")
            self._condition.wait(remaining)

    def pause(self, timeout: float | None = None) -> None:
        deadline = self._drain_deadline(timeout)
        with self._condition:
            self._disable_and_drain(deadline)

    def stop(self, timeout: float | None = None) -> None:
        """Drain and join; retain a live thread handle if shutdown times out."""
        deadline = self._drain_deadline(timeout)
        with self._condition:
            self._stopped = True
            self._disable_and_drain(deadline)
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - monotonic()))
            if thread.is_alive():
                raise TimeoutError("health monitor thread has not stopped")
            with self._condition:
                self._thread = None

    def is_checking_enabled(self) -> bool:
        with self._condition:
            return self._enabled and not self._stopped and self._failure is None

    def check_health(self) -> None:
        with self._condition:
            self._raise_failure()
            if self._thread is None or self._stopped:
                raise RuntimeError("health monitor is not running")

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("engine health monitor failed") from self._failure

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stopped:
                        delay = self._next_check - monotonic()
                        if self._enabled and delay <= 0:
                            break
                        self._condition.wait(max(0, delay) if self._enabled else None)
                    if self._stopped:
                        return
                    epoch = self._epoch
                    self._active = True
                try:
                    self._check_targets(epoch)
                except BaseException as exc:
                    # Make failure visible before releasing the drain barrier.
                    with self._condition:
                        self._failure = exc
                        self._enabled = False
                    raise
                finally:
                    with self._condition:
                        self._active = False
                        if self._epoch == epoch:
                            self._next_check = monotonic() + self._config.interval
                        self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self._enabled = False
                self._condition.notify_all()
            logger.exception("Engine health monitor stopped after an internal failure")

    def _check_targets(self, epoch: int) -> None:
        for target in self._checks.targets():
            with self._condition:
                if not self._enabled or self._stopped or self._epoch != epoch:
                    return
            try:
                target.check(self._config.timeout)
            except Exception:
                with self._condition:
                    if not self._enabled or self._stopped or self._epoch != epoch:
                        return
                # _active remains true through retirement. A concurrent pause
                # cannot return until all mutations to this snapshot finish.
                logger.warning("Inference engine health probe failed; retiring the captured engine")
                target.retire(self._config.timeout)
