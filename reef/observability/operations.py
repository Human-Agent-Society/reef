"""Small, process-local measurements that remain readable during slow operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock


@dataclass(eq=False)
class OperationMeasurement:
    """One in-flight call, including calls whose lifetime spans streaming APIs."""

    metrics: OperationMetrics
    operation: str
    started_monotonic_seconds: float

    def finish(self, *, succeeded: bool) -> None:
        """Record an outcome once; repeated cleanup does not count twice."""
        with self.metrics.lock:
            if self not in self.metrics.active_operations:
                return
            self.metrics.active_operations.remove(self)
            duration_seconds = time.monotonic() - self.started_monotonic_seconds
            values = self.metrics.metrics
            values[f"{self.operation}/last_duration_seconds"] = duration_seconds
            outcome_metric = f"{self.operation}/completed_total" if succeeded else f"{self.operation}/failed_total"
            values[outcome_metric] += 1
            values[f"{self.operation}/duration_seconds_total"] += duration_seconds


class OperationMetrics:
    """Measure concurrent operations without holding a lock while they execute.

    Counters belong to this instance and reset when its owner is rebuilt.
    Sampling performs no provider calls and never consumes recorded values.
    """

    def __init__(self, operations: tuple[str, ...], *, counters: tuple[str, ...] = ()) -> None:
        self.lock = Lock()
        self.started_at_seconds = time.time()
        self.metrics: dict[str, float | int] = dict.fromkeys(counters, 0)
        for operation in operations:
            for quantity in (
                "active",
                "elapsed_seconds",
                "started_total",
                "completed_total",
                "failed_total",
                "duration_seconds_total",
            ):
                self.metrics[f"{operation}/{quantity}"] = 0
        self.active_operations: set[OperationMeasurement] = set()

    def start(self, operation: str) -> OperationMeasurement:
        measurement = OperationMeasurement(self, operation, time.monotonic())
        with self.lock:
            self.metrics[f"{operation}/started_total"] += 1
            self.active_operations.add(measurement)
        return measurement

    def increment(self, counter: str) -> None:
        """Count an event using a fixed metric name chosen by its caller."""
        with self.lock:
            self.metrics[counter] = self.metrics.get(counter, 0) + 1

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        measurement = self.start(operation)
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            measurement.finish(succeeded=succeeded)

    def snapshot(self) -> dict[str, float | int]:
        with self.lock:
            metrics = {"started_at_seconds": self.started_at_seconds, **self.metrics}
            now_monotonic_seconds = time.monotonic()
            for measurement in self.active_operations:
                operation = measurement.operation
                metrics[f"{operation}/active"] += 1
                metrics[f"{operation}/elapsed_seconds"] = max(
                    metrics[f"{operation}/elapsed_seconds"],
                    now_monotonic_seconds - measurement.started_monotonic_seconds,
                )
            return metrics
