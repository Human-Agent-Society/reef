"""Usage reporting: measured resource usage of each scenario operation, sent to an optional reporter.

Execution components measure the work they do, such as one model call, and
send one :class:`UsageEvent` per operation to :class:`UsageReporting`. The
reporting queue hands each event to the configured :class:`UsageReporter` on
a worker thread, so a slow or failing reporter never delays or fails
execution. A full queue or a reporter failure is logged and counted, and the
event is lost. Reporting is off unless ``observability.usage.reporter`` names
a reporter.

Events hold identifiers and measurements only: never prompts, completions or
credentials. A measurement that the operation did not report is omitted, never
reported as zero. Deployment systems calculate prices and totals from the
events; the events alone are not a billing record.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from threading import Event, Lock, Thread
from typing import Any, Self, TypeGuard

from reef.core.config import Settings, config_option
from reef.core.plugins import ConfiguredPlugin, PluginSettings

logger = logging.getLogger(__name__)

#: The kinds of work an event can measure.
USAGE_SOURCES = ("inference", "proposer", "evaluation", "judge", "generator", "sandbox", "training")
#: How an operation ended: ``cancelled`` when it was stopped before it finished.
USAGE_OUTCOMES = ("succeeded", "failed", "cancelled")


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """What a provider reported about one model call; a field is ``None`` when the provider did not report it.

    ``provider_cost`` and ``currency`` are kept as the provider returns them, for
    example OpenRouter's ``usage.cost``. A provider that returns no currency
    leaves ``currency`` empty; Reef does not guess one.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_cost: float | None = None
    currency: str | None = None
    provider_request_id: str | None = None
    model: str | None = None

    @classmethod
    def from_response(cls, response: object) -> ModelUsage:
        """Read a provider response object, or a recorded stream response (``stream`` true and a text ``body``).

        Chat Completions counts ``prompt_tokens`` and ``completion_tokens``;
        Messages and Responses count ``input_tokens`` and ``output_tokens``.
        """
        if not isinstance(response, Mapping):
            return cls()
        body = response.get("body")
        if not isinstance(response.get("usage"), Mapping) and response.get("stream") is True and isinstance(body, str):
            return cls.from_text(body)
        return cls().merged(response)

    @classmethod
    def from_sse(cls, body: str) -> ModelUsage:
        """Read the usage that a provider SSE response reported, including a partial stream.

        Streaming providers report cumulative totals, so a later count replaces
        an earlier one; counts are never summed across chunks.
        """
        usage = cls()
        for frame in body.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
            data = "\n".join(line[5:].removeprefix(" ") for line in frame.split("\n") if line.startswith("data:"))
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            if event.get("type") == "message_start":
                message = event.get("message")
            elif str(event.get("type", "")).startswith("response."):
                message = event.get("response")
            else:
                message = event
            if isinstance(message, Mapping):
                usage = usage.merged(message)
        return usage

    @classmethod
    def from_text(cls, text: str) -> ModelUsage:
        """Read a provider reply body that is either one JSON object or an SSE stream."""
        try:
            response = json.loads(text)
        except json.JSONDecodeError:
            return cls.from_sse(text)
        return cls.from_response(response)

    def merged(self, message: Mapping[str, object]) -> ModelUsage:
        """This usage updated with the values one response object or stream event reports."""
        usage = message.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        input_tokens = token_count(usage, "input_tokens", "prompt_tokens")
        output_tokens = token_count(usage, "output_tokens", "completion_tokens")
        cost = usage.get("cost")
        provider_cost = float(cost) if is_measurement(cost) else None
        currency = usage.get("currency")
        request_id = message.get("id")
        model = message.get("model")
        return ModelUsage(
            input_tokens=self.input_tokens if input_tokens is None else input_tokens,
            output_tokens=self.output_tokens if output_tokens is None else output_tokens,
            provider_cost=self.provider_cost if provider_cost is None else provider_cost,
            currency=(
                currency if provider_cost is not None and isinstance(currency, str) and currency else self.currency
            ),
            provider_request_id=request_id if isinstance(request_id, str) and request_id else self.provider_request_id,
            model=model if isinstance(model, str) and model else self.model,
        )


def token_count(usage: Mapping[str, object], *keys: str) -> int | None:
    """The first non-negative integer count among ``keys``, or ``None``."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def is_measurement(value: object) -> TypeGuard[int | float]:
    """True for a finite, non-negative number that is not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """The measured usage of one resource operation in one scenario.

    ``event_id`` identifies this report: a reporter that sends the event again
    keeps it, so a receiver can drop the duplicate. ``operation_id``
    identifies the operation; every new attempt of a retried operation gets a
    new one. ``step`` is the scenario's committed step when the operation
    started. ``elapsed_seconds`` is the wall-clock time of the operation,
    which does not have to match a provider's billing unit. Unknown
    measurements stay ``None`` and are left out of :meth:`to_dict`.
    """

    scenario: str
    source: str
    outcome: str
    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_id: str = field(default_factory=lambda: f"usage-{uuid.uuid4().hex}")
    finished_at: float = field(default_factory=time.time)
    step: int | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    provider_request_id: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    provider_cost: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        if not self.scenario:
            raise ValueError("a usage event requires a scenario")
        if self.source not in USAGE_SOURCES:
            raise ValueError(f"usage source must be one of {', '.join(USAGE_SOURCES)}, got {self.source!r}")
        if self.outcome not in USAGE_OUTCOMES:
            raise ValueError(f"usage outcome must be one of {', '.join(USAGE_OUTCOMES)}, got {self.outcome!r}")
        for name, count in (
            ("step", self.step),
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
                raise ValueError(f"usage {name} must be a non-negative integer")
        for name, amount in (("elapsed_seconds", self.elapsed_seconds), ("provider_cost", self.provider_cost)):
            if amount is not None and not is_measurement(amount):
                raise ValueError(f"usage {name} must be a finite, non-negative number")
        if self.currency is not None and self.provider_cost is None:
            raise ValueError("a usage currency requires a provider_cost")

    @classmethod
    def for_model_call(
        cls,
        usage: ModelUsage,
        *,
        scenario: str,
        source: str,
        outcome: str,
        elapsed_seconds: float,
        model: str | None = None,
        step: int | None = None,
    ) -> UsageEvent:
        """One model call's event: what the provider reported, the elapsed time, and the model it answered with.

        ``model`` is the requested model, used when the provider names none.
        """
        return cls(
            scenario=scenario,
            source=source,
            outcome=outcome,
            step=step,
            provider_request_id=usage.provider_request_id,
            model=usage.model or model or None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            elapsed_seconds=max(0.0, elapsed_seconds),
            provider_cost=usage.provider_cost,
            currency=usage.currency,
        )

    def to_dict(self) -> dict[str, Any]:
        """The event as a JSON object, without the measurements that are unknown."""
        return {name: value for name, value in asdict(self).items() if value is not None}


class UsageReporter(ConfiguredPlugin, ABC):
    """Where usage events go: a log, a metrics system, or a deployment's billing intake.

    ``observability.usage.reporter`` names the class as
    ``package.module:ClassName``, or as a mapping with ``module`` and
    ``config``. Reef calls :meth:`report_usage` from one worker thread, never
    from the thread that did the work. A reporter that raises loses that event;
    Reef logs and counts the failure and continues.
    """

    @abstractmethod
    def report_usage(self, event: UsageEvent) -> None:
        """Send one event. A retry of the same event keeps its ``event_id``."""

    def close(self) -> None:
        """Release what the reporter holds; called once, after the last event."""


@dataclass(frozen=True)
class JsonLogSettings(PluginSettings):
    """The config of :class:`JsonLogUsageReporter`."""

    path: str | None = config_option(None, help="File that receives one JSON line per event; stderr when unset.")


class JsonLogUsageReporter(UsageReporter):
    """Write each event as one JSON object per line on the ``reef.usage`` logger.

    The reporter owns the logger's handler: the lines go to ``path``, appended,
    or to stderr. They do not pass to the service's other log handlers.
    """

    def __init__(self, path: str | None = None) -> None:
        self.handler: logging.Handler = (
            logging.StreamHandler(sys.stderr) if path is None else logging.FileHandler(path, encoding="utf-8")
        )
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.usage_logger = logging.getLogger("reef.usage")
        self.usage_logger.setLevel(logging.INFO)
        self.usage_logger.propagate = False
        self.usage_logger.addHandler(self.handler)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        settings = JsonLogSettings.parse(config, plugin_class=cls)
        return cls(path=settings.path)

    def report_usage(self, event: UsageEvent) -> None:
        self.usage_logger.info("%s", json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")))

    def close(self) -> None:
        self.usage_logger.removeHandler(self.handler)
        self.handler.close()


@dataclass(frozen=True)
class UsageConfig(Settings):
    """Validated ``observability.usage`` configuration; reporting is off while ``reporter`` is unset."""

    reporter: str | Mapping[str, Any] | None = config_option(
        None, help="The UsageReporter plugin: 'package.module:ClassName' or module and config."
    )
    max_queued_events: int = config_option(
        1000, help="Most events waiting for the reporter; a new event is dropped while the queue is full."
    )

    def __post_init__(self) -> None:
        if self.max_queued_events <= 0:
            raise ValueError("observability.usage.max_queued_events must be a positive integer")
        if self.reporter is not None:
            UsageReporter.reference(self.reporter, setting="observability.usage.reporter")

    @classmethod
    def from_mapping(cls, value: object) -> UsageConfig:
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError("observability.usage must be a mapping")
        return cls.from_config(value, setting="observability.usage")


class UsageReporting:
    """The process's usage reporting: a bounded queue and one worker thread that calls the reporter.

    Built without a reporter it is off: :meth:`report` returns at once and no
    thread starts. Execution components call :meth:`report` and never wait for
    the reporter. :meth:`counters` reports how many events the reporter took,
    how many it failed on, and how many a full queue dropped.
    """

    #: The shortest time between two log lines about the same kind of lost event.
    WARNING_INTERVAL_SECONDS = 60.0

    def __init__(self, reporter: UsageReporter | None = None, *, max_queued_events: int = 1000) -> None:
        if max_queued_events <= 0:
            raise ValueError("max_queued_events must be a positive integer")
        self.reporter = reporter
        self.max_queued_events = max_queued_events
        self.events: queue.Queue[UsageEvent | None] = queue.Queue(maxsize=max_queued_events)
        self.lock = Lock()
        self.counts = {"reported_total": 0, "failed_total": 0, "dropped_total": 0}
        self.warned_at_monotonic_seconds: dict[str, float] = {}
        self.closed = Event()
        self.worker: Thread | None = None
        if reporter is not None:
            self.worker = Thread(target=self.run, args=(reporter,), name="reef-usage-reporting", daemon=True)
            self.worker.start()

    @classmethod
    def from_config(cls, config: UsageConfig) -> UsageReporting:
        if config.reporter is None:
            return cls(max_queued_events=config.max_queued_events)
        reporter = UsageReporter.from_setting(config.reporter, setting="observability.usage.reporter")
        return cls(reporter, max_queued_events=config.max_queued_events)

    @property
    def enabled(self) -> bool:
        return self.reporter is not None

    def report(self, event: UsageEvent) -> None:
        """Queue ``event`` for the reporter; a full or closed queue drops it and counts the loss."""
        if self.reporter is None:
            return
        if not self.closed.is_set():
            try:
                self.events.put_nowait(event)
                return
            except queue.Full:
                pass
        self.count("dropped_total")
        self.warn(
            "dropped_total",
            "usage reporting dropped event %s for scenario %r: the queue of %d events is full or closed",
            event.event_id,
            event.scenario,
            self.max_queued_events,
        )

    def run(self, reporter: UsageReporter) -> None:
        while True:
            event = self.events.get()
            if event is None:
                return
            try:
                reporter.report_usage(event)
            except Exception:
                self.count("failed_total")
                self.warn(
                    "failed_total",
                    "usage reporter %s failed on event %s",
                    type(reporter).__name__,
                    event.event_id,
                    exc_info=True,
                )
            else:
                self.count("reported_total")

    def count(self, counter: str) -> None:
        with self.lock:
            self.counts[counter] += 1

    def warn(self, kind: str, message: str, *args: object, exc_info: bool = False) -> None:
        """Log a lost event, at most once per interval for each kind; the counters still count every loss."""
        now = time.monotonic()
        with self.lock:
            last = self.warned_at_monotonic_seconds.get(kind)
            if last is not None and now - last < self.WARNING_INTERVAL_SECONDS:
                return
            self.warned_at_monotonic_seconds[kind] = now
        logger.warning(message, *args, exc_info=exc_info)

    def counters(self) -> dict[str, int]:
        """Process totals since start, as ``usage_reporting/*`` operational metrics."""
        with self.lock:
            counters = {f"usage_reporting/{name}": value for name, value in self.counts.items()}
        counters["usage_reporting/queued"] = self.events.qsize()
        return counters

    def close(self, timeout_seconds: float = 10.0) -> None:
        """Stop taking events, give the queued ones to the reporter, then close it."""
        if self.closed.is_set():
            return
        self.closed.set()
        if self.worker is None or self.reporter is None:
            return
        try:
            self.events.put(None, timeout=timeout_seconds)
        except queue.Full:
            logger.warning("usage reporting queue did not drain in %g s; queued events are lost", timeout_seconds)
            return
        self.worker.join(timeout_seconds)
        if self.worker.is_alive():
            logger.warning("usage reporter %s did not finish in %g s", type(self.reporter).__name__, timeout_seconds)
            return
        try:
            self.reporter.close()
        except Exception:
            logger.exception("usage reporter %s failed to close", type(self.reporter).__name__)


__all__ = [
    "USAGE_OUTCOMES",
    "USAGE_SOURCES",
    "JsonLogUsageReporter",
    "ModelUsage",
    "UsageConfig",
    "UsageEvent",
    "UsageReporter",
    "UsageReporting",
]
