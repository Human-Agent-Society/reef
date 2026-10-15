"""Usage events, the reporter plugin, and the reporting queue of ``observability.usage``."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from reef.core.plugins import PluginConfigError
from reef.observability import (
    JsonLogUsageReporter,
    ModelUsage,
    UsageConfig,
    UsageEvent,
    UsageReporter,
    UsageReporting,
    build_usage_reporting,
)

pytestmark = pytest.mark.unit


class CapturingReporter(UsageReporter):
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []
        self.closed = False

    def report_usage(self, event: UsageEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


class FailingReporter(UsageReporter):
    def report_usage(self, event: UsageEvent) -> None:
        raise ConnectionError("billing intake is down")


class BlockedReporter(UsageReporter):
    """Holds the worker in its first call until the test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.events: list[UsageEvent] = []

    def report_usage(self, event: UsageEvent) -> None:
        self.entered.set()
        self.release.wait(5)
        self.events.append(event)


def sse(*events: dict[str, object]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


def event(**fields: object) -> UsageEvent:
    return UsageEvent(**{"scenario": "alice-coding", "source": "inference", "outcome": "succeeded", **fields})


def test_model_usage_reads_json_responses_of_every_dialect_and_keeps_provider_cost() -> None:
    chat = ModelUsage.from_response(
        {
            "id": "gen-1",
            "model": "example-model",
            "usage": {"prompt_tokens": 18234, "completion_tokens": 912, "cost": 0.0071, "currency": "USD"},
        }
    )
    messages = ModelUsage.from_response({"id": "msg_1", "usage": {"input_tokens": 20, "output_tokens": 5}})

    assert chat == ModelUsage(18234, 912, 0.0071, "USD", "gen-1", "example-model")
    assert messages == ModelUsage(input_tokens=20, output_tokens=5, provider_request_id="msg_1")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            sse(
                {"id": "chatcmpl-1", "model": "m", "choices": [{"delta": {"content": "hi"}}]},
                {
                    "id": "chatcmpl-1",
                    "choices": [],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "cost": 0.5},
                },
            ),
            ModelUsage(7, 2, 0.5, None, "chatcmpl-1", "m"),
        ),
        (
            sse(
                {"type": "message_start", "message": {"id": "msg_1", "model": "claude", "usage": {"input_tokens": 9}}},
                {"type": "message_delta", "usage": {"output_tokens": 4}},
            ),
            ModelUsage(9, 4, None, None, "msg_1", "claude"),
        ),
        (
            sse(
                {"type": "response.created", "response": {"id": "resp_1", "model": "gpt", "usage": None}},
                {"type": "response.output_text.delta", "delta": "hi"},
                {
                    "type": "response.completed",
                    "response": {"id": "resp_1", "usage": {"input_tokens": 3, "output_tokens": 1}},
                },
            ),
            ModelUsage(3, 1, None, None, "resp_1", "gpt"),
        ),
    ],
)
def test_model_usage_reads_cumulative_stream_usage_without_summing_chunks(body: str, expected: ModelUsage) -> None:
    assert ModelUsage.from_sse(body) == expected
    assert ModelUsage.from_response({"stream": True, "body": body, "complete": True}) == expected
    assert ModelUsage.from_text(body) == expected


def test_missing_usage_stays_unknown_instead_of_zero() -> None:
    usage = ModelUsage.from_response({"choices": [{"message": {"content": "hi"}}], "usage": {"cost": -1}})
    partial = ModelUsage.from_sse('data: {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}\n\n')

    assert usage == ModelUsage() and ModelUsage.from_response(None) == ModelUsage()
    # A stream cut before its output tokens keeps the input count it already reported.
    assert partial == ModelUsage(input_tokens=5)
    assert event(input_tokens=None).to_dict().keys() == {
        "event_id",
        "scenario",
        "source",
        "outcome",
        "operation_id",
        "finished_at",
    }


def test_an_event_keeps_its_id_and_each_attempt_gets_a_new_operation_id() -> None:
    first = UsageEvent.for_model_call(
        ModelUsage(10, 2), scenario="alice-coding", source="inference", outcome="cancelled", elapsed_seconds=1.0
    )
    retry = UsageEvent.for_model_call(
        ModelUsage(10, 4), scenario="alice-coding", source="inference", outcome="succeeded", elapsed_seconds=1.5
    )
    reporter = CapturingReporter()
    reporting = UsageReporting(reporter)
    reporting.report(first)
    reporting.report(first)
    reporting.report(retry)
    reporting.close()

    assert [item.event_id for item in reporter.events] == [first.event_id, first.event_id, retry.event_id]
    assert first.event_id != retry.event_id and first.operation_id != retry.operation_id
    assert reporter.closed and reporting.counters()["usage_reporting/reported_total"] == 3


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"source": "billing"}, "usage source"),
        ({"outcome": "partial"}, "usage outcome"),
        ({"input_tokens": -1}, "input_tokens"),
        ({"elapsed_seconds": float("nan")}, "elapsed_seconds"),
        ({"currency": "USD"}, "requires a provider_cost"),
        ({"scenario": ""}, "requires a scenario"),
    ],
)
def test_events_refuse_invalid_measurements(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        event(**fields)


def test_a_failing_reporter_is_counted_and_reporting_continues(caplog: pytest.LogCaptureFixture) -> None:
    reporting = UsageReporting(FailingReporter())
    reporting.report(event())
    reporting.report(event())
    reporting.close()

    counters = reporting.counters()
    assert counters["usage_reporting/failed_total"] == 2 and counters["usage_reporting/reported_total"] == 0
    # The log line repeats at most once per interval; the counter counts every failure.
    assert sum("usage reporter FailingReporter failed" in record.getMessage() for record in caplog.records) == 1


def test_a_full_queue_drops_new_events_without_blocking_the_caller() -> None:
    reporter = BlockedReporter()
    reporting = UsageReporting(reporter, max_queued_events=1)
    held, queued, dropped = event(), event(), event()
    reporting.report(held)
    assert reporter.entered.wait(5)
    reporting.report(queued)
    reporting.report(dropped)
    assert reporting.counters()["usage_reporting/dropped_total"] == 1
    reporter.release.set()
    reporting.close()

    assert [item.event_id for item in reporter.events] == [held.event_id, queued.event_id]
    assert reporting.counters() == {
        "usage_reporting/reported_total": 2,
        "usage_reporting/failed_total": 0,
        "usage_reporting/dropped_total": 1,
        "usage_reporting/queued": 0,
    }


def test_reporting_without_a_reporter_is_off() -> None:
    reporting = build_usage_reporting(None)
    reporting.report(event())
    reporting.close()

    assert not reporting.enabled and reporting.worker is None
    assert reporting.counters()["usage_reporting/dropped_total"] == 0


def test_json_log_reporter_writes_one_json_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    reporting = build_usage_reporting(
        {"reporter": {"module": "reef.observability.usage:JsonLogUsageReporter", "config": {"path": str(path)}}}
    )
    reported = event(step=12, model="example-model", input_tokens=18234, provider_cost=0.0071, currency="USD")
    reporting.report(reported)
    reporting.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [reported.to_dict()]
    assert isinstance(reporting.reporter, JsonLogUsageReporter)


def test_usage_config_names_a_reporter_plugin_and_bounds_the_queue() -> None:
    assert UsageConfig.from_mapping(None) == UsageConfig()
    assert UsageConfig.from_mapping({"max-queued-events": 5}).max_queued_events == 5
    with pytest.raises(ValueError, match="max_queued_events must be a positive integer"):
        UsageConfig.from_mapping({"max_queued_events": 0})
    with pytest.raises(PluginConfigError, match="must name a class that inherits UsageReporter"):
        UsageConfig.from_mapping({"reporter": "reef.observability.usage:UsageEvent"})
    with pytest.raises(ValueError, match="unknown config fields: endpoint"):
        UsageConfig.from_mapping({"endpoint": "https://billing.example"})
    with pytest.raises(PluginConfigError, match="JsonLogUsageReporter"):
        build_usage_reporting(
            {"reporter": {"module": "reef.observability.usage:JsonLogUsageReporter", "config": {"level": "debug"}}}
        )
