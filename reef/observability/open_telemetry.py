"""OpenTelemetry exporter for Reef's record tracing contract.

Importing this module imports the OpenTelemetry SDK. The factory loads it only
when ``observability.tracing.enabled`` is true, so the base installation does
not need the ``opentelemetry`` extra.

Every accepted inference record becomes the root span of its own trace, whose
ids derive from the scenario and record id. A feedback record joins the trace
of the first inference it references and links the others; a committed
training step adds one child span below every record it consumed, so a trace
viewer shows which version an exchange trained. The span attributes follow the
OpenTelemetry GenAI semantic conventions (``gen_ai.*``); Reef's own identifiers
use the ``reef.*`` prefix and the scenario also travels as ``session.id``.
Spans carry the exchange text itself, so the backend sees the scenario's traffic.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from threading import Lock
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator
from opentelemetry.trace import Link, NonRecordingSpan, SpanContext, SpanKind, TraceFlags, set_span_in_context
from opentelemetry.util.types import AttributeValue

from reef.core.artifact_ref import ArtifactRef, LiveWeightArtifactRef
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import exchange_messages
from reef.core.version import __version__
from reef.observability.tracing import TracingConfig
from reef.storage.commits import CommitRecord
from reef.storage.observer import RecordObserver

#: Instrumentation scope name reported with every span.
INSTRUMENTATION_SCOPE = "reef"


def record_span_context(scenario: str, agent_record_id: str) -> SpanContext:
    """The trace and span ids a record's span carries; stable across restarts and hosts."""
    return _span_context("record", scenario, agent_record_id)


def commit_span_context(scenario: str, step: int) -> SpanContext:
    """The trace and span ids a committed step's span carries."""
    return _span_context("commit", scenario, str(step))


def _span_context(*identity: str) -> SpanContext:
    digest = hashlib.sha256("\x00".join(identity).encode("utf-8")).digest()
    # The SDK treats zero ids as invalid; a digest prefix of all zeros is astronomically unlikely, but stay valid.
    trace_id = int.from_bytes(digest[:16], "big") or 1
    span_id = int.from_bytes(digest[16:24], "big") or 1
    return SpanContext(trace_id, span_id, is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED))


class PlannedIdGenerator(IdGenerator):
    """Hands the SDK the ids planned for the next span; random ids otherwise.

    The observer sets ``planned`` under its lock right before starting a span,
    so a record's span keeps the same ids on every host and after a restart.
    """

    def __init__(self) -> None:
        self.planned: SpanContext | None = None
        self._random = RandomIdGenerator()

    def generate_trace_id(self) -> int:
        if self.planned is None:
            return self._random.generate_trace_id()
        return self.planned.trace_id

    def generate_span_id(self) -> int:
        if self.planned is None:
            return self._random.generate_span_id()
        return self.planned.span_id


class OpenTelemetryRecordObserver(RecordObserver):
    """Export accepted records and committed steps as OpenTelemetry spans over OTLP/HTTP."""

    def __init__(self, config: TracingConfig, *, span_exporter: SpanExporter | None = None) -> None:
        self.config = config
        if span_exporter is None:
            span_exporter = OTLPSpanExporter(endpoint=config.endpoint, headers=config.request_headers())
        self._ids = PlannedIdGenerator()
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": config.service_name, "service.version": __version__}),
            id_generator=self._ids,
            shutdown_on_exit=False,
        )
        self._provider.add_span_processor(BatchSpanProcessor(span_exporter))
        self._tracer = self._provider.get_tracer(INSTRUMENTATION_SCOPE, __version__)
        self._lock = Lock()

    def record_accepted(self, item: AgentRecord) -> None:
        attributes: dict[str, AttributeValue] = {
            "session.id": item.scenario,
            "reef.scenario": item.scenario,
            "reef.agent_record_id": item.agent_record_id,
            "reef.request_type": item.request_type.value,
        }
        if item.artifact_ref is not None:
            attributes.update(_artifact_attributes(item.artifact_ref))
        tags = _tags(item.payload)
        if tags:
            attributes["reef.tags"] = tags
        parent: SpanContext | None = None
        links: list[Link] = []
        kind = SpanKind.INTERNAL
        if item.request_type is RequestType.INFERENCE:
            kind = SpanKind.CLIENT
            attributes.update(self.inference_attributes(item.payload))
            model = attributes.get("gen_ai.request.model")
            name = f"chat {model}" if isinstance(model, str) else "chat"
        elif item.request_type is RequestType.REPORT:
            name = "feedback"
            attributes.update(self._report_attributes(item.payload))
            if item.references:
                attributes["reef.references"] = list(item.references)
                parent = record_span_context(item.scenario, item.references[0])
                links = [
                    Link(record_span_context(item.scenario, reference), {"reef.agent_record_id": reference})
                    for reference in item.references[1:]
                ]
        else:
            name = "training request"
            attributes.update(self._training_request_attributes(item.payload))
        self._emit(
            name,
            planned=record_span_context(item.scenario, item.agent_record_id),
            parent=parent,
            kind=kind,
            attributes=attributes,
            links=links,
            timestamp_seconds=item.created_at,
        )

    def record_committed(self, commit: CommitRecord) -> None:
        attributes: dict[str, AttributeValue] = {
            "session.id": commit.scenario,
            "reef.scenario": commit.scenario,
            "reef.step": commit.step,
            "reef.commit_operation": commit.operation,
            "reef.checkpoint": commit.checkpoint,
            "reef.pending": commit.pending,
            "reef.consumed_record_count": len(commit.consumed_ids),
            "reef.compacted_record_count": len(commit.compacted_ids),
            **_artifact_attributes(commit.artifact_ref),
        }
        if commit.training_job_id is not None:
            attributes["reef.training_job_id"] = commit.training_job_id
        if commit.component is not None:
            attributes["reef.component"] = commit.component
        if commit.rollback_target_release_id is not None:
            attributes["reef.rollback_target_release_id"] = commit.rollback_target_release_id
        for key, value in (commit.metrics or {}).items():
            metric = _attribute_value(value)
            if metric is not None:
                attributes[f"reef.metrics.{key}"] = metric
        commit_context = commit_span_context(commit.scenario, commit.step)
        self._emit(
            f"{commit.operation} step {commit.step}",
            planned=commit_context,
            parent=None,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
            links=[],
            timestamp_seconds=commit.recorded_at,
        )
        consumed_attributes: dict[str, AttributeValue] = {
            "session.id": commit.scenario,
            "reef.scenario": commit.scenario,
            "reef.step": commit.step,
            **_artifact_attributes(commit.artifact_ref),
        }
        if commit.training_job_id is not None:
            consumed_attributes["reef.training_job_id"] = commit.training_job_id
        if commit.component is not None:
            consumed_attributes["reef.component"] = commit.component
        for agent_record_id in sorted(commit.consumed_ids):
            self._emit(
                f"trained in step {commit.step}",
                planned=_span_context("consumed", commit.scenario, str(commit.step), agent_record_id),
                parent=record_span_context(commit.scenario, agent_record_id),
                kind=SpanKind.INTERNAL,
                attributes={**consumed_attributes, "reef.agent_record_id": agent_record_id},
                links=[Link(commit_context)],
                timestamp_seconds=commit.recorded_at,
            )

    def close(self) -> None:
        self._provider.shutdown()

    def _emit(
        self,
        name: str,
        *,
        planned: SpanContext,
        parent: SpanContext | None,
        kind: SpanKind,
        attributes: Mapping[str, AttributeValue],
        links: Sequence[Link],
        timestamp_seconds: float,
    ) -> None:
        """Start and end one span at ``timestamp_seconds`` with the planned ids.

        Records carry one timestamp, so their spans have zero duration; the
        time is the moment the record was created, not when it was exported.
        """
        context = None if parent is None else set_span_in_context(NonRecordingSpan(parent))
        timestamp_nanoseconds = int(timestamp_seconds * 1_000_000_000)
        with self._lock:
            self._ids.planned = planned
            try:
                span = self._tracer.start_span(
                    name,
                    context=context,
                    kind=kind,
                    attributes=attributes,
                    links=links,
                    start_time=timestamp_nanoseconds,
                )
            finally:
                self._ids.planned = None
        span.end(end_time=timestamp_nanoseconds)

    def inference_attributes(self, payload: Mapping[str, object]) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {"gen_ai.operation.name": "chat"}
        model = payload.get("model")
        if isinstance(model, str) and model:
            attributes["gen_ai.request.model"] = model
        runtime_load_id = payload.get("runtime_load_id")
        if isinstance(runtime_load_id, str) and runtime_load_id:
            attributes["reef.runtime_load_id"] = runtime_load_id
        response = payload.get("response")
        response = response if isinstance(response, Mapping) else {}
        response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            attributes["gen_ai.response.model"] = response_model
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            attributes["gen_ai.response.id"] = response_id
        usage = response.get("usage")
        body = response.get("body")
        if not isinstance(usage, Mapping) and response.get("stream") is True and isinstance(body, str):
            usage = stream_token_usage(body)
        if isinstance(usage, Mapping):
            input_tokens = token_count(usage, "input_tokens", "prompt_tokens")
            if input_tokens is not None:
                attributes["gen_ai.usage.input_tokens"] = input_tokens
            output_tokens = token_count(usage, "output_tokens", "completion_tokens")
            if output_tokens is not None:
                attributes["gen_ai.usage.output_tokens"] = output_tokens
        finish_reasons = _finish_reasons(response)
        if finish_reasons:
            attributes["gen_ai.response.finish_reasons"] = finish_reasons
        request_messages, response_messages = exchange_messages(payload)
        attributes["gen_ai.input.messages"] = _json_text(request_messages)
        attributes["gen_ai.output.messages"] = _json_text(response_messages)
        return attributes

    def _report_attributes(self, payload: Mapping[str, Any]) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {}
        score = _attribute_value(payload.get("score"))
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            attributes["reef.score"] = score
        feedback = payload.get("feedback")
        if feedback is not None:
            attributes["reef.feedback"] = feedback if isinstance(feedback, str) else _json_text(feedback)
        return attributes

    def _training_request_attributes(self, payload: Mapping[str, Any]) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {}
        for key in ("session", "release_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                attributes[f"reef.{key}"] = value
        text = payload.get("text")
        if isinstance(text, str):
            attributes["reef.instruction"] = text
        return attributes


def _artifact_attributes(ref: ArtifactRef) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {"reef.release_id": ref.release_id, "reef.content_id": ref.content_id}
    if isinstance(ref, LiveWeightArtifactRef):
        attributes["reef.runtime_load_id"] = ref.runtime_load_id
    return attributes


def _tags(payload: Mapping[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    if isinstance(tags, str) or not isinstance(tags, Sequence):
        return []
    return [tag for tag in tags if isinstance(tag, str)]


def stream_token_usage(body: str) -> dict[str, int]:
    """Read cumulative token counts from a recorded provider SSE response."""
    counts: dict[str, int] = {}
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
            response = event.get("message")
        elif event.get("type") in ("response.completed", "response.incomplete"):
            response = event.get("response")
        else:
            response = event
        if not isinstance(response, Mapping):
            continue
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            continue
        # Streaming providers report cumulative totals, not per-chunk increments.
        for name, alias in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
            count = token_count(usage, name, alias)
            if count is not None:
                counts[name] = count
    return counts


def token_count(usage: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _finish_reasons(response: Mapping[str, Any]) -> list[str]:
    choices = response.get("choices")
    if isinstance(choices, list):
        reasons = [choice.get("finish_reason") for choice in choices if isinstance(choice, Mapping)]
        return [reason for reason in reasons if isinstance(reason, str)]
    stop_reason = response.get("stop_reason")
    return [stop_reason] if isinstance(stop_reason, str) else []


def _attribute_value(value: object) -> AttributeValue | None:
    """A metric or score as an attribute: finite numbers, booleans and strings pass, everything else is dropped."""
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return value if isinstance(value, int) else number
    return None


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


__all__ = [
    "INSTRUMENTATION_SCOPE",
    "OpenTelemetryRecordObserver",
    "PlannedIdGenerator",
    "commit_span_context",
    "record_span_context",
]
