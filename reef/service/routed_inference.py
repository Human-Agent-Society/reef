"""Serve a request through its selected external target and record every provider attempt."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from aiohttp import ClientError

from reef.core.errors import ReefError, UnknownScenario, UpstreamStatusError, inference_aborted
from reef.core.records_types import AgentRecord, RequestType
from reef.core.values import JsonValue
from reef.dispatcher import Dispatcher
from reef.inference.http import HttpInferenceHandler
from reef.inference.interfaces import InferenceStream
from reef.inference.provider_protocol import stream_events
from reef.observability.operations import OperationMeasurement, OperationMetrics
from reef.service.lifecycle import PendingWrites
from reef.service.routing import RoutingAdapter, RoutingDecision, RoutingScenario
from reef.service.routing_preview import RoutingPreviewClient
from reef.service.wire import RequestHeaders, parse_request_headers, with_request_tags

WriteResult = TypeVar("WriteResult")


@dataclass(frozen=True)
class InferenceRetryPolicy:
    """Retry delays start at ``initial_s`` and grow to ``max_s``; ``timeout_s`` bounds the retry loop."""

    initial_s: float = 0.05
    max_s: float = 1.0
    timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not 0 < self.initial_s <= self.max_s or self.timeout_s <= 0:
            raise ValueError("inference retry policy requires 0 < initial_s <= max_s and timeout_s > 0")


class InferenceRetryTimeout(ReefError):
    """The routed request exhausted its provider deadline."""


@dataclass(frozen=True)
class RoutedAttempt:
    """One external provider attempt under its request's fixed routing decision.

    The record and decision name the same Scenario. The nonempty request ID and the earlier attempt IDs
    link retries. The start time is monotonic. A stream keeps its measurement until its record is stored.
    """

    item: AgentRecord
    decision: RoutingDecision
    request_id: str
    prior_attempt_ids: tuple[str, ...]
    started: float
    measurement: OperationMeasurement

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RoutingDecision) or self.decision.scenario != self.item.scenario:
            raise ValueError("a routed attempt needs its own Scenario's decision")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("a routed attempt needs a request ID")
        if type(self.started) not in (float, int) or not math.isfinite(self.started):
            raise ValueError("a routed attempt's start must be a finite monotonic reading")

    def result(
        self,
        response: Mapping[str, Any],
        *,
        execution: str,
        delivery: str,
        client_visible: bool,
    ) -> AgentRecord:
        """Capture the provider response and authoritative routing metadata in this attempt's record."""
        payload = dict(self.item.payload)
        if execution != "completed":
            payload.pop("training", None)
        payload["response"] = {key: value for key, value in response.items() if key != "training"}
        raw = payload.get("metadata")
        metadata = dict(raw) if isinstance(raw, Mapping) else {}
        metadata["routing"] = {
            **self.decision.metadata(),
            "request_id": self.request_id,
            "attempt_index": len(self.prior_attempt_ids) + 1,
            "prior_attempt_ids": list(self.prior_attempt_ids),
            "purpose": "serve",
            "execution_outcome": execution,
            "delivery_outcome": delivery,
            "elapsed_seconds": max(0.0, time.monotonic() - self.started),
            "client_visible": client_visible,
        }
        return replace(self.item, payload={**payload, "metadata": metadata})


@dataclass(frozen=True)
class RoutedScenario:
    """What one routed request reads of its Scenario when it starts.

    Values: ``operations`` measures the request. ``serving_config`` is the override that each provider call of the
    request carries. Purpose: ``RoutedRecords.open_scenario`` makes it; ``RoutedInference`` reads it.
    """

    operations: OperationMetrics
    serving_config: Mapping[str, JsonValue]


class RoutedRecords(ABC):
    """Where routed requests read their Scenario and store their records.

    ``DispatcherRecords`` serves a process that runs control. ``ReplicaRecords`` serves the inference service, which
    has no dispatcher. Both operations can block, so ``RoutedInference`` calls them on a worker thread.
    """

    @abstractmethod
    def open_scenario(self, name: str) -> RoutedScenario:
        """The Scenario ``name``. Raise ``UnknownScenario`` when the deployment cannot serve it."""

    @abstractmethod
    def accept_record(self, item: AgentRecord) -> AgentRecord:
        """Store ``item`` in its Scenario and return the stored record."""


class DispatcherRecords(RoutedRecords):
    """The Scenarios and records of ``dispatcher``, which also creates a Scenario that does not exist yet."""

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    def open_scenario(self, name: str) -> RoutedScenario:
        scenario = self.dispatcher.get_or_create_scenario(name)
        if scenario is None:
            raise UnknownScenario(f"unknown scenario {name!r}")
        return RoutedScenario(scenario.operations, scenario.served_serving_config())

    def accept_record(self, item: AgentRecord) -> AgentRecord:
        return self.dispatcher.accept_record(item)


class RoutedInference:
    """Routed calls of a record-only deployment.

    ``records`` owns Scenario records. Pins resolve locally; automatic requests use the preview client.
    ``writes`` tracks submitted writes across cancellation when it is given. The retry policy bounds aborted
    buffered calls.
    """

    def __init__(
        self,
        records: RoutedRecords,
        preview: RoutingPreviewClient | None,
        retry: InferenceRetryPolicy,
        writes: PendingWrites | None,
    ) -> None:
        self.records = records
        self.preview = preview
        self.retry = retry
        self.writes = writes

    async def persist(self, write: Awaitable[WriteResult]) -> WriteResult:
        """Retain submitted writes through request cancellation and drain."""
        if self.writes is not None:
            return await self.writes.run(write)
        return await write

    async def select_route(
        self, routing: RoutingAdapter, scenario: RoutingScenario, payload: Mapping[str, object]
    ) -> RoutingDecision:
        """Resolve a pin locally or make one preview against the request's captured routing policy."""
        decision = routing.explicit_decision(scenario, payload)
        if decision is not None:
            return decision
        if self.preview is None:
            raise RuntimeError("routing has no Semantic Router preview client")
        return await self.preview.select(routing, scenario, payload)

    async def scenario(
        self, headers: Mapping[str, str], decision: RoutingDecision
    ) -> tuple[RequestHeaders, RoutedScenario]:
        """Resolve the request's Scenario before its first provider call. Track implicit creation as a write."""
        parsed = parse_request_headers(headers, RequestType.INFERENCE)
        if parsed.scenario != decision.scenario:
            raise ValueError("a routed request's headers name another Scenario than its routing decision")
        return parsed, await self.persist(asyncio.to_thread(self.records.open_scenario, parsed.scenario))

    @staticmethod
    def attempt(
        parsed: RequestHeaders,
        provider_payload: Mapping[str, Any],
        decision: RoutingDecision,
        request_id: str,
        prior_attempt_ids: tuple[str, ...],
        measurement: OperationMeasurement,
    ) -> RoutedAttempt:
        # Storage also reads artifact_id from legacy payloads. A routed external model has no Reef artifact.
        recorded_payload = {key: value for key, value in provider_payload.items() if key != "artifact_id"}
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=RequestType.INFERENCE,
            payload=with_request_tags(recorded_payload, parsed),
        )
        return RoutedAttempt(item, decision, request_id, prior_attempt_ids, time.monotonic(), measurement)

    async def failed(self, attempt: RoutedAttempt, exc: BaseException) -> None:
        """Persist a failure before any provider output reached the client. Translate transport errors after the write."""
        execution = "error"
        if isinstance(exc, asyncio.CancelledError):
            execution = "cancelled"
        elif isinstance(exc, TimeoutError):
            execution = "timeout"
        item = attempt.result(
            {"error_type": type(exc).__name__}, execution=execution, delivery="not_started", client_visible=False
        )
        await self.persist(asyncio.to_thread(self.records.accept_record, item))
        # ServerTimeoutError is both a TimeoutError and a ClientError. Its timeout takes precedence.
        if isinstance(exc, TimeoutError):
            raise InferenceRetryTimeout("routed inference timed out") from exc
        if isinstance(exc, ClientError):
            raise UpstreamStatusError("routed inference upstream connection failed", status=502) from exc

    async def infer_routed(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        decision: RoutingDecision,
        handler: HttpInferenceHandler,
    ) -> tuple[dict[str, Any], AgentRecord]:
        """Serve a buffered call. Only a provider abort retries; each retry records its preceding attempts."""
        parsed, scenario = await self.scenario(headers, decision)
        measurement = scenario.operations.start("serve/request")
        succeeded = False
        deadline = time.monotonic() + self.retry.timeout_s
        delay = self.retry.initial_s
        request_id = uuid.uuid4().hex
        prior_ids: tuple[str, ...] = ()
        try:
            while True:
                sent = handler.request_payload(payload, serving_config=scenario.serving_config)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InferenceRetryTimeout("routed inference retry deadline exceeded")
                attempt = self.attempt(parsed, sent, decision, request_id, prior_ids, measurement)
                try:
                    response = await asyncio.wait_for(handler.post_json(path, sent), timeout=remaining)
                except BaseException as exc:
                    await self.failed(attempt, exc)
                    raise
                aborted = inference_aborted(response)
                item = attempt.result(
                    response,
                    execution="aborted" if aborted else "completed",
                    delivery="not_started" if aborted else "unconfirmed",
                    client_visible=not aborted,
                )
                stored = await self.persist(asyncio.to_thread(self.records.accept_record, item))
                if not aborted:
                    succeeded = True
                    return dict(item.payload["response"]), stored
                prior_ids += (stored.agent_record_id,)
                scenario.operations.increment("serve/retries_total")
                await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(delay * 2, self.retry.max_s)
        finally:
            measurement.finish(succeeded=succeeded)

    async def start_routed_stream(
        self,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        path: str,
        handler: HttpInferenceHandler,
        *,
        decision: RoutingDecision,
    ) -> tuple[InferenceStream, RoutedAttempt]:
        """Open one SSE stream and transfer its lifetime and request measurement to the relay."""
        parsed, scenario = await self.scenario(headers, decision)
        measurement = scenario.operations.start("serve/request")
        try:
            sent = handler.request_payload(payload, serving_config=scenario.serving_config)
            attempt = self.attempt(parsed, sent, decision, uuid.uuid4().hex, (), measurement)
            stream = None
            try:
                stream = await handler.post_stream(path, sent)
                content_type = next((v for k, v in stream.headers.items() if k.lower() == "content-type"), "")
                if "text/event-stream" not in content_type.lower():
                    raise UpstreamStatusError("routed streaming requires an SSE response", status=502)
                return stream, attempt
            except BaseException as exc:
                try:
                    if stream is not None:
                        await stream.close()
                finally:
                    await self.failed(attempt, exc)
                raise
        except BaseException:
            measurement.finish(succeeded=False)
            raise

    def record_routed_stream(self, pending: RoutedAttempt, response: Mapping[str, Any]) -> AgentRecord:
        """Store the relayed stream once, tracking model aborts separately from interrupted delivery."""
        succeeded = False
        try:
            delivery = response.get("stream_delivery", response)
            complete = isinstance(delivery, Mapping) and delivery.get("complete") is True and not delivery.get("error")
            body = response.get("body")
            aborted = inference_aborted(response) or (
                isinstance(body, str) and any(inference_aborted(event) for event in stream_events([body.encode()]))
            )
            item = pending.result(
                response,
                execution="aborted" if aborted else "completed" if complete else "unknown",
                delivery="unconfirmed" if complete else "incomplete",
                client_visible=bool(body) or complete,
            )
            stored = self.records.accept_record(item)
            succeeded = complete
            return stored
        finally:
            pending.measurement.finish(succeeded=succeeded)
