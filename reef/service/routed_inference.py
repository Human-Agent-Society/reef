"""Serve a request through its selected external target and record every provider attempt."""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from aiohttp import ClientError

from reef.core.errors import ReefError, UnknownScenario, UpstreamStatusError, inference_aborted
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectory_index import SHADOW_PURPOSE
from reef.dispatcher import Dispatcher
from reef.inference.http import HttpInferenceHandler
from reef.inference.interfaces import InferenceStream
from reef.inference.provider_protocol import stream_events
from reef.observability.operations import OperationMeasurement
from reef.scenario.scenario import Scenario
from reef.service.lifecycle import INFERENCE_FINALIZATION, PendingWrites
from reef.service.routed_shadow import ShadowTasks
from reef.service.routing import RoutingAdapter, RoutingDecision, RoutingScenario
from reef.service.routing_preview import RoutingPreviewClient
from reef.service.wire import RequestHeaders, parse_request_headers, with_request_tags

WriteResult = TypeVar("WriteResult")
logger = logging.getLogger(__name__)


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
    ``primary_request_id`` is None for an attempt that serves the client (purpose ``serve``). For a shadow
    attempt (purpose ``shadow``) it is the request ID of the primary attempts of the same client request: then
    the decision's reason is ``shadow_target``, there is no earlier attempt, and ``result`` refuses a client
    visible or delivered outcome. Each purpose has its own request ID.
    """

    item: AgentRecord
    decision: RoutingDecision
    request_id: str
    prior_attempt_ids: tuple[str, ...]
    started: float
    measurement: OperationMeasurement
    primary_request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RoutingDecision) or self.decision.scenario != self.item.scenario:
            raise ValueError("a routed attempt needs its own Scenario's decision")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("a routed attempt needs a request ID")
        if type(self.started) not in (float, int) or not math.isfinite(self.started):
            raise ValueError("a routed attempt's start must be a finite monotonic reading")
        shadow = self.primary_request_id is not None
        if shadow != (self.decision.reason == "shadow_target"):
            raise ValueError("exactly a shadow decision's attempt names its primary request")
        if shadow and (
            not isinstance(self.primary_request_id, str)
            or not self.primary_request_id
            or self.primary_request_id == self.request_id
            or self.prior_attempt_ids
        ):
            raise ValueError("a shadow attempt is the only attempt of its own request ID")

    def result(
        self,
        response: Mapping[str, Any],
        *,
        execution: str,
        delivery: str,
        client_visible: bool,
    ) -> AgentRecord:
        """Capture the provider response and authoritative routing metadata in this attempt's record.

        The record's ``metadata.routing`` is Reef's alone: it replaces whatever the client sent there, shadow
        marks included. A shadow record keeps no training data, whatever its outcome, and adds
        ``primary_request_id``.
        """
        shadow = self.primary_request_id is not None
        if shadow and (client_visible or delivery != "not_started"):
            raise ValueError("a shadow attempt is never delivered to the client")
        payload = dict(self.item.payload)
        if execution != "completed" or shadow:
            payload.pop("training", None)
        payload["response"] = {key: value for key, value in response.items() if key != "training"}
        raw = payload.get("metadata")
        metadata = dict(raw) if isinstance(raw, Mapping) else {}
        metadata["routing"] = {
            **self.decision.metadata(),
            "request_id": self.request_id,
            "attempt_index": len(self.prior_attempt_ids) + 1,
            "prior_attempt_ids": list(self.prior_attempt_ids),
            "purpose": SHADOW_PURPOSE if shadow else "serve",
            "execution_outcome": execution,
            "delivery_outcome": delivery,
            "elapsed_seconds": max(0.0, time.monotonic() - self.started),
            "client_visible": client_visible,
            **({"primary_request_id": self.primary_request_id} if shadow else {}),
        }
        return replace(self.item, payload={**payload, "metadata": metadata})


class RoutedInference:
    """Routed calls of a record-only deployment.

    The dispatcher owns Scenario records. Pins resolve locally; automatic requests use the preview client.
    Managed children track submitted writes across cancellation. The retry policy bounds aborted buffered calls.
    ``shadows`` runs the shadow calls of routed requests, with its limits.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        preview: RoutingPreviewClient | None,
        retry: InferenceRetryPolicy,
        writes: PendingWrites | None,
        shadows: ShadowTasks,
    ) -> None:
        self.dispatcher = dispatcher
        self.preview = preview
        self.retry = retry
        self.writes = writes
        self.shadows = shadows

    async def persist(self, write: Awaitable[WriteResult]) -> WriteResult:
        """Retain a managed child's submitted writes through request cancellation and drain."""
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

    async def scenario(self, headers: Mapping[str, str], decision: RoutingDecision) -> tuple[RequestHeaders, Scenario]:
        """Resolve the request's Scenario before its first provider call. Track implicit creation as a write."""
        parsed = parse_request_headers(headers, RequestType.INFERENCE)
        if parsed.scenario != decision.scenario:
            raise ValueError("a routed request's headers name another Scenario than its routing decision")
        scenario = await self.persist(asyncio.to_thread(self.dispatcher.get_or_create_scenario, parsed.scenario))
        if scenario is None:
            raise UnknownScenario(f"unknown scenario {parsed.scenario!r}")
        return parsed, scenario

    @staticmethod
    def attempt(
        parsed: RequestHeaders,
        provider_payload: Mapping[str, Any],
        decision: RoutingDecision,
        request_id: str,
        prior_attempt_ids: tuple[str, ...],
        measurement: OperationMeasurement,
        primary_request_id: str | None = None,
    ) -> RoutedAttempt:
        # Storage also reads artifact_id from legacy payloads. A routed external model has no Reef artifact.
        recorded_payload = {key: value for key, value in provider_payload.items() if key != "artifact_id"}
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=RequestType.INFERENCE,
            payload=with_request_tags(recorded_payload, parsed),
        )
        return RoutedAttempt(
            item, decision, request_id, prior_attempt_ids, time.monotonic(), measurement, primary_request_id
        )

    @staticmethod
    def failure(attempt: RoutedAttempt, exc: BaseException) -> AgentRecord:
        """The record of an attempt that ``exc`` ended before any provider output reached the client."""
        execution = "error"
        if isinstance(exc, asyncio.CancelledError):
            execution = "cancelled"
        elif isinstance(exc, TimeoutError):
            execution = "timeout"
        return attempt.result(
            {"error_type": type(exc).__name__}, execution=execution, delivery="not_started", client_visible=False
        )

    async def failed(self, attempt: RoutedAttempt, exc: BaseException) -> None:
        """Persist a failure before any provider output reached the client. Translate transport errors after the write."""
        await self.persist(asyncio.to_thread(self.dispatcher.accept_record, self.failure(attempt, exc)))
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
        *,
        request_id: str | None = None,
    ) -> tuple[dict[str, Any], AgentRecord]:
        """Serve a buffered call. Only a provider abort retries; each retry records its preceding attempts.

        Every attempt records ``request_id``, or a new one when it is None.
        """
        parsed, scenario = await self.scenario(headers, decision)
        measurement = scenario.operations.start("serve/request")
        succeeded = False
        deadline = time.monotonic() + self.retry.timeout_s
        delay = self.retry.initial_s
        request_id = request_id or uuid.uuid4().hex
        prior_ids: tuple[str, ...] = ()
        try:
            while True:
                sent = handler.request_payload(payload, serving_config=scenario.served_serving_config())
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
                stored = await self.persist(asyncio.to_thread(self.dispatcher.accept_record, item))
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
        request_id: str | None = None,
    ) -> tuple[InferenceStream, RoutedAttempt]:
        """Open one SSE stream and transfer its lifetime and request measurement to the relay.

        The attempt records ``request_id``, or a new one when it is None.
        """
        parsed, scenario = await self.scenario(headers, decision)
        measurement = scenario.operations.start("serve/request")
        try:
            sent = handler.request_payload(payload, serving_config=scenario.served_serving_config())
            attempt = self.attempt(parsed, sent, decision, request_id or uuid.uuid4().hex, (), measurement)
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
            stored = self.dispatcher.accept_record(item)
            succeeded = complete
            return stored
        finally:
            pending.measurement.finish(succeeded=succeeded)

    def start_shadow(
        self,
        headers: Mapping[str, str],
        client_payload: Mapping[str, Any],
        path: str,
        decision: RoutingDecision,
        handler: HttpInferenceHandler,
        primary_request_id: str,
    ) -> None:
        """Start the shadow call of a routed request without waiting for it, or skip it and log why.

        The request's primary call never awaits the shadow task, and no failure of it reaches the client.
        """
        started = self.shadows.start(self.shadow(headers, client_payload, path, decision, handler, primary_request_id))
        if started != "started":
            logger.warning(
                "routed shadow call to target %r of Scenario %r skipped: %s",
                decision.target.target_id,
                decision.scenario,
                started,
            )

    async def shadow(
        self,
        headers: Mapping[str, str],
        client_payload: Mapping[str, Any],
        path: str,
        decision: RoutingDecision,
        handler: HttpInferenceHandler,
        primary_request_id: str,
    ) -> None:
        """Make one shadow call and store its one record, under a request ID of its own.

        The provider gets ``client_payload`` with the shadow target's provider model and, when the client
        streams, ``stream`` false and no ``stream_options``. The call is buffered, never retried, and bounded
        by the limits of ``shadows``. Its record keeps the outcome: ``completed`` or ``aborted`` with the
        response, or ``error``, ``timeout`` or ``cancelled`` with the error type.

        Raises:
            asyncio.CancelledError: The call was cancelled; its ``cancelled`` record was written first.
            Exception: The Scenario or the record could not be stored.
        """
        # Its writes belong to inference finalization, also when this task outlives the request that started it.
        INFERENCE_FINALIZATION.set(True)
        limits = self.shadows.limits
        parsed, scenario = await self.scenario(headers, decision)
        measurement = scenario.operations.start("shadow/request")
        succeeded = False
        try:
            body = {key: value for key, value in client_payload.items() if key != "stream_options"}
            body["model"] = decision.target.provider_model
            if "stream" in body:
                body["stream"] = False
            sent = handler.request_payload(body, serving_config=scenario.served_serving_config())
            attempt = self.attempt(
                parsed, sent, decision, uuid.uuid4().hex, (), measurement, primary_request_id=primary_request_id
            )
            try:
                response = await asyncio.wait_for(
                    handler.post_json(path, sent, max_response_bytes=limits.max_response_bytes),
                    timeout=limits.timeout_seconds,
                )
            except BaseException as exc:
                scenario.operations.increment("shadow/failures_total")
                await self.persist(asyncio.to_thread(self.dispatcher.accept_record, self.failure(attempt, exc)))
                if isinstance(exc, Exception):
                    return
                raise
            aborted = inference_aborted(response)
            item = attempt.result(
                response,
                execution="aborted" if aborted else "completed",
                delivery="not_started",
                client_visible=False,
            )
            await self.persist(asyncio.to_thread(self.dispatcher.accept_record, item))
            succeeded = not aborted
        finally:
            measurement.finish(succeeded=succeeded)
