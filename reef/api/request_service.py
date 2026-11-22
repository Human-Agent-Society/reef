"""Transport-independent request handling behind the aiohttp routes.

``RequestService`` normalizes typed Reef payloads, resolves the artifact
version before every provider call so concurrent publication cannot change
what gets recorded, and stores each exchange as a record. No aiohttp types appear here;
``reef.api.routes`` adapts these methods to HTTP. The harness operations are in
``reef.control.operations.harness``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Mapping
from typing import Any, TypeVar

from reef.api.lifecycle import PendingWrites
from reef.api.routed_inference import (
    DispatcherRecords,
    InferenceRetryPolicy,
    InferenceRetryTimeoutError,
    RoutedInference,
)
from reef.api.wire import ReportPayload, RequestHeaders, parse_request_headers, scenario_header, with_request_tags
from reef.control.dispatcher import Dispatcher
from reef.core.records_types import AgentRecord, RequestType
from reef.routing.manifest import RoutingAdapter
from reef.routing.preview import RoutingPreviewClient

logger = logging.getLogger(__name__)
WriteResult = TypeVar("WriteResult")


def normalize_request_payload(
    request_type: RequestType,
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Normalize a typed Reef payload; a native provider body passes through."""
    if request_type is not RequestType.REPORT:
        return dict(payload), ()
    report = ReportPayload.from_dict(payload)
    return report.to_dict(), report.references


class RequestService:
    """The request handling of one service app over ``dispatcher``.

    Values: ``inference_url`` is the address at which clients reach the inference service
    (``harness.inference_url``), or ``None``; a client's harness layer sends its model calls there, else to the
    address that the client gave.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        *,
        inference_url: str | None = None,
        routing: RoutingAdapter | None = None,
        writes: PendingWrites | None = None,
        preview: RoutingPreviewClient | None = None,
        retry_policy: InferenceRetryPolicy | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self.routing = routing
        self.preview = preview
        self.routed = RoutedInference(
            DispatcherRecords(dispatcher), preview, retry_policy or InferenceRetryPolicy(), writes
        )
        self.inference_url = None if inference_url is None else inference_url.rstrip("/")

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    async def persist(self, write: Awaitable[WriteResult]) -> WriteResult:
        """Retain a managed child's submitted writes through request cancellation and drain."""
        return await self.routed.persist(write)

    def accept_request(
        self, headers: Mapping[str, str], payload: Mapping[str, Any], *, request_id: str | None = None
    ) -> tuple[str, str]:
        """Queue a training request (``Dispatcher.accept_request``); return its scenario and ID."""
        item = self._dispatcher.accept_request(scenario_header(headers), payload, request_id=request_id)
        return item.scenario, item.id

    def accept(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        *,
        request_type: RequestType,
        agent_record_id: str | None = None,
    ) -> AgentRecord:
        if request_type is RequestType.INFERENCE:
            raise ValueError("inference requests must use infer()")
        parsed = parse_request_headers(headers, request_type)
        return self._accept(parsed, payload, agent_record_id=agent_record_id)

    def import_record(self, headers: Mapping[str, str], body: Mapping[str, object]) -> AgentRecord:
        """Persist an existing inference or report through ordinary admission.

        A stable client ID makes uploads retryable. The scenario comes from
        the headers, and importing never invokes an inference handler.
        """
        return self._dispatcher.accept_record(self.prepare_import_record(headers, body))

    def import_records(self, headers: Mapping[str, str], body: Mapping[str, object]) -> tuple[AgentRecord, ...]:
        """Import a bounded, atomic batch using the existing record contract."""
        records = body.get("records")
        if body.keys() != {"records"} or not isinstance(records, list) or not 1 <= len(records) <= 1000:
            raise ValueError("body must contain only 'records', an array of 1 to 1000 record objects")
        items: list[AgentRecord] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("each record must be an object")
            items.append(self.prepare_import_record(headers, record))
        return self._dispatcher.accept_records(items)

    def prepare_import_record(self, headers: Mapping[str, str], body: Mapping[str, object]) -> AgentRecord:
        if body.keys() - {"agent_record_id", "request_type", "payload"}:
            raise ValueError("record fields must be agent_record_id, request_type and payload")
        record_id = body.get("agent_record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("agent_record_id must be a non-empty string")
        request_type = body.get("request_type")
        if not isinstance(request_type, str) or request_type not in ("inference", "report"):
            raise ValueError("import request_type must be 'inference' or 'report'")
        payload = body.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("record payload must be an object")
        parsed = parse_request_headers(headers, RequestType(request_type))
        normalized, references = normalize_request_payload(parsed.request_type, payload)
        return AgentRecord.create(
            scenario=parsed.scenario,
            request_type=parsed.request_type,
            payload=with_request_tags(normalized, parsed),
            agent_record_id=record_id,
            references=references,
            artifact_id=normalized.get("artifact_id"),
        )

    def _accept(
        self,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        *,
        agent_record_id: str | None = None,
    ) -> AgentRecord:
        normalized_payload, references = normalize_request_payload(parsed.request_type, payload)
        normalized_payload = with_request_tags(normalized_payload, parsed)
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=parsed.request_type,
            payload=normalized_payload,
            agent_record_id=agent_record_id,
            references=references,
            artifact_id=normalized_payload.get("artifact_id"),
        )
        try:
            return self._dispatcher.accept_record(item)
        except Exception as exc:
            logger.warning(
                "dispatcher rejected %s record for scenario %r (record %s): %s: %s",
                parsed.request_type.value,
                parsed.scenario,
                item.agent_record_id,
                type(exc).__name__,
                exc,
            )
            raise


__all__ = [
    "InferenceRetryPolicy",
    "InferenceRetryTimeoutError",
    "RequestService",
    "normalize_request_payload",
]
