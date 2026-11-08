"""The chat completions route for deployments that select external models with Semantic Router."""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from reef.core.errors import INFERENCE_ABORT_ERROR, inference_aborted
from reef.core.records_types import ROLLOUT_HEADER
from reef.inference.interfaces import InferenceStream
from reef.service.inference_service import ARTIFACT_HEADER, RECORD_HEADER, SSERelay
from reef.service.lifecycle import ADMITTED_ROUTING, REQUEST_ACCEPTANCE
from reef.service.request_service import RequestService
from reef.service.routed_inference import RoutedAttempt, RoutedInference
from reef.service.routes.payload import read_object
from reef.service.routing import RoutingAdapter, RoutingRequestError
from reef.service.streaming import receipt_sse_events, stream_record

logger = logging.getLogger(__name__)


async def relay_routed_stream(
    request: web.Request, routed: RoutedInference, for_codex: bool, upstream: InferenceStream, pending: RoutedAttempt
) -> web.StreamResponse:
    """Relay an open provider SSE stream and store its attempt before sending its receipt.

    A stream that ends early still gets one record. Closing the stream and recording the attempt also run
    when the client disconnects or the deployment cancels this request.
    """
    body = bytearray()
    error: str | None = None
    record_attempted = False
    response: web.StreamResponse | None = None
    try:
        # Only Reef issues receipts and artifact headers. This request selected an external target.
        headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower()
            not in (
                RECORD_HEADER,
                ARTIFACT_HEADER,
                "x-reef-release-id",
                "x-reef-artifact-version",
                "content-length",
                "transfer-encoding",
            )
        }
        response = web.StreamResponse(status=upstream.status, headers=headers)
        await response.prepare(request)
        relay = SSERelay(request.path, response, for_codex=for_codex)
        async for chunk in upstream.chunks:
            body.extend(chunk)
            if await relay.feed(chunk):
                break
        if relay.terminal is None:
            await relay.finish()
        if relay.terminal is None:
            error = "upstream SSE ended without a terminal event"
        elif upstream.record_response is not None and inference_aborted(upstream.record_response):
            error = INFERENCE_ABORT_ERROR
            await response.write(relay.terminal)
        else:
            record_attempted = True
            captured = stream_record(upstream, bytes(body), complete=True)
            item = await routed.persist(asyncio.to_thread(routed.record_routed_stream, pending, captured))
            for frame in receipt_sse_events(request.path, relay.chat_identity, relay.terminal, item.agent_record_id):
                await response.write(frame)
    except ConnectionResetError as exc:
        if response is None:
            error = f"{type(exc).__name__}: {exc}"
            raise
        error = "client disconnected"
    except asyncio.CancelledError:
        error = "request cancelled"
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if error is not None:
            logger.warning("routed stream for record %s ended early: %s", pending.item.agent_record_id, error)
        try:
            await upstream.close()
        finally:
            if not record_attempted:
                captured = stream_record(upstream, bytes(body), complete=False, error=error)
                await routed.persist(asyncio.to_thread(routed.record_routed_stream, pending, captured))
    return response


async def serve_routed_chat(
    request: web.Request, routing: RoutingAdapter | None, routed: RoutedInference, *, for_codex: bool
) -> web.StreamResponse:
    """Select a target under ``routing``, the policy that this request keeps, call it, and store each attempt."""
    try:
        if routing is None:
            raise RoutingRequestError("routing is unavailable", code="routing_unpublished", status=503)
        if request.headers.get(ROLLOUT_HEADER, "").strip().lower() in ("1", "true"):
            raise RoutingRequestError("routed deployments do not serve rollout calls", code="unsupported_rollout")
        scenario = routing.check_request(request.headers.items(), request.path)
        client_payload = await read_object(request, strict=True)
        decision = await routed.select_route(routing, scenario, client_payload)
        handler = routing.handler_for(decision)
    finally:
        # The request has either failed or captured everything it needs from this routing policy.
        acceptance = request.app.get(REQUEST_ACCEPTANCE)
        task = asyncio.current_task()
        if acceptance is not None and task is not None:
            acceptance.release_selection(task)
    payload = {**client_payload, "model": decision.target.provider_model}
    if payload.get("stream") is True:
        stream, pending = await routed.start_routed_stream(
            request.headers, payload, request.path, handler, decision=decision
        )
        return await relay_routed_stream(request, routed, for_codex, stream, pending)
    response, item = await routed.infer_routed(request.headers, payload, request.path, decision, handler)
    return web.json_response(response, headers={RECORD_HEADER: item.agent_record_id})


def register_routed_chat(app: web.Application, *, request_service: RequestService) -> None:
    async def chat(request: web.Request) -> web.StreamResponse:
        return await serve_routed_chat(
            request,
            # Before the upload: a managed deployment already captured its adapter when it admitted this request.
            request.get(ADMITTED_ROUTING, request_service.routing),
            request_service.routed,
            for_codex=request_service.dispatcher.recipe.harness_adapter == "codex",
        )

    app.router.add_post("/v1/chat/completions", chat)
