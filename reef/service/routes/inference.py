from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from aiohttp import web

from reef.runtime.interfaces import MULTIMODAL_ROUTES, InferenceHandler
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.streaming import (
    SSEFrameDecoder,
    chat_completion_chunk_identity,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)
from reef.service.wire import RELEASE_ID_HEADER, SCENARIO_HEADER

logger = logging.getLogger(__name__)


async def _release_header(request_service: RequestService, headers: Mapping[str, str]) -> dict[str, str]:
    """``x-reef-release-id`` on a file serving scenario's inference responses, so a resident harness learns of a new head on its next call."""
    head = await asyncio.to_thread(request_service.harness_head, headers)
    return {} if head is None else {RELEASE_ID_HEADER: head}


class _SSERelay:
    """Relay upstream SSE frames to the client, holding back the terminal event.

    The terminal event is withheld so Reef can append its receipt frames before
    it, and the chat-completion identity those frames need is picked up from the
    first frame that carries one.
    """

    def __init__(self, path: str, response: web.StreamResponse) -> None:
        self._path = path
        self._response = response
        self._decoder = SSEFrameDecoder()
        self.chat_identity: dict[str, Any] = {}
        self.terminal: bytes | None = None

    async def feed(self, chunk: bytes) -> bool:
        """Write the frames completed by ``chunk``; return whether the terminal arrived."""
        return await self._write(self._decoder.feed(chunk))

    async def finish(self) -> bool:
        """Flush what the decoder still holds; return whether the terminal arrived.

        A trailing partial frame is passed through as-is: the stream ended
        without a terminal event, so the client gets whatever upstream sent.
        """
        frames, remainder = self._decoder.finalize()
        if await self._write(frames):
            return True
        if remainder:
            await self._response.write(remainder)
        return False

    async def _write(self, frames: Iterable[bytes]) -> bool:
        for frame in frames:
            if is_terminal_sse_event(self._path, frame):
                self.terminal = frame
                return True
            if not self.chat_identity:
                self.chat_identity = chat_completion_chunk_identity(frame)
            await self._response.write(frame)
        return False


#: The provider routes Reef serves; an evaluation call names one of them under its scenario.
INFERENCE_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens")


async def _relay_inference_stream(
    request: web.Request,
    payload: dict[str, Any],
    *,
    headers: Mapping[str, str],
    path: str,
    record: bool,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
) -> web.StreamResponse:
    """Stream one upstream inference to the client and, unless it is an evaluation call, record what it sent."""
    upstream, pending = await request_service.start_stream(headers, payload, path, inference_handler, record=record)
    response_headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() not in ("x-reef-agent-record-id", RELEASE_ID_HEADER)
    }
    response_headers.update(await _release_header(request_service, headers))
    content_type = next(
        (value for name, value in response_headers.items() if name.lower() == "content-type"),
        "",
    )
    is_sse = "text/event-stream" in content_type.lower()
    if not is_sse and record:
        response_headers["x-reef-agent-record-id"] = pending.item.agent_record_id
    response = web.StreamResponse(status=upstream.status, headers=response_headers)
    body = bytearray()
    complete = False
    error = None
    record_attempted = False
    try:
        await response.prepare(request)
        if is_sse:
            relay = _SSERelay(path, response)
            async for chunk in upstream.chunks:
                body.extend(chunk)
                if await relay.feed(chunk):
                    break
            if relay.terminal is None:
                await relay.finish()
            if relay.terminal is None:
                error = "upstream SSE ended without a terminal event"
            else:
                record_attempted = True
                item = request_service.record_stream(
                    pending,
                    stream_record(upstream, bytes(body), complete=True),
                )
                frames = (
                    receipt_sse_events(path, relay.chat_identity, relay.terminal, item.agent_record_id)
                    if record
                    else (relay.terminal,)
                )
                for frame in frames:
                    await response.write(frame)
                complete = True
        else:
            async for chunk in upstream.chunks:
                body.extend(chunk)
                await response.write(chunk)
            complete = True
    except ConnectionResetError:
        error = "client disconnected"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if error is not None:
            logger.warning(
                "stream for %s (record %s) ended early: %s",
                request.path,
                pending.item.agent_record_id,
                error,
            )
        try:
            await upstream.close()
        finally:
            if not record_attempted:
                request_service.record_stream(
                    pending,
                    stream_record(upstream, bytes(body), complete=complete, error=error),
                )
    return response


def register_inference_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    inference_handler: InferenceHandler | None,
) -> None:
    async def serve(
        request: web.Request, headers: Mapping[str, str], path: str, *, record: bool
    ) -> web.StreamResponse:
        payload = await read_object(request)
        if payload.get("stream") is True:
            return await _relay_inference_stream(
                request,
                payload,
                headers=headers,
                path=path,
                record=record,
                request_service=request_service,
                inference_handler=inference_handler,
            )
        response_payload, item = await request_service.infer_with_data(
            headers, payload, path, inference_handler, record=record
        )
        response_headers = {} if item is None else {"x-reef-agent-record-id": item.agent_record_id}
        response_headers.update(await _release_header(request_service, headers))
        return web.json_response(response_payload, headers=response_headers)

    async def inference(request: web.Request) -> web.StreamResponse:
        return await serve(request, request.headers, request.path, record=True)

    async def evaluation(request: web.Request) -> web.StreamResponse:
        """An evaluation episode's call: the scenario's release, served like any other and kept by nobody."""
        path = "/v1/" + request.match_info["route"]
        if path not in INFERENCE_PATHS:
            raise web.HTTPNotFound(text=f"{path} is not an inference route")
        headers = {**dict(request.headers), SCENARIO_HEADER: request.match_info["scenario"]}
        return await serve(request, headers, path, record=False)

    async def multimodal(request: web.Request) -> web.StreamResponse:
        """Relay a multimodal call through the scenario's recipe; the answer streams back unchanged, unrecorded."""
        payload = await read_object(request)
        if payload.get("stream") is True:
            raise web.HTTPBadRequest(text=f"{request.path} does not stream; send the request without stream")
        upstream = await request_service.relay_multimodal(request.headers, payload, request.path)
        response = web.StreamResponse(status=upstream.status, headers=upstream.headers)
        try:
            await response.prepare(request)
            async for chunk in upstream.chunks:
                await response.write(chunk)
        finally:
            await upstream.close()
        return response

    app.router.add_post("/v1/chat/completions", inference)
    app.router.add_post("/v1/responses", inference)
    app.router.add_post("/v1/messages", inference)
    app.router.add_post("/v1/messages/count_tokens", inference)
    app.router.add_post("/reef/scenarios/{scenario}/evaluation/v1/{route:.+}", evaluation)
    for path in MULTIMODAL_ROUTES:
        app.router.add_post(path, multimodal)


__all__ = ["register_inference_routes"]
