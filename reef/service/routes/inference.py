from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from aiohttp import web

from reef.runtime.inference import InferenceBackend
from reef.service.request_service import RequestService
from reef.service.routes.payload import read_object
from reef.service.streaming import (
    SSEFrameDecoder,
    chat_completion_chunk_identity,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)

logger = logging.getLogger(__name__)


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


async def _relay_inference_stream(
    request: web.Request,
    payload: dict[str, Any],
    *,
    request_service: RequestService,
    inference_backend: InferenceBackend | None,
) -> web.StreamResponse:
    """Stream one upstream inference to the client and record what it sent."""
    upstream, pending = await request_service.start_stream(
        request.headers,
        payload,
        request.path,
        inference_backend,
    )
    response_headers = {
        name: value for name, value in upstream.headers.items() if name.lower() != "x-reef-agent-record-id"
    }
    content_type = next(
        (value for name, value in response_headers.items() if name.lower() == "content-type"),
        "",
    )
    is_sse = "text/event-stream" in content_type.lower()
    if not is_sse:
        response_headers["x-reef-agent-record-id"] = pending.item.agent_record_id
    response = web.StreamResponse(status=upstream.status, headers=response_headers)
    body = bytearray()
    complete = False
    error = None
    record_attempted = False
    try:
        await response.prepare(request)
        if is_sse:
            relay = _SSERelay(request.path, response)
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
                for frame in receipt_sse_events(
                    request.path,
                    relay.chat_identity,
                    relay.terminal,
                    item.agent_record_id,
                ):
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
    inference_backend: InferenceBackend | None,
) -> None:
    async def inference(request: web.Request) -> web.StreamResponse:
        payload = await read_object(request)
        if payload.get("stream") is True:
            return await _relay_inference_stream(
                request,
                payload,
                request_service=request_service,
                inference_backend=inference_backend,
            )
        response_payload, item = await request_service.infer_with_data(
            request.headers,
            payload,
            request.path,
            inference_backend,
        )
        return web.json_response(response_payload, headers={"x-reef-agent-record-id": item.agent_record_id})

    app.router.add_post("/v1/chat/completions", inference)
    app.router.add_post("/v1/messages", inference)
    app.router.add_post("/v1/messages/count_tokens", inference)


__all__ = ["register_inference_routes"]
