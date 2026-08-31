from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from reef.runtime.inference import InferenceBackend
from reef.service.request_service import RequestService
from reef.service.streaming import (
    SSEFrameDecoder,
    chat_completion_chunk_identity,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)

logger = logging.getLogger(__name__)


def register_inference_routes(
    app: web.Application,
    *,
    request_service: RequestService,
    inference_backend: InferenceBackend | None,
) -> None:
    async def inference(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        if payload.get("stream") is True:
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
            if not is_sse and pending.item is not None:
                response_headers["x-reef-agent-record-id"] = pending.item.agent_record_id
            response = web.StreamResponse(status=upstream.status, headers=response_headers)
            body = bytearray()
            chat_identity: dict[str, Any] = {}
            complete = False
            error = None
            settled = False
            try:
                await response.prepare(request)
                if is_sse:
                    decoder = SSEFrameDecoder()
                    terminal = None
                    async for chunk in upstream.chunks:
                        body.extend(chunk)
                        for frame in decoder.feed(chunk):
                            if is_terminal_sse_event(request.path, frame):
                                terminal = frame
                                break
                            if not chat_identity:
                                chat_identity = chat_completion_chunk_identity(frame)
                            await response.write(frame)
                        if terminal is not None:
                            break
                    if terminal is None:
                        remainder = decoder.finish()
                        if remainder:
                            await response.write(remainder)
                        error = "upstream SSE ended without a terminal event"
                    else:
                        settled = True
                        item = request_service.record_stream(
                            pending,
                            stream_record(upstream, bytes(body), complete=True),
                        )
                        terminal_events = (
                            receipt_sse_events(
                                request.path,
                                chat_identity,
                                terminal,
                                item.agent_record_id,
                            )
                            if item is not None
                            else (terminal,)
                        )
                        for frame in terminal_events:
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
                    stream_context = (
                        f"record {pending.item.agent_record_id}" if pending.item is not None else "serve-only request"
                    )
                    logger.warning(
                        "stream for %s (%s) ended early: %s",
                        request.path,
                        stream_context,
                        error,
                    )
                try:
                    await upstream.close()
                finally:
                    if not settled:
                        request_service.record_stream(
                            pending,
                            stream_record(upstream, bytes(body), complete=complete, error=error),
                        )
            return response

        response_payload, item = await request_service.infer_with_data(
            request.headers,
            payload,
            request.path,
            inference_backend,
        )
        response_headers = {} if item is None else {"x-reef-agent-record-id": item.agent_record_id}
        return web.json_response(response_payload, headers=response_headers)

    app.router.add_post("/v1/chat/completions", inference)
    app.router.add_post("/v1/messages", inference)
    app.router.add_post("/v1/messages/count_tokens", inference)


__all__ = ["register_inference_routes"]
