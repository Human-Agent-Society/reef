from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from reef.runtime.inference import InferenceStream


class SSEFrameDecoder:
    """Split arbitrarily chunked SSE bytes without changing event bytes."""

    _SEPARATORS = (b"\r\n\r\n", b"\n\n", b"\r\r")

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while True:
            boundaries = tuple(
                (index, separator) for separator in self._SEPARATORS if (index := self._buffer.find(separator)) >= 0
            )
            if not boundaries:
                return tuple(frames)
            index, separator = min(boundaries, key=lambda boundary: boundary[0])
            end = index + len(separator)
            frames.append(bytes(self._buffer[:end]))
            self._buffer[:end] = b""

    def finish(self) -> bytes:
        remainder = bytes(self._buffer)
        self._buffer.clear()
        return remainder


def _sse_data(frame: bytes) -> str:
    lines = frame.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line[len("data:") :].removeprefix(" ") for line in lines if line.startswith("data:"))


def is_terminal_sse_event(path: str, frame: bytes) -> bool:
    """Whether ``frame`` is the provider protocol's successful terminator."""

    data = _sse_data(frame)
    if path == "/v1/chat/completions":
        return data == "[DONE]"
    if path == "/v1/messages":
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and payload.get("type") == "message_stop"
    return False


def chat_completion_chunk_identity(frame: bytes) -> dict[str, Any]:
    """Return the stable wire identity fields from one OpenAI chunk."""

    try:
        payload = json.loads(_sse_data(frame))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict) or payload.get("object") != "chat.completion.chunk":
        return {}
    return {
        key: payload[key]
        for key in ("id", "object", "created", "model", "system_fingerprint", "service_tier")
        if key in payload
    }


def receipt_sse_events(
    path: str,
    chat_identity: Mapping[str, Any],
    terminal: bytes,
    agent_record_id: str,
) -> tuple[bytes, ...]:
    """Return terminal event(s) carrying a receipt for a persisted record.

    Chat Completions gets an empty-choice metadata chunk immediately before
    ``[DONE]``, following the same placement as OpenAI's final usage chunk.
    Anthropic's existing ``message_stop`` event carries the metadata directly.
    """

    reef = {"agent_record_id": agent_record_id}
    if path == "/v1/chat/completions":
        base = {"object": "chat.completion.chunk", **chat_identity}
        metadata = {**base, "choices": [], "reef": reef}
        return (f"data: {json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(), terminal)

    if path == "/v1/messages":
        try:
            payload = json.loads(_sse_data(terminal))
        except json.JSONDecodeError:
            payload = {"type": "message_stop"}
        if not isinstance(payload, dict):
            payload = {"type": "message_stop"}
        payload["reef"] = reef
        return (
            b"event: message_stop\n"
            + f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(),
        )

    return (terminal,)


def stream_record(
    stream: InferenceStream,
    body: bytes,
    *,
    complete: bool,
    error: str | None = None,
) -> dict[str, Any]:
    record_response = getattr(stream, "record_response", None)
    if record_response is not None:
        captured_response = dict(record_response)
        captured_response["stream_delivery"] = {
            "status": stream.status,
            "headers": stream.headers,
            "complete": complete,
            **({"error": error} if error is not None else {}),
        }
        return captured_response
    response: dict[str, Any] = {
        "stream": True,
        "status": stream.status,
        "headers": stream.headers,
        "complete": complete,
    }
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        response["body_base64"] = base64.b64encode(body).decode("ascii")
    else:
        response["body"] = text
        if complete:
            aggregated = aggregate_sse_text(text)
            if aggregated is not None:
                response["message"] = {"role": "assistant", "content": aggregated}
    if error is not None:
        response["error"] = error
    return response


def sse_events(body: str):
    """Yield each SSE event's joined data payload, splitting on real newlines only."""

    data_lines: list[str] = []
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].removeprefix(" "))
        elif not line.strip() and data_lines:
            yield "\n".join(data_lines)
            data_lines = []
    if data_lines:
        yield "\n".join(data_lines)


def aggregate_sse_text(body: str) -> str | None:
    """Concatenate the primary choice's text deltas from an SSE body (OpenAI and
    Anthropic shapes); None for tool-using turns or when no text is recognized."""

    parts: list[str] = []
    for data in sse_events(body):
        if data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if choice.get("index") not in (None, 0):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if delta.get("tool_calls"):
                    return None
                if isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
            continue
        kind = event.get("type")
        if kind == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return None
        elif kind == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, dict):
                if delta.get("type") == "input_json_delta":
                    return None
                if isinstance(delta.get("text"), str):
                    parts.append(delta["text"])
    return "".join(parts) if parts else None


__all__ = [
    "SSEFrameDecoder",
    "aggregate_sse_text",
    "chat_completion_chunk_identity",
    "is_terminal_sse_event",
    "receipt_sse_events",
    "sse_events",
    "stream_record",
]
