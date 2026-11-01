from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from reef.inference.provider_protocol import SSE_DONE, frame_data, stream_events
from reef.runtime.interfaces import InferenceStream

#: Responses events that end a stream the model finished; ``incomplete`` is a normal stop at a token limit.
RESPONSES_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete"})

#: The output item type OpenRouter streams for its web search server tool, and the item type Codex reads
#: for a web search (actions ``search``, ``open_page``, ``find_in_page``).
OPENROUTER_WEB_SEARCH_ITEM = "openrouter:web_search"
WEB_SEARCH_CALL_ITEM = "web_search_call"


def is_terminal_sse_event(path: str, frame: bytes) -> bool:
    """Whether ``frame`` is the provider protocol's successful terminator."""

    data = frame_data(frame)
    if path == "/v1/chat/completions":
        return data == SSE_DONE
    if path in ("/v1/messages", "/v1/responses"):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        if path == "/v1/messages":
            return payload.get("type") == "message_stop"
        return payload.get("type") in RESPONSES_TERMINAL_EVENTS
    return False


def frame_for_codex(frame: bytes) -> bytes:
    """Rewrite one Responses SSE frame so a Codex client reads OpenRouter's web search.

    OpenRouter streams its web search server tool as an output item of type
    ``openrouter:web_search``; Codex only parses ``web_search_call`` and drops
    an item of an unknown type, so the search never shows in the TUI, never
    lands in the session rollout, and never replays on the next turn. The two
    item shapes share the OpenAI action object, so the rewrite renames the
    item's type on ``response.output_item.added`` and ``response.output_item.done``,
    and inside the ``response.completed`` output. A frame without the item,
    including a frame of any other protocol, comes back unchanged.

    Scope note (follow-up, #621): this rewrite covers the client-facing frames
    only, and only within one turn. Whether OpenRouter accepts the replayed
    ``web_search_call`` item on the *next* turn's input is unverified offline —
    it needs a live OpenRouter key and Codex — so the replay leg stays an open
    follow-up, not something this function settles.
    """

    if OPENROUTER_WEB_SEARCH_ITEM.encode("ascii") not in frame:
        return frame
    try:
        payload = json.loads(frame_data(frame))
    except json.JSONDecodeError:
        return frame
    if not isinstance(payload, dict):
        return frame
    kind = payload.get("type")
    if kind in ("response.output_item.added", "response.output_item.done") and isinstance(payload.get("item"), dict):
        item = payload["item"]
        if item.get("type") != OPENROUTER_WEB_SEARCH_ITEM:
            return frame
        payload["item"] = {**item, "type": WEB_SEARCH_CALL_ITEM}
    elif kind == "response.completed" and isinstance(payload.get("response"), dict):
        output = payload["response"].get("output")
        if not isinstance(output, list) or not any(
            isinstance(item, dict) and item.get("type") == OPENROUTER_WEB_SEARCH_ITEM for item in output
        ):
            return frame
        payload["response"]["output"] = [
            (
                {**item, "type": WEB_SEARCH_CALL_ITEM}
                if isinstance(item, dict) and item.get("type") == OPENROUTER_WEB_SEARCH_ITEM
                else item
            )
            for item in output
        ]
    else:
        return frame
    return _rewritten_sse_frame(frame, payload)


def _rewritten_sse_frame(frame: bytes, payload: dict[str, Any]) -> bytes:
    """The frame with its data lines replaced by one line carrying ``payload``; every other line kept."""

    lines = frame.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    head = [line for line in lines if not line.startswith("data:")]
    while head and not head[-1]:
        head.pop()
    data = f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    return ("\n".join([*head, data]) + "\n\n").encode()


def chat_completion_chunk_identity(frame: bytes) -> dict[str, Any]:
    """Return the stable wire identity fields from one OpenAI chunk."""

    try:
        payload = json.loads(frame_data(frame))
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
    Anthropic's existing ``message_stop`` event and the Responses terminal
    event (``response.completed``) carry the metadata directly.
    """

    reef = {"agent_record_id": agent_record_id}
    if path == "/v1/chat/completions":
        base = {"object": "chat.completion.chunk", **chat_identity}
        metadata = {**base, "choices": [], "reef": reef}
        return (f"data: {json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(), terminal)

    if path == "/v1/messages":
        try:
            payload = json.loads(frame_data(terminal))
        except json.JSONDecodeError:
            payload = {"type": "message_stop"}
        if not isinstance(payload, dict):
            payload = {"type": "message_stop"}
        payload["reef"] = reef
        return (
            b"event: message_stop\n"
            + f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(),
        )

    if path == "/v1/responses":
        try:
            payload = json.loads(frame_data(terminal))
        except json.JSONDecodeError:
            return (terminal,)
        if not isinstance(payload, dict):
            return (terminal,)
        payload["reef"] = reef
        return (
            f"event: {payload.get('type')}\n".encode()
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
    record_response = stream.record_response
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


def aggregate_sse_text(body: str) -> str | None:
    """Concatenate the primary choice's text deltas from an SSE body (Chat
    Completions, Responses and Anthropic shapes); None for tool-using turns or
    when no text is recognized."""

    parts: list[str] = []
    for event in stream_events([body.encode()]):
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
        if kind == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") not in ("message", "reasoning"):
                return None
        elif kind == "response.output_text.delta":
            if isinstance(event.get("delta"), str):
                parts.append(event["delta"])
        elif kind == "content_block_start":
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
    "aggregate_sse_text",
    "chat_completion_chunk_identity",
    "frame_for_codex",
    "is_terminal_sse_event",
    "receipt_sse_events",
    "stream_record",
]
