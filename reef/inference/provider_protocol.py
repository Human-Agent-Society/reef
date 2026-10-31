"""The provider wire protocol that Reef's proxies, model clients and recorders share.

Reef reads and writes provider traffic in the provider's own format: OpenAI
Chat Completions, OpenAI Responses and Anthropic Messages. The rules of that
format live here, so the proxy, the model clients and the recorders apply each
rule the same way:

- :func:`provider_headers` gives the credential and version headers of a provider route.
- :class:`SSEFrameDecoder` splits a Server-Sent Events byte stream into event
  frames without changing a byte, and :func:`frame_data` reads one frame's data.
- :func:`stream_events` reads the JSON events of an SSE stream as it arrives,
  and :func:`response_events` reads the JSON objects of a body, streamed or not.
- :func:`fold_chat_stream`, :func:`fold_anthropic_stream` and
  :func:`fold_responses_stream` turn the events of a streamed reply into the
  response object that the request returns without streaming.
- :func:`usage_of` reads the token counts of a response in any dialect.

The module imports only the standard library, so a model client can use it
without loading an inference backend.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

#: The Anthropic Messages API version that Reef sends when its client names none.
ANTHROPIC_VERSION = "2023-06-01"

#: Routes that speak the Anthropic Messages API and take its native auth headers.
ANTHROPIC_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})

#: The Anthropic request headers that a relay passes on from its client: the API version and the beta features.
ANTHROPIC_CLIENT_HEADERS = ("anthropic-version", "anthropic-beta")

#: The data of the event that ends a Chat Completions stream.
SSE_DONE = "[DONE]"


def provider_headers(path: str, api_key: str | None) -> dict[str, str]:
    """The provider headers for a request on the route ``path``: the credential and the API version.

    An Anthropic route (:data:`ANTHROPIC_PATHS`) gets ``anthropic-version``,
    and the key as ``x-api-key`` when ``api_key`` is not empty. Any other route
    gets the key as a bearer ``Authorization`` header, and no header when
    ``api_key`` is empty.
    """
    if path in ANTHROPIC_PATHS:
        headers = {"x-api-key": api_key} if api_key else {}
        headers["anthropic-version"] = ANTHROPIC_VERSION
        return headers
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


class SSEFrameDecoder:
    """Split an arbitrarily chunked Server-Sent Events byte stream into frames without changing a byte.

    Values: a frame is the bytes of one event up to and including the empty
    line that ends it, where a line ends with CR, LF or CRLF as the SSE spec
    allows. ``buffer`` holds the bytes fed since the last frame was returned.
    ``after_line_end`` is true when the last byte of ``buffer`` ends a line;
    ``pending_cr`` is true when that byte is a CR that a following LF extends
    into CRLF; ``pending_frame`` is true when ``buffer`` is a whole frame that
    ends with such a CR, so the next byte decides whether an LF belongs to it.
    All three flags are false while ``buffer`` is empty.
    Purpose: a reader of a live stream feeds each chunk as it arrives and acts
    on each whole frame. The frames that :meth:`feed` returns, followed by the
    bytes that :meth:`finish` returns, are the fed bytes in their order.
    """

    CR = ord("\r")
    LF = ord("\n")

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.after_line_end = False
        self.pending_cr = False
        self.pending_frame = False

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        """Append ``chunk`` and return the frames that it completed, in order."""
        frames: list[bytes] = []
        for value in chunk:
            # A CR that ends an empty line may still be the first byte of CRLF.
            # Delay that frame until the following byte makes the choice clear.
            if self.pending_frame:
                if value == self.LF:
                    self.buffer.append(value)
                    frames.append(self.take_frame())
                    continue
                frames.append(self.take_frame())

            self.buffer.append(value)
            if self.pending_cr:
                self.pending_cr = False
                if value == self.LF:
                    continue

            if value == self.CR:
                self.pending_frame = self.after_line_end
                self.after_line_end = True
                self.pending_cr = True
            elif value == self.LF:
                if self.after_line_end:
                    frames.append(self.take_frame())
                else:
                    self.after_line_end = True
            else:
                self.after_line_end = False
        return tuple(frames)

    def finalize(self) -> tuple[tuple[bytes, ...], bytes]:
        """End the stream: the frame held back by a trailing CR, if any, and the bytes of an unfinished event.

        The decoder is empty afterwards.
        """
        frames = (bytes(self.buffer),) if self.pending_frame else ()
        remainder = b"" if frames else bytes(self.buffer)
        self.reset()
        return frames, remainder

    def finish(self) -> bytes:
        """Return every byte not already returned, including a final whole frame; the decoder is empty afterwards."""
        frames, remainder = self.finalize()
        return b"".join(frames) + remainder

    def take_frame(self) -> bytes:
        """Return ``buffer`` as one frame and empty the decoder."""
        frame = bytes(self.buffer)
        self.reset()
        return frame

    def reset(self) -> None:
        """Empty ``buffer`` and clear the line-end flags."""
        self.buffer = bytearray()
        self.after_line_end = False
        self.pending_cr = False
        self.pending_frame = False


def frame_data(frame: bytes) -> str:
    """The data of one SSE event ``frame``: the values of its ``data:`` lines, joined by LF.

    Each value loses one leading space. The result is empty when the frame has
    no ``data:`` line. The frame is read as UTF-8 with invalid bytes replaced,
    as the SSE spec reads a stream.
    """
    lines = frame.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line[len("data:") :].removeprefix(" ") for line in lines if line.startswith("data:"))


def stream_events(chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """The JSON objects in the events of an SSE byte stream, in order, up to the ``[DONE]`` event.

    ``chunks`` are read one at a time, and no chunk is read after the one that
    ends the ``[DONE]`` event, so a sender that keeps the connection open
    after it does not hold the reader. :class:`SSEFrameDecoder` splits the
    chunks into events, and :func:`frame_data` reads the data of each event.
    The data of an event is one JSON value, as the SSE spec reads it. A sender
    that leaves out the empty lines between events puts several values in one
    event, one on each data line, so data of several lines that is not one
    JSON value is read line by line. Data that is not JSON, and a value that
    is not an object, is skipped.
    """

    def is_one_value(data: str) -> bool:
        try:
            json.loads(data)
        except json.JSONDecodeError:
            return False
        return True

    def frames() -> Iterator[bytes]:
        decoder = SSEFrameDecoder()
        for chunk in chunks:
            yield from decoder.feed(chunk)
        yield decoder.finish()

    for frame in frames():
        data = frame_data(frame)
        payloads = data.split("\n") if "\n" in data and not is_one_value(data) else [data]
        for payload in payloads:
            if payload == SSE_DONE:
                return
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def response_events(body: str) -> list[dict[str, Any]]:
    """The JSON objects of a provider response body, streamed or not.

    A body that starts with ``{`` is one JSON object: the list holds it, or is
    empty when the body does not parse. Any other body is an SSE stream, read
    by :func:`stream_events`.
    """
    if body.lstrip().startswith("{"):
        try:
            return [json.loads(body)]
        except json.JSONDecodeError:
            return []
    return list(stream_events([body.encode()]))


def merge_reasoning_details(details: list[Any], fragments: Sequence[Any]) -> None:
    """Join provider detail fragments by index (or id), leaving opaque blocks opaque."""
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            details.append(fragment)
            continue
        key = "index" if fragment.get("index") is not None else "id" if fragment.get("id") is not None else None
        target = next(
            (
                item
                for item in details
                if isinstance(item, dict) and key is not None and item.get(key) == fragment[key]
            ),
            None,
        )
        if target is None:
            details.append(dict(fragment))
            continue
        for name, value in fragment.items():
            if value is None and target.get(name) is not None:
                continue
            if name in ("text", "summary", "signature", "data") and isinstance(value, str):
                previous = target.get(name)
                target[name] = (previous if isinstance(previous, str) else "") + value
            else:
                target[name] = value


def fold_chat_stream(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The Chat Completions response that one streamed reply's chunks ``events`` fold into.

    The message joins the content deltas of every choice in order and keeps
    the reasoning fields that the deltas carry; the role, finish reason, model
    and usage are the last ones seen.
    """

    pieces: list[str] = []
    role = "assistant"
    finish_reason = None
    model = None
    usage: dict[str, Any] | None = None
    reasoning: dict[str, Any] = {}
    for chunk in events:
        model = chunk.get("model", model)
        if isinstance(chunk.get("usage"), Mapping):
            usage = dict(chunk["usage"])
        for choice in chunk.get("choices", ()):
            delta = choice.get("delta", {})
            role = delta.get("role", role)
            content = delta.get("content")
            if isinstance(content, str):
                pieces.append(content)
            for key in ("reasoning", "reasoning_content", "thinking"):
                if isinstance(delta.get(key), str):
                    reasoning[key] = reasoning.get(key, "") + delta[key]
            if isinstance(delta.get("reasoning_details"), list):
                merge_reasoning_details(reasoning.setdefault("reasoning_details", []), delta["reasoning_details"])
            finish_reason = choice.get("finish_reason", finish_reason)
    folded: dict[str, Any] = {
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": "".join(pieces), **reasoning},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        folded["usage"] = usage
    return folded


def fold_anthropic_stream(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The Anthropic Messages response that one streamed reply's ``events`` fold into.

    Each content block takes its deltas in order. A tool input is the JSON its
    pieces form, or stays as ``partial_json`` when they do not parse. The usage
    of ``message_start`` is updated by each ``message_delta``.
    """

    blocks: dict[int, dict[str, Any]] = {}
    partial_inputs: dict[int, str] = {}
    model = None
    stop_reason = None
    usage: dict[str, Any] = {}
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            model = event.get("message", {}).get("model", model)
            if isinstance(event.get("message", {}).get("usage"), Mapping):
                usage.update(event["message"]["usage"])
        elif kind == "content_block_start" and isinstance(event.get("content_block"), Mapping):
            blocks[event.get("index", 0)] = dict(event["content_block"])
        elif kind == "content_block_delta":
            delta = event.get("delta", {})
            fields = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}
            field_name = fields.get(delta.get("type"))
            if field_name is not None and isinstance(delta.get(field_name), str):
                block = blocks.setdefault(
                    event.get("index", 0), {"type": "text" if field_name == "text" else "thinking"}
                )
                block[field_name] = str(block.get(field_name, "")) + delta[field_name]
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                index = event.get("index", 0)
                partial_inputs[index] = partial_inputs.get(index, "") + delta["partial_json"]
        elif kind == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason", stop_reason)
            if isinstance(event.get("usage"), Mapping):
                usage.update(event["usage"])
    for index, partial_json in partial_inputs.items():
        block = blocks.setdefault(index, {"type": "tool_use"})
        try:
            block["input"] = json.loads(partial_json)
        except json.JSONDecodeError:
            # Preserve an interrupted tool argument as fragments, never the start block's empty input placeholder.
            block.pop("input", None)
            block["partial_json"] = partial_json
    folded: dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [blocks[index] for index in sorted(blocks)],
        "stop_reason": stop_reason,
    }
    if usage:
        folded["usage"] = usage
    return folded


def fold_responses_stream(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The OpenAI Responses response that one streamed reply's ``events`` fold into.

    It is the response of the ``response.completed`` event when the stream has
    one. Otherwise it holds the finished output items, plus an assistant
    message with the text deltas when no item is one.
    """

    pieces: list[str] = []
    output: dict[int, dict[str, Any]] = {}
    completed: dict[str, Any] | None = None
    for event in events:
        if event.get("type") == "response.output_text.delta" and isinstance(event.get("delta"), str):
            pieces.append(event["delta"])
        elif event.get("type") == "response.output_item.done" and isinstance(event.get("item"), dict):
            output[event.get("output_index", len(output))] = event["item"]
        elif event.get("type") == "response.completed" and isinstance(event.get("response"), dict):
            completed = event["response"]
    if completed is not None:
        return completed
    items = [output[index] for index in sorted(output)]
    if not any(item.get("type") == "message" and item.get("role") == "assistant" for item in items):
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(pieces)}],
            }
        )
    return {
        "object": "response",
        "status": "completed",
        "output": items,
    }


def usage_of(response: object) -> dict[str, int] | None:
    """The tokens a response reports, as ``{"input_tokens", "output_tokens"}``
    whatever the dialect (Chat Completions counts ``prompt_tokens`` /
    ``completion_tokens``; Messages and Responses count ``input_tokens`` /
    ``output_tokens``); None when the response carries no usage.
    """

    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return None

    def count(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    inputs = count("input_tokens", "prompt_tokens")
    outputs = count("output_tokens", "completion_tokens")
    if inputs is None and outputs is None:
        return None
    return {"input_tokens": inputs or 0, "output_tokens": outputs or 0}


__all__ = [
    "ANTHROPIC_CLIENT_HEADERS",
    "ANTHROPIC_PATHS",
    "SSE_DONE",
    "SSEFrameDecoder",
    "fold_anthropic_stream",
    "fold_chat_stream",
    "fold_responses_stream",
    "frame_data",
    "merge_reasoning_details",
    "provider_headers",
    "response_events",
    "stream_events",
    "usage_of",
]
