"""OpenAI and Anthropic chat protocol helpers that every inference handler shares.

The helpers turn sampled text into assistant messages, split thinking from
visible text while a response streams, and encode the protocol's SSE frames.
They do not depend on an engine, so token-native handlers and in-process
backends use the same code.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The reasoning tags and the tool-call start that SGLang's Qwen3 reasoning
# parser reads; a tool call that starts before ``</think>`` ends the reasoning.
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"


@dataclass(frozen=True)
class ParsedToolCall:
    """One tool call in the shape the chat facade reads: a name and its JSON argument text."""

    name: str
    parameters: str


def held_suffix(value: str, delimiter: str) -> int:
    """How many trailing characters of ``value`` could be the start of ``delimiter``."""
    for size in range(min(len(value), len(delimiter) - 1), 0, -1):
        if delimiter.startswith(value[-size:]):
            return size
    return 0


class ReasoningStreamSplitter:
    """Split sampled thinking tags without waiting for the full completion.

    Follows SGLang's Qwen3 reasoning parser: the first ``<think>`` starts
    reasoning and is dropped, the first ``</think>`` or ``tool_call_start``
    ends it, and a stream that ends inside it is reasoning. A slice that may
    be the start of one of these tags is held until the next delta. With
    ``tool_call_start`` set to ``None``, only ``</think>`` ends the reasoning.
    """

    def __init__(self, *, enabled: bool, force_reasoning: bool, tool_call_start: str | None = TOOL_CALL_START) -> None:
        self._enabled = enabled
        self._in_reasoning = enabled and force_reasoning
        self._stripped_think_start = False
        self._buffer = ""
        self._tool_call_start = tool_call_start

    def feed(self, delta: str) -> list[tuple[str, str]]:
        if not delta:
            return []
        if not self._enabled:
            return [("text", delta)]
        self._buffer += delta
        end_tags = [THINK_END] if self._tool_call_start is None else [THINK_END, self._tool_call_start]
        if any(tag.startswith(self._buffer) and tag != self._buffer for tag in (THINK_START, *end_tags)):
            return []
        if not self._stripped_think_start and THINK_START in self._buffer:
            self._buffer = self._buffer.replace(THINK_START, "", 1)
            self._stripped_think_start = True
            self._in_reasoning = True
        if not self._in_reasoning:
            value, self._buffer = self._buffer, ""
            return [("text", value)]

        before, separator, after = self._buffer.partition(THINK_END)
        if separator:
            self._buffer = ""
            self._in_reasoning = False
            parts = [("thinking", before)] if before else []
            visible = after.lstrip("\n")
            if visible:
                parts.append(("text", visible))
            return parts
        tool_start = -1 if self._tool_call_start is None else self._buffer.find(self._tool_call_start)
        if tool_start != -1:
            before, after = self._buffer[:tool_start], self._buffer[tool_start:]
            self._buffer = ""
            self._in_reasoning = False
            return [("thinking", before), ("text", after)] if before else [("text", after)]
        held_tags = end_tags if self._stripped_think_start else [THINK_START, *end_tags]
        held = max(held_suffix(self._buffer, tag) for tag in held_tags)
        if held:
            value = self._buffer[:-held]
            self._buffer = self._buffer[-held:]
        else:
            value, self._buffer = self._buffer, ""
        return [("thinking", value)] if value else []

    def finish(self) -> list[tuple[str, str]]:
        if not self._buffer:
            return []
        value, self._buffer = self._buffer, ""
        if self._in_reasoning:
            value = value.removeprefix(THINK_START)
            return [("thinking", value)] if value else []
        return [("text", value)]


def sse_data(event: Mapping[str, Any]) -> bytes:
    """One OpenAI-style SSE frame: a bare ``data:`` line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def sse_event(event: Mapping[str, Any]) -> bytes:
    """One Anthropic-style SSE frame: the event type names the payload's ``type``."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


class StreamedToolCalls:
    """Tool calls seen so far in one stream: their ids, names and the argument text already sent."""

    def __init__(self) -> None:
        self.ids: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.arguments: dict[int, str] = {}

    def announce(self, index: int, name: Any) -> bool:
        """Record a call's name on first sight; returns whether this chunk carried the name."""
        if name:
            self.ids.setdefault(index, new_call_id())
            self.names[index] = str(name)
        if index not in self.ids:
            raise ValueError("tool stream emitted arguments before the tool name")
        return bool(name)

    def add_arguments(self, index: int, arguments: str) -> None:
        self.arguments[index] = self.arguments.get(index, "") + arguments


class StreamWriter(ABC):
    """Encode one protocol's streamed events; the relay decides what to send and when."""

    @abstractmethod
    def start(self) -> list[bytes]: ...

    @abstractmethod
    def thinking(self, value: str) -> list[bytes]: ...

    @abstractmethod
    def text(self, value: str) -> list[bytes]: ...

    @abstractmethod
    def tool_call(self, index: int, call_id: str, name: str, arguments: str, *, announced: bool) -> list[bytes]:
        """Send tool-call progress; ``announced`` marks the chunk that first named the call."""

    def logprobs(self, content: list[dict[str, Any]]) -> list[bytes]:
        return []

    @abstractmethod
    def finish(self, response: Mapping[str, Any], *, output_tokens: int) -> list[bytes]:
        """Close the stream with the protocol's own view of the completed response."""


class OpenAIChunkWriter(StreamWriter):
    """``chat.completion.chunk`` events sharing one id, model and timestamp."""

    def __init__(self, common: Mapping[str, Any]) -> None:
        self._common = dict(common)

    def _chunk(self, delta: Mapping[str, Any], **choice_fields: Any) -> bytes:
        finish_reason = choice_fields.pop("finish_reason", None)
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason, **choice_fields}
        return sse_data({**self._common, "choices": [choice]})

    def start(self) -> list[bytes]:
        return [self._chunk({"role": "assistant"})]

    def thinking(self, value: str) -> list[bytes]:
        return [self._chunk({"reasoning_content": value})]

    def text(self, value: str) -> list[bytes]:
        return [self._chunk({"content": value})]

    def tool_call(self, index: int, call_id: str, name: str, arguments: str, *, announced: bool) -> list[bytes]:
        tool_delta: dict[str, Any] = {"index": index, "function": {"arguments": arguments}}
        if announced:
            tool_delta.update(id=call_id, type="function")
            tool_delta["function"]["name"] = name
        return [self._chunk({"tool_calls": [tool_delta]})]

    def logprobs(self, content: list[dict[str, Any]]) -> list[bytes]:
        return [self._chunk({}, logprobs={"content": content})]

    def finish(self, response: Mapping[str, Any], *, output_tokens: int) -> list[bytes]:
        choice = response["choices"][0]
        return [
            self._chunk({}, finish_reason=choice["finish_reason"], meta_info=choice["meta_info"]),
            b"data: [DONE]\n\n",
        ]


class AnthropicEventWriter(StreamWriter):
    """Anthropic Messages events; content blocks open and close as the sampled text changes kind."""

    def __init__(self, message_id: str, *, model: str, input_tokens: int) -> None:
        self._message_id = message_id
        self._model = model
        self._input_tokens = input_tokens
        self._block_index = 0
        self._open_block: tuple[str, int | None] | None = None

    def start(self) -> list[bytes]:
        return [
            sse_event(
                {
                    "type": "message_start",
                    "message": {
                        "id": self._message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self._model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": self._input_tokens, "output_tokens": 0},
                    },
                }
            )
        ]

    def _open(self, key: tuple[str, int | None], content_block: Mapping[str, Any]) -> list[bytes]:
        if self._open_block == key:
            return []
        output = self._close()
        output.append(
            sse_event({"type": "content_block_start", "index": self._block_index, "content_block": content_block})
        )
        self._open_block = key
        return output

    def _close(self) -> list[bytes]:
        if self._open_block is None:
            return []
        output = [sse_event({"type": "content_block_stop", "index": self._block_index})]
        self._block_index += 1
        self._open_block = None
        return output

    def _delta(self, delta: Mapping[str, Any]) -> bytes:
        return sse_event({"type": "content_block_delta", "index": self._block_index, "delta": delta})

    def thinking(self, value: str) -> list[bytes]:
        output = self._open(("thinking", None), {"type": "thinking", "thinking": "", "signature": ""})
        output.append(self._delta({"type": "thinking_delta", "thinking": value}))
        return output

    def text(self, value: str) -> list[bytes]:
        output = self._open(("text", None), {"type": "text", "text": ""})
        output.append(self._delta({"type": "text_delta", "text": value}))
        return output

    def tool_call(self, index: int, call_id: str, name: str, arguments: str, *, announced: bool) -> list[bytes]:
        output = self._open(("tool", index), {"type": "tool_use", "id": call_id, "name": name, "input": {}})
        if arguments:
            output.append(self._delta({"type": "input_json_delta", "partial_json": arguments}))
        return output

    def finish(self, response: Mapping[str, Any], *, output_tokens: int) -> list[bytes]:
        return [
            *self._close(),
            sse_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": response["stop_reason"], "stop_sequence": response["stop_sequence"]},
                    "usage": {"output_tokens": output_tokens},
                }
            ),
            sse_event({"type": "message_stop"}),
        ]


class ChatStreamRelay:
    """Turn one request's sampled deltas into client frames and remember what was sent.

    Tool markers are parsed out of the visible text as it streams; at the end
    :meth:`reconcile` compares the canonical parse of the full sample with
    what already went out and sends only the difference.
    """

    def __init__(self, writer: StreamWriter, stream_tool_parser: Any) -> None:
        self.writer = writer
        self._parser = stream_tool_parser
        self._tools = StreamedToolCalls()
        self._reasoning = ""
        self._text = ""

    def piece(self, kind: str, value: str, *, parse_tools: bool = True) -> list[bytes]:
        if not value:
            return []
        if kind == "thinking":
            self._reasoning += value
            return self.writer.thinking(value)
        normal_text, calls = value, []
        if self._parser is not None and parse_tools:
            parse = getattr(self._parser, "parse_stream_chunk", None)
            if parse is None:
                # Third-party test/fallback parsers without an incremental
                # API cannot safely distinguish a partial tool marker from
                # visible text. Reconciliation emits the final parsed tool
                # call at stream end.
                normal_text = ""
            else:
                normal_text, calls = parse(value)
        output: list[bytes] = []
        if normal_text:
            self._text += normal_text
            output.extend(self.writer.text(normal_text))
        for fallback_index, call in enumerate(calls):
            index = getattr(call, "tool_index", fallback_index)
            index = fallback_index if index is None else int(index)
            announced = self._tools.announce(index, getattr(call, "name", None))
            arguments = argument_text(getattr(call, "parameters", ""))
            self._tools.add_arguments(index, arguments)
            output.extend(
                self.writer.tool_call(
                    index, self._tools.ids[index], self._tools.names[index], arguments, announced=announced
                )
            )
        return output

    def reconcile(self, message: dict[str, Any]) -> list[bytes]:
        """Send what the canonical parse of the full sample holds beyond what streamed.

        The message's tool calls are also given the ids the stream already
        announced, so the recorded response and the streamed one agree.
        """
        tool_calls = message.get("tool_calls") or []
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            call["id"] = self._tools.ids.setdefault(index, str(call.get("id") or new_call_id()))
            function = call.get("function")
            if isinstance(function, Mapping):
                self._tools.names.setdefault(index, str(function.get("name", "")))
        output: list[bytes] = []
        final_reasoning = message.get("reasoning_content")
        if isinstance(final_reasoning, str) and final_reasoning.startswith(self._reasoning):
            output.extend(self.piece("thinking", final_reasoning[len(self._reasoning) :], parse_tools=False))
        final_text = message.get("content")
        if isinstance(final_text, str) and final_text.startswith(self._text):
            output.extend(self.piece("text", final_text[len(self._text) :], parse_tools=False))
        for index, call in enumerate(tool_calls):
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, Mapping):
                continue
            arguments = argument_text(function.get("arguments", ""))
            emitted = self._tools.arguments.get(index, "")
            missing = arguments[len(emitted) :] if arguments.startswith(emitted) else arguments
            if not missing and index in self._tools.arguments:
                continue
            output.extend(
                self.writer.tool_call(index, self._tools.ids[index], self._tools.names[index], missing, announced=True)
            )
        return output


def argument_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def stream_ids() -> tuple[str, str, int]:
    """The chat id, message id and timestamp one streamed response shares across its frames."""
    return f"chatcmpl-{uuid.uuid4().hex}", f"msg_{uuid.uuid4().hex}", int(time.time())


def template_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Copy chat messages for the chat template, parsing JSON-string tool-call arguments into objects.

    Raises ``ValueError`` for a tool call that is not a function object or
    whose arguments are not valid JSON.
    """
    normalized: list[dict[str, Any]] = []
    for message in messages:
        entry = dict(message)
        tool_calls = entry.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls = []
            for call in tool_calls:
                if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                    raise ValueError("assistant tool_calls must contain function objects")
                normalized_call = dict(call)
                function = dict(call["function"])
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError("assistant tool-call arguments must be valid JSON") from exc
                normalized_call["function"] = function
                normalized_calls.append(normalized_call)
            entry["tool_calls"] = normalized_calls
        normalized.append(entry)
    return normalized


def assistant_message(
    text: str,
    tool_parser: Any,
    *,
    split_reasoning: bool = True,
    force_reasoning: bool = False,
    tool_call_start: str | None = TOOL_CALL_START,
) -> tuple[dict[str, Any], bool]:
    """Split sampled text into an OpenAI assistant message; ``True`` when it carries tool calls.

    OpenAI-shaped clients render ``content`` verbatim, so reasoning goes to
    ``reasoning_content``. The split follows SGLang's Qwen3 reasoning parser:
    leading ``<think>`` tags are dropped and the first ``</think>`` ends the
    reasoning. The chat template often inserts the opening tag, so the sample
    may carry only the closing tag. In a sample with no ``</think>``, a
    ``tool_call_start`` ends the reasoning; ``None`` turns that rule off. With
    ``force_reasoning`` the template opened the block. A sample that opens
    reasoning and never closes it hit the token cap: all of it is
    ``reasoning_content`` and ``content`` is empty.

    The training tensors come from the raw token stream; this split changes
    only what the client reads. A parser that cannot parse text it
    recognised as a call raises ``ValueError``.
    """
    reasoning: str | None = None
    if split_reasoning and (force_reasoning or THINK_START in text or THINK_END in text):
        body = text
        while body.startswith(THINK_START):
            body = body[len(THINK_START) :]
        head, separator, tail = body.partition(THINK_END)
        tool_start = -1 if tool_call_start is None else body.find(tool_call_start)
        if separator:
            reasoning, text = head, tail.lstrip("\n")
        elif tool_start != -1:
            reasoning, text = body[:tool_start], body[tool_start:]
        else:
            reasoning, text = body, ""
        reasoning = reasoning.strip()
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_parser is None or not tool_parser.has_tool_call(text):
        return message, False
    try:
        remaining_text, parsed = tool_parser.parse_non_stream(text)
    except Exception as exc:
        raise ValueError("tool-call parser could not parse the sampled output") from exc
    if not parsed:
        return message, False
    tool_calls = []
    for call_item in parsed:
        arguments = call_item.parameters
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": new_call_id(),
                "type": "function",
                "function": {"name": call_item.name, "arguments": arguments},
            }
        )
    message["content"] = remaining_text or None
    message["tool_calls"] = tool_calls
    return message, True


def normalize_finish_reason(value: Any) -> str:
    """Map an engine finish reason (a string or a ``{"type": ...}`` object) to stop, length or abort."""
    reason = value.get("type") if isinstance(value, Mapping) else value
    if reason == "length":
        return "length"
    if reason == "abort":
        return "abort"
    return "stop"


def positive_max_tokens(request: Mapping[str, Any], defaults: Mapping[str, Any], default_key: str) -> int:
    """The completion budget from the OpenAI fields, else the handler defaults, else 512."""
    value = request.get(
        "max_completion_tokens",
        request.get("max_tokens", request.get("max_new_tokens", defaults.get(default_key, 512))),
    )
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_completion_tokens must be a positive integer")
    return value


__all__ = [
    "AnthropicEventWriter",
    "ChatStreamRelay",
    "OpenAIChunkWriter",
    "ParsedToolCall",
    "ReasoningStreamSplitter",
    "StreamWriter",
    "StreamedToolCalls",
    "argument_text",
    "assistant_message",
    "held_suffix",
    "new_call_id",
    "normalize_finish_reason",
    "positive_max_tokens",
    "sse_data",
    "sse_event",
    "stream_ids",
    "template_messages",
]
