"""OpenAI and Anthropic chat facades over an engine's token-native generate API.

Clients use either ``/v1/chat/completions`` or ``/v1/messages`` while Reef
normalizes the provider request, renders the prompt once, calls the selected
engine's native generate route, and answers with the provider response. The
response carries no training tensors: a training component renders a recorded
call with its own chat template and computes the log probabilities that it
needs with its own weights.

The engine-specific part is a :class:`NativeGenerateClient`: it shapes the
native request and parses the native response into a
:class:`CapturedGeneration`. Everything else here is shared by every engine.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.reasoning import PROMPT_TAIL_TOKEN_COUNT, is_reasoning_pre_opened, split_reasoning
from reef.core.values import JsonValue, require_integer
from reef.inference.anthropic import NATIVE_REQUEST_KEYS, anthropic_request, anthropic_response
from reef.inference.http import HttpInferenceHandler, RequestHeadersFactory
from reef.inference.interfaces import InferenceArtifact, InferenceStream, parse_full_weight_name
from reef.inference.provider_protocol import SSE_DONE, SSEFrameDecoder, frame_data

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"


@dataclass(frozen=True)
class CapturedGeneration:
    """One request's exact engine output.

    ``output_ids`` and ``rollout_log_probs`` have one entry per sampled token.
    ``text`` is the engine's own decoding when it provides one; ``None`` makes
    the handler decode the ids with its tokenizer. ``public_meta`` is the
    engine metadata exposed to clients as ``choice["meta_info"]``; it must not
    carry tensors.
    """

    output_ids: list[int]
    rollout_log_probs: list[float]
    finish_reason: str
    text: str | None
    public_meta: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.output_ids)
        if len(self.rollout_log_probs) != count:
            raise ValueError("captured response tensors have inconsistent lengths")
        if not count and self.finish_reason != "abort":
            raise ValueError("captured response has no sampled tokens")
        if self.finish_reason not in {"stop", "length", "abort"}:
            raise ValueError(f"captured finish reason must be stop, length or abort, got {self.finish_reason!r}")


class NativeGenerateClient(ABC):
    """One engine's token-native generate API: prompt ids in, exact sample out."""

    #: Route under the engine URL that accepts token ids and returns sampled ids.
    generate_path: str
    #: Label for upstream errors, naming the engine and route.
    error_label: str

    @abstractmethod
    def sampling_params(self, request: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
        """Translate provider sampling fields to the engine's names, applying ``defaults`` for omitted ones."""

    @abstractmethod
    def payload(
        self,
        request: Mapping[str, Any],
        prompt_ids: list[int],
        sampling_params: Mapping[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """The native request body for ``prompt_ids``, asking for the sampled tokens' log-probs."""

    @abstractmethod
    def parse(self, response: Mapping[str, Any]) -> CapturedGeneration:
        """Validate one buffered native response and read the exact sample out of it."""

    @abstractmethod
    def tool_parser(self, tools: list[dict[str, Any]], parser_name: str, tokenizer: Any) -> Any:
        """Build the engine's tool-call parser for one request's ``tools``.

        The parser exposes ``has_tool_call(text)`` and
        ``parse_non_stream(text) -> (remaining_text, calls)`` where each call
        has ``name`` and ``parameters``; one with ``parse_stream_chunk(delta)``
        also supports incremental streaming. ``tokenizer`` is the handler's,
        for parsers that need one.
        """


async def sse_json_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Decode SSE JSON events and consume the transport through EOF, including after ``[DONE]``.

    An event without data is skipped. An event whose data is not a JSON object,
    or that reports an ``error``, raises :class:`ValueError`. Data after ``[DONE]`` is invalid.
    Exhaustion means the transport completed, so a trailing transport error cannot be hidden by the marker.
    """

    def decode_event(payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("native stream event must be a JSON object")
        if isinstance(value.get("error"), Mapping):
            raise ValueError(f"native stream failed: {value['error'].get('message', value['error'])}")
        return value

    decoder = SSEFrameDecoder()
    done = False
    async for chunk in chunks:
        for frame in decoder.feed(chunk):
            payload = frame_data(frame)
            if done and payload:
                raise ValueError("native stream sent data after [DONE]")
            if payload == SSE_DONE:
                done = True
            elif payload:
                yield decode_event(payload)

    payload = frame_data(decoder.finish())
    if done and payload:
        raise ValueError("native stream sent data after [DONE]")
    if payload and payload != SSE_DONE:
        yield decode_event(payload)


def _sse_data(event: Mapping[str, Any]) -> bytes:
    """One OpenAI-style SSE frame: a bare ``data:`` line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _sse_event(event: Mapping[str, Any]) -> bytes:
    """One Anthropic-style SSE frame: the event type names the payload's ``type``."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass(frozen=True)
class StreamedToolCall:
    """One tool argument delta supplied by a native parser adapter to the chat relay.

    ``index`` is the parser's nonnegative integer call position. ``name`` is a nonempty tool name or None when omitted.
    ``arguments`` is the argument fragment, which may be empty. The relay checks the position and name across deltas.
    """

    index: int
    name: str | None
    arguments: str

    def __post_init__(self) -> None:
        require_integer(self.index, "the streamed tool-call index")
        if not isinstance(self.arguments, str):
            raise ValueError("a streamed tool-call argument fragment must be a string")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("a streamed tool-call name must be a nonempty string or None")


class StreamingToolParser(ABC):
    """Read one tool parser's incremental text and emit typed tool call deltas.

    ``parse_stream_chunk`` returns the visible text and tool call deltas in the order read from that chunk.
    An adapter that has only a buffered parser returns no visible text or tool deltas.
    """

    @abstractmethod
    def parse_stream_chunk(self, text: str) -> tuple[str, list[StreamedToolCall]]:
        """Parse the next text fragment of this reply."""


class _StreamedToolCalls:
    """Keep the tool calls that ``ChatStreamRelay`` announced in one stream.

    ``ids`` and ``names`` have the same nonnegative integer indices. Each index is a tool's position in the
    completed reply. Its ID and name are nonempty strings and remain fixed after announcement. Each key of
    ``arguments`` has an announced call. Its value is a nonempty list of strings in send order, including empty
    fragments. An absent key means no fragments have been sent for that call.
    """

    def __init__(self) -> None:
        self.ids: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.arguments: dict[int, list[str]] = {}

    def announce(self, index: int, name: str | None) -> bool:
        """Return true and assign an ID when a named call first appears.

        Return false when an existing call has the same name or the fragment omits its name. Raise
        ``ValueError`` for a negative index, a changed name, or a new call without a name.
        """
        if index < 0:
            raise ValueError("a streamed tool-call index must be nonnegative")
        if index in self.ids:
            if name and self.names[index] != name:
                raise ValueError("the streamed tool-call name changed")
            return False
        if not name:
            raise ValueError("tool stream emitted arguments before the tool name")
        self.ids[index] = _new_call_id()
        self.names[index] = name
        return True

    def add_arguments(self, index: int, arguments: str) -> None:
        """Append the argument fragment for already announced ``index``."""
        self.arguments.setdefault(index, []).append(arguments)


class _StreamWriter(ABC):
    """Encode one protocol's streamed events; the relay decides what to send and when.

    ``accepts_raw_arguments`` is false for protocols that require the final tool arguments to be a JSON object.
    """

    accepts_raw_arguments: bool = False

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


class _OpenAIChunkWriter(_StreamWriter):
    """Encode one OpenAI Chat Completions stream for ``ChatStreamRelay``.

    The response ID and model are strings. The creation timestamp is a nonnegative integer in seconds.
    The writer sets the object value to ``chat.completion.chunk``. The boolean ``include_usage`` adds null
    usage to each choice chunk. It also adds
    the response totals in one empty-choice chunk after the final choice.

    ``start`` emits the assistant role. ``thinking``, ``text``, and ``logprobs`` put their inputs in the
    corresponding choice fields. ``tool_call`` emits the argument fragment at its index. Its first fragment also
    names the call and includes its ID. ``finish`` emits the final choice, optional usage chunk, and ``[DONE]``.
    """

    accepts_raw_arguments = True

    def __init__(self, response_id: str, *, model: str, created: int, include_usage: bool) -> None:
        if not isinstance(response_id, str) or not isinstance(model, str):
            raise ValueError("The response ID and model must be strings.")
        require_integer(created, "the creation timestamp")
        if not isinstance(include_usage, bool):
            raise ValueError("The include_usage flag must be a boolean.")
        self._common: dict[str, JsonValue] = {
            "id": response_id,
            "model": model,
            "created": created,
            "object": "chat.completion.chunk",
        }
        self.include_usage = include_usage
        if include_usage:
            self._common["usage"] = None

    def _chunk(self, delta: Mapping[str, Any], **choice_fields: Any) -> bytes:
        finish_reason = choice_fields.pop("finish_reason", None)
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason, **choice_fields}
        return _sse_data({**self._common, "choices": [choice]})

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
        """Use the completed first choice's finish reason and metadata in the final choice chunk.

        If the response has token IDs, put the output IDs in the choice and the prompt IDs beside ``choices``.
        """
        choice = response["choices"][0]
        closing = {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"], "meta_info": choice["meta_info"]}
        if "token_ids" in choice:
            closing["token_ids"] = choice["token_ids"]
        prompt = {"prompt_token_ids": response["prompt_token_ids"]} if "prompt_token_ids" in response else {}
        frames = [_sse_data({**self._common, **prompt, "choices": [closing]})]
        if self.include_usage:
            frames.append(_sse_data({**self._common, "choices": [], "usage": response["usage"]}))
        frames.append(b"data: [DONE]\n\n")
        return frames


#: The fields of a Messages answer that Reef adds when a call asks for them (``anthropic.TOKEN_REQUEST_KEYS``).
ANTHROPIC_TOKEN_FIELDS = ("logprobs", "token_ids", "prompt_token_ids")


class _AnthropicEventWriter(_StreamWriter):
    """Encode one Anthropic Messages stream for ``ChatStreamRelay``.

    The message ID and model are strings. Input tokens are a nonnegative count. The block index counts closed
    blocks, starting at zero. The open block is absent, or a pair of a block kind and a call index. The kind is
    ``text``, ``thinking``, or ``tool``. Only a tool has a call index, its nonnegative position among the reply's calls.

    ``start`` emits the header. ``thinking`` and ``text`` use the current block when its kind matches. Otherwise
    they close it and open a block for the new kind. ``tool_call`` opens a block for a new call. An argument
    fragment for an announced call requires its block to remain open. ``finish`` closes the block and emits the
    stop reason, stop sequence, and output token count before ``message_stop``.
    """

    def __init__(self, message_id: str, *, model: str, input_tokens: int) -> None:
        require_integer(input_tokens, "the input token count")
        self._message_id = message_id
        self._model = model
        self._input_tokens = input_tokens
        self._block_index = 0
        self._open_block: tuple[str, int | None] | None = None

    def start(self) -> list[bytes]:
        return [
            _sse_event(
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
            _sse_event({"type": "content_block_start", "index": self._block_index, "content_block": content_block})
        )
        self._open_block = key
        return output

    def _close(self) -> list[bytes]:
        if self._open_block is None:
            return []
        output = [_sse_event({"type": "content_block_stop", "index": self._block_index})]
        self._block_index += 1
        self._open_block = None
        return output

    def _delta(self, delta: Mapping[str, Any]) -> bytes:
        return _sse_event({"type": "content_block_delta", "index": self._block_index, "delta": delta})

    def thinking(self, value: str) -> list[bytes]:
        output = self._open(("thinking", None), {"type": "thinking", "thinking": "", "signature": ""})
        output.append(self._delta({"type": "thinking_delta", "thinking": value}))
        return output

    def text(self, value: str) -> list[bytes]:
        output = self._open(("text", None), {"type": "text", "text": ""})
        output.append(self._delta({"type": "text_delta", "text": value}))
        return output

    def tool_call(self, index: int, call_id: str, name: str, arguments: str, *, announced: bool) -> list[bytes]:
        key = ("tool", index)
        if not announced and self._open_block != key:
            raise ValueError("streamed tool-call arguments arrived after the content block closed")
        output = self._open(key, {"type": "tool_use", "id": call_id, "name": name, "input": {}})
        if arguments:
            output.append(self._delta({"type": "input_json_delta", "partial_json": arguments}))
        return output

    def finish(self, response: Mapping[str, Any], *, output_tokens: int) -> list[bytes]:
        return [
            *self._close(),
            _sse_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": response["stop_reason"], "stop_sequence": response["stop_sequence"]},
                    "usage": {"output_tokens": output_tokens},
                    **{field: response[field] for field in ANTHROPIC_TOKEN_FIELDS if field in response},
                }
            ),
            _sse_event({"type": "message_stop"}),
        ]


class ChatStreamRelay:
    """Translate one model reply into protocol frames and check the completed parse.

    The writer encodes one protocol. The parser is absent or a ``StreamingToolParser``. The tool calls hold the
    indices, names, IDs, and argument fragments already sent. Text and reasoning fragments are nonempty strings
    in send order. Pending reasoning whitespace holds only whitespace after the last emitted reasoning character.

    ``piece`` takes one text or thinking delta. It emits no frame for empty input and holds trailing thinking
    whitespace until more reasoning arrives. For text with a parser, it emits the text and tool calls that the parser returns. A parser without an incremental method defers text and tool output to ``reconcile``.

    ``reconcile`` takes the completed assistant message. It emits missing suffixes and assigns the streamed IDs
    to matching completed calls. It raises ``ValueError`` if any emitted field and its completed value disagree
    or the protocol cannot append to an earlier block. The caller sends a terminal event only on success.
    """

    def __init__(self, writer: _StreamWriter, stream_tool_parser: StreamingToolParser | None) -> None:
        self.writer = writer
        self._parser = stream_tool_parser
        self._tools = _StreamedToolCalls()
        self.reasoning_fragments: list[str] = []
        self.reasoning_whitespace: list[str] = []
        self.text_fragments: list[str] = []

    def piece(self, kind: str, value: str) -> list[bytes]:
        if not value:
            return []
        if kind == "thinking":
            # The complete parse trims reasoning. Keep whitespace until another reasoning fragment follows it.
            if not self.reasoning_fragments:
                value = value.lstrip()
            trimmed = value.rstrip()
            if trimmed:
                delta = "".join(self.reasoning_whitespace) + trimmed
                self.reasoning_whitespace.clear()
                self.reasoning_fragments.append(delta)
                frames = self.writer.thinking(delta)
            else:
                frames = []
            if len(trimmed) < len(value):
                self.reasoning_whitespace.append(value[len(trimmed) :])
            return frames
        calls: list[StreamedToolCall]
        if self._parser is None:
            normal_text, calls = value, []
        else:
            normal_text, calls = self._parser.parse_stream_chunk(value)
        output: list[bytes] = []
        if normal_text:
            self.text_fragments.append(normal_text)
            output.extend(self.writer.text(normal_text))
        for call in calls:
            index = call.index
            announced = self._tools.announce(index, call.name)
            self._tools.add_arguments(index, call.arguments)
            output.extend(
                self.writer.tool_call(
                    index, self._tools.ids[index], self._tools.names[index], call.arguments, announced=announced
                )
            )
        return output

    def reconcile(self, message: dict[str, Any]) -> list[bytes]:
        """Send what the canonical parse of the full sample holds beyond what streamed.

        The message's tool calls are also given the ids the stream already
        announced, so the recorded response and the streamed one agree.
        """
        tool_calls = message.get("tool_calls") or []
        if any(index >= len(tool_calls) for index in self._tools.ids):
            raise ValueError("a streamed tool call is missing from the completed response")
        output: list[bytes] = []
        for name, field, fragments, write in (
            ("reasoning", "reasoning_content", self.reasoning_fragments, self.writer.thinking),
            ("text", "content", self.text_fragments, self.writer.text),
        ):
            final = message.get(field) or ""
            emitted = "".join(fragments)
            if not final.startswith(emitted):
                raise ValueError(f"streamed {name} differs from the completed response")
            if len(final) > len(emitted):
                output.extend(write(final[len(emitted) :]))
        for index, call in enumerate(tool_calls):
            function = call["function"]
            name = function["name"]
            announced = index in self._tools.ids
            if announced:
                if self._tools.names[index] != name:
                    raise ValueError("the streamed tool-call name differs from the completed response")
                call["id"] = self._tools.ids[index]
            arguments = argument_text(function["arguments"])
            emitted = "".join(self._tools.arguments.get(index, ()))
            missing = remaining_tool_arguments(emitted, arguments, allow_raw_prefix=self.writer.accepts_raw_arguments)
            if missing or not announced:
                output.extend(self.writer.tool_call(index, call["id"], name, missing, announced=not announced))
        return output


def argument_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def without_json_whitespace(value: str) -> str:
    """Remove JSON whitespace outside strings, including in a partial object."""
    inside_string = False
    escaped = False
    characters: list[str] = []
    for character in value:
        if not inside_string and character in " \t\r\n":
            continue
        characters.append(character)
        if escaped:
            escaped = False
        elif inside_string and character == "\\":
            escaped = True
        elif character == '"':
            inside_string = not inside_string
    return "".join(characters)


def remaining_tool_arguments(emitted: str, arguments: str, *, allow_raw_prefix: bool = False) -> str:
    """Return the suffix that completes emitted arguments, or raise ValueError on a mismatch.

    When allow_raw_prefix is true, a literal prefix match suffices even for invalid JSON.
    Other matches must complete a JSON object equal to the final object.
    """
    has_prefix = arguments.startswith(emitted)
    if allow_raw_prefix and has_prefix:
        return arguments[len(emitted) :]
    try:
        expected = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("completed tool-call arguments are not valid JSON") from exc
    if not isinstance(expected, dict):
        raise ValueError("completed tool-call arguments must be a JSON object")
    if has_prefix:
        return arguments[len(emitted) :]

    compact = without_json_whitespace(arguments)
    prefix = without_json_whitespace(emitted)
    suffix = compact[len(prefix) :] if compact.startswith(prefix) else ""
    try:
        actual = json.loads(emitted + suffix)
    except json.JSONDecodeError as exc:
        raise ValueError("streamed tool-call arguments cannot be completed") from exc
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("streamed tool-call arguments differ from the completed response")
    return suffix


@dataclass(frozen=True)
class ChatCall:
    """One validated chat request, rendered and ready for the engine's generate route."""

    anthropic: bool
    payload: Mapping[str, Any]
    request: dict[str, Any]
    prompt_ids: list[int]
    sampling_params: dict[str, Any]
    tool_parser: Any


def stream_ids() -> tuple[str, str, int]:
    """The chat id, message id and timestamp one streamed response shares across its frames."""
    return f"chatcmpl-{uuid.uuid4().hex}", f"msg_{uuid.uuid4().hex}", int(time.time())


def template_kwargs_of(value: Any, label: str) -> dict[str, Any]:
    """Validate chat template arguments: an object without ``tools``, which the top-level tools field owns."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if "tools" in value:
        raise ValueError(f"{label}.tools is reserved; use the top-level tools field")
    return dict(value)


#: The sampling fields that a scenario may set for the token-native chat handler (RFC #1097). OpenAI chat and
#: Anthropic messages requests both carry them, so a scenario's value applies alike to either protocol.
SERVING_SAMPLING_FIELDS = ("temperature", "top_p", "top_k")


class TokenNativeChatHandler(HttpInferenceHandler):
    """Serve OpenAI or Anthropic chat over an engine's token-native generate API.

    Subclasses supply the :class:`NativeGenerateClient` for their engine and
    may override :meth:`inference_stream` with an incremental relay. The
    default streams the buffered result as one burst of protocol frames, so
    clients that requested a stream still receive a valid one.
    """

    #: A harness that parses raw thinking tags itself opts out by overriding
    #: this class attribute with False, which keeps the tags in the content.
    SPLIT_REASONING = True
    # Set when the chat template pre-opens ``<think>``: the sample then
    # carries only the closing tag, so a sample with no ``</think>`` is
    # reasoning that ran out of tokens, not an answer. Read from the end of
    # each request's rendered prompt; ``force_reasoning`` in the handler
    # config pins it either way.
    force_reasoning: bool | None = None

    @classmethod
    def from_config(
        cls, upstream_url: str, *, model_path: str, timeout_s: float, **config: Any
    ) -> TokenNativeChatHandler:
        """Construct the tokenizer-aware handler selected by deployment configuration."""
        return cls(upstream_url, model_path=model_path, timeout_s=timeout_s, **config)

    def __init__(
        self,
        upstream_url: str,
        *,
        client: NativeGenerateClient,
        model_path: str,
        timeout_s: float = 300.0,
        tokenizer: Any = None,
        tool_call_parser: str | None = None,
        sampling_defaults: Mapping[str, Any] | None = None,
        force_reasoning: bool | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        request_headers: RequestHeadersFactory | None = None,
    ) -> None:
        super().__init__(
            upstream_url, request_headers=request_headers, timeout_s=timeout_s, error_label=client.error_label
        )
        if tokenizer is None and not model_path:
            raise ValueError("token-native chat requires model_path")
        self._client = client
        self._model_path = model_path
        self._tokenizer = tokenizer
        self._tool_call_parser = tool_call_parser.strip() if tool_call_parser else None
        # Sampling values applied when the client omits a parameter. Native
        # generate routes have engine defaults (temperature 1.0), NOT the
        # model generation_config the chat endpoint would use — a recipe
        # that reproduces a reference run pins its effective temperature
        # here (e.g. OpenClaw-RL: 0.6/0.95/20).
        self._sampling_defaults = dict(sampling_defaults or {})
        # None = read the end of each rendered prompt.
        self.force_reasoning = force_reasoning
        # Chat template arguments for a request that omits them, for example
        # {"enable_thinking": false}. A request's own chat_template_kwargs win key by key.
        self._chat_template_kwargs = template_kwargs_of(chat_template_kwargs, "chat_template_kwargs")

    @property
    def client(self) -> NativeGenerateClient:
        return self._client

    @property
    def model_path(self) -> str:
        return self._model_path

    def serving_config(self, override: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """The sampling fields ``temperature``, ``top_p`` and ``top_k``, the ones that both chat protocols carry:
        the override's value of each, else the handler's ``sampling_defaults`` value.

        ``temperature`` is a number of at least 0, ``top_p`` a number in (0, 1], and ``top_k`` -1 (no limit) or a
        positive integer. ``ValueError`` for another field or value."""
        unknown = sorted(set(override) - set(SERVING_SAMPLING_FIELDS))
        if unknown:
            raise ValueError(
                f"unknown serving fields {unknown}: the chat handler takes {list(SERVING_SAMPLING_FIELDS)}"
            )
        for name, value in override.items():
            number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            if name == "temperature" and not (number is not None and number >= 0):
                raise ValueError("serving field temperature must be a number of at least 0")
            if name == "top_p" and not (number is not None and 0 < number <= 1):
                raise ValueError("serving field top_p must be a number in (0, 1]")
            if name == "top_k" and not (isinstance(number, int) and (number == -1 or number > 0)):
                raise ValueError("serving field top_k must be -1 or a positive integer")
        return {
            name: override.get(name, self._sampling_defaults.get(name))
            for name in SERVING_SAMPLING_FIELDS
            if name in override or name in self._sampling_defaults
        }

    def request_payload(
        self, payload: Mapping[str, Any], *, serving_config: Mapping[str, JsonValue]
    ) -> dict[str, Any]:
        """``payload`` with each sampling field of ``serving_config`` that the client did not set: the scenario's
        value is a default, as ``sampling_defaults`` is."""
        body = dict(payload)
        for name in SERVING_SAMPLING_FIELDS:
            if name in serving_config and body.get(name) is None:
                body[name] = serving_config[name]
        return body

    def address_model(self, payload: Mapping[str, Any], model_name: str) -> dict[str, Any]:
        """The engine serves the base model under its path and each adapter under its own name, which a native
        request names in ``lora_path``."""
        pinned = parse_full_weight_name(model_name)
        if pinned is not None:
            model_name = self.model_path
        addressed = {key: value for key, value in payload.items() if key != "lora_path"}
        if model_name == self.model_path:
            return {**addressed, "model": model_name}
        return {**addressed, "model": model_name, "lora_path": model_name}

    # -- Buffered requests ------------------------------------------------------

    async def inference(self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == ANTHROPIC_COUNT_TOKENS_PATH:
            return self._anthropic_count_tokens(artifact, payload)
        if path not in {CHAT_COMPLETIONS_PATH, ANTHROPIC_MESSAGES_PATH}:
            raise ValueError(
                "chat training inference supports only "
                f"{CHAT_COMPLETIONS_PATH}, {ANTHROPIC_MESSAGES_PATH}, and {ANTHROPIC_COUNT_TOKENS_PATH}"
            )
        call = self._chat_call(path, payload)
        response, provider_response = await self._complete(artifact, call)
        return provider_response if call.anthropic else response

    async def _complete(self, artifact: InferenceArtifact, call: ChatCall) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one buffered generation; return the OpenAI-shaped response and the provider-shaped one."""
        native = await super().inference(
            artifact,
            self._client.generate_path,
            self._client.payload(call.request, call.prompt_ids, call.sampling_params, stream=False),
        )
        captured = self._client.parse(native)
        response = self._response_from_capture(call, captured, call.tool_parser)
        if not call.anthropic:
            return response, response
        return response, anthropic_response(call.payload, response)

    def _chat_call(self, path: str, payload: dict[str, Any]) -> ChatCall:
        """Normalize the provider request and render the prompt once for either endpoint."""
        anthropic = path == ANTHROPIC_MESSAGES_PATH
        request = anthropic_request(payload) if anthropic else dict(payload)
        if request.get("n", 1) != 1:
            raise ValueError("token-native chat requires n=1")
        return ChatCall(
            anthropic=anthropic,
            payload=payload,
            request=request,
            prompt_ids=self._render_prompt(request),
            sampling_params=self._client.sampling_params(request, self._sampling_defaults),
            tool_parser=self._configured_tool_parser(request),
        )

    def _response_from_capture(self, call: ChatCall, captured: CapturedGeneration, tool_parser: Any) -> dict[str, Any]:
        """The OpenAI-shaped response of one exact sample."""
        text = captured.text
        if text is None:
            text = self._require_tokenizer().decode(captured.output_ids, skip_special_tokens=True)
        message, parsed_tool_calls = self._assistant_message(
            text, tool_parser, force_reasoning=self.reasoning_is_pre_opened(call.prompt_ids)
        )
        # Tool parsing must not hide an engine abort from the service's abort check.
        if parsed_tool_calls and captured.finish_reason != "abort":
            finish_reason = "tool_calls"
        else:
            finish_reason = captured.finish_reason
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "meta_info": dict(captured.public_meta),
        }
        if call.request.get("logprobs") is True:
            choice["logprobs"] = {"content": self._openai_logprobs(captured.output_ids, captured.rollout_log_probs)}
        # vLLM's ``return_token_ids`` extension: the rendered prompt's ids and the sampled ids, as the client asked.
        return_token_ids = call.request.get("return_token_ids") is True
        if return_token_ids:
            choice["token_ids"] = list(captured.output_ids)
        response: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(call.request.get("model", self._model_path)),
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(call.prompt_ids),
                "completion_tokens": len(captured.output_ids),
                "total_tokens": len(call.prompt_ids) + len(captured.output_ids),
            },
        }
        if return_token_ids:
            response["prompt_token_ids"] = list(call.prompt_ids)
        return response

    # -- Streaming --------------------------------------------------------------

    def _validate_stream_path(self, path: str) -> None:
        if path == ANTHROPIC_COUNT_TOKENS_PATH:
            raise ValueError("Anthropic count_tokens does not support streaming")
        if path not in {CHAT_COMPLETIONS_PATH, ANTHROPIC_MESSAGES_PATH}:
            raise ValueError(
                "chat training inference supports streaming only for "
                f"{CHAT_COMPLETIONS_PATH} and {ANTHROPIC_MESSAGES_PATH}"
            )

    def _stream_writer(self, call: ChatCall, chat_id: str, message_id: str, created: int) -> _StreamWriter:
        if call.anthropic:
            return _AnthropicEventWriter(
                message_id,
                model=str(call.payload.get("model", self._model_path)),
                input_tokens=len(call.prompt_ids),
            )
        options = call.request.get("stream_options")
        return _OpenAIChunkWriter(
            chat_id,
            model=str(call.request.get("model", self._model_path)),
            created=created,
            include_usage=isinstance(options, Mapping) and options.get("include_usage") is True,
        )

    def _finish_frames(
        self,
        call: ChatCall,
        relay: ChatStreamRelay,
        response: dict[str, Any],
        chat_id: str,
        message_id: str,
        created: int,
    ) -> tuple[dict[str, Any], list[bytes]]:
        """Reconcile the canonical response with what streamed; return the record and closing frames."""
        response["id"] = chat_id
        response["created"] = created
        frames = relay.reconcile(response["choices"][0]["message"])
        output_tokens = response["usage"]["completion_tokens"]
        if call.anthropic:
            provider_response = anthropic_response(call.payload, response)
            provider_response["id"] = message_id
        else:
            provider_response = response
        frames.extend(relay.writer.finish(provider_response, output_tokens=output_tokens))
        return provider_response, frames

    async def inference_stream(
        self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]
    ) -> InferenceStream:
        """Stream the buffered result as one burst of protocol frames.

        The engine is asked for a buffered generation, so the exact sample is
        recorded atomically; the client receives the whole turn at once but
        in the stream format it asked for.
        """
        self._validate_stream_path(path)
        call = self._chat_call(path, payload)
        response, _ = await self._complete(artifact, call)
        chat_id, message_id, created = stream_ids()
        relay = ChatStreamRelay(self._stream_writer(call, chat_id, message_id, created), None)
        frames = list(relay.writer.start())
        if not call.anthropic and call.request.get("logprobs") is True:
            logprobs = response["choices"][0].get("logprobs")
            if isinstance(logprobs, Mapping):
                frames.extend(relay.writer.logprobs(list(logprobs["content"])))
        provider_response, closing = self._finish_frames(call, relay, response, chat_id, message_id, created)
        frames.extend(closing)

        async def chunks() -> AsyncIterator[bytes]:
            for frame in frames:
                yield frame

        return InferenceStream(
            status=200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            chunks=chunks(),
            record_response=provider_response,
        )

    # -- Anthropic Messages -----------------------------------------------------

    def _anthropic_count_tokens(self, artifact: InferenceArtifact, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Count the exact templated input."""
        request = anthropic_request({**payload, "max_tokens": 1})
        return {"input_tokens": len(self._render_prompt(request))}

    # -- Prompt rendering and tool parsing -------------------------------------

    def _require_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        return self._tokenizer

    def _render_prompt(self, payload: Mapping[str, Any]) -> list[int]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("chat training requests require a non-empty messages list")
        if any(
            not isinstance(message, Mapping)
            or not isinstance(message.get("role"), str)
            or (message.get("content") is not None and not isinstance(message.get("content"), str))
            for message in messages
        ):
            raise ValueError("exact chat capture currently requires text-only message content")
        template_kwargs = template_kwargs_of(payload.get("chat_template_kwargs"), "chat_template_kwargs")
        template_messages = self._template_messages(messages)
        tools = self._request_tools(payload)
        render_kwargs = {**self._chat_template_kwargs, **template_kwargs}
        if tools is not None and payload.get("tool_choice") != "none":
            render_kwargs["tools"] = tools
        tokenizer = self._require_tokenizer()
        rendered = tokenizer.apply_chat_template(
            template_messages,
            tokenize=True,
            add_generation_prompt=True,
            **render_kwargs,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered.get("input_ids")
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if isinstance(rendered, list) and rendered and isinstance(rendered[0], list):
            if len(rendered) != 1:
                raise ValueError("chat template unexpectedly returned a batched token tensor")
            rendered = rendered[0]
        return integer_tokens(rendered, "chat template input_ids")

    def _configured_tool_parser(self, request: Mapping[str, Any]) -> Any:
        tools = self._request_tools(request)
        if tools is None or request.get("tool_choice") == "none":
            return None
        if not self._tool_call_parser:
            raise ValueError(
                "tool calls require inference_handler_config.tool_call_parser to match the engine's configuration"
            )
        return self._client.tool_parser(tools, self._tool_call_parser, self._require_tokenizer())

    @staticmethod
    def _request_tools(request: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        tools = request.get("tools")
        if tools is None:
            return None
        if not isinstance(tools, list) or not tools or any(not isinstance(tool, Mapping) for tool in tools):
            raise ValueError("tools must be a non-empty list of objects")
        return [dict(tool) for tool in tools]

    @staticmethod
    def _template_messages(messages: list[Any]) -> list[dict[str, Any]]:
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

    def reasoning_is_pre_opened(self, prompt_ids: list[int]) -> bool:
        """Whether the rendered prompt ends with an open ``<think>`` for the model.

        Some templates end the generation prompt with ``<think>``: Qwen3.5 and
        Qwen3.8 by default, Qwen3-*-Thinking always. The sample then carries
        only the closing tag, so a sample without one is truncated reasoning.
        Hybrid templates close the tag at once for ``enable_thinking: false``,
        so a sample without a tag is then the answer. The check reads the
        request's own prompt, so it follows its ``chat_template_kwargs``,
        tools and messages. An explicit ``force_reasoning`` in the handler
        config wins.
        """
        if self.force_reasoning is not None:
            return self.force_reasoning
        tail = self._require_tokenizer().decode(prompt_ids[-PROMPT_TAIL_TOKEN_COUNT:], skip_special_tokens=False)
        return is_reasoning_pre_opened(tail)

    @classmethod
    def _assistant_message(
        cls,
        text: str,
        tool_parser: Any,
        *,
        force_reasoning: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        # Thinking-template models emit reasoning before </think>. The
        # opening tag is often inserted by the chat template, so it is
        # usually absent from the sampled text. OpenAI-shaped clients render
        # content verbatim, so reasoning must ride the standard
        # reasoning_content field — otherwise every consumer downstream
        # (agents, judges, user simulators) sees chain-of-thought as the
        # reply. Training tensors are captured from the raw token stream and
        # are unaffected by this presentation split. The split follows
        # SGLang's Qwen3 reasoning parser (``split_reasoning``). SGLang forces
        # reasoning whenever enable_thinking is not false, so a closing tag
        # also starts the split: it only appears when thinking is on. A reply
        # without a closing tag hit its token cap mid-thought, so all of it is
        # reasoning: a judge that scored it as content would score the wrong text.
        reasoning = ""
        if cls.SPLIT_REASONING:
            reasoning, text = split_reasoning(text, pre_opened=force_reasoning)
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
        for call in parsed:
            arguments = call.parameters
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            )
        message["content"] = remaining_text or None
        message["tool_calls"] = tool_calls
        return message, True

    def _openai_logprobs(self, token_ids: list[int], log_probs: list[float]) -> list[dict[str, Any]]:
        tokenizer = self._require_tokenizer()
        values = []
        for token_id, log_prob in zip(token_ids, log_probs, strict=True):
            token = tokenizer.decode([token_id], skip_special_tokens=False)
            selected = {
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": log_prob,
            }
            values.append(
                {
                    **selected,
                    "top_logprobs": [selected],
                }
            )
        return values


def integer_tokens(value: Any, label: str) -> list[int]:
    """Validate a non-empty list of token ids."""
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(token, int) or isinstance(token, bool) for token in value)
    ):
        raise ValueError(f"{label} must be a non-empty integer list")
    return list(value)


def finite_log_prob(value: Any, label: str) -> float:
    """Validate one rollout log probability."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


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
    "ANTHROPIC_COUNT_TOKENS_PATH",
    "ANTHROPIC_MESSAGES_PATH",
    "CHAT_COMPLETIONS_PATH",
    "NATIVE_REQUEST_KEYS",
    "CapturedGeneration",
    "ChatCall",
    "ChatStreamRelay",
    "NativeGenerateClient",
    "TokenNativeChatHandler",
    "finite_log_prob",
    "integer_tokens",
    "normalize_finish_reason",
    "positive_max_tokens",
    "sse_json_events",
    "stream_ids",
]
