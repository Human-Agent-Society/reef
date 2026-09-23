"""OpenAI and Anthropic chat facades over an engine's token-native generate API.

Clients use either ``/v1/chat/completions`` or ``/v1/messages`` while Reef
normalizes the provider request, renders the prompt once, calls the selected
engine's native generate route, and records the exact sampled token ids beside
their rollout log probabilities and the weight version that produced each
token. Decoded text is only the client-facing view; it is never re-tokenized
to construct a training sample.

The engine-specific part is a :class:`NativeGenerateClient`: it shapes the
native request and parses the native response into a
:class:`CapturedGeneration`. Everything else here is shared by every engine.
"""

from __future__ import annotations

import codecs
import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact
from reef.inference.http import HttpInferenceHandler
from reef.runtime.interfaces import InferenceStream

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"

#: Request keys forwarded from the provider request to the native payload
#: unchanged; each engine reads the ones it understands.
NATIVE_REQUEST_KEYS = ("lora_path", "rid", "sglang_sampling_params")


@dataclass(frozen=True)
class CapturedGeneration:
    """One request's exact engine output, in the shape training records need.

    ``token_runtime_load_ids`` names the weight version that produced each
    sampled token; a request paused for a weight update and resumed spans two.
    ``text`` is the engine's own decoding when it provides one; ``None`` makes
    the handler decode the ids with its tokenizer. ``public_meta`` is the
    engine metadata exposed to clients as ``choice["meta_info"]``; it must not
    carry tensors.
    """

    output_ids: list[int]
    rollout_log_probs: list[float]
    token_runtime_load_ids: list[str]
    finish_reason: str
    text: str | None
    public_meta: dict[str, Any]
    topk_indices: list[list[int]] | None = None
    topk_log_probs: list[list[float]] | None = None

    def __post_init__(self) -> None:
        count = len(self.output_ids)
        if len(self.rollout_log_probs) != count or len(self.token_runtime_load_ids) != count:
            raise ValueError("captured response tensors have inconsistent lengths")
        if not count and self.finish_reason != "abort":
            raise ValueError("captured response has no sampled tokens")
        if self.finish_reason not in {"stop", "length", "abort"}:
            raise ValueError(f"captured finish reason must be stop, length or abort, got {self.finish_reason!r}")
        if (self.topk_indices is None) != (self.topk_log_probs is None):
            raise ValueError("captured top-k indices and log-probs must be given together")


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
        capture_topk: int,
        stream: bool,
    ) -> dict[str, Any]:
        """The native request body for ``prompt_ids``, asking for log-probs and, if ``capture_topk``, top-k."""

    @abstractmethod
    def parse(self, artifact: Artifact, response: Mapping[str, Any], *, capture_topk: int) -> CapturedGeneration:
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
    """Decode arbitrarily chunked SSE bytes into JSON data events."""

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    def complete_events(*, final: bool = False) -> tuple[list[str], str]:
        nonlocal buffer
        # Engines use LF, but accepting CRLF keeps the proxy correct behind
        # HTTP middleware that normalizes line endings.
        buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
        blocks = buffer.split("\n\n")
        buffer = blocks.pop() if not final else ""
        payloads = []
        for block in blocks:
            data = [line[5:].removeprefix(" ") for line in block.split("\n") if line.startswith("data:")]
            if data:
                payloads.append("\n".join(data))
        return payloads, buffer

    def decode_event(payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("native stream event must be a JSON object")
        if isinstance(value.get("error"), Mapping):
            raise ValueError(f"native stream failed: {value['error'].get('message', value['error'])}")
        return value

    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        payloads, _ = complete_events()
        for payload in payloads:
            if payload == "[DONE]":
                return
            yield decode_event(payload)

    buffer += decoder.decode(b"", final=True)
    payloads, _ = complete_events(final=True)
    for payload in payloads:
        if payload == "[DONE]":
            return
        yield decode_event(payload)


class ReasoningStreamSplitter:
    """Split sampled thinking tags without waiting for the full completion."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, *, enabled: bool, force_reasoning: bool) -> None:
        self._mode = "thinking" if enabled and force_reasoning else ("undecided" if enabled else "text")
        self._buffer = ""

    @staticmethod
    def _held_suffix(value: str, delimiter: str) -> int:
        for size in range(min(len(value), len(delimiter) - 1), 0, -1):
            if delimiter.startswith(value[-size:]):
                return size
        return 0

    def feed(self, delta: str) -> list[tuple[str, str]]:
        if not delta:
            return []
        if self._mode == "text":
            return [("text", delta)]
        self._buffer += delta
        if self._mode == "undecided":
            if self._OPEN.startswith(self._buffer) and len(self._buffer) < len(self._OPEN):
                return []
            if self._buffer.startswith(self._OPEN):
                self._buffer = self._buffer[len(self._OPEN) :]
                self._mode = "thinking"
            else:
                value, self._buffer = self._buffer, ""
                self._mode = "text"
                return [("text", value)]

        before, separator, after = self._buffer.partition(self._CLOSE)
        if separator:
            self._buffer = ""
            self._mode = "text"
            parts = [("thinking", before)] if before else []
            visible = after.lstrip("\n")
            if visible:
                parts.append(("text", visible))
            return parts
        held = self._held_suffix(self._buffer, self._CLOSE)
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
        kind = "thinking" if self._mode == "thinking" else "text"
        return [(kind, value)]


def _sse_data(event: Mapping[str, Any]) -> bytes:
    """One OpenAI-style SSE frame: a bare ``data:`` line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _sse_event(event: Mapping[str, Any]) -> bytes:
    """One Anthropic-style SSE frame: the event type names the payload's ``type``."""
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


class _StreamedToolCalls:
    """Tool calls seen so far in one stream: their ids, names and the argument text already sent."""

    def __init__(self) -> None:
        self.ids: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.arguments: dict[int, str] = {}

    def announce(self, index: int, name: Any) -> bool:
        """Record a call's name on first sight; returns whether this chunk carried the name."""
        if name:
            self.ids.setdefault(index, _new_call_id())
            self.names[index] = str(name)
        if index not in self.ids:
            raise ValueError("tool stream emitted arguments before the tool name")
        return bool(name)

    def add_arguments(self, index: int, arguments: str) -> None:
        self.arguments[index] = self.arguments.get(index, "") + arguments


class _StreamWriter(ABC):
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


class _OpenAIChunkWriter(_StreamWriter):
    """``chat.completion.chunk`` events sharing one id, model and timestamp."""

    def __init__(self, common: Mapping[str, Any]) -> None:
        self._common = dict(common)

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
        choice = response["choices"][0]
        return [
            self._chunk({}, finish_reason=choice["finish_reason"], meta_info=choice["meta_info"]),
            b"data: [DONE]\n\n",
        ]


class _AnthropicEventWriter(_StreamWriter):
    """Anthropic Messages events; content blocks open and close as the sampled text changes kind."""

    def __init__(self, message_id: str, *, model: str, input_tokens: int) -> None:
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
        output = self._open(("tool", index), {"type": "tool_use", "id": call_id, "name": name, "input": {}})
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
                }
            ),
            _sse_event({"type": "message_stop"}),
        ]


class ChatStreamRelay:
    """Turn one request's sampled deltas into client frames and remember what was sent.

    Tool markers are parsed out of the visible text as it streams; at the end
    :meth:`reconcile` compares the canonical parse of the full sample with
    what already went out and sends only the difference.
    """

    def __init__(self, writer: _StreamWriter, stream_tool_parser: Any) -> None:
        self.writer = writer
        self._parser = stream_tool_parser
        self._tools = _StreamedToolCalls()
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
            arguments = _argument_text(getattr(call, "parameters", ""))
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
            call["id"] = self._tools.ids.setdefault(index, str(call.get("id") or _new_call_id()))
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
            arguments = _argument_text(function.get("arguments", ""))
            emitted = self._tools.arguments.get(index, "")
            missing = arguments[len(emitted) :] if arguments.startswith(emitted) else arguments
            if not missing and index in self._tools.arguments:
                continue
            output.extend(
                self.writer.tool_call(index, self._tools.ids[index], self._tools.names[index], missing, announced=True)
            )
        return output


def _argument_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


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


class TokenNativeChatHandler(HttpInferenceHandler):
    """Serve OpenAI or Anthropic chat with engine-native policy tensors.

    Subclasses supply the :class:`NativeGenerateClient` for their engine and
    may override :meth:`inference_stream` with an incremental relay. The
    default streams the buffered result as one burst of protocol frames, so
    clients that requested a stream still receive a valid one.
    """

    # A harness that parses raw thinking tags itself opts out by overriding
    # this class attribute.
    SPLIT_REASONING = True
    # Set when the chat template pre-opens ``<think>``: the sample then
    # carries only the closing tag, so a sample with no ``</think>`` is
    # reasoning that ran out of tokens, not an answer. Sniffed from the
    # rendered template at first use; ``force_reasoning`` in the handler
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
        capture_topk: int = 0,
        sampling_defaults: Mapping[str, Any] | None = None,
        force_reasoning: bool | None = None,
    ) -> None:
        super().__init__(upstream_url, timeout_s=timeout_s, error_label=client.error_label)
        if tokenizer is None and not model_path:
            raise ValueError("chat training inference requires model_path")
        self._client = client
        self._model_path = model_path
        self._tokenizer = tokenizer
        self._tool_call_parser = tool_call_parser.strip() if tool_call_parser else None
        # OpenClaw-RL top-K objective: capture the generation-time top-K
        # vocab log-probs per sampled token (the old-policy ``ell_old`` on
        # S^q) into ``response.training``. 0 disables the capture.
        self._capture_topk = int(capture_topk)
        # Sampling values applied when the client omits a parameter. Native
        # generate routes have engine defaults (temperature 1.0), NOT the
        # model generation_config the chat endpoint would use — a recipe
        # that reproduces a reference run pins its effective temperature
        # here (e.g. OpenClaw-RL: 0.6/0.95/20).
        self._sampling_defaults = dict(sampling_defaults or {})
        # None = sniff the chat template on first render.
        self.force_reasoning = force_reasoning

    @property
    def client(self) -> NativeGenerateClient:
        return self._client

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def capture_topk(self) -> int:
        return self._capture_topk

    # -- Buffered requests ------------------------------------------------------

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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

    async def _complete(self, artifact: Artifact, call: ChatCall) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one buffered generation; return the OpenAI-shaped response and the provider-shaped one."""
        native = await super().inference(
            artifact,
            self._client.generate_path,
            self._client.payload(
                call.request, call.prompt_ids, call.sampling_params, capture_topk=self._capture_topk, stream=False
            ),
        )
        captured = self._client.parse(artifact, native, capture_topk=self._capture_topk)
        response = self._response_from_capture(call, captured, call.tool_parser)
        if not call.anthropic:
            return response, response
        return response, self._anthropic_response(call.payload, call.request, response)

    def _chat_call(self, path: str, payload: dict[str, Any]) -> ChatCall:
        """Normalize the provider request and render the prompt once for either endpoint."""
        anthropic = path == ANTHROPIC_MESSAGES_PATH
        request = self._anthropic_request(payload) if anthropic else dict(payload)
        if request.get("n", 1) != 1:
            raise ValueError("exact training capture currently requires n=1")
        return ChatCall(
            anthropic=anthropic,
            payload=payload,
            request=request,
            prompt_ids=self._render_prompt(request),
            sampling_params=self._client.sampling_params(request, self._sampling_defaults),
            tool_parser=self._configured_tool_parser(request),
        )

    def _response_from_capture(self, call: ChatCall, captured: CapturedGeneration, tool_parser: Any) -> dict[str, Any]:
        """The OpenAI-shaped response, with training tensors, for one exact sample."""
        text = captured.text
        if text is None:
            text = self._require_tokenizer().decode(captured.output_ids, skip_special_tokens=True)
        message, parsed_tool_calls = self._assistant_message(
            text, tool_parser, force_reasoning=self.reasoning_is_pre_opened()
        )
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if parsed_tool_calls else captured.finish_reason,
            "meta_info": dict(captured.public_meta),
        }
        if call.request.get("logprobs") is True:
            choice["logprobs"] = {"content": self._openai_logprobs(captured.output_ids, captured.rollout_log_probs)}
        runtime_load_spans = self._runtime_load_spans(captured.token_runtime_load_ids)
        versions = {span["runtime_load_id"] for span in runtime_load_spans}
        runtime_load_id = versions.pop() if len(versions) == 1 else None
        training: dict[str, Any] = {
            "tokens": [*call.prompt_ids, *captured.output_ids],
            "loss_mask": [1] * len(captured.output_ids),
            "rollout_log_probs": list(captured.rollout_log_probs),
            "prompt_length": len(call.prompt_ids),
            "response_length": len(captured.output_ids),
            "runtime_load_id": None if runtime_load_id is None else str(runtime_load_id),
            "runtime_load_spans": runtime_load_spans,
        }
        if captured.topk_indices is not None and captured.topk_log_probs is not None:
            training["topk_indices"] = captured.topk_indices
            training["topk_log_probs"] = captured.topk_log_probs
        return {
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
            "training": training,
        }

    @staticmethod
    def _runtime_load_spans(token_runtime_load_ids: list[str]) -> list[dict[str, Any]]:
        """Collapse one version per token into contiguous ``[start, end)`` spans."""
        spans: list[dict[str, Any]] = []
        for index, version in enumerate(token_runtime_load_ids):
            if spans and spans[-1]["runtime_load_id"] == version:
                spans[-1]["end"] = index + 1
            else:
                spans.append({"start": index, "end": index + 1, "runtime_load_id": version})
        return spans

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
        return _OpenAIChunkWriter(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": str(call.request.get("model", self._model_path)),
            }
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
            provider_response = self._anthropic_response(call.payload, call.request, response)
            provider_response["id"] = message_id
        else:
            provider_response = response
        frames.extend(relay.writer.finish(provider_response, output_tokens=output_tokens))
        return provider_response, frames

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
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

    def _anthropic_count_tokens(self, artifact: Artifact, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Count the exact templated input without creating policy tensors."""
        request = self._anthropic_request({**payload, "max_tokens": 1})
        prompt_ids = self._render_prompt(request)
        runtime_load_id = getattr(artifact.ref, "runtime_load_id", None) or artifact.ref.release_id
        return {
            "input_tokens": len(prompt_ids),
            # RequestService strips this block from the client response. It
            # supplies the producing version required by durable-request validation, but
            # deliberately contains no tokens/loss mask/log-probabilities, so
            # a token-count call can never become a policy sample.
            "training": {
                "runtime_load_id": str(runtime_load_id),
                "request_messages": list(request.get("messages") or []),
                "request_tools": request.get("tools"),
            },
        }

    @classmethod
    def _anthropic_request(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize an Anthropic Messages request to the internal chat shape.

        The normalized request is what the tokenizer sees and what the private
        training record retains. Keeping this boundary provider-independent is
        important for multi-turn policy assembly and OpenClaw-RL's judge, both
        of which must render the exact same transcript as the rollout backend.
        """

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Anthropic messages requires a non-empty model")
        max_tokens = payload.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("Anthropic messages max_tokens must be a positive integer")

        messages: list[dict[str, Any]] = []
        system = cls._anthropic_text_content(payload.get("system"), "system", allow_missing=True)
        if system:
            messages.append({"role": "system", "content": system})

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("Anthropic messages requires a non-empty messages list")
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise ValueError(f"Anthropic message at index {index} must be an object")
            role = raw_message.get("role")
            if role == "user":
                messages.extend(cls._anthropic_user_messages(raw_message.get("content"), index))
            elif role == "assistant":
                messages.append(cls._anthropic_assistant_message(raw_message.get("content"), index))
            elif role == "system":
                text = cls._anthropic_text_content(raw_message.get("content"), f"message {index} system")
                messages.append({"role": "system", "content": text})
            else:
                raise ValueError(f"Anthropic message at index {index} has unsupported role {role!r}")

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        for key in ("temperature", "top_p", "top_k", "stream", "chat_template_kwargs", *NATIVE_REQUEST_KEYS):
            if key in payload and payload[key] is not None:
                request[key] = payload[key]
        stop_sequences = payload.get("stop_sequences")
        if stop_sequences is not None:
            if not isinstance(stop_sequences, list) or any(
                not isinstance(value, str) or not value for value in stop_sequences
            ):
                raise ValueError("Anthropic stop_sequences must be a list of non-empty strings")
            request["stop"] = list(stop_sequences)

        tools = cls._anthropic_tools(payload.get("tools"))
        if tools is not None:
            request["tools"] = tools
            request["tool_choice"] = cls._anthropic_tool_choice(payload.get("tool_choice"))
        elif payload.get("tool_choice") is not None:
            raise ValueError("Anthropic tool_choice requires tools")
        return request

    @classmethod
    def _anthropic_user_messages(cls, content: Any, message_index: int) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if not isinstance(content, list):
            raise ValueError(f"Anthropic user message {message_index} content must be text or content blocks")

        normalized: list[dict[str, Any]] = []
        text_parts: list[str] = []

        def flush_text() -> None:
            if text_parts:
                normalized.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts.clear()

        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                raise ValueError(
                    f"Anthropic user message {message_index} content block {block_index} must be an object"
                )
            kind = block.get("type")
            if kind == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("Anthropic text blocks require string text")
                text_parts.append(text)
                continue
            if kind != "tool_result":
                raise ValueError(
                    f"Anthropic content block type {kind!r} is not supported by exact text-token training"
                )
            flush_text()
            tool_call_id = block.get("tool_use_id", block.get("id"))
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("Anthropic tool_result requires tool_use_id")
            tool_content = cls._anthropic_text_content(
                block.get("content"),
                f"tool_result {tool_call_id}",
                allow_missing=True,
            )
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content or "",
                }
            )
        flush_text()
        if not normalized:
            normalized.append({"role": "user", "content": ""})
        return normalized

    @classmethod
    def _anthropic_assistant_message(cls, content: Any, message_index: int) -> dict[str, Any]:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        if not isinstance(content, list):
            raise ValueError(f"Anthropic assistant message {message_index} content must be text or content blocks")
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                raise ValueError(
                    f"Anthropic assistant message {message_index} content block {block_index} must be an object"
                )
            kind = block.get("type")
            if kind == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("Anthropic text blocks require string text")
                text_parts.append(text)
            elif kind == "thinking":
                thinking = block.get("thinking")
                if not isinstance(thinking, str):
                    raise ValueError("Anthropic thinking blocks require string thinking")
                reasoning_parts.append(thinking)
            elif kind == "redacted_thinking":
                raise ValueError("Anthropic redacted_thinking history is not supported")
            elif kind == "tool_use":
                tool_id = block.get("id")
                name = block.get("name")
                arguments = block.get("input", {})
                if not isinstance(tool_id, str) or not tool_id or not isinstance(name, str) or not name:
                    raise ValueError("Anthropic tool_use requires non-empty id and name")
                if not isinstance(arguments, Mapping):
                    raise ValueError("Anthropic tool_use input must be an object")
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(dict(arguments), ensure_ascii=False),
                        },
                    }
                )
            else:
                raise ValueError(f"Anthropic assistant content block type {kind!r} is not supported by exact training")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if reasoning_parts:
            message["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    @staticmethod
    def _anthropic_text_content(content: Any, label: str, *, allow_missing: bool = False) -> str:
        if content is None and allow_missing:
            return ""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise ValueError(f"Anthropic {label} content must be text or text blocks")
        texts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "text" or not isinstance(block.get("text"), str):
                raise ValueError(f"Anthropic {label} supports only text blocks")
            texts.append(block["text"])
        return "\n".join(texts)

    @staticmethod
    def _anthropic_tools(value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError("Anthropic tools must be a non-empty list of objects")
        tools: list[dict[str, Any]] = []
        for index, tool in enumerate(value):
            if not isinstance(tool, Mapping):
                raise ValueError(f"Anthropic tool at index {index} must be an object")
            name = tool.get("name")
            schema = tool.get("input_schema")
            if not isinstance(name, str) or not name or not isinstance(schema, Mapping):
                raise ValueError("Anthropic custom tools require non-empty name and input_schema")
            function: dict[str, Any] = {"name": name, "parameters": dict(schema)}
            description = tool.get("description")
            if description is not None:
                if not isinstance(description, str):
                    raise ValueError("Anthropic tool description must be a string")
                function["description"] = description
            tools.append({"type": "function", "function": function})
        return tools

    @staticmethod
    def _anthropic_tool_choice(value: Any) -> Any:
        if value is None:
            return "auto"
        if not isinstance(value, Mapping):
            raise ValueError("Anthropic tool_choice must be an object")
        kind = value.get("type")
        if kind == "auto":
            return "auto"
        if kind == "any":
            return "required"
        if kind == "none":
            return "none"
        if kind == "tool":
            name = value.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Anthropic tool_choice type 'tool' requires name")
            return {"type": "function", "function": {"name": name}}
        raise ValueError(f"unsupported Anthropic tool_choice type {kind!r}")

    @classmethod
    def _anthropic_response(
        cls,
        original_request: Mapping[str, Any],
        normalized_request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        choice = response["choices"][0]
        message = choice["message"]
        content: list[dict[str, Any]] = []
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning,
                    # Local reasoning parsers do not produce Anthropic's
                    # cryptographic signature. Engines' own compatibility
                    # layers likewise emit an empty signature for local models.
                    "signature": "",
                }
            )
        text = message.get("content")
        if isinstance(text, str) and text:
            content.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
                raise ValueError("internal assistant tool call is malformed")
            function = call["function"]
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("sampled tool-call arguments are not valid JSON") from exc
            if not isinstance(arguments, Mapping):
                raise ValueError("sampled tool-call arguments must decode to an object")
            content.append(
                {
                    "type": "tool_use",
                    "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": str(function.get("name", "")),
                    "input": dict(arguments),
                }
            )

        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        training = dict(response.get("training") or {})
        # Provider-neutral transcript fields are private (client responses
        # strip the whole training block) and let processors render Anthropic
        # traffic with the exact same chat template used for generation.
        training.update(
            {
                "request_messages": list(normalized_request.get("messages") or []),
                "request_tools": normalized_request.get("tools"),
                "response_message": dict(message),
                "finish_reason": choice.get("finish_reason"),
            }
        )
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": str(original_request.get("model", response.get("model", ""))),
            "content": content,
            "stop_reason": cls._anthropic_stop_reason(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
            },
            "training": training,
        }

    @staticmethod
    def _anthropic_stop_reason(value: Any) -> str:
        if value == "tool_calls":
            return "tool_use"
        if value == "length":
            return "max_tokens"
        return "end_turn"

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
        template_kwargs = payload.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, Mapping):
            raise ValueError("chat_template_kwargs must be an object")
        if "tools" in template_kwargs:
            raise ValueError("chat_template_kwargs.tools is reserved; use the top-level tools field")
        template_messages = self._template_messages(messages)
        tools = self._request_tools(payload)
        render_kwargs = dict(template_kwargs)
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

    def reasoning_is_pre_opened(self) -> bool:
        """Whether this model's chat template opens ``<think>`` for the model.

        Qwen3-*-Thinking templates end the generation prompt with an open
        ``<think>``, so the sample carries only the closing tag and a sample
        without one is truncated reasoning. Sniffed once from the rendered
        prompt; an explicit ``force_reasoning`` in the handler config wins.
        Rendering errors propagate without caching a result.
        """
        if self.force_reasoning is not None:
            return self.force_reasoning
        rendered = self._require_tokenizer().apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
        )
        self.force_reasoning = str(rendered).rstrip().endswith("<think>")
        return self.force_reasoning

    @classmethod
    def _assistant_message(
        cls,
        text: str,
        tool_parser: Any,
        *,
        force_reasoning: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        # Thinking-template models emit reasoning before the last </think>
        # (the opening tag is auto-inserted by the chat template, so it is
        # usually absent from the sampled text). OpenAI-shaped clients render
        # content verbatim, so reasoning must ride the standard
        # reasoning_content field — otherwise every consumer downstream
        # (agents, judges, user simulators) sees chain-of-thought as the
        # reply. Training tensors are captured from the raw token stream and
        # are unaffected by this presentation split.
        reasoning: str | None = None
        if cls.SPLIT_REASONING and "</think>" in text:
            head, _, tail = text.rpartition("</think>")
            reasoning = head.replace("<think>", "").strip()
            text = tail.lstrip("\n")
        elif cls.SPLIT_REASONING and force_reasoning:
            # A pre-opened template with no closing tag: the generation hit
            # its token cap mid-thought. The whole sample is reasoning, so
            # fail closed — handing it back as content is exactly the
            # chain-of-thought-as-reply this split exists to prevent, and a
            # judge scoring it scores the wrong text. SGLang's own parser
            # takes the same branch (force_reasoning => normal_text empty).
            reasoning = text.replace("<think>", "").strip()
            text = ""
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
    "ReasoningStreamSplitter",
    "TokenNativeChatHandler",
    "finite_log_prob",
    "integer_tokens",
    "normalize_finish_reason",
    "positive_max_tokens",
    "sse_json_events",
    "stream_ids",
]
