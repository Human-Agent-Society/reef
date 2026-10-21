"""OpenAI chat completions served in this process by the MLX engine.

A response carries a private ``training`` block with the exact token ids the
engine sampled and their log probabilities, so a served exchange can become
a training sample without tokenizing decoded text again. The public
``message`` is the client's view of the same sample: reasoning moves to
``reasoning_content`` and tool calls in the template's syntax become
``tool_calls``. The split and the stream encoding are the ones the
token-native chat facade in :mod:`reef.inference.chat` uses.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.chat_protocol import (
    ChatStreamRelay,
    OpenAIChunkWriter,
    ParsedToolCall,
    ReasoningStreamSplitter,
    assistant_message,
    held_suffix,
    normalize_finish_reason,
    positive_max_tokens,
    stream_ids,
)
from reef.runtime.interfaces import InferenceHandler, InferenceStream, UpstreamStatusError
from reef.surface.base import WeightRuntime
from reef.train.mlx_backend.contracts import GenerationListener, ModelEngine, Rollout

#: Request fields that would change sampling or the response, which the engine does not implement.
UNSUPPORTED_REQUEST_FIELDS = (
    "min_p",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
)


class MLXToolCallParser:
    """Find the tool calls in sampled text and parse each one with the engine's parser.

    The engine's parser is the one mlx-lm chose for the chat template.
    ``parse_stream_chunk`` shows streamed text up to the first call marker
    and holds the rest; the stream's reconciliation sends the parsed calls.
    """

    def __init__(self, engine: ModelEngine, markers: tuple[str, str], tools: Sequence[Mapping[str, Any]]) -> None:
        self._engine = engine
        self.tool_call_start, self._tool_call_end = markers
        self._tools = [dict(tool) for tool in tools]
        self._buffer = ""
        self._holding = False

    @classmethod
    def for_request(cls, engine: ModelEngine, tools: Sequence[Mapping[str, Any]] | None) -> MLXToolCallParser | None:
        """The parser for one request, or ``None`` when the request declares no tools or the template has no call syntax."""
        markers = engine.tool_call_markers
        if not tools or markers is None:
            return None
        return cls(engine, markers, tools)

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start in text

    def parse_non_stream(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        """The text outside every call, and the calls in order; raise ``ValueError`` for a call that never closes."""
        outside: list[str] = []
        calls: list[ParsedToolCall] = []
        cursor = 0
        while (start := text.find(self.tool_call_start, cursor)) != -1:
            outside.append(text[cursor:start])
            body_start = start + len(self.tool_call_start)
            end = text.find(self._tool_call_end, body_start)
            if end == -1:
                raise ValueError(f"a tool call opened with {self.tool_call_start!r} never closed")
            calls.append(self._engine.parse_tool_call(text[body_start:end].strip(), self._tools))
            cursor = end + len(self._tool_call_end)
        outside.append(text[cursor:])
        return "".join(outside).strip(), calls

    def parse_stream_chunk(self, delta: str) -> tuple[str, list[ParsedToolCall]]:
        """The streamed text that is safe to show: nothing from the first call marker on.

        A tail that could still become the marker, and whitespace before it,
        wait for the next delta.
        """
        if self._holding:
            return "", []
        self._buffer += delta
        index = self._buffer.find(self.tool_call_start)
        if index != -1:
            self._holding = True
            visible, self._buffer = self._buffer[:index].rstrip(), ""
            return visible, []
        cut = len(self._buffer) - held_suffix(self._buffer, self.tool_call_start)
        while cut > 0 and self._buffer[cut - 1].isspace():
            cut -= 1
        visible, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return visible, []


@dataclass(frozen=True)
class ChatRequest:
    """One validated chat-completions request, rendered to the engine's prompt tokens."""

    payload: Mapping[str, Any]
    prompt_tokens: list[int]
    opens_reasoning: bool
    max_tokens: int
    temperature: float | None
    tools: list[dict[str, Any]] | None


class LoopListener(GenerationListener):
    """Hand the pieces of one streamed generation from the engine thread to the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stop = threading.Event()
        #: ``None`` marks the end of the generation.
        self.pieces: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(self, piece: str) -> None:
        self._loop.call_soon_threadsafe(self.pieces.put_nowait, piece)

    def cancelled(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()


class MLXInferenceHandler(InferenceHandler):
    """Answer chat completions from the engine's live adapter.

    ``runtime`` reports the version of the served weights. Requests take
    the engine one at a time.
    """

    def __init__(self, engine: ModelEngine, runtime: WeightRuntime) -> None:
        self._engine = engine
        self._runtime = runtime
        self._engine_lock = asyncio.Lock()

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream") is True:
            raise UpstreamStatusError("a streaming completion is served by inference_stream", status=400)
        request = await self._request(path, payload)
        async with self._engine_lock:
            rollout = await asyncio.to_thread(
                self._engine.generate,
                request.prompt_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        return self._response(request, rollout, self._tool_parser(request))

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        """Stream one chat completion as OpenAI chunks while it is sampled.

        The record is the response the buffered path builds, attached once
        the sample is complete. Closing the stream early stops the generation.
        """
        request = await self._request(path, payload)
        chat_id, _, created = stream_ids()
        writer = OpenAIChunkWriter(
            {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": self._model(request)}
        )
        stream_parser = self._tool_parser(request)
        relay = ChatStreamRelay(writer, stream_parser)
        splitter = ReasoningStreamSplitter(
            enabled=True,
            force_reasoning=request.opens_reasoning,
            tool_call_start=None if stream_parser is None else stream_parser.tool_call_start,
        )

        async def frames() -> AsyncGenerator[bytes, None]:
            async with self._engine_lock:
                listener = LoopListener(asyncio.get_running_loop())
                generation = asyncio.ensure_future(
                    asyncio.to_thread(
                        self._engine.generate,
                        request.prompt_tokens,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        listener=listener,
                    )
                )
                generation.add_done_callback(lambda _: listener.pieces.put_nowait(None))
                try:
                    for frame in writer.start():
                        yield frame
                    while (piece := await listener.pieces.get()) is not None:
                        for kind, value in splitter.feed(piece):
                            for frame in relay.piece(kind, value):
                                yield frame
                    rollout = await generation
                    for kind, value in splitter.finish():
                        for frame in relay.piece(kind, value):
                            yield frame
                    # The canonical response parses the whole sample with a
                    # fresh parser; the stream parser holds state.
                    response = self._response(request, rollout, self._tool_parser(request))
                    response["id"] = chat_id
                    response["created"] = created
                    closing = relay.reconcile(response["choices"][0]["message"])
                    closing.extend(writer.finish(response, output_tokens=len(rollout.output_tokens)))
                    stream.record_response = response
                    for frame in closing:
                        yield frame
                finally:
                    # Reached on completion, on failure and on an early close.
                    # The engine lock opens only after the generation stops.
                    listener.stop()
                    if not generation.done():
                        await asyncio.gather(generation, return_exceptions=True)

        chunks = frames()
        stream = InferenceStream(
            status=200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            chunks=chunks,
            close=chunks.aclose,
            record_response_pending=True,
        )
        return stream

    async def _request(self, path: str, payload: Mapping[str, Any]) -> ChatRequest:
        """Validate a chat-completions request and render its prompt; raise a client error for a bad one."""
        if not path.rstrip("/").endswith("chat/completions"):
            raise UpstreamStatusError(f"the mlx backend serves chat completions, not {path!r}", status=404)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or any(not isinstance(item, Mapping) for item in messages):
            raise UpstreamStatusError("chat completions require a non-empty list of message objects", status=400)
        for name in UNSUPPORTED_REQUEST_FIELDS:
            if payload.get(name) not in (None, False):
                raise UpstreamStatusError(f"the mlx backend does not support the request field {name!r}", status=400)
        if payload.get("n", 1) != 1:
            raise UpstreamStatusError("the mlx backend samples one choice per request; n must be 1", status=400)
        for name, configured in (("top_p", self._engine.config.top_p), ("top_k", self._engine.config.top_k)):
            requested = payload.get(name)
            if requested is not None and requested != configured:
                raise UpstreamStatusError(
                    f"the mlx backend samples every request at training.options.{name} ({configured})", status=400
                )
        try:
            max_tokens = positive_max_tokens(payload, {"max_tokens": self._engine.config.max_tokens}, "max_tokens")
        except ValueError as exc:
            raise UpstreamStatusError(str(exc), status=400) from exc
        temperature = payload.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise UpstreamStatusError("temperature must be a finite non-negative number", status=400)
        template_kwargs = payload.get("chat_template_kwargs")
        if template_kwargs is not None and not isinstance(template_kwargs, Mapping):
            raise UpstreamStatusError("chat_template_kwargs must be an object", status=400)
        tools = payload.get("tools")
        if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, Mapping) for tool in tools)):
            raise UpstreamStatusError("tools must be an array of objects", status=400)
        try:
            prompt_tokens = await asyncio.to_thread(
                self._engine.render_prompt, messages, tools=tools, template_kwargs=template_kwargs
            )
        except ValueError as exc:
            raise UpstreamStatusError(str(exc), status=400) from exc
        max_context_tokens = self._engine.config.max_context_tokens
        if max_context_tokens:
            if len(prompt_tokens) >= max_context_tokens:
                raise UpstreamStatusError(
                    f"the prompt has {len(prompt_tokens)} tokens; training.options.max_context_tokens is "
                    f"{max_context_tokens}",
                    status=400,
                )
            # Like a context-length limit, this shortens the completion budget rather than refusing the request.
            max_tokens = min(max_tokens, max_context_tokens - len(prompt_tokens))
        return ChatRequest(
            payload=payload,
            prompt_tokens=prompt_tokens,
            opens_reasoning=self._engine.opens_reasoning(prompt_tokens),
            max_tokens=max_tokens,
            temperature=None if temperature is None else float(temperature),
            # ``tool_choice: none`` declares the tools for context only.
            tools=None if tools is None or payload.get("tool_choice") == "none" else [dict(tool) for tool in tools],
        )

    def _tool_parser(self, request: ChatRequest) -> MLXToolCallParser | None:
        return MLXToolCallParser.for_request(self._engine, request.tools)

    def _model(self, request: ChatRequest) -> str:
        return str(request.payload.get("model") or self._engine.model_path)

    def _response(self, request: ChatRequest, rollout: Rollout, parser: MLXToolCallParser | None) -> dict[str, Any]:
        """The OpenAI response with the training block; raise a server error for an unusable sample."""
        if not rollout.output_tokens:
            # A completion without response tokens can never become a policy
            # sample; the caller retries the rollout.
            raise UpstreamStatusError("the model produced no response tokens", status=502)
        try:
            message, called = assistant_message(
                rollout.text,
                parser,
                force_reasoning=request.opens_reasoning,
                tool_call_start=None if parser is None else parser.tool_call_start,
            )
        except ValueError as exc:
            raise UpstreamStatusError(str(exc), status=502) from exc
        runtime_load_id = self._runtime.serving_runtime_load_id()
        prompt_tokens = list(request.prompt_tokens)
        output_tokens = list(rollout.output_tokens)
        training: dict[str, Any] = {
            "tokens": [*prompt_tokens, *output_tokens],
            # Every generated token is a policy action: a completion has no environment tokens.
            "loss_mask": [1] * len(output_tokens),
            "rollout_log_probs": list(rollout.rollout_log_probs),
            "prompt_length": len(prompt_tokens),
            "response_length": len(output_tokens),
            "runtime_load_id": runtime_load_id,
        }
        if rollout.topk_indices:
            training["topk_indices"] = [list(row) for row in rollout.topk_indices]
            training["topk_log_probs"] = [list(row) for row in rollout.topk_log_probs]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model(request),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if called else normalize_finish_reason(rollout.finish_reason),
                    # The weight surface reads the engine's version from meta_info.
                    "meta_info": {"runtime_load_id": runtime_load_id},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(output_tokens),
                "total_tokens": len(prompt_tokens) + len(output_tokens),
            },
            "training": training,
        }


__all__ = ["MLXInferenceHandler", "MLXToolCallParser"]
