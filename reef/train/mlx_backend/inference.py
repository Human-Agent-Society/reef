"""OpenAI-shaped chat inference executed in this process by MLX.

The response carries a private ``training`` block holding the exact token ids
the engine sampled and their log-probabilities. That block is what makes a
served exchange trainable: Reef never re-tokenizes decoded text to reconstruct
policy tensors, so a rollout that was not captured token-natively at
generation time can never become a policy sample.

The public ``message`` is a presentation of the same sample: reasoning split
off into ``reasoning_content`` and tool calls parsed into ``tool_calls`` in
the syntax the model's own chat template asked for. The split reads the
decoded text and never touches the tensors.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact
from reef.runtime.assistant_message import (
    THINK_OPEN,
    ReasoningStreamSplitter,
    ToolMarkerStreamHold,
    split_assistant_message,
)
from reef.runtime.inference import InferenceBackend, InferenceStream, UpstreamStatusError


@dataclass(frozen=True)
class ParsedToolCall:
    """One call the parser recovered: what ``split_assistant_message`` reads."""

    name: str
    parameters: Any


class MLXToolCallParser:
    """mlx-lm's per-model tool parser behind the interface the message split reads.

    mlx-lm chooses the parser from the chat template when it loads the
    tokenizer (Qwen3.8's ``<tool_call>``-wrapped XML, Hermes-style JSON,
    Mistral's ``[TOOL_CALLS]``, ...) and exposes the markers that bound a call
    plus a ``parse_tool_call(text, tools)`` that coerces each argument by the
    type its declared schema gives it. This adapter finds the bounded spans,
    hands each to that function, and keeps the text outside them as the reply.
    """

    def __init__(self, tokenizer: Any, tools: Sequence[Mapping[str, Any]]) -> None:
        self.tool_call_start: str = tokenizer.tool_call_start
        self._tool_call_end: str = tokenizer.tool_call_end
        self._parse = tokenizer.tool_parser
        self._tools = [dict(tool) for tool in tools]

    @classmethod
    def for_tokenizer(cls, tokenizer: Any, tools: Sequence[Mapping[str, Any]] | None) -> MLXToolCallParser | None:
        """The parser for this request, or ``None`` when nothing could be a call.

        No declared toolset means the prompt stated no call syntax, so any
        markup in the reply is text. A template mlx-lm found no parser for
        leaves the reply as text too, rather than guessing at a syntax.
        """
        if not tools or not getattr(tokenizer, "has_tool_calling", False):
            return None
        return cls(tokenizer, tools)

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start in text

    def parse_non_stream(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        """The text outside every call, and the calls in order.

        A call that opens and never closes is a sample that hit the token cap
        mid-call; it raises, and the request fails so the agent retries a
        rollout instead of reading half a call as a reply.
        """
        outside: list[str] = []
        calls: list[ParsedToolCall] = []
        cursor = 0
        while True:
            start = text.find(self.tool_call_start, cursor)
            if start == -1:
                outside.append(text[cursor:])
                break
            outside.append(text[cursor:start])
            body_start = start + len(self.tool_call_start)
            end = text.find(self._tool_call_end, body_start)
            if end == -1:
                raise ValueError(f"a tool call opened with {self.tool_call_start!r} never closed")
            parsed = self._parse(text[body_start:end].strip(), self._tools)
            name = parsed.get("name") if isinstance(parsed, Mapping) else None
            if not isinstance(name, str) or not name:
                raise ValueError("a tool call names no function")
            arguments = parsed.get("arguments", parsed.get("parameters"))
            calls.append(ParsedToolCall(name=name, parameters={} if arguments is None else arguments))
            cursor = end + len(self._tool_call_end)
        return "".join(outside).strip(), calls


def prompt_opens_reasoning(tokenizer: Any, prompt_tokens: Sequence[int]) -> bool:
    """Whether this rendered prompt ends by opening the model's thinking block.

    Decided per request rather than once per model: the same Qwen3 template
    opens ``<think>`` by default and opens-and-closes it at once when a
    request sets ``enable_thinking`` false, and the split has to know which
    one produced the sample it is reading. The last few prompt tokens are
    enough: the marker is the final thing the template writes.
    """
    tail = tokenizer.decode(list(prompt_tokens[-8:]))
    return str(tail).rstrip().endswith(THINK_OPEN)


@dataclass(frozen=True)
class _ChatRequest:
    """One validated chat-completions request, buffered or streamed alike."""

    messages: list[Any]
    max_tokens: int | None
    temperature: float | None
    template_kwargs: Mapping[str, Any] | None
    tools: list[Any] | None
    parser: MLXToolCallParser | None


class _LoopListener:
    """The engine's ``GenerationListener`` for one stream, bridging its thread to the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stop = threading.Event()
        self.pieces: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(self, piece: str) -> None:
        self._loop.call_soon_threadsafe(self.pieces.put_nowait, piece)

    def cancelled(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()


def _unsent(final: Any, emitted: str) -> str:
    """What the canonical field still owes the wire, given what the stream showed of it."""
    if not isinstance(final, str) or not final.startswith(emitted):
        return ""
    return final[len(emitted) :]


def _sse(event: Mapping[str, Any]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


class MLXInferenceBackend(InferenceBackend):
    """Answer chat completions from the runtime's resident MLX model."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        # MLX evaluates on one process-wide stream, and training mutates the
        # same parameters generation reads. Both completion paths — buffered and
        # streamed — drive the engine as a single sequence, so this lock keeps
        # the two from running against that stream at once. Concurrent requests
        # serialize here rather than sharing a batch: mlx-lm cannot continuously
        # batch the Qwen3.5 hybrid cache (splicing a fresh prompt into a live
        # decode corrupts the mixed GatedDeltaNet/attention cache), so a shared
        # batch bought corruption, not throughput.
        self._engine_lock = asyncio.Lock()

    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if payload.get("stream") is True:
            # The buffered path answers with one JSON body. A caller that
            # asked for a stream and got that would parse something that is
            # not SSE; the service routes streams to ``inference_stream``.
            raise UpstreamStatusError("a streaming completion is served by inference_stream", status=400)
        request = self._parse_request(path, payload)
        prompt_tokens, opens_reasoning = await asyncio.to_thread(self._render, request)
        rollout = await self._submit(prompt_tokens, request)
        if not rollout.output_tokens:
            # A completion with no response tokens can never be a policy
            # sample: `policy_row_violation` rejects an empty loss mask, the
            # processor marks the report terminally unusable, and a grid-based
            # recipe would then wait forever for a slot that can never fill.
            # Fail the request instead, so the caller retries a rollout.
            raise UpstreamStatusError("the model produced no response tokens", status=502)
        return self._response(payload, rollout, request.parser, force_reasoning=opens_reasoning)

    async def _submit(self, prompt_tokens: Sequence[int], request: _ChatRequest) -> Any:
        """Sample this request's rollout as a single sequence on the engine thread.

        Held under ``_engine_lock`` so a buffered completion and a streamed one
        never drive the one MLX stream at once; concurrent buffered requests
        serialize through it in turn.
        """
        async with self._engine_lock:
            engine = self._runtime.engine
            return await asyncio.to_thread(
                engine.generate,
                prompt_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

    async def inference_stream(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> InferenceStream:
        """Serve one chat completion as OpenAI chunks while it is being sampled.

        The wire carries deltas as the engine settles them; the trainable
        record is the same response the buffered path builds, attached as
        ``record_response`` once the sample is complete and parsed, so the
        service records exactly what a buffered request would have recorded.
        """
        request = self._parse_request(path, payload)
        holder: dict[str, InferenceStream] = {}
        chunks = self._stream_chunks(payload, request, holder)
        stream = InferenceStream(
            status=200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            chunks=chunks,
            # Closing the stream early must stop the engine, not just drop
            # the bytes: the generator's cleanup cancels the generation and
            # waits for it before the engine lock is released.
            close=chunks.aclose,
            record_response_pending=True,
        )
        holder["stream"] = stream
        return stream

    def _parse_request(self, path: str, payload: Mapping[str, Any]) -> _ChatRequest:
        if not path.rstrip("/").endswith("chat/completions"):
            raise UpstreamStatusError(f"the mlx backend serves chat completions, not {path!r}", status=404)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise UpstreamStatusError("chat completions require a non-empty messages list", status=400)
        max_tokens = payload.get("max_completion_tokens", payload.get("max_tokens"))
        temperature = payload.get("temperature")
        # The field OpenAI-compatible servers use to steer a chat template —
        # `enable_thinking` above all, which decides whether a reasoning
        # model's `<think>` block becomes response tokens.
        template_kwargs = payload.get("chat_template_kwargs")
        if template_kwargs is not None and not isinstance(template_kwargs, Mapping):
            raise UpstreamStatusError("chat_template_kwargs must be an object", status=400)
        # The toolset the caller declared. The chat template renders it into
        # the prompt — the schemas and the call syntax the model is meant to
        # answer in — so dropping it leaves the model inventing both.
        tools = payload.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise UpstreamStatusError("tools must be an array", status=400)
        # The same toolset decides how the reply is read back: the parser
        # mlx-lm matched to the template turns the model's call markup into
        # `tool_calls`. `tool_choice: none` declares the tools for context
        # only, as the OpenAI dialect defines it.
        parser = None if payload.get("tool_choice") == "none" else self._tool_call_parser(tools)
        return _ChatRequest(
            messages=messages,
            max_tokens=None if max_tokens is None else int(max_tokens),
            temperature=None if temperature is None else float(temperature),
            template_kwargs=template_kwargs,
            tools=tools,
            parser=parser,
        )

    def _tool_call_parser(self, tools: list[Any] | None) -> MLXToolCallParser | None:
        return MLXToolCallParser.for_tokenizer(self._runtime.engine.tokenizer, tools)

    def _render(self, request: _ChatRequest) -> tuple[list[int], bool]:
        engine = self._runtime.engine
        prompt_tokens = engine.render_prompt(
            request.messages, tools=request.tools, template_kwargs=request.template_kwargs
        )
        return prompt_tokens, prompt_opens_reasoning(engine.tokenizer, prompt_tokens)

    async def _stream_chunks(
        self,
        payload: Mapping[str, Any],
        request: _ChatRequest,
        holder: Mapping[str, InferenceStream],
    ) -> AsyncGenerator[bytes, None]:
        common = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": str(payload.get("model") or self._runtime.engine.config.model_path),
        }

        def chunk(delta: Mapping[str, Any], *, finish_reason: str | None = None, **extra: Any) -> bytes:
            return _sse(
                {**common, "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason, **extra}]}
            )

        async with self._engine_lock:
            engine = self._runtime.engine
            prompt_tokens, opens_reasoning = await asyncio.to_thread(self._render, request)
            marker = None if request.parser is None else request.parser.tool_call_start
            reasoning = ReasoningStreamSplitter(enabled=True, force_reasoning=opens_reasoning, tool_call_start=marker)
            hold = ToolMarkerStreamHold(marker)
            emitted_reasoning = ""
            emitted_text = ""

            # The engine thread hands each settled piece of text to the event
            # loop; ``None`` marks the end of the generation, whatever ended it.
            listener = _LoopListener(asyncio.get_running_loop())
            pieces = listener.pieces
            generation = asyncio.ensure_future(
                asyncio.to_thread(
                    engine.generate_stream,
                    prompt_tokens,
                    listener=listener,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            generation.add_done_callback(lambda _: pieces.put_nowait(None))

            def show(kind: str, value: str) -> bytes | None:
                nonlocal emitted_reasoning, emitted_text
                if kind == "thinking":
                    emitted_reasoning += value
                    return chunk({"reasoning_content": value})
                visible = hold.feed(value)
                if not visible:
                    return None
                emitted_text += visible
                return chunk({"content": visible})

            try:
                yield chunk({"role": "assistant"})
                while True:
                    piece = await pieces.get()
                    if piece is None:
                        break
                    for kind, value in reasoning.feed(piece):
                        if (out := show(kind, value)) is not None:
                            yield out
                rollout = await generation
                for kind, value in reasoning.finish():
                    if (out := show(kind, value)) is not None:
                        yield out
                tail = hold.finish()
                if tail:
                    emitted_text += tail
                    yield chunk({"content": tail})
                if not rollout.output_tokens:
                    raise UpstreamStatusError("the model produced no response tokens", status=502)

                # The canonical message, parsed whole: what the record holds
                # and what the wire is reconciled to. Whatever the stream has
                # not shown yet — the held tool-call markup as `tool_calls`,
                # a final piece of text — goes out now, before the terminal.
                response = self._response(payload, rollout, request.parser, force_reasoning=opens_reasoning)
                response["id"] = common["id"]
                response["created"] = common["created"]
                choice = response["choices"][0]
                message = choice["message"]
                if rest := _unsent(message.get("reasoning_content"), emitted_reasoning):
                    yield chunk({"reasoning_content": rest})
                if rest := _unsent(message.get("content"), emitted_text):
                    yield chunk({"content": rest})
                for index, call in enumerate(message.get("tool_calls") or []):
                    yield chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": dict(call["function"]),
                                }
                            ]
                        }
                    )
                holder["stream"].record_response = response
                terminal = {"finish_reason": choice["finish_reason"]}
                if "meta_info" in choice:
                    terminal["meta_info"] = choice["meta_info"]
                yield chunk({}, **terminal)
                yield b"data: [DONE]\n\n"
            finally:
                # Reached on completion, on failure, and when the consumer
                # closes early. The engine is told to stop and waited for,
                # so the lock never opens on a generation still running.
                listener.stop()
                if not generation.done():
                    await asyncio.gather(generation, return_exceptions=True)

    def _response(
        self,
        request: Mapping[str, Any],
        rollout: Any,
        parser: MLXToolCallParser | None,
        *,
        force_reasoning: bool,
    ) -> dict[str, Any]:
        runtime_load_id = self._runtime.serving_runtime_load_id()
        prompt_tokens = list(rollout.prompt_tokens)
        output_tokens = list(rollout.output_tokens)
        try:
            message, called = split_assistant_message(
                rollout.text,
                parser,
                force_reasoning=force_reasoning,
                tool_call_start=None if parser is None else parser.tool_call_start,
                parser_label="the mlx tool-call",
            )
        except ValueError as exc:
            # Broken call markup is not a reply. Failing the request is what
            # the SGLang path does too; the agent's request loop retries.
            raise UpstreamStatusError(str(exc), status=502) from exc
        response: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(request.get("model") or self._runtime.engine.config.model_path),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if called else rollout.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(output_tokens),
                "total_tokens": len(prompt_tokens) + len(output_tokens),
            },
            "training": {
                "tokens": [*prompt_tokens, *output_tokens],
                # Every generated token is a policy action here: this backend
                # runs single-turn completions with no environment tokens.
                "loss_mask": [1] * len(output_tokens),
                "rollout_log_probs": list(rollout.rollout_log_probs),
                "prompt_length": len(prompt_tokens),
                "response_length": len(output_tokens),
                "runtime_load_id": runtime_load_id,
                # Present only when the engine was asked to capture them. A
                # distillation objective trains on the candidate set the
                # policy actually considered, and nothing downstream can
                # reconstruct it after generation.
                **(
                    {
                        "topk_indices": [list(row) for row in rollout.topk_indices],
                        "topk_log_probs": [list(row) for row in rollout.topk_log_probs],
                    }
                    if rollout.topk_indices
                    else {}
                ),
            },
        }
        if runtime_load_id is not None:
            # Surface verification reads the engine-reported version from
            # meta_info; without it a live-weight artifact cannot prove which
            # weights answered.
            response["choices"][0]["meta_info"] = {"runtime_load_id": runtime_load_id}
        return response


__all__ = ["MLXInferenceBackend", "MLXToolCallParser", "ParsedToolCall", "prompt_opens_reasoning"]
