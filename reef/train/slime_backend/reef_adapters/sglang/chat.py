"""OpenAI and Anthropic chat facades backed by SGLang's token-native API.

Clients use either ``/v1/chat/completions`` or ``/v1/messages`` while Reef
normalizes the provider request, renders the prompt once, calls SGLang
``/generate``, and records the exact sampled token ids beside their rollout log
probabilities. Decoded text is only the client-facing view; it is never
re-tokenized to construct a training sample.
"""

from __future__ import annotations

import codecs
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from reef.artifact.artifact import Artifact, is_local_version
from reef.runtime.inference import HttpInferenceBackend, InferenceStream

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
SGLANG_GENERATE_PATH = "/generate"

ToolParserFactory = Callable[[list[dict[str, Any]], str], Any]


async def _sse_json_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Decode arbitrarily chunked SSE bytes into JSON data events."""

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    def complete_events(*, final: bool = False) -> tuple[list[str], str]:
        nonlocal buffer
        # SGLang uses LF, but accepting CRLF keeps the proxy correct behind
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

    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        payloads, _ = complete_events()
        for payload in payloads:
            if payload == "[DONE]":
                return
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("SGLang stream event must be a JSON object")
            if isinstance(value.get("error"), Mapping):
                raise ValueError(f"SGLang stream failed: {value['error'].get('message', value['error'])}")
            yield value

    buffer += decoder.decode(b"", final=True)
    payloads, _ = complete_events(final=True)
    for payload in payloads:
        if payload == "[DONE]":
            return
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("SGLang stream event must be a JSON object")
        if isinstance(value.get("error"), Mapping):
            raise ValueError(f"SGLang stream failed: {value['error'].get('message', value['error'])}")
        yield value


class _NativeStreamCapture:
    """Accumulate Reef's required disjoint SGLang stream events."""

    _LIST_META_KEYS = (
        "output_token_logprobs",
        "output_top_logprobs",
        "output_token_ids_logprobs",
        "output_token_sampling_mask",
        "output_token_sampling_logprobs",
        "_reef_token_weight_versions",
    )

    def __init__(self) -> None:
        self.text = ""
        self.output_ids: list[int] = []
        self.meta: dict[str, Any] = {}
        self.last_output_logprobs: list[Any] = []
        self.seen = False

    def accept(self, event: Mapping[str, Any]) -> str:
        meta = event.get("meta_info")
        if not isinstance(meta, Mapping):
            raise ValueError("SGLang stream event lacks meta_info")
        self.seen = True
        self.last_output_logprobs = []
        total = meta.get("completion_tokens")
        pairs = meta.get("output_token_logprobs")
        if not isinstance(pairs, list):
            raise ValueError("incremental SGLang stream event lacks output_token_logprobs")
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or total != len(self.meta.get("output_token_logprobs", [])) + len(pairs)
        ):
            raise ValueError(
                "Reef requires SGLang --incremental-streaming-output; received a cumulative or malformed event"
            )

        raw_ids = event.get("output_ids")
        if isinstance(raw_ids, list):
            ids = [int(token_id) for token_id in raw_ids]
            if len(ids) != len(pairs):
                raise ValueError("incremental SGLang stream event has inconsistent output ids and log-probs")
            self.output_ids.extend(ids)

        for key, value in meta.items():
            if key in self._LIST_META_KEYS and isinstance(value, list):
                accumulated = self.meta.setdefault(key, [])
                if not isinstance(accumulated, list):
                    raise ValueError(f"incremental SGLang metadata {key} changed type")
                accumulated.extend(value)
                if key == "output_token_logprobs":
                    self.last_output_logprobs = list(value)
            else:
                self.meta[key] = value

        current_text = event.get("text")
        if current_text is None:
            return ""
        if not isinstance(current_text, str):
            raise ValueError("SGLang stream event text must be a string")
        self.text += current_text
        return current_text

    def response(self) -> dict[str, Any]:
        if not self.seen:
            raise ValueError("SGLang stream completed without a generation event")
        response: dict[str, Any] = {"text": self.text, "meta_info": dict(self.meta)}
        versions = self.meta.get("_reef_token_weight_versions")
        if versions is not None:
            response["_reef_token_weight_versions"] = versions
        return response


class _ReasoningStreamSplitter:
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


def _build_sglang_tool_parser(tools: list[dict[str, Any]], parser_name: str) -> Any:
    """Load the parser shipped with the same SGLang runtime as Slime."""

    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    return FunctionCallParser([Tool.model_validate(tool) for tool in tools], parser_name)


class SGLangChatTrainingInferenceBackend(HttpInferenceBackend):
    """Serve OpenAI or Anthropic chat with engine-native policy tensors."""

    def __init__(
        self,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float = 300.0,
        tokenizer: Any = None,
        tool_call_parser: str | None = None,
        tool_parser_factory: ToolParserFactory = _build_sglang_tool_parser,
        capture_topk: int = 0,
        sampling_defaults: Mapping[str, Any] | None = None,
        force_reasoning: bool | None = None,
    ) -> None:
        super().__init__(upstream_url, timeout_s=timeout_s, error_label="SGLang /generate")
        if tokenizer is None and not model_path:
            raise ValueError("SGLang chat training inference requires model_path")
        self._model_path = model_path
        self._tokenizer = tokenizer
        self._tool_call_parser = tool_call_parser.strip() if tool_call_parser else None
        self._tool_parser_factory = tool_parser_factory
        # OpenClaw-RL top-K objective: capture the generation-time top-K
        # vocab log-probs per sampled token (the old-policy ``ell_old`` on
        # S^q) into ``response.training``. 0 disables the capture.
        self._capture_topk = int(capture_topk)
        # Sampling values applied when the client omits a parameter. The
        # native /generate path has engine defaults (temperature 1.0), NOT
        # the model generation_config the chat endpoint would use — a
        # recipe that reproduces a reference run pins its effective
        # temperature here (e.g. OpenClaw-RL: 0.6/0.95/20).
        self._sampling_defaults = dict(sampling_defaults or {})
        # None = sniff the chat template on first render.
        self._force_reasoning = force_reasoning

    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if path == ANTHROPIC_COUNT_TOKENS_PATH:
            return self._anthropic_count_tokens(artifact, payload)
        if path not in {CHAT_COMPLETIONS_PATH, ANTHROPIC_MESSAGES_PATH}:
            raise ValueError(
                "SGLang chat training inference supports only "
                f"{CHAT_COMPLETIONS_PATH}, {ANTHROPIC_MESSAGES_PATH}, and {ANTHROPIC_COUNT_TOKENS_PATH}"
            )
        anthropic = path == ANTHROPIC_MESSAGES_PATH
        request = self._anthropic_request(payload) if anthropic else dict(payload)
        if request.get("n", 1) != 1:
            raise ValueError("exact training capture currently requires n=1")

        tool_parser = self._configured_tool_parser(request)
        prompt_ids = self._render_prompt(request)
        sampling_params = self._sampling_params(request)
        native_payload = self._native_payload(request, prompt_ids, sampling_params)
        native = await self._native_inference(artifact, native_payload)
        output_ids, rollout_log_probs = self._output_tensors(native)
        version_spans = self._weight_version_spans(native, len(output_ids))

        response = self._chat_response(
            request,
            native,
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            rollout_log_probs=rollout_log_probs,
            loss_mask=[1] * len(output_ids),
            weight_version_spans=version_spans,
            tool_parser=tool_parser,
        )
        if not anthropic:
            return response
        return self._anthropic_response(payload, request, response)

    def _anthropic_count_tokens(self, artifact: Artifact, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Count the exact templated input without creating policy tensors."""
        request = self._anthropic_request({**payload, "max_tokens": 1})
        prompt_ids = self._render_prompt(request)
        weight_version = getattr(artifact.ref, "weight_version", None) or artifact.ref.version
        return {
            "input_tokens": len(prompt_ids),
            # RequestService strips this block from the client response. It
            # supplies provenance required by durable-request validation, but
            # deliberately contains no tokens/loss mask/log-probabilities, so
            # a token-count call can never become a policy sample.
            "training": {
                "weight_version": str(weight_version),
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
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "stream",
            "lora_path",
            "rid",
            "sglang_sampling_params",
            "chat_template_kwargs",
        ):
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

    def _native_payload(
        self,
        request: Mapping[str, Any],
        prompt_ids: list[int],
        sampling_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        native_payload: dict[str, Any] = {
            "input_ids": prompt_ids,
            "sampling_params": dict(sampling_params),
            "return_logprob": True,
            # Buffered inference is the default; inference_stream flips this
            # request to native SSE after installing its incremental capture.
            "stream": False,
        }
        top_logprobs = request.get("top_logprobs")
        if isinstance(top_logprobs, int) and not isinstance(top_logprobs, bool) and top_logprobs >= 0:
            native_payload["top_logprobs_num"] = top_logprobs
        if self._capture_topk > 0:
            native_payload["top_logprobs_num"] = max(
                self._capture_topk, int(native_payload.get("top_logprobs_num") or 0)
            )
        for key in ("lora_path", "rid"):
            if key in request:
                native_payload[key] = request[key]
        return native_payload

    async def _native_inference(
        self,
        artifact: Artifact,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = await super().inference(artifact, SGLANG_GENERATE_PATH, payload)
        return self._normalize_native_response(artifact, response)

    @staticmethod
    def _normalize_native_response(
        artifact: Artifact,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        exact_weight_versions_required = not is_local_version(artifact.ref.version)
        meta = response.get("meta_info")
        if not isinstance(meta, Mapping):
            raise ValueError("SGLang /generate response lacks meta_info")
        pairs = meta.get("output_token_logprobs")
        if not isinstance(pairs, list):
            raise ValueError("SGLang /generate response lacks output token log probabilities")
        stamped = meta.get("_reef_token_weight_versions")
        if stamped is None and not exact_weight_versions_required:
            version = meta.get("weight_version")
            stamped = [version] * len(pairs)
        if (
            not isinstance(stamped, list)
            or len(stamped) != len(pairs)
            or any(not isinstance(version, str) or not version for version in stamped)
        ):
            raise ValueError("SGLang /generate response lacks scheduler-stamped token weight versions")
        normalized_meta = dict(meta)
        if stamped:
            normalized_meta["weight_version"] = stamped[-1]
        response["meta_info"] = normalized_meta
        response["_reef_token_weight_versions"] = stamped
        return response

    async def inference_stream(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> InferenceStream:
        if path == ANTHROPIC_COUNT_TOKENS_PATH:
            raise ValueError("Anthropic count_tokens does not support streaming")
        if path not in {CHAT_COMPLETIONS_PATH, ANTHROPIC_MESSAGES_PATH}:
            raise ValueError(
                "SGLang chat training inference supports streaming only for "
                f"{CHAT_COMPLETIONS_PATH} and {ANTHROPIC_MESSAGES_PATH}"
            )

        anthropic = path == ANTHROPIC_MESSAGES_PATH
        request = self._anthropic_request(payload) if anthropic else dict(payload)
        if request.get("n", 1) != 1:
            raise ValueError("exact training capture currently requires n=1")
        stream_tool_parser = self._configured_tool_parser(request)
        prompt_ids = self._render_prompt(request)
        sampling_params = self._sampling_params(request)
        # Force the scheduler to flush every available decode step. This is a
        # request-local knob; deployments using SGLang's disjoint incremental
        # output mode are supported by _NativeStreamCapture as well.
        sampling_params.setdefault("stream_interval", 1)
        native_payload = self._native_payload(request, prompt_ids, sampling_params)
        native_payload["stream"] = True
        upstream = await super().inference_stream(artifact, SGLANG_GENERATE_PATH, native_payload)

        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"
        created = int(time.time())
        common = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": str(request.get("model", self._model_path)),
        }
        stream_holder: dict[str, InferenceStream] = {}

        async def chunks() -> AsyncIterator[bytes]:
            capture = _NativeStreamCapture()
            reasoning = _ReasoningStreamSplitter(
                enabled=self._SPLIT_REASONING,
                force_reasoning=self._reasoning_is_pre_opened(),
            )
            emitted_reasoning = ""
            emitted_text = ""
            tool_ids: dict[int, str] = {}
            tool_names: dict[int, str] = {}
            emitted_tool_arguments: dict[int, str] = {}
            block_index = 0
            open_block: tuple[str, int | None] | None = None

            def frame(event: Mapping[str, Any]) -> bytes:
                event_type = str(event["type"])
                return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

            def open_anthropic_block(kind: str, tool_index: int | None = None) -> list[bytes]:
                nonlocal block_index, open_block
                output: list[bytes] = []
                target = (kind, tool_index)
                if open_block == target:
                    return output
                if open_block is not None:
                    output.append(frame({"type": "content_block_stop", "index": block_index}))
                    block_index += 1
                    open_block = None
                content_block: dict[str, Any]
                if kind == "thinking":
                    content_block = {"type": "thinking", "thinking": "", "signature": ""}
                elif kind == "text":
                    content_block = {"type": "text", "text": ""}
                else:
                    if tool_index is None:
                        raise RuntimeError("tool block is missing its tool index")
                    content_block = {
                        "type": "tool_use",
                        "id": tool_ids[tool_index],
                        "name": tool_names[tool_index],
                        "input": {},
                    }
                output.append(
                    frame(
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": content_block,
                        }
                    )
                )
                open_block = target
                return output

            def close_anthropic_block() -> list[bytes]:
                nonlocal block_index, open_block
                if open_block is None:
                    return []
                output = [frame({"type": "content_block_stop", "index": block_index})]
                block_index += 1
                open_block = None
                return output

            def emit_piece(kind: str, value: str, *, parse_tools: bool = True) -> list[bytes]:
                nonlocal emitted_reasoning, emitted_text
                if not value:
                    return []
                if kind == "thinking":
                    emitted_reasoning += value
                    if anthropic:
                        return [
                            *open_anthropic_block("thinking"),
                            frame(
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "thinking_delta", "thinking": value},
                                }
                            ),
                        ]
                    event = {
                        **common,
                        "choices": [{"index": 0, "delta": {"reasoning_content": value}, "finish_reason": None}],
                    }
                    return [f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()]

                normal_text = value
                calls: list[Any] = []
                if stream_tool_parser is not None and parse_tools:
                    parser = getattr(stream_tool_parser, "parse_stream_chunk", None)
                    if parser is None:
                        # Third-party test/fallback parsers without an
                        # incremental API cannot safely distinguish a partial
                        # tool marker from visible text. Reconciliation below
                        # emits the final parsed tool call at stream end.
                        normal_text = ""
                    else:
                        normal_text, calls = parser(value)
                output: list[bytes] = []
                if normal_text:
                    emitted_text += normal_text
                    if anthropic:
                        output.extend(open_anthropic_block("text"))
                        output.append(
                            frame(
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": normal_text},
                                }
                            )
                        )
                    else:
                        event = {
                            **common,
                            "choices": [{"index": 0, "delta": {"content": normal_text}, "finish_reason": None}],
                        }
                        output.append(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                for fallback_index, call in enumerate(calls):
                    tool_index = getattr(call, "tool_index", fallback_index)
                    tool_index = fallback_index if tool_index is None else int(tool_index)
                    name = getattr(call, "name", None)
                    if name:
                        tool_ids.setdefault(tool_index, f"call_{uuid.uuid4().hex[:24]}")
                        tool_names[tool_index] = str(name)
                    if tool_index not in tool_ids:
                        raise ValueError("SGLang tool stream emitted arguments before the tool name")
                    arguments = getattr(call, "parameters", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    emitted_tool_arguments[tool_index] = emitted_tool_arguments.get(tool_index, "") + arguments
                    if anthropic:
                        output.extend(open_anthropic_block("tool", tool_index))
                        if arguments:
                            output.append(
                                frame(
                                    {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                                    }
                                )
                            )
                    else:
                        tool_delta: dict[str, Any] = {
                            "index": tool_index,
                            "function": {"arguments": arguments},
                        }
                        if name:
                            tool_delta.update(
                                {
                                    "id": tool_ids[tool_index],
                                    "type": "function",
                                }
                            )
                            tool_delta["function"]["name"] = str(name)
                        event = {
                            **common,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [tool_delta]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        output.append(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                return output

            try:
                if anthropic:
                    yield frame(
                        {
                            "type": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "model": str(payload.get("model", self._model_path)),
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": len(prompt_ids), "output_tokens": 0},
                            },
                        }
                    )
                else:
                    event = {
                        **common,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

                async for native_event in _sse_json_events(upstream.chunks):
                    raw_delta = capture.accept(native_event)
                    for kind, value in reasoning.feed(raw_delta):
                        for output in emit_piece(kind, value):
                            yield output
                    if not anthropic and request.get("logprobs") is True and capture.last_output_logprobs:
                        token_ids, log_probs = self._output_tensors(
                            {"meta_info": {"output_token_logprobs": capture.last_output_logprobs}}
                        )
                        event = {
                            **common,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None,
                                    "logprobs": {"content": self._openai_logprobs(token_ids, log_probs)},
                                }
                            ],
                        }
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                for kind, value in reasoning.finish():
                    for output in emit_piece(kind, value):
                        yield output

                native = self._normalize_native_response(artifact, capture.response())
                output_ids, rollout_log_probs = self._output_tensors(native)
                version_spans = self._weight_version_spans(native, len(output_ids))
                # Streaming parsers are stateful. Parse the complete sample
                # with a fresh instance for the canonical training response.
                final_tool_parser = self._configured_tool_parser(request)
                response = self._chat_response(
                    request,
                    native,
                    prompt_ids=prompt_ids,
                    output_ids=output_ids,
                    rollout_log_probs=rollout_log_probs,
                    loss_mask=[1] * len(output_ids),
                    weight_version_spans=version_spans,
                    tool_parser=final_tool_parser,
                )
                response["id"] = chat_id
                response["created"] = created
                message = response["choices"][0]["message"]

                for index, call in enumerate(message.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    call_id = tool_ids.setdefault(index, str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}"))
                    call["id"] = call_id
                    function = call.get("function")
                    if isinstance(function, Mapping):
                        tool_names.setdefault(index, str(function.get("name", "")))

                final_reasoning = message.get("reasoning_content")
                if isinstance(final_reasoning, str) and final_reasoning.startswith(emitted_reasoning):
                    for output in emit_piece("thinking", final_reasoning[len(emitted_reasoning) :], parse_tools=False):
                        yield output
                final_text = message.get("content")
                if isinstance(final_text, str) and final_text.startswith(emitted_text):
                    for output in emit_piece("text", final_text[len(emitted_text) :], parse_tools=False):
                        yield output

                for index, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function") if isinstance(call, Mapping) else None
                    if not isinstance(function, Mapping):
                        continue
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    emitted = emitted_tool_arguments.get(index, "")
                    missing = arguments[len(emitted) :] if arguments.startswith(emitted) else arguments
                    if not missing and index in emitted_tool_arguments:
                        continue
                    if anthropic:
                        for output in open_anthropic_block("tool", index):
                            yield output
                        if missing:
                            yield frame(
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "input_json_delta", "partial_json": missing},
                                }
                            )
                    else:
                        tool_delta = {
                            "index": index,
                            "id": tool_ids[index],
                            "type": "function",
                            "function": {"name": tool_names[index], "arguments": missing},
                        }
                        event = {
                            **common,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [tool_delta]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

                if anthropic:
                    provider_response = self._anthropic_response(payload, request, response)
                    provider_response["id"] = message_id
                    stream_holder["stream"].record_response = provider_response
                    for output in close_anthropic_block():
                        yield output
                    yield frame(
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": provider_response["stop_reason"],
                                "stop_sequence": provider_response["stop_sequence"],
                            },
                            "usage": {"output_tokens": len(output_ids)},
                        }
                    )
                    yield frame({"type": "message_stop"})
                else:
                    stream_holder["stream"].record_response = response
                    choice = response["choices"][0]
                    event = {
                        **common,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": choice["finish_reason"],
                                "meta_info": choice["meta_info"],
                            }
                        ],
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
            finally:
                await upstream.close()

        stream = InferenceStream(
            status=200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            chunks=chunks(),
            close=upstream.close,
            record_response_pending=True,
        )
        stream_holder["stream"] = stream
        return stream

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
                    # cryptographic signature. SGLang's compatibility layer
                    # likewise emits an empty signature for local models.
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
            raise ValueError("exact SGLang chat capture currently requires text-only message content")
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
        return self._integer_tokens(rendered, "chat template input_ids")

    def _sampling_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        max_new_tokens = payload.get(
            "max_completion_tokens",
            payload.get(
                "max_tokens",
                payload.get("max_new_tokens", self._sampling_defaults.get("max_new_tokens", 512)),
            ),
        )
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_completion_tokens must be a positive integer")
        sampling: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
            "ignore_eos",
            "skip_special_tokens",
            "logit_bias",
        ):
            if key in payload:
                sampling[key] = payload[key]
        if "seed" in payload:
            sampling["sampling_seed"] = payload["seed"]
        for key, value in self._sampling_defaults.items():
            if key != "max_new_tokens":
                sampling.setdefault(key, value)
        extra = payload.get("sglang_sampling_params")
        if extra is not None:
            if not isinstance(extra, Mapping):
                raise ValueError("sglang_sampling_params must be an object")
            sampling.update(extra)
        return sampling

    def _chat_response(
        self,
        request: Mapping[str, Any],
        native: Mapping[str, Any],
        *,
        prompt_ids: list[int],
        output_ids: list[int],
        rollout_log_probs: list[float],
        loss_mask: list[int],
        weight_version_spans: list[dict[str, Any]],
        tool_parser: Any = None,
    ) -> dict[str, Any]:
        if len(output_ids) != len(rollout_log_probs) or len(output_ids) != len(loss_mask):
            raise ValueError("captured SGLang response tensors have inconsistent lengths")
        text = native.get("text")
        if not isinstance(text, str):
            raise ValueError("SGLang response is missing generated text")
        meta = native.get("meta_info")
        if not isinstance(meta, Mapping):
            raise ValueError("SGLang response is missing meta_info")
        public_meta = {
            key: value
            for key, value in meta.items()
            if key
            not in {
                "_reef_token_weight_versions",
                "input_token_logprobs",
                "output_token_logprobs",
            }
        }
        message, parsed_tool_calls = self._assistant_message(
            text, tool_parser, force_reasoning=self._reasoning_is_pre_opened()
        )
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": (
                "tool_calls" if parsed_tool_calls else self._openai_finish_reason(meta.get("finish_reason"))
            ),
            "meta_info": public_meta,
        }
        if request.get("logprobs") is True:
            choice["logprobs"] = {"content": self._openai_logprobs(output_ids, rollout_log_probs)}
        versions = {span["weight_version"] for span in weight_version_spans}
        weight_version = versions.pop() if len(versions) == 1 else None
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(request.get("model", self._model_path)),
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(output_ids),
                "total_tokens": len(prompt_ids) + len(output_ids),
            },
            "training": {
                "tokens": [*prompt_ids, *output_ids],
                "loss_mask": loss_mask,
                "rollout_log_probs": rollout_log_probs,
                "prompt_length": len(prompt_ids),
                "response_length": len(output_ids),
                "weight_version": None if weight_version is None else str(weight_version),
                "weight_version_spans": weight_version_spans,
                **self._captured_topk(meta, len(output_ids)),
            },
        }

    @staticmethod
    def _weight_version_spans(response: Mapping[str, Any], response_length: int) -> list[dict[str, Any]]:
        raw = response.get("_reef_token_weight_versions")
        if raw is None:
            meta = response.get("meta_info")
            version = meta.get("weight_version") if isinstance(meta, Mapping) else None
            raw = [version] * response_length
        if (
            not isinstance(raw, list)
            or len(raw) != response_length
            or any(not isinstance(version, str) or not version for version in raw)
        ):
            raise ValueError("SGLang response has incomplete token weight versions")
        spans: list[dict[str, Any]] = []
        for index, version in enumerate(raw):
            if spans and spans[-1]["weight_version"] == version:
                spans[-1]["end"] = index + 1
            else:
                spans.append({"start": index, "end": index + 1, "weight_version": version})
        return spans

    def _configured_tool_parser(self, request: Mapping[str, Any]) -> Any:
        tools = self._request_tools(request)
        if tools is None or request.get("tool_choice") == "none":
            return None
        if not self._tool_call_parser:
            raise ValueError(
                "tool calls require inference_backend_config.tool_call_parser to match the SGLang server configuration"
            )
        return self._tool_parser_factory(tools, self._tool_call_parser)

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

    # A harness that parses raw thinking tags itself opts out by overriding
    # this class attribute.
    _SPLIT_REASONING = True
    # Set when the chat template pre-opens ``<think>``: the sample then
    # carries only the closing tag, so a sample with no ``</think>`` is
    # reasoning that ran out of tokens, not an answer. Sniffed from the
    # rendered template at first use; ``force_reasoning`` in the backend
    # config pins it either way.
    _force_reasoning: bool | None = None

    def _reasoning_is_pre_opened(self) -> bool:
        """Whether this model's chat template opens ``<think>`` for the model.

        Qwen3-*-Thinking templates end the generation prompt with an open
        ``<think>``, so the sample carries only the closing tag and a sample
        without one is truncated reasoning. Sniffed once from the rendered
        prompt; an explicit ``force_reasoning`` in the backend config wins.
        """
        if self._force_reasoning is not None:
            return self._force_reasoning
        try:
            rendered = self._require_tokenizer().apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
            )
        except Exception:  # a template we cannot render tells us nothing
            self._force_reasoning = False
            return False
        self._force_reasoning = str(rendered).rstrip().endswith("<think>")
        return self._force_reasoning

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
        if cls._SPLIT_REASONING and "</think>" in text:
            head, _, tail = text.rpartition("</think>")
            reasoning = head.replace("<think>", "").strip()
            text = tail.lstrip("\n")
        elif cls._SPLIT_REASONING and force_reasoning:
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
            raise ValueError("SGLang tool-call parser could not parse the sampled output") from exc
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

    def _captured_topk(self, meta: Mapping[str, Any], response_length: int) -> dict[str, Any]:
        """Generation-time top-K capture for the OpenClaw-RL top-K objective.

        SGLang's ``output_top_logprobs`` lists, per generated position, the
        top-N ``[logprob, token_id, ...]`` entries. Rows are truncated or
        absent when the engine cannot provide them; a partial capture
        disables the channel for the whole response (all-or-nothing keeps
        the training contract rectangular).
        """
        if self._capture_topk <= 0:
            return {}
        entries = meta.get("output_top_logprobs")
        if not isinstance(entries, list) or len(entries) < response_length:
            return {}
        k = self._capture_topk
        indices: list[list[int]] = []
        log_probs: list[list[float]] = []
        for position in entries[:response_length]:
            if not isinstance(position, list) or len(position) < k:
                return {}
            row_idx: list[int] = []
            row_lp: list[float] = []
            for entry in position[:k]:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2 or entry[0] is None:
                    return {}
                row_lp.append(float(entry[0]))
                row_idx.append(int(entry[1]))
            indices.append(row_idx)
            log_probs.append(row_lp)
        return {"topk_indices": indices, "topk_log_probs": log_probs}

    @staticmethod
    def _output_tensors(response: Mapping[str, Any]) -> tuple[list[int], list[float]]:
        meta = response.get("meta_info")
        pairs = meta.get("output_token_logprobs") if isinstance(meta, Mapping) else None
        finish = meta.get("finish_reason") if isinstance(meta, Mapping) else None
        finish_type = finish.get("type") if isinstance(finish, Mapping) else finish
        if finish_type == "abort" and not pairs:
            return [], []
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("SGLang response is missing non-empty meta_info.output_token_logprobs")
        output_ids: list[int] = []
        log_probs: list[float] = []
        for index, pair in enumerate(pairs):
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) < 2
                or not isinstance(pair[0], (int, float))
                or isinstance(pair[0], bool)
                or not math.isfinite(float(pair[0]))
                or not isinstance(pair[1], int)
                or isinstance(pair[1], bool)
            ):
                raise ValueError(f"invalid SGLang output_token_logprobs entry at index {index}")
            log_probs.append(float(pair[0]))
            output_ids.append(pair[1])
        return output_ids, log_probs

    @staticmethod
    def _integer_tokens(value: Any, label: str) -> list[int]:
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(token, int) or isinstance(token, bool) for token in value)
        ):
            raise ValueError(f"{label} must be a non-empty integer list")
        return list(value)

    @staticmethod
    def _openai_finish_reason(value: Any) -> str:
        reason = value.get("type") if isinstance(value, Mapping) else value
        if reason == "length":
            return "length"
        if reason == "abort":
            return "abort"
        return "stop"


__all__ = ["SGLangChatTrainingInferenceBackend"]
