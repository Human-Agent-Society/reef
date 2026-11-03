"""Convert Anthropic Messages payloads to and from Reef's internal chat shape.

These functions validate provider fields and translate text, reasoning, tools,
and usage. They do not render prompts or manage inference and stream state.
"""

import json
import uuid
from collections.abc import Mapping
from typing import Any

from reef.inference.provider_protocol import usage_of

#: Provider fields forwarded unchanged to native engines.
NATIVE_REQUEST_KEYS = ("lora_path", "rid", "sglang_sampling_params", "vllm_sampling_params")


def anthropic_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an Anthropic Messages request to the internal chat shape.

    The normalized request is what the tokenizer sees. Keeping this
    boundary provider-independent lets one chat template render OpenAI
    and Anthropic traffic alike.
    """

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Anthropic messages requires a non-empty model")
    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("Anthropic messages max_tokens must be a positive integer")

    messages: list[dict[str, Any]] = []
    system = anthropic_text_content(payload.get("system"), "system", allow_missing=True)
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
            messages.extend(anthropic_user_messages(raw_message.get("content"), index))
        elif role == "assistant":
            messages.append(anthropic_assistant_message(raw_message.get("content"), index))
        elif role == "system":
            text = anthropic_text_content(raw_message.get("content"), f"message {index} system")
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

    tools = anthropic_tools(payload.get("tools"))
    if tools is not None:
        request["tools"] = tools
        request["tool_choice"] = anthropic_tool_choice(payload.get("tool_choice"))
    elif payload.get("tool_choice") is not None:
        raise ValueError("Anthropic tool_choice requires tools")
    return request


def anthropic_user_messages(content: Any, message_index: int) -> list[dict[str, Any]]:
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
            raise ValueError(f"Anthropic user message {message_index} content block {block_index} must be an object")
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Anthropic text blocks require string text")
            text_parts.append(text)
            continue
        if kind != "tool_result":
            raise ValueError(f"Anthropic content block type {kind!r} is not supported by exact text-token training")
        flush_text()
        tool_call_id = block.get("tool_use_id", block.get("id"))
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("Anthropic tool_result requires tool_use_id")
        tool_content = anthropic_text_content(
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


def anthropic_assistant_message(content: Any, message_index: int) -> dict[str, Any]:
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


def anthropic_text_content(content: Any, label: str, *, allow_missing: bool = False) -> str:
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


def anthropic_tools(value: Any) -> list[dict[str, Any]] | None:
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


def anthropic_tool_choice(value: Any) -> Any:
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


def anthropic_response(original_request: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
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

    # Anthropic has no stop reason for an engine abort: the response names it in its meta_info, which the
    # service reads (inference_aborted); an aborted response never reaches the client.
    aborted = {"meta_info": {"finish_reason": "abort"}} if choice.get("finish_reason") == "abort" else {}
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": str(original_request.get("model", response.get("model", ""))),
        "content": content,
        "stop_reason": anthropic_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": usage_of(response) or {"input_tokens": 0, "output_tokens": 0},
        **aborted,
    }


def anthropic_stop_reason(value: Any) -> str:
    if value == "tool_calls":
        return "tool_use"
    if value == "length":
        return "max_tokens"
    return "end_turn"
