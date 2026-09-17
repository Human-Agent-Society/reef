"""The chat request a recorded inference carries, in the form a chat template consumes.

A record's payload is the provider request as the harness sent it; a rollout
backend also retains a provider-neutral copy of the messages and tools under
``response.training`` (``request_messages`` and ``request_tools``).
:func:`recorded_request` reads whichever the record has, and
:func:`normalize_messages_for_template` turns OpenAI-style messages into the
plain form Hugging Face chat templates expect: text content, chat roles, and
tool-call arguments as objects; :func:`recorded_response` reads the text the
model answered with. Recipes that render a recorded request again, with a
demonstration added or as the turn to train, share these readers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def recorded_request(payload: Mapping[str, Any]) -> tuple[list[Any], list[Any] | None]:
    """The messages and tools of a recorded inference request.

    The rollout backend retains its provider-neutral copy under
    ``response.training``; a record without it carries the request body.
    """
    response = payload.get("response")
    training = response.get("training") if isinstance(response, Mapping) else None
    if isinstance(training, Mapping) and isinstance(training.get("request_messages"), list):
        messages = list(training["request_messages"])
        tools = training.get("request_tools", payload.get("tools"))
    else:
        messages = list(payload.get("messages") or [])
        tools = payload.get("tools")
    return messages, list(tools) if isinstance(tools, list) and tools else None


def recorded_response(payload: Mapping[str, Any]) -> str:
    """The text of a recorded inference's response: the final assistant message, or empty when it has none."""
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return ""
    training = response.get("training")
    message = training.get("response_message") if isinstance(training, Mapping) else None
    if isinstance(message, Mapping):
        return flatten_content(message.get("content"))
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            return flatten_content(message.get("content"))
        return flatten_content(choices[0].get("text"))
    return ""


def flatten_content(content: Any) -> str:
    """The plain text of an OpenAI-style message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, Mapping) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def normalize_messages_for_template(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Messages as a chat template expects them: text content, chat roles, tool arguments as objects."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        entry = dict(message)
        if entry.get("role") == "developer":
            entry["role"] = "system"
        content = entry.get("content")
        if content is not None and not isinstance(content, str):
            entry["content"] = flatten_content(content)
        if entry.get("tool_calls"):
            entry["tool_calls"] = [normalize_tool_call(call) for call in entry["tool_calls"]]
        normalized.append(entry)
    return normalized


def normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """A tool call with its function arguments as an object, as chat templates render them."""
    normalized = dict(call)
    function = normalized.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                function["arguments"] = {}
        normalized["function"] = function
    return normalized
