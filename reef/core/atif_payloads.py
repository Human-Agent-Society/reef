"""Provider payloads as ATIF, in both directions and without loss (RFC #1054).

A payload is the body of one model call with its ``response``: an OpenAI chat completion (``messages``), an
Anthropic message (``messages`` with ``system``), or an OpenAI response (``input`` with ``instructions``). This
module turns such a payload into ATIF objects and back:

* Each request message, and a top-level ``system`` or ``instructions`` before them, becomes an entry: a list of ATIF
  elements. An element is an ATIF step without ``step_id`` (it has ``source``), or an ATIF observation result (a
  tool result; it has ``source_call_id``). A trajectory numbers its steps and attaches each result to the step that
  made its tool call (``folded_steps``).
* ``tools`` becomes an ATIF agent fragment: function tools in ``tool_definitions``, other tools in
  ``extra.reef.other_tools``.
* The reply in ``response`` becomes agent steps; the other request fields and the rest of the response stay with
  the call.

The ATIF fields cannot hold every provider field: ``cache_control``, a tool call's ``arguments`` string, an
Anthropic thinking ``signature``, and the like. Each conversion therefore renders its ATIF objects back to the
provider form and keeps the difference to the original as a patch (``patch_of``) in ``extra.reef.patch`` of the
entry's first element, of the agent fragment, or of the call. Rendering and then applying the patch gives the
original value exactly, so ``provider_payload(atif_call(P))`` equals ``P``.
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The providers whose payload shapes this module knows.
OPENAI_CHAT = "openai-chat"
ANTHROPIC = "anthropic"
OPENAI_RESPONSES = "openai-responses"

#: The media types that an ATIF image part can name.
ATIF_IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def canonical_json(value: object) -> str:
    """The JSON of ``value`` with sorted keys, compact separators and non-ASCII characters kept."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def patch_of(original: Any, rendered: Any) -> dict[str, Any] | None:
    """The patch that turns the JSON value ``rendered`` into ``original``, or ``None`` when they are equal.

    A patch is ``{"=": v}`` (the value is ``v``), ``{"~": {key: patch}}`` (patch these keys of an object; a key's
    patch ``{"-": true}`` removes it), or ``{"[]": {index: patch}}`` (patch these items of a list of the same
    length). Values are compared by their canonical JSON, so ``1`` and ``True`` differ.
    """
    if canonical_json(original) == canonical_json(rendered):
        return None
    if isinstance(original, Mapping) and isinstance(rendered, Mapping):
        changes: dict[str, Any] = {}
        for key, value in original.items():
            if key not in rendered:
                changes[key] = {"=": value}
            else:
                change = patch_of(value, rendered[key])
                if change is not None:
                    changes[key] = change
        for key in rendered:
            if key not in original:
                changes[key] = {"-": True}
        return {"~": changes}
    if isinstance(original, list) and isinstance(rendered, list) and len(original) == len(rendered):
        items = {}
        for index, (value, rendered_value) in enumerate(zip(original, rendered, strict=True)):
            change = patch_of(value, rendered_value)
            if change is not None:
                items[str(index)] = change
        return {"[]": items}
    return {"=": original}


def patched(rendered: Any, patch: Mapping[str, Any] | None) -> Any:
    """``rendered`` with ``patch`` applied (``patch_of``); ``rendered`` itself when ``patch`` is ``None``.

    Raise ``ValueError`` when the patch does not fit the value.
    """
    if patch is None:
        return rendered
    if "=" in patch:
        return patch["="]
    if "~" in patch:
        if not isinstance(rendered, Mapping):
            raise ValueError("an object patch applies to an object")
        result = dict(rendered)
        for key, change in patch["~"].items():
            if "-" in change:
                result.pop(key, None)
            else:
                result[key] = patched(result.get(key), change)
        return result
    if "[]" in patch:
        if not isinstance(rendered, list):
            raise ValueError("a list patch applies to a list")
        items = list(rendered)
        for index, change in patch["[]"].items():
            items[int(index)] = patched(items[int(index)], change)
        return items
    raise ValueError(f"unknown patch {patch!r}")


def element_patch(element: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The patch that an element, an agent fragment or a call keeps in ``extra.reef.patch``, if any."""
    extra = element.get("extra")
    reef = extra.get("reef") if isinstance(extra, Mapping) else None
    return reef.get("patch") if isinstance(reef, Mapping) else None


def with_patch(element: dict[str, Any], patch: Mapping[str, Any] | None) -> dict[str, Any]:
    """``element`` with ``patch`` in ``extra.reef.patch``; ``element`` itself when there is no patch."""
    if patch is None:
        return element
    return {**element, "extra": {"reef": {"patch": dict(patch)}}}


def without_patch(element: Mapping[str, Any]) -> dict[str, Any]:
    """``element`` without the patch that ``with_patch`` added."""
    return {key: value for key, value in element.items() if key != "extra"}


# --- Content ---------------------------------------------------------------------------------------------------


def image_path(part: Mapping[str, Any]) -> tuple[str, str] | None:
    """The ``(media_type, path)`` of a provider image part that ATIF can name, or ``None``."""
    source = part.get("source")
    image_url = part.get("image_url")
    path = image_url.get("url") if isinstance(image_url, Mapping) else image_url
    if isinstance(source, Mapping):
        if source.get("type") == "base64":
            path = f"data:{source.get('media_type')};base64,{source.get('data')}"
        else:
            path = path or source.get("url") or source.get("path")
    if not isinstance(path, str):
        return None
    media_type = (source.get("media_type") if isinstance(source, Mapping) else None) or (
        path[5:].split(";", 1)[0] if path.startswith("data:") else mimetypes.guess_type(path.split("?", 1)[0])[0]
    )
    return (media_type, path) if media_type in ATIF_IMAGE_TYPES else None


def atif_message(content: Any) -> str | list[dict[str, Any]]:
    """The ATIF message of a provider message's content: a string, or a list of ATIF content parts.

    A text part keeps its text, an image part its media type and path, and any other part becomes a text part of
    its JSON. ``None`` becomes ``""``.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return canonical_json(content)
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, Mapping) and part.get("type") in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "text": str(part.get("text", ""))})
            continue
        image = image_path(part) if isinstance(part, Mapping) else None
        if image is not None:
            parts.append({"type": "image", "source": {"media_type": image[0], "path": image[1]}})
        else:
            parts.append({"type": "text", "text": canonical_json(part)})
    return parts


def provider_content(message: Any, provider: str, *, assistant: bool = False) -> Any:
    """The provider content of an ATIF message: a string stays a string, and each part takes the provider's form."""
    if not isinstance(message, list):
        return message
    content: list[dict[str, Any]] = []
    for part in message:
        if part.get("type") == "image":
            path = part["source"]["path"]
            if provider == ANTHROPIC:
                if path.startswith("data:") and ";base64," in path:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part["source"]["media_type"],
                                "data": path.split(";base64,", 1)[1],
                            },
                        }
                    )
                else:
                    content.append({"type": "image", "source": {"type": "url", "url": path}})
            elif provider == OPENAI_RESPONSES:
                content.append({"type": "input_image", "image_url": path})
            else:
                content.append({"type": "image_url", "image_url": {"url": path}})
        else:
            kind = "text"
            if provider == OPENAI_RESPONSES:
                kind = "output_text" if assistant else "input_text"
            content.append({"type": kind, "text": part.get("text", "")})
    return content


# --- Tool calls --------------------------------------------------------------------------------------------------


def atif_tool_call(call_id: Any, name: Any, arguments: Any) -> dict[str, Any]:
    """The ATIF tool call of a provider call; a JSON-string ``arguments`` is parsed into an object."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments}
    if not isinstance(arguments, Mapping):
        arguments = {"raw_arguments": arguments}
    return {"tool_call_id": str(call_id), "function_name": str(name), "arguments": dict(arguments)}


def arguments_json(arguments: Mapping[str, Any]) -> str:
    """The JSON string of a tool call's arguments, as a provider sends it."""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


# --- Messages ----------------------------------------------------------------------------------------------------


def step(source: str, message: Any, *, copied: bool = False, **fields: Any) -> dict[str, Any]:
    """An ATIF step without ``step_id``; ``copied`` marks an agent step that a request resent."""
    result: dict[str, Any] = {"source": source, "message": message}
    result.update({key: value for key, value in fields.items() if value})
    if copied and source == "agent":
        result["is_copied_context"] = True
    return result


def source_of(role: Any) -> str:
    """The ATIF source of a provider role."""
    if role == "assistant":
        return "agent"
    return "system" if role in ("system", "developer") else "user"


def openai_chat_elements(message: Mapping[str, Any], *, copied: bool) -> list[dict[str, Any]]:
    role = message.get("role")
    if role == "tool":
        return [{"source_call_id": str(message.get("tool_call_id")), "content": atif_message(message.get("content"))}]
    calls = [
        atif_tool_call(
            call.get("id"), (call.get("function") or {}).get("name"), (call.get("function") or {}).get("arguments")
        )
        for call in message.get("tool_calls") or []
        if isinstance(call, Mapping)
    ]
    source = source_of(role)
    reasoning = message.get("reasoning_content") if source == "agent" else None
    return [
        step(
            source,
            atif_message(message.get("content")),
            copied=copied,
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
            tool_calls=calls if source == "agent" else None,
        )
    ]


def openai_chat_message(elements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    element = elements[0]
    if "source_call_id" in element:
        return {
            "role": "tool",
            "tool_call_id": element["source_call_id"],
            "content": provider_content(element.get("content"), OPENAI_CHAT),
        }
    role = {"agent": "assistant", "system": "system"}.get(element["source"], "user")
    # An assistant message that only calls tools has no content.
    content = None if element["message"] == "" and element.get("tool_calls") else element["message"]
    message: dict[str, Any] = {"role": role, "content": provider_content(content, OPENAI_CHAT)}
    if element.get("tool_calls"):
        message["tool_calls"] = [
            {
                "id": call["tool_call_id"],
                "type": "function",
                "function": {"name": call["function_name"], "arguments": arguments_json(call["arguments"])},
            }
            for call in element["tool_calls"]
        ]
    if element.get("reasoning_content"):
        message["reasoning_content"] = element["reasoning_content"]
    return message


def anthropic_elements(message: Mapping[str, Any], *, copied: bool) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(content, list):
        return [step(source_of(role), atif_message(content), copied=copied)]
    blocks = [block for block in content if isinstance(block, Mapping)]
    results = [
        {"source_call_id": str(block.get("tool_use_id")), "content": atif_message(block.get("content"))}
        for block in blocks
        if block.get("type") == "tool_result"
    ]
    calls = [
        atif_tool_call(block.get("id"), block.get("name"), block.get("input", {}))
        for block in blocks
        if block.get("type") == "tool_use"
    ]
    reasoning = "".join(str(block.get("thinking", "")) for block in blocks if block.get("type") == "thinking")
    parts = [
        block
        for block in content
        if not isinstance(block, Mapping)
        or block.get("type") not in ("tool_result", "tool_use", "thinking", "redacted_thinking")
    ]
    source = source_of(role)
    if results and not parts and source == "user":
        return list(results)
    elements: list[dict[str, Any]] = list(results)
    elements.append(
        step(
            source,
            atif_message(parts) if parts else "",
            copied=copied,
            reasoning_content=reasoning if source == "agent" else None,
            tool_calls=calls if source == "agent" else None,
        )
    )
    return elements


def anthropic_message(elements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    content: list[Any] = []
    role = "user"
    for element in elements:
        if "source_call_id" in element:
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": element["source_call_id"],
                    "content": provider_content(element.get("content"), ANTHROPIC),
                }
            )
            continue
        role = "assistant" if element["source"] == "agent" else "user"
        if element.get("reasoning_content"):
            content.append({"type": "thinking", "thinking": element["reasoning_content"]})
        message = element["message"]
        if isinstance(message, str):
            if len(elements) == 1 and not element.get("tool_calls") and not element.get("reasoning_content"):
                return {"role": role, "content": message}
            if message:
                content.append({"type": "text", "text": message})
        else:
            content.extend(provider_content(message, ANTHROPIC))
        content.extend(
            {"type": "tool_use", "id": call["tool_call_id"], "name": call["function_name"], "input": call["arguments"]}
            for call in element.get("tool_calls") or []
        )
    return {"role": role, "content": content}


def openai_responses_elements(item: Mapping[str, Any], *, copied: bool) -> list[dict[str, Any]]:
    kind = item.get("type")
    if kind == "function_call":
        return [
            step(
                "agent",
                "",
                copied=copied,
                tool_calls=[atif_tool_call(item.get("call_id"), item.get("name"), item.get("arguments"))],
            )
        ]
    if kind == "function_call_output":
        return [{"source_call_id": str(item.get("call_id")), "content": atif_message(item.get("output"))}]
    if kind == "reasoning":
        summary = item["summary"] if isinstance(item.get("summary"), list) else []
        reasoning = "\n".join(str(part.get("text", "")) for part in summary if isinstance(part, Mapping))
        return [step("agent", "", copied=copied, reasoning_content=reasoning or None)]
    if "role" in item:
        return [step(source_of(item.get("role")), atif_message(item.get("content")), copied=copied)]
    return [step("user", "")]


def openai_responses_item(elements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    element = elements[0]
    if "source_call_id" in element:
        return {
            "type": "function_call_output",
            "call_id": element["source_call_id"],
            "output": provider_content(element.get("content"), OPENAI_RESPONSES),
        }
    calls = element.get("tool_calls") or []
    if element["source"] == "agent" and calls:
        call = calls[0]
        return {
            "type": "function_call",
            "call_id": call["tool_call_id"],
            "name": call["function_name"],
            "arguments": arguments_json(call["arguments"]),
        }
    if element["source"] == "agent" and element.get("reasoning_content") and element["message"] == "":
        return {"type": "reasoning", "summary": [{"type": "summary_text", "text": element["reasoning_content"]}]}
    role = {"agent": "assistant", "system": "system"}.get(element["source"], "user")
    return {
        "role": role,
        "content": provider_content(element["message"], OPENAI_RESPONSES, assistant=role == "assistant"),
    }


ELEMENTS = {
    OPENAI_CHAT: openai_chat_elements,
    ANTHROPIC: anthropic_elements,
    OPENAI_RESPONSES: openai_responses_elements,
}
RENDER = {OPENAI_CHAT: openai_chat_message, ANTHROPIC: anthropic_message, OPENAI_RESPONSES: openai_responses_item}


def message_entry(message: Any, provider: str, *, copied: bool) -> list[dict[str, Any]]:
    """The ATIF elements of one provider message, with the patch that restores the message exactly."""
    elements = ELEMENTS[provider](message, copied=copied) if isinstance(message, Mapping) else [step("user", "")]
    return [with_patch(elements[0], patch_of(message, RENDER[provider](elements))), *elements[1:]]


def provider_message(entry: Sequence[Mapping[str, Any]], provider: str) -> Any:
    """The provider message of an entry (``message_entry``)."""
    elements = [without_patch(element) for element in entry]
    return patched(RENDER[provider](elements), element_patch(entry[0]))


def system_entry(value: Any) -> list[dict[str, Any]]:
    """The ATIF elements of a top-level ``system`` or ``instructions`` value: one system step."""
    element = step("system", atif_message(value))
    return [with_patch(element, patch_of(value, system_value(element)))]


def system_value(element: Mapping[str, Any]) -> Any:
    """The provider value of a top-level system step, before its patch."""
    message = element["message"]
    return message if isinstance(message, str) else provider_content(message, ANTHROPIC)


# --- Tools -------------------------------------------------------------------------------------------------------


def atif_agent(tools: Any, provider: str) -> dict[str, Any]:
    """The ATIF agent fragment of a payload's ``tools``, with the patch that restores them exactly.

    Function tools take OpenAI's function schema in ``tool_definitions``; other tools, such as server tools, stay
    in ``extra.reef.other_tools`` with their positions.
    """
    definitions: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for index, tool in enumerate(tools if isinstance(tools, list) else []):
        function = tool_function(tool, provider) if isinstance(tool, Mapping) else None
        if function is None:
            others.append({"index": index, "tool": tool})
        else:
            definitions.append({"type": "function", "function": function})
    agent: dict[str, Any] = {"tool_definitions": definitions}
    reef: dict[str, Any] = {"other_tools": others} if others else {}
    patch = patch_of(tools, provider_tools({**agent, "extra": {"reef": reef}}, provider))
    if patch is not None:
        reef["patch"] = patch
    return {**agent, "extra": {"reef": reef}} if reef else agent


def tool_function(tool: Mapping[str, Any], provider: str) -> dict[str, Any] | None:
    """The OpenAI function schema of a provider tool, or ``None`` for a tool that is not a function."""
    if provider == ANTHROPIC:
        if "input_schema" not in tool:
            return None
        function = {"name": tool.get("name"), "parameters": tool.get("input_schema")}
    elif tool.get("type") != "function":
        return None
    elif provider == OPENAI_RESPONSES:
        function = {"name": tool.get("name"), "parameters": tool.get("parameters")}
    else:
        inner = tool.get("function")
        if not isinstance(inner, Mapping):
            return None
        function = {"name": inner.get("name"), "parameters": inner.get("parameters")}
        tool = inner
    if function["parameters"] is None:
        # A tool without parameters has none in its schema either, as a chat template expects.
        function.pop("parameters")
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    return function


def provider_tools(agent: Mapping[str, Any], provider: str) -> Any:
    """The provider ``tools`` of an ATIF agent fragment (``atif_agent``)."""
    tools: list[Any] = []
    for definition in agent.get("tool_definitions") or []:
        function = definition["function"]
        if provider == ANTHROPIC:
            tool = {"name": function.get("name"), "input_schema": function.get("parameters")}
        elif provider == OPENAI_RESPONSES:
            tool = {"type": "function", "name": function.get("name"), "parameters": function.get("parameters")}
        else:
            tool = {"type": "function", "function": dict(function)}
        if provider != OPENAI_CHAT and "description" in function:
            tool["description"] = function["description"]
        tools.append(tool)
    extra = agent.get("extra")
    reef = extra.get("reef", {}) if isinstance(extra, Mapping) else {}
    for other in reef.get("other_tools", []):
        tools.insert(other["index"], other["tool"])
    return patched(tools, reef.get("patch"))


# --- Replies -----------------------------------------------------------------------------------------------------


def reply_messages(response: Any) -> list[Any] | None:
    """The messages of a response that a client appends to its next request, or ``None``.

    A chat completion gives the message of its first choice, a response its ``output`` items, and an Anthropic
    message its role and content. A response without one of these shapes has no reply messages.
    """
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        return [message] if isinstance(message, Mapping) else None
    output = response.get("output")
    if isinstance(output, list):
        return output
    if response.get("type") == "message" and isinstance(response.get("content"), list):
        return [{"role": response.get("role", "assistant"), "content": response["content"]}]
    return None


def reply_skeleton(messages: list[Any], provider: str) -> dict[str, Any]:
    """The response of the reply messages ``messages``, before the call's patch."""
    if provider == OPENAI_RESPONSES:
        return {"output": messages}
    if provider == ANTHROPIC:
        message = messages[0] if messages else {"role": "assistant", "content": []}
        return {"type": "message", "role": message.get("role"), "content": message.get("content")}
    return {"choices": [{"message": message} for message in messages]}


def usage_metrics(response: Mapping[str, Any]) -> dict[str, int] | None:
    """The ATIF metrics of a response's ``usage``: prompt, completion and cached tokens."""
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    candidates = {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "cached_tokens": (details.get("cached_tokens") if isinstance(details, Mapping) else None)
        or usage.get("cache_read_input_tokens"),
    }
    metrics = {
        key: value for key, value in candidates.items() if isinstance(value, int) and not isinstance(value, bool)
    }
    return metrics or None


# --- Calls -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallShape:
    """Where a payload keeps its parts.

    Values: ``provider`` is ``OPENAI_CHAT``, ``ANTHROPIC`` or ``OPENAI_RESPONSES``; ``message_field`` is
    ``"messages"`` or ``"input"``; ``system_field`` is ``"system"`` or ``"instructions"`` when the payload has that
    top-level field, and ``None`` otherwise; ``has_tools`` says whether the payload has ``tools``. The calls of
    one trajectory have one shape. The constructor checks the names.

    Purpose: ``atif_call`` finds it, and ``provider_payload`` puts the parts back where they were.
    """

    provider: str
    message_field: str
    system_field: str | None
    has_tools: bool

    def __post_init__(self) -> None:
        if self.provider not in RENDER:
            raise ValueError(f"unknown provider {self.provider!r}")
        if self.message_field not in ("messages", "input"):
            raise ValueError(f"a message list is in 'messages' or 'input', not {self.message_field!r}")
        if self.system_field not in (None, "system", "instructions"):
            raise ValueError(f"a system field is 'system' or 'instructions', not {self.system_field!r}")

    def as_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "message_field": self.message_field,
            "system_field": self.system_field,
            "has_tools": self.has_tools,
        }


@dataclass(frozen=True)
class AtifCall:
    """One model call as ATIF objects, from which its payload renders exactly.

    Values: ``shape`` says where the payload keeps its parts. ``entries`` holds one entry per request message, in
    order, after an entry of the system field when ``shape.system_field`` names one (``message_entry``,
    ``system_entry``). ``agent`` is the ATIF agent fragment of ``tools`` (``atif_agent``), or ``None`` without
    tools. ``reply`` holds the ATIF elements of the response's reply messages, one entry per message, without
    patches. ``call`` holds the other request fields (``request``) and, when the payload has a response, the patch
    that turns the reply's skeleton into it (``response_patch``); the reply's ATIF ``metrics`` are a view of its
    usage.

    Purpose: ``atif_call`` makes it from a payload; a SQL record store keeps it in its trajectory nodes, and
    ``make_trajectory`` folds calls into an ATIF document. ``provider_payload`` renders the payload back.
    """

    shape: CallShape
    entries: tuple[list[dict[str, Any]], ...]
    agent: dict[str, Any] | None
    reply: tuple[list[dict[str, Any]], ...]
    call: dict[str, Any] = field(default_factory=dict)


def payload_provider(payload: Mapping[str, Any]) -> str | None:
    """The provider of a payload with a message list, by its fields, or ``None`` for a payload without one."""
    if isinstance(payload.get("messages"), list):
        response = payload.get("response")
        if isinstance(response, Mapping) and response.get("type") == "message":
            return ANTHROPIC
        if "system" in payload:
            return ANTHROPIC
        blocks = (
            block
            for message in payload["messages"]
            if isinstance(message, Mapping) and isinstance(message.get("content"), list)
            for block in message["content"]
        )
        anthropic_blocks = ("tool_use", "tool_result", "thinking", "redacted_thinking")
        if any(isinstance(block, Mapping) and block.get("type") in anthropic_blocks for block in blocks):
            return ANTHROPIC
        return OPENAI_CHAT
    if isinstance(payload.get("input"), list):
        return OPENAI_RESPONSES
    return None


def call_shape(payload: Mapping[str, Any]) -> CallShape | None:
    """Where ``payload`` keeps its parts, or ``None`` when the payload has no message list."""
    provider = payload_provider(payload)
    if provider is None:
        return None
    system_field = next(
        (name for name in ("system", "instructions") if name in payload and provider != OPENAI_CHAT), None
    )
    message_field = "input" if provider == OPENAI_RESPONSES else "messages"
    return CallShape(provider, message_field, system_field, "tools" in payload)


def atif_entries(payload: Mapping[str, Any], shape: CallShape, start: int = 0) -> list[list[dict[str, Any]]]:
    """The entries of ``payload``'s request values from position ``start`` on: the system field's value first, when
    ``shape`` names one, then each request message (``system_entry``, ``message_entry``)."""
    values: list[Any] = list(payload[shape.message_field])
    if shape.system_field is not None:
        values.insert(0, payload[shape.system_field])
    entries: list[list[dict[str, Any]]] = []
    for index in range(start, len(values)):
        if index == 0 and shape.system_field is not None:
            entries.append(system_entry(values[0]))
        else:
            entries.append(message_entry(values[index], shape.provider, copied=True))
    return entries


def atif_reply(
    payload: Mapping[str, Any], shape: CallShape
) -> tuple[tuple[list[dict[str, Any]], ...], dict[str, Any]]:
    """The reply entries of ``payload`` and its call fields, as ``AtifCall.reply`` and ``AtifCall.call`` hold them."""
    reply: list[list[dict[str, Any]]] = []
    call: dict[str, Any] = {
        "request": {
            key: value
            for key, value in payload.items()
            if key not in (shape.message_field, shape.system_field, "tools", "response")
        }
    }
    if "response" in payload:
        response = payload["response"]
        messages = reply_messages(response) or []
        reply = [
            [without_patch(element) for element in message_entry(message, shape.provider, copied=False)]
            for message in messages
        ]
        rendered = (
            reply_skeleton([RENDER[shape.provider](entry) for entry in reply], shape.provider) if messages else {}
        )
        # A response without a patch keeps the key, so that the call still has a response.
        call["response_patch"] = patch_of(response, rendered)
        metrics = usage_metrics(response) if isinstance(response, Mapping) else None
        if metrics and reply and reply[-1] and reply[-1][-1].get("source") == "agent":
            reply[-1][-1] = {**reply[-1][-1], "metrics": metrics}
    return tuple(reply), call


def atif_call(payload: Mapping[str, Any]) -> AtifCall | None:
    """The ATIF call of ``payload``, or ``None`` when the payload has no message list."""
    shape = call_shape(payload)
    if shape is None:
        return None
    agent = atif_agent(payload["tools"], shape.provider) if shape.has_tools else None
    reply, call = atif_reply(payload, shape)
    return AtifCall(shape, tuple(atif_entries(payload, shape)), agent, reply, call)


def provider_payload(call: AtifCall) -> dict[str, Any]:
    """The payload of an ATIF call (``atif_call``), exactly as it was given."""
    shape = call.shape
    entries = list(call.entries)
    payload: dict[str, Any] = dict(call.call.get("request", {}))
    if shape.system_field is not None:
        first = entries.pop(0)
        payload[shape.system_field] = patched(system_value(without_patch(first[0])), element_patch(first[0]))
    payload[shape.message_field] = [provider_message(entry, shape.provider) for entry in entries]
    if shape.has_tools:
        payload["tools"] = provider_tools(call.agent or {}, shape.provider)
    if "response_patch" in call.call:
        messages = [RENDER[shape.provider]([without_metrics(element) for element in entry]) for entry in call.reply]
        rendered = reply_skeleton(messages, shape.provider) if messages else {}
        payload["response"] = patched(rendered, call.call.get("response_patch"))
    return dict(sorted(payload.items()))


def without_metrics(element: Mapping[str, Any]) -> dict[str, Any]:
    """An element without its ATIF ``metrics``, which are a view of the response's usage."""
    return {key: value for key, value in element.items() if key != "metrics"}


# --- Trajectories ------------------------------------------------------------------------------------------------


def plain_message(message: Any) -> Any:
    """An ATIF message whose parts are all text as one string, as ATIF writes text-only content."""
    if isinstance(message, list) and all(part.get("type") == "text" for part in message):
        return "".join(str(part.get("text", "")) for part in message)
    return message


def folded_steps(entries: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """The ATIF steps of entries in order, numbered from 1.

    Each observation result joins the observation of the latest step that made its tool call. A result whose call
    no step made becomes a ``user`` step with its content, and keeps the result in ``extra.reef.observation``.
    """
    steps: list[dict[str, Any]] = []
    for entry in entries:
        for element in entry:
            if "source" in element:
                steps.append({"step_id": len(steps) + 1, **element, "message": plain_message(element["message"])})
                continue
            call_id = element.get("source_call_id")
            owner = next(
                (
                    candidate
                    for candidate in reversed(steps)
                    if any(call["tool_call_id"] == call_id for call in candidate.get("tool_calls") or [])
                ),
                None,
            )
            if owner is None:
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "user",
                        "message": element.get("content") if element.get("content") is not None else "",
                        "extra": {"reef": {"observation": dict(element)}},
                    }
                )
            else:
                owner.setdefault("observation", {"results": []})["results"].append(dict(element))
    return steps


__all__ = [
    "ANTHROPIC",
    "OPENAI_CHAT",
    "OPENAI_RESPONSES",
    "AtifCall",
    "CallShape",
    "atif_agent",
    "atif_call",
    "atif_entries",
    "atif_message",
    "atif_reply",
    "call_shape",
    "canonical_json",
    "folded_steps",
    "message_entry",
    "patch_of",
    "patched",
    "provider_message",
    "provider_payload",
    "provider_tools",
    "reply_messages",
]
