"""SDFT's processor: the shared distillation processor with the demonstration appended to the request."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from reef.train.processors import DistillProcessor
from reef.train.processors.common import flatten_content
from reef.train.types import ProcessorContext

#: The reference implementation's demonstration block (idanshen/Self-Distillation, ``main.py``).
DEFAULT_CONTEXT_TEMPLATE = (
    "This is an example for a response to the question:\n{context}\n\n"
    "Now answer with a response of your own, including the thinking process."
)
CONTEXT_PLACEHOLDER = "{context}"


def context_block_template(config: Mapping[str, Any]) -> str:
    """The demonstration block's template from the processor config, checked for its placeholder."""
    template = str(config.get("context_template", DEFAULT_CONTEXT_TEMPLATE))
    if CONTEXT_PLACEHOLDER not in template:
        raise ValueError(f"context_template must contain {CONTEXT_PLACEHOLDER}")
    return template


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


class SDFTProcessor(DistillProcessor):
    """One rollout and its demonstration, one training unit.

    The teacher reads the student's request with the demonstration block
    appended to its final user message, as ``main.py`` of the reference
    implementation rewrites the question; an agent request that ends in a
    tool result gets the block as a new user message instead, so the
    demonstration still sits right before the response it conditions. The
    block is ``context_template`` with ``{context}`` replaced.
    """

    batch_label = "sdft"

    def __init__(self, context: ProcessorContext) -> None:
        self._template = context_block_template(context.config)
        super().__init__(context)

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        block = self._template.replace(CONTEXT_PLACEHOLDER, teacher_context)
        rendered = normalize_messages_for_template(messages)
        if rendered and rendered[-1].get("role") == "user":
            last = dict(rendered[-1])
            content = str(last.get("content") or "")
            last["content"] = f"{content}\n\n{block}" if content else block
            return [*rendered[:-1], last], tools
        return [*rendered, {"role": "user", "content": block}], tools


__all__ = ["CONTEXT_PLACEHOLDER", "DEFAULT_CONTEXT_TEMPLATE", "SDFTProcessor", "context_block_template"]
