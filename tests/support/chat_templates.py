"""A chat template for tests: each message is its role token and one token per character."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.chat_rendering import ChatTemplate

ROLES = {"system": 1, "user": 2, "assistant": 9, "tool": 4}


class CharTemplate(ChatTemplate):
    """Each message is its role ID followed by one token per character; the generation prompt is token 9, the
    assistant role, as a real template opens the assistant turn with its generation prompt.

    ``tools`` adds token 8 at the start, and ``drop_history`` leaves out every assistant message but the last, as a
    template that drops earlier turns' reasoning does.
    """

    def __init__(self, drop_history: bool = False) -> None:
        self.drop_history = drop_history
        self.tools: list[Any] = []
        #: The template arguments of the last ``with_arguments``.
        self.arguments: dict[str, Any] = {}

    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None, *, prompt: bool
    ) -> list[int]:
        self.tools.append(tools)
        last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        tokens = [8] if tools else []
        for index, message in enumerate(messages):
            if self.drop_history and message["role"] == "assistant" and index != last_assistant:
                continue
            tokens += [ROLES[message["role"]], *(100 + ord(char) for char in message["content"])]
        return tokens + ([9] if prompt else [])

    def with_arguments(self, arguments: Mapping[str, Any]) -> ChatTemplate:
        """The same template, which records ``arguments``; the rendering ignores them."""
        self.arguments = dict(arguments)
        return self
