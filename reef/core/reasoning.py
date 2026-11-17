"""The split of a thinking model's sampled reply into its reasoning and its answer, as SGLang's Qwen3 reasoning parser
splits it.

A thinking chat template has the model write its reasoning between ``<think>`` and ``</think>``, before its answer.
A template that ends the generation prompt with ``<think>`` leaves only the closing tag to the sample
(``is_reasoning_pre_opened``). ``split_reasoning`` splits a whole reply, and ``ReasoningStreamSplitter`` splits a
reply while it streams, so that an OpenAI chat answer carries the reasoning in ``reasoning_content`` and not in
``content``. The token-native chat handlers (``reef.inference.chat``) and Tinker's chat (``reef.tinker.sampling``)
both split with this module. It is in ``reef.core`` because ``reef.tinker`` serves training too, and training never
imports ``reef.inference``.
"""

from __future__ import annotations

#: The tokens at the end of a rendered prompt that hold an open ``<think>``: the tag and the newline after it can be
#: several tokens.
PROMPT_TAIL_TOKEN_COUNT = 8
#: The tag that starts reasoning.
THINK_START = "<think>"
#: The tag that ends reasoning.
THINK_END = "</think>"
#: The start of a tool call, which ends reasoning that has no ``</think>``.
TOOL_CALL_START = "<tool_call>"


def is_reasoning_pre_opened(prompt_tail: str) -> bool:
    """Whether ``prompt_tail``, the last ``PROMPT_TAIL_TOKEN_COUNT`` tokens of a rendered prompt decoded with their
    special tokens, ends with an open ``<think>``: the sample then carries only the closing tag."""
    return prompt_tail.rstrip().endswith(THINK_START)


def split_reasoning(text: str, *, pre_opened: bool) -> tuple[str, str]:
    """The reasoning and the answer of the sampled reply ``text``; the reasoning is "" when the reply has none.

    ``pre_opened`` holds when the prompt ended with an open ``<think>``. Without it, a reply with no ``<think>`` and
    no ``</think>`` is all answer. Otherwise the leading ``<think>`` tags drop, and the first ``</think>`` ends the
    reasoning, with the newlines after it dropped. Without a ``</think>``, a ``<tool_call>`` ends the reasoning, and a
    reply with neither is all reasoning: its generation stopped while it thought. The reasoning has no leading or
    trailing whitespace.
    """
    if not (pre_opened or THINK_START in text or THINK_END in text):
        return "", text
    body = text
    while body.startswith(THINK_START):
        body = body[len(THINK_START) :]
    reasoning, separator, answer = body.partition(THINK_END)
    if separator:
        return reasoning.strip(), answer.lstrip("\n")
    tool_start = body.find(TOOL_CALL_START)
    if tool_start != -1:
        return body[:tool_start].strip(), body[tool_start:]
    return body.strip(), ""


class ReasoningStreamSplitter:
    """The split of one streamed reply into reasoning and answer pieces, without waiting for the whole reply.

    A value holds the text it has read but not given out (``buffer``) and its position in the reply: whether it reads
    reasoning (``in_reasoning``) and whether it has dropped the first ``<think>`` (``stripped_think_start``). A
    splitter with ``enabled`` false gives every delta out as answer. Otherwise the first ``<think>`` starts reasoning
    and drops, and with ``force_reasoning`` the reply starts in reasoning. The first ``</think>`` ends the reasoning,
    with the newlines after it dropped, and a ``<tool_call>`` before it ends the reasoning too. A reply that ends in
    reasoning is all reasoning. ``buffer`` keeps only an end of the text that can be the start of one of these tags.
    One splitter reads one reply; the streaming chat handlers make it and read its pieces.
    """

    def __init__(self, *, enabled: bool, force_reasoning: bool) -> None:
        self.enabled = enabled
        self.in_reasoning = enabled and force_reasoning
        self.stripped_think_start = False
        self.buffer = ""

    @staticmethod
    def held_suffix_length(value: str, tag: str) -> int:
        """The length of the longest end of ``value`` that is a proper start of ``tag``; 0 when there is none."""
        for size in range(min(len(value), len(tag) - 1), 0, -1):
            if tag.startswith(value[-size:]):
                return size
        return 0

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """The pieces that ``delta`` completes, in reply order, as ``("thinking", text)`` for reasoning and
        ``("text", text)`` for the answer; a part of a tag that is still open stays in ``buffer``."""
        if not delta:
            return []
        if not self.enabled:
            return [("text", delta)]
        self.buffer += delta
        if any(
            tag.startswith(self.buffer) and tag != self.buffer for tag in (THINK_START, THINK_END, TOOL_CALL_START)
        ):
            return []
        if not self.stripped_think_start and THINK_START in self.buffer:
            self.buffer = self.buffer.replace(THINK_START, "", 1)
            self.stripped_think_start = True
            self.in_reasoning = True
        if not self.in_reasoning:
            value, self.buffer = self.buffer, ""
            return [("text", value)]

        before, separator, after = self.buffer.partition(THINK_END)
        if separator:
            self.buffer = ""
            self.in_reasoning = False
            parts = [("thinking", before)] if before else []
            visible = after.lstrip("\n")
            if visible:
                parts.append(("text", visible))
            return parts
        tool_start = self.buffer.find(TOOL_CALL_START)
        if tool_start != -1:
            before, after = self.buffer[:tool_start], self.buffer[tool_start:]
            self.buffer = ""
            self.in_reasoning = False
            return [("thinking", before), ("text", after)] if before else [("text", after)]
        held_tags = (
            [THINK_END, TOOL_CALL_START] if self.stripped_think_start else [THINK_START, THINK_END, TOOL_CALL_START]
        )
        held = max(self.held_suffix_length(self.buffer, tag) for tag in held_tags)
        if held:
            value = self.buffer[:-held]
            self.buffer = self.buffer[-held:]
        else:
            value, self.buffer = self.buffer, ""
        return [("thinking", value)] if value else []

    def finish(self) -> list[tuple[str, str]]:
        """The piece that ``buffer`` holds when the reply ends, as reasoning when the reply ends in reasoning."""
        if not self.buffer:
            return []
        value, self.buffer = self.buffer, ""
        if self.in_reasoning:
            value = value.removeprefix(THINK_START)
            return [("thinking", value)] if value else []
        return [("text", value)]


__all__ = [
    "PROMPT_TAIL_TOKEN_COUNT",
    "ReasoningStreamSplitter",
    "is_reasoning_pre_opened",
    "split_reasoning",
]
