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

import re

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


REASONING_BOUNDARY = re.compile(f"{re.escape(THINK_END)}|{re.escape(TOOL_CALL_START)}")
LEADING_THINK_TAG = re.compile(rf"\s*{re.escape(THINK_START)}")


def leading_think_tags_end(text: str) -> int:
    """Return the end of the leading opening tags and any whitespace before them."""
    end = 0
    while match := LEADING_THINK_TAG.match(text, end):
        end = match.end()
    return end


def split_reasoning(text: str, *, pre_opened: bool) -> tuple[str, str]:
    """The reasoning and the answer of the sampled reply ``text``; the reasoning is "" when the reply has none.

    The reply starts in reasoning when the prompt pre-opened it or when the reply begins with opening tags.
    An unmarked prefix also counts as reasoning if the first boundary is a closing tag. A tool call ends known
    reasoning, and otherwise belongs to the answer. Opening tags inside visible text are literal.
    """
    start = leading_think_tags_end(text)
    body = text[start:]
    boundary = REASONING_BOUNDARY.search(body)
    if boundary is None:
        return (body.strip(), "") if pre_opened or start else ("", text)
    if boundary.group() == THINK_END:
        return body[: boundary.start()].strip(), body[boundary.end() :].lstrip("\n")
    if pre_opened or start:
        return body[: boundary.start()].strip(), body[boundary.start() :]
    return "", text


class ReasoningStreamSplitter:
    """The split of one streamed reply into reasoning and answer pieces, without waiting for the whole reply.

    A value holds the text it has read but not given out (``buffer``) and its position in the reply: whether it reads
    reasoning (``in_reasoning``) and whether only leading tags have appeared (``at_start``). A splitter with
    ``enabled`` false gives every delta out as answer. Otherwise leading ``<think>`` tags start reasoning and drop.
    With ``force_reasoning`` the reply starts in reasoning. The first ``</think>`` ends the reasoning.
    ``strip_answer_newlines`` drops the newlines after that tag when the answer starts in a later chunk.
    A ``<tool_call>`` before the closing tag ends the reasoning too. A reply that ends in reasoning is all reasoning.
    While ``at_start`` is true, ``leading_whitespace`` holds nonempty whitespace fragments in their input order.
    ``buffer`` can hold a partial opening tag after those fragments. A complete opening tag discards them.
    Other text or the end of the reply releases them in order. After ``at_start``, ``buffer`` holds only a tag suffix.
    Unmarked answer text streams immediately. A later closing tag cannot reclassify text already sent.
    One splitter reads one reply; the streaming chat handlers make it and read its pieces.
    """

    def __init__(self, *, enabled: bool, force_reasoning: bool) -> None:
        self.enabled = enabled
        self.in_reasoning = enabled and force_reasoning
        self.at_start = True
        self.strip_answer_newlines = False
        self.leading_whitespace: list[str] = []
        self.buffer = ""

    @staticmethod
    def held_suffix_length(value: str, tag: str) -> int:
        """The length of the longest end of ``value`` that is a proper start of ``tag``; 0 when there is none."""
        for size in range(min(len(value), len(tag) - 1), 0, -1):
            if tag.startswith(value[-size:]):
                return size
        return 0

    def restore_leading_whitespace(self) -> None:
        """Put pending whitespace back in the buffer before it is emitted."""
        if self.leading_whitespace:
            self.leading_whitespace.append(self.buffer)
            self.buffer = "".join(self.leading_whitespace)
            self.leading_whitespace.clear()

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """The pieces that ``delta`` completes, in reply order, as ``("thinking", text)`` for reasoning and
        ``("text", text)`` for the answer; a part of a tag that is still open stays in ``buffer``."""
        if not delta:
            return []
        if not self.enabled:
            return [("text", delta)]
        self.buffer += delta
        if self.at_start:
            start = leading_think_tags_end(self.buffer)
            if start:
                self.buffer = self.buffer[start:]
                self.in_reasoning = True
                self.leading_whitespace.clear()
            possible_tag = self.buffer.lstrip()
            if THINK_START.startswith(possible_tag):
                whitespace = len(self.buffer) - len(possible_tag)
                if whitespace:
                    self.leading_whitespace.append(self.buffer[:whitespace])
                self.buffer = possible_tag
                return []
            self.at_start = False
            self.restore_leading_whitespace()
        if not self.in_reasoning:
            value, self.buffer = self.buffer, ""
            if self.strip_answer_newlines:
                value = value.lstrip("\n")
                self.strip_answer_newlines = not value
            return [("text", value)] if value else []

        boundary = REASONING_BOUNDARY.search(self.buffer)
        if boundary is not None:
            before = self.buffer[: boundary.start()]
            if boundary.group() == THINK_END:
                parts = [("thinking", before)] if before else []
                visible = self.buffer[boundary.end() :].lstrip("\n")
                self.strip_answer_newlines = not visible
            else:
                parts = [("thinking", before)] if before else []
                visible = self.buffer[boundary.start() :]
            self.buffer = ""
            self.in_reasoning = False
            if visible:
                parts.append(("text", visible))
            return parts
        held = max(self.held_suffix_length(self.buffer, tag) for tag in (THINK_END, TOOL_CALL_START))
        if held:
            value = self.buffer[:-held]
            self.buffer = self.buffer[-held:]
        else:
            value, self.buffer = self.buffer, ""
        return [("thinking", value)] if value else []

    def finish(self) -> list[tuple[str, str]]:
        """Emit any pending whitespace and text in the reply's last channel."""
        self.restore_leading_whitespace()
        if not self.buffer:
            return []
        value, self.buffer = self.buffer, ""
        return [("thinking" if self.in_reasoning else "text", value)]


__all__ = [
    "PROMPT_TAIL_TOKEN_COUNT",
    "ReasoningStreamSplitter",
    "is_reasoning_pre_opened",
    "split_reasoning",
]
