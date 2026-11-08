"""The split of a thinking model's sampled reply into its reasoning and its answer, as SGLang's Qwen3 reasoning parser
splits it.

A thinking chat template has the model write its reasoning between ``<think>`` and ``</think>``, before its answer.
A template that ends the generation prompt with ``<think>`` leaves only the closing tag to the sample
(``is_reasoning_pre_opened``). ``split_reasoning`` gives the reasoning and the answer, so that an OpenAI chat answer
carries the reasoning in ``reasoning_content`` and not in ``content``. The token-native chat handlers
(``reef.inference.chat``) and Tinker's chat (``reef.tinker.sampling``) both split with it. It is here because
``reef.tinker`` serves training too, and training never imports ``reef.inference``.
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


__all__ = [
    "PROMPT_TAIL_TOKEN_COUNT",
    "THINK_END",
    "THINK_START",
    "TOOL_CALL_START",
    "is_reasoning_pre_opened",
    "split_reasoning",
]
