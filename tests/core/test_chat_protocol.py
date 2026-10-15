"""The assistant-message split and the streamed reasoning split that the token-native chat backends share."""

from __future__ import annotations

import pytest

from reef.core.chat_protocol import ParsedToolCall, ReasoningStreamSplitter, assistant_message, template_messages

pytestmark = pytest.mark.unit


class FixedCallParser:
    """Recognises ``<tool_call>`` and returns one fixed call."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def has_tool_call(self, text: str) -> bool:
        return "<tool_call>" in text

    def parse_non_stream(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        if self.fail:
            raise RuntimeError("bad markup")
        before, _, _ = text.partition("<tool_call>")
        return before.strip(), [ParsedToolCall("read_file", '{"path": "README.md"}')]


def test_reasoning_before_the_closing_tag_goes_to_reasoning_content() -> None:
    message, called = assistant_message("<think>pondering</think>\nThe answer is 36.", None)

    assert message == {"role": "assistant", "content": "The answer is 36.", "reasoning_content": "pondering"}
    assert called is False


@pytest.mark.parametrize(
    ("force_reasoning", "expected"),
    [
        (False, {"role": "assistant", "content": "First I should read the file"}),
        (True, {"role": "assistant", "content": "", "reasoning_content": "First I should read the file"}),
    ],
    ids=["template-left-it-closed", "template-opened-it"],
)
def test_unclosed_reasoning_is_reasoning_only_when_the_template_opened_it(
    force_reasoning: bool, expected: dict
) -> None:
    message, _ = assistant_message("First I should read the file", None, force_reasoning=force_reasoning)

    assert message == expected


def test_a_handler_that_reads_raw_tags_keeps_the_text_whole() -> None:
    message, _ = assistant_message("<think>x</think>y", None, split_reasoning=False, force_reasoning=True)

    assert message == {"role": "assistant", "content": "<think>x</think>y"}


def test_tool_calls_come_back_structurally_with_the_remaining_text() -> None:
    message, called = assistant_message("I'll read it.\n<tool_call>...</tool_call>", FixedCallParser())

    assert called is True
    assert message["content"] == "I'll read it."
    [call] = message["tool_calls"]
    assert call["id"].startswith("call_")
    assert call["function"] == {"name": "read_file", "arguments": '{"path": "README.md"}'}


@pytest.mark.parametrize(
    ("tool_call_start", "reasoning", "called"),
    [(None, "I should look first.\n<tool_call>...</tool_call>", False), ("<tool_call>", "I should look first.", True)],
    ids=["no-marker", "marker"],
)
def test_a_call_opened_inside_thinking_ends_the_thinking_only_with_the_marker(
    tool_call_start: str | None, reasoning: str, called: bool
) -> None:
    text = "I should look first.\n<tool_call>...</tool_call>"

    message, parsed = assistant_message(text, FixedCallParser(), force_reasoning=True, tool_call_start=tool_call_start)

    assert parsed is called
    assert message["reasoning_content"] == reasoning


def test_a_closed_think_block_wins_over_the_call_rule() -> None:
    text = "<think>plan</think>\nDoing it.\n<tool_call>...</tool_call>"

    message, called = assistant_message(text, FixedCallParser(), force_reasoning=True, tool_call_start="<tool_call>")

    assert called is True
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "Doing it."


def test_unparseable_call_markup_raises_a_value_error() -> None:
    with pytest.raises(ValueError, match="tool-call parser could not parse"):
        assistant_message("<tool_call>", FixedCallParser(fail=True))


def test_template_messages_parse_string_tool_call_arguments() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"path": "a.txt"}'}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"k": 1}}}]},
        {"role": "user", "content": "hi"},
    ]

    rendered = template_messages(messages)

    assert rendered[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"k": 1}
    assert rendered[2] == {"role": "user", "content": "hi"}


def test_template_messages_refuse_arguments_that_are_not_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        template_messages(
            [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}]}]
        )


def drain(splitter: ReasoningStreamSplitter, *deltas: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for delta in deltas:
        parts.extend(splitter.feed(delta))
    parts.extend(splitter.finish())
    return parts


@pytest.mark.parametrize(
    ("force_reasoning", "deltas", "expected"),
    [
        (True, ("work</thi", "nk>\nanswer"), [("thinking", "work"), ("text", "answer")]),
        (True, ("still ", "thinking"), [("thinking", "still "), ("thinking", "thinking")]),
        (False, ("<th", "e answer"), [("text", "<the answer")]),
    ],
    ids=["split-closing-tag", "never-closed", "not-a-tag"],
)
def test_the_stream_splitter_finds_the_thinking_boundary(
    force_reasoning: bool, deltas: tuple[str, ...], expected: list[tuple[str, str]]
) -> None:
    assert drain(ReasoningStreamSplitter(enabled=True, force_reasoning=force_reasoning), *deltas) == expected


def test_a_call_opened_inside_streamed_thinking_ends_the_thinking() -> None:
    splitter = ReasoningStreamSplitter(enabled=True, force_reasoning=True, tool_call_start="<tool_call>")

    parts = drain(splitter, "look first\n<tool_", "call>\n<function=read_file>", "\n</tool_call>")

    assert parts[0] == ("thinking", "look first\n")
    assert all(kind == "text" for kind, _ in parts[1:])
    assert "".join(value for _, value in parts[1:]) == "<tool_call>\n<function=read_file>\n</tool_call>"
