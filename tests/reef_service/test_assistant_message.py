"""The presentation split every trainable chat backend shares."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.runtime.assistant_message import (
    ReasoningStreamSplitter,
    ToolMarkerStreamHold,
    reasoning_is_pre_opened,
    split_assistant_message,
)


class _Parser:
    """A parser that recognises ``<tool_call>`` and answers with one fixed call."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def has_tool_call(self, text: str) -> bool:
        return "<tool_call>" in text

    def parse_non_stream(self, text: str):
        if self.fail:
            raise RuntimeError("bad markup")
        before, _, _ = text.partition("<tool_call>")
        return before.strip(), [SimpleNamespace(name="read_file", parameters={"path": "README.md"})]


@pytest.mark.unit
def test_reasoning_before_the_closing_tag_rides_reasoning_content() -> None:
    message, called = split_assistant_message("<think>pondering</think>\nThe answer is 36.", None)
    assert message == {"role": "assistant", "content": "The answer is 36.", "reasoning_content": "pondering"}
    assert called is False


@pytest.mark.unit
def test_truncated_reasoning_is_reasoning_only_when_the_template_pre_opened_it() -> None:
    truncated = "Okay, the user wants me to solve this. First I should read the file"
    leaked, _ = split_assistant_message(truncated, None)
    assert leaked["content"] == truncated  # nothing says this was thinking

    caught, _ = split_assistant_message(truncated, None, force_reasoning=True)
    assert caught["content"] == ""
    assert caught["reasoning_content"] == truncated


@pytest.mark.unit
def test_a_harness_that_reads_raw_tags_can_opt_out() -> None:
    message, _ = split_assistant_message("<think>x</think>y", None, force_reasoning=True, split_reasoning=False)
    assert message == {"role": "assistant", "content": "<think>x</think>y"}


@pytest.mark.unit
def test_tool_calls_come_back_structurally_with_the_remaining_text() -> None:
    message, called = split_assistant_message("I'll read it.\n<tool_call>...</tool_call>", _Parser())
    assert called is True
    assert message["content"] == "I'll read it."
    [call] = message["tool_calls"]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"] == {"name": "read_file", "arguments": '{"path": "README.md"}'}


@pytest.mark.unit
def test_a_call_opened_inside_thinking_ends_the_thinking() -> None:
    """Ollama's Qwen3 parser makes the same call: a model that starts acting has stopped thinking.

    Without the rule the whole turn is truncated reasoning and the call is lost."""
    text = "I should look first.\n<tool_call>...</tool_call>"
    lost, called = split_assistant_message(text, _Parser(), force_reasoning=True)
    assert called is False and lost["content"] == ""

    kept, called = split_assistant_message(text, _Parser(), force_reasoning=True, tool_call_start="<tool_call>")
    assert called is True
    assert kept["reasoning_content"] == "I should look first."
    assert kept["content"] is None
    assert kept["tool_calls"][0]["function"]["name"] == "read_file"


@pytest.mark.unit
def test_a_closed_think_block_still_wins_over_the_call_rule() -> None:
    text = "<think>plan</think>\nDoing it.\n<tool_call>...</tool_call>"
    message, called = split_assistant_message(text, _Parser(), force_reasoning=True, tool_call_start="<tool_call>")
    assert called is True
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "Doing it."


@pytest.mark.unit
def test_unparseable_call_markup_raises_with_the_parser_named() -> None:
    with pytest.raises(ValueError, match="the mlx tool-call parser could not parse"):
        split_assistant_message("<tool_call>", _Parser(fail=True), parser_label="the mlx tool-call")


@pytest.mark.unit
def test_the_pre_opened_sniff_reads_the_generation_prompt_tail() -> None:
    def template(tail: str):
        return SimpleNamespace(apply_chat_template=lambda *a, **k: f"<|im_start|>user\n<|im_end|>\n{tail}")

    assert reasoning_is_pre_opened(template("<|im_start|>assistant\n<think>\n")) is True
    assert reasoning_is_pre_opened(template("<|im_start|>assistant\n")) is False
    assert reasoning_is_pre_opened(SimpleNamespace()) is False  # no template at all


# ------------------------------------------------------------- streaming halves


def _drain(splitter: ReasoningStreamSplitter, *deltas: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for delta in deltas:
        parts.extend(splitter.feed(delta))
    parts.extend(splitter.finish())
    return parts


@pytest.mark.unit
def test_a_closing_tag_split_across_deltas_is_still_one_boundary() -> None:
    parts = _drain(ReasoningStreamSplitter(enabled=True, force_reasoning=True), "work</thi", "nk>\nanswer")
    assert parts == [("thinking", "work"), ("text", "answer")]


@pytest.mark.unit
def test_a_pre_opened_block_that_never_closes_stays_reasoning() -> None:
    parts = _drain(ReasoningStreamSplitter(enabled=True, force_reasoning=True), "still ", "thinking")
    assert parts == [("thinking", "still "), ("thinking", "thinking")]


@pytest.mark.unit
def test_an_undecided_stream_becomes_text_once_it_cannot_be_a_tag() -> None:
    parts = _drain(ReasoningStreamSplitter(enabled=True, force_reasoning=False), "<th", "e answer")
    assert parts == [("text", "<the answer")]


@pytest.mark.unit
def test_a_call_opened_inside_streamed_thinking_ends_the_thinking() -> None:
    splitter = ReasoningStreamSplitter(enabled=True, force_reasoning=True, tool_call_start="<tool_call>")
    parts = _drain(splitter, "look first\n<tool_", "call>\n<function=read_file>", "\n</tool_call>")
    assert parts[0] == ("thinking", "look first\n")
    assert (
        "".join(value for kind, value in parts if kind == "text") == "<tool_call>\n<function=read_file>\n</tool_call>"
    )
    assert all(kind == "text" for kind, _ in parts[1:])


@pytest.mark.unit
def test_the_hold_shows_text_up_to_the_marker_and_nothing_after() -> None:
    hold = ToolMarkerStreamHold("<tool_call>")
    shown = [hold.feed("I'll read it.\n<tool_"), hold.feed("call>\n<function=read_file>"), hold.feed("</tool_call>")]
    assert shown == ["I'll read it.", "", ""]
    assert hold.finish() == ""


@pytest.mark.unit
def test_the_hold_releases_a_false_alarm_and_trailing_whitespace_at_the_end() -> None:
    hold = ToolMarkerStreamHold("<tool_call>")
    assert hold.feed("a < b ") == "a < b"  # a "<" already followed by a space is text; trailing space waits
    assert hold.feed("and <to") == " and"  # "<to" could still become the marker
    assert hold.feed("p!\n") == " <top!"
    assert hold.finish() == "\n"


@pytest.mark.unit
def test_the_hold_is_transparent_without_a_marker() -> None:
    hold = ToolMarkerStreamHold(None)
    assert hold.feed("<tool_call> as text ") == "<tool_call> as text "
    assert hold.finish() == ""
