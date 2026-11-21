"""The split of a thinking model's sampled reply into its reasoning and its answer."""

from __future__ import annotations

import pytest

from reef.core.reasoning import ReasoningStreamSplitter, is_reasoning_pre_opened, split_reasoning


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "pre_opened", "split"),
    [
        ("plan</think>\n\nanswer", True, ("plan", "answer")),
        ("<think>plan</think>answer", False, ("plan", "answer")),
        ("<think><think> plan\n</think>answer", False, ("plan", "answer")),
        ("plan</think>answer", False, ("plan", "answer")),
        ("plan<tool_call>{}", True, ("plan", "<tool_call>{}")),
        ("cut off while thinking", True, ("cut off while thinking", "")),
        ("answer", False, ("", "answer")),
    ],
)
def test_the_reasoning_split_follows_sglangs_qwen3_parser(text, pre_opened, split) -> None:
    assert split_reasoning(text, pre_opened=pre_opened) == split


@pytest.mark.unit
def test_a_prompt_that_ends_with_an_open_think_tag_pre_opens_the_reasoning() -> None:
    assert is_reasoning_pre_opened("<|im_start|>assistant\n<think>\n")
    assert not is_reasoning_pre_opened("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("deltas", "force_reasoning", "parts"),
    [
        (["<thi", "nk>a", "</th", "ink>\n\nb"], False, [("thinking", "a"), ("text", "b")]),
        (["<think>a</think>b"], True, [("thinking", "a"), ("text", "b")]),
        (["<think><think>a</think>b"], False, [("thinking", "a"), ("text", "b")]),
        (["<think><thi", "nk>a", "</think>b"], True, [("thinking", "a"), ("text", "b")]),
        (["a", "<tool_", "call>x"], True, [("thinking", "a"), ("text", "<tool_call>x")]),
        (["still thin", "king"], True, [("thinking", "still thin"), ("thinking", "king")]),
        (["plain ", "text"], False, [("text", "plain "), ("text", "text")]),
    ],
)
def test_reasoning_stream_split_follows_sglang_qwen3_parser(deltas, force_reasoning, parts) -> None:
    splitter = ReasoningStreamSplitter(enabled=True, force_reasoning=force_reasoning)
    streamed = [part for delta in deltas for part in splitter.feed(delta)]
    assert [*streamed, *splitter.finish()] == parts


@pytest.mark.unit
@pytest.mark.parametrize("force_reasoning", [False, True])
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("\n<think>work</think>answer", ("work", "answer")),
        (" \t<think> \n<think>work</think>\n   answer", ("work", "   answer")),
        ("\n<think></think>\nanswer", ("", "answer")),
        ("\n<think>unfinished", ("unfinished", "")),
        ("", ("", "")),
        ("\n  ", None),
        ("\n <thi", None),
        ("\n <thing>literal", None),
        ("\n   ordinary answer", None),
    ],
)
def test_leading_whitespace_preserves_reasoning_and_literal_answer_text(reply, expected, force_reasoning) -> None:
    if expected is None:
        expected = (reply.strip(), "") if force_reasoning else ("", reply)
    assert split_reasoning(reply, pre_opened=force_reasoning) == expected
    for offset in range(len(reply) + 1):
        splitter = ReasoningStreamSplitter(enabled=True, force_reasoning=force_reasoning)
        parts = splitter.feed(reply[:offset]) + splitter.feed(reply[offset:]) + splitter.finish()
        reasoning = "".join(value for kind, value in parts if kind == "thinking").strip()
        answer = "".join(value for kind, value in parts if kind == "text")
        assert (reasoning, answer) == expected, offset


@pytest.mark.unit
def test_an_opening_reasoning_tag_split_after_whitespace_leaks_no_answer_text() -> None:
    splitter = ReasoningStreamSplitter(enabled=True, force_reasoning=False)
    assert splitter.feed("\n ") == []
    assert splitter.feed("<thi") == []
    assert splitter.feed("nk>work") == [("thinking", "work")]
    assert splitter.feed("</think>answer") == [("text", "answer")]
