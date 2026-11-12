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
        (["a", "<tool_", "call>x"], True, [("thinking", "a"), ("text", "<tool_call>x")]),
        (["still thin", "king"], True, [("thinking", "still thin"), ("thinking", "king")]),
        (["plain ", "text"], False, [("text", "plain "), ("text", "text")]),
    ],
)
def test_reasoning_stream_split_follows_sglang_qwen3_parser(deltas, force_reasoning, parts) -> None:
    splitter = ReasoningStreamSplitter(enabled=True, force_reasoning=force_reasoning)
    streamed = [part for delta in deltas for part in splitter.feed(delta)]
    assert [*streamed, *splitter.finish()] == parts
