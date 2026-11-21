"""Shared streaming protocol contracts for native chat handlers."""

import pytest

from reef.inference.chat import ChatStreamRelay, StreamedToolCall, _AnthropicEventWriter


@pytest.mark.unit
def test_an_anthropic_stream_writer_refuses_a_negative_input_count() -> None:
    with pytest.raises(ValueError, match="input token count"):
        _AnthropicEventWriter("message", model="model", input_tokens=-1)


@pytest.mark.unit
def test_a_streamed_tool_delta_refuses_an_empty_name() -> None:
    with pytest.raises(ValueError, match="tool-call name must be a nonempty string"):
        StreamedToolCall(0, "", "{}")


@pytest.mark.unit
def test_the_relay_rejects_reasoning_that_disagrees_with_the_completed_parse() -> None:
    relay = ChatStreamRelay(_AnthropicEventWriter("message", model="model", input_tokens=1), None)
    assert relay.piece("thinking", "first")
    with pytest.raises(ValueError, match="streamed reasoning differs"):
        relay.reconcile({"reasoning_content": "second"})
