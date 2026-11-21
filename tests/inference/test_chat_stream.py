"""Shared streaming protocol contracts for native chat handlers."""

import pytest

from reef.inference.chat import _AnthropicEventWriter


@pytest.mark.unit
def test_an_anthropic_stream_writer_refuses_a_negative_input_count() -> None:
    with pytest.raises(ValueError, match="input token count"):
        _AnthropicEventWriter("message", model="model", input_tokens=-1)
