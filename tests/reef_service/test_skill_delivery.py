from __future__ import annotations

from reef.service.streaming import (
    SSEFrameDecoder,
    aggregate_sse_text,
    chat_completion_chunk_identity,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)


class _FakeStream:
    status = 200
    headers = {"content-type": "text/event-stream"}


OPENAI_SSE = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"The answer"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":" is 7."}}]}\n\n'
    "data: [DONE]\n\n"
)
ANTHROPIC_SSE = (
    'data: {"type":"message_start","message":{}}\n\n'
    'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"The answer"}}\n\n'
    'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" is 7."}}\n\n'
    'data: {"type":"message_stop"}\n\n'
)


def test_aggregate_sse_text_reads_both_wire_shapes() -> None:
    assert aggregate_sse_text(OPENAI_SSE) == "The answer is 7."
    assert aggregate_sse_text(ANTHROPIC_SSE) == "The answer is 7."
    assert aggregate_sse_text("not sse at all") is None
    assert aggregate_sse_text("data: {broken json}\n\n") is None


def test_aggregate_sse_text_keeps_only_the_primary_choice() -> None:
    interleaved = (
        'data: {"choices":[{"index":0,"delta":{"content":"Paris"}}]}\n\n'
        'data: {"choices":[{"index":1,"delta":{"content":"The capital"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":" is the capital."}}]}\n\n'
    )
    assert aggregate_sse_text(interleaved) == "Paris is the capital."


def test_aggregate_sse_text_refuses_tool_using_turns() -> None:
    openai_tools = (
        'data: {"choices":[{"index":0,"delta":{"content":"Let me check."}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}\n\n'
    )
    anthropic_tools = (
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Let me check."}}\n\n'
        'data: {"type":"content_block_start","content_block":{"type":"tool_use","name":"f"}}\n\n'
    )
    assert aggregate_sse_text(openai_tools) is None
    assert aggregate_sse_text(anthropic_tools) is None


def test_aggregate_sse_text_survives_unicode_line_separators_and_multiline_data() -> None:
    with_separator = 'data: {"choices":[{"index":0,"delta":{"content":"a\u2028b"}}]}\n\n'
    assert aggregate_sse_text(with_separator) == "a\u2028b"
    multiline = 'data: {"choices":[{"index":0,\ndata: "delta":{"content":"joined"}}]}\n\n'
    assert aggregate_sse_text(multiline) == "joined"


def test_stream_record_stores_the_aggregated_message_next_to_the_raw_body() -> None:
    record = stream_record(_FakeStream(), OPENAI_SSE.encode("utf-8"), complete=True)
    assert record["body"] == OPENAI_SSE
    assert record["message"] == {"role": "assistant", "content": "The answer is 7."}


def test_stream_record_skips_the_message_for_incomplete_streams() -> None:
    record = stream_record(_FakeStream(), OPENAI_SSE.encode("utf-8"), complete=False, error="client disconnected")
    assert record["body"] == OPENAI_SSE
    assert "message" not in record


def test_sse_frame_decoder_finds_split_openai_terminator_and_builds_receipt_chunk() -> None:
    decoder = SSEFrameDecoder()
    frames = list(decoder.feed(b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'))
    frames.extend(decoder.feed(b'"created":1,"model":"m","choices":[]}\n\ndata: [DO'))
    frames.extend(decoder.feed(b"NE]\n\n"))

    assert len(frames) == 2
    assert is_terminal_sse_event("/v1/chat/completions", frames[-1])
    identity = chat_completion_chunk_identity(frames[0])
    metadata, terminal = receipt_sse_events("/v1/chat/completions", identity, frames[-1], "record-1")
    payload = __import__("json").loads(metadata.removeprefix(b"data: "))
    assert payload == {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "reef": {"agent_record_id": "record-1"},
    }
    assert terminal == b"data: [DONE]\n\n"


def test_anthropic_receipt_is_attached_to_message_stop() -> None:
    terminal = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    assert is_terminal_sse_event("/v1/messages", terminal)
    (with_receipt,) = receipt_sse_events("/v1/messages", {}, terminal, "record-2")
    assert with_receipt == (
        b'event: message_stop\ndata: {"type":"message_stop","reef":{"agent_record_id":"record-2"}}\n\n'
    )
