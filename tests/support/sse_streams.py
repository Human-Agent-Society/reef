"""A canned OpenAI server-sent-event stream, for tests that record a streamed response."""

from __future__ import annotations

from collections.abc import AsyncIterator

from reef.inference.interfaces import InferenceStream


class FakeStream(InferenceStream):
    def __init__(self) -> None:
        super().__init__(status=200, headers={"content-type": "text/event-stream"}, chunks=self.chunks_iter())

    async def chunks_iter(self) -> AsyncIterator[bytes]:
        yield OPENAI_SSE.encode("utf-8")


OPENAI_SSE = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"The answer"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":" is 7."}}]}\n\n'
    "data: [DONE]\n\n"
)
