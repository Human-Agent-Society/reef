"""The MLX chat handler against a fake engine: request checks, the trainable response and streaming."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from tests.support.mlx_engine_stubs import DEFAULT_ROLLOUT, FakeMLXEngine

from reef.artifact.artifact import Artifact
from reef.runtime.interfaces import UpstreamStatusError
from reef.train.mlx_backend.config import MLXEngineConfig
from reef.train.mlx_backend.inference import MLXToolCallParser
from reef.train.mlx_backend.serving import MLXServingRuntime

pytestmark = pytest.mark.unit

ARTIFACT = Artifact.local(Path("."))
CHAT = "/v1/chat/completions"
HI = [{"role": "user", "content": "hi"}]
READ_FILE = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]
THINKING_TAIL = "<|im_start|>assistant\n<think>\n"
XML_CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>"


def serve(payload: dict, engine: FakeMLXEngine | None = None) -> dict:
    serving = MLXServingRuntime(engine or FakeMLXEngine())
    return asyncio.run(serving.inference_handler.inference(ARTIFACT, CHAT, payload))


def engine_with(text: str, **options) -> FakeMLXEngine:
    return FakeMLXEngine(rollout=replace(DEFAULT_ROLLOUT, text=text), **options)


def test_a_response_carries_the_tensors_that_make_it_trainable() -> None:
    engine = FakeMLXEngine()
    serving = MLXServingRuntime(engine)

    response = asyncio.run(serving.inference_handler.inference(ARTIFACT, CHAT, {"messages": HI, "max_tokens": 8}))

    training = response["training"]
    assert training["tokens"] == [11, 12, 13, 21, 22]
    assert training["loss_mask"] == [1, 1]
    assert training["rollout_log_probs"] == [-0.5, -0.25]
    assert training["runtime_load_id"] == serving.serving_runtime_load_id()
    assert response["choices"][0]["meta_info"] == {"runtime_load_id": serving.serving_runtime_load_id()}
    assert training["topk_indices"] == [[21, 99], [22, 98]]
    assert engine.generated[0]["max_tokens"] == 8


def test_max_context_tokens_shortens_the_completion_and_refuses_a_prompt_that_fills_it() -> None:
    # The fake engine renders every prompt as three tokens.
    engine = FakeMLXEngine(engine_config=MLXEngineConfig(max_context_tokens=7))
    serve({"messages": HI, "max_tokens": 8}, engine)
    serve({"messages": HI, "max_tokens": 2}, engine)
    assert [call["max_tokens"] for call in engine.generated] == [4, 2]

    with pytest.raises(UpstreamStatusError, match="max_context_tokens is 3") as refused:
        serve({"messages": HI}, FakeMLXEngine(engine_config=MLXEngineConfig(max_context_tokens=3)))
    assert refused.value.status == 400


def test_no_candidate_channel_when_the_engine_captured_none() -> None:
    engine = FakeMLXEngine(rollout=replace(DEFAULT_ROLLOUT, topk_indices=(), topk_log_probs=()))

    training = serve({"messages": HI}, engine)["training"]

    assert "topk_indices" not in training
    assert "topk_log_probs" not in training


def test_concurrent_completions_each_get_their_own_generation() -> None:
    engine = FakeMLXEngine()
    handler = MLXServingRuntime(engine).inference_handler

    async def wave(count: int) -> list[dict]:
        return await asyncio.gather(
            *(
                handler.inference(ARTIFACT, CHAT, {"messages": [{"role": "user", "content": f"q{index}"}]})
                for index in range(count)
            )
        )

    responses = asyncio.run(wave(4))

    assert [response["training"]["tokens"] for response in responses] == [[11, 12, 13, 21, 22]] * 4
    assert len(engine.generated) == 4


@pytest.mark.parametrize(
    ("payload", "rendered"),
    [
        ({"messages": HI}, {"tools": None, "template_kwargs": None}),
        (
            {"messages": HI, "chat_template_kwargs": {"enable_thinking": False}, "tools": READ_FILE},
            {"tools": READ_FILE, "template_kwargs": {"enable_thinking": False}},
        ),
    ],
    ids=["defaults", "request-settings"],
)
def test_the_request_steers_the_chat_template(payload: dict, rendered: dict) -> None:
    engine = FakeMLXEngine()

    serve(payload, engine)

    [call] = engine.rendered
    assert {key: call[key] for key in rendered} == rendered


@pytest.mark.parametrize(
    ("payload", "status", "message"),
    [
        ({"messages": HI, "stream": True}, 400, "inference_stream"),
        ({"messages": HI, "tools": {"name": "read_file"}}, 400, "tools"),
        ({"messages": HI, "chat_template_kwargs": "no-think"}, 400, "chat_template_kwargs"),
        ({"messages": []}, 400, "message objects"),
        ({"messages": HI, "max_tokens": 0}, 400, "max_completion_tokens"),
        ({"messages": HI, "max_tokens": "8"}, 400, "max_completion_tokens"),
        ({"messages": HI, "temperature": "hot"}, 400, "temperature"),
        ({"messages": HI, "seed": 3}, 400, "'seed'"),
        ({"messages": HI, "stop": ["\n"]}, 400, "'stop'"),
        ({"messages": HI, "n": 2}, 400, "n must be 1"),
        ({"messages": HI, "top_p": 0.9}, 400, "training.options.top_p"),
        ({"messages": HI, "top_k": 20}, 400, "training.options.top_k"),
        (
            {"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{"}}]}]},
            400,
            "valid JSON",
        ),
    ],
    ids=[
        "stream",
        "tools",
        "template-kwargs",
        "no-messages",
        "zero-max-tokens",
        "text-max-tokens",
        "temperature",
        "seed",
        "stop",
        "n",
        "top-p",
        "top-k",
        "tool-call-arguments",
    ],
)
def test_a_request_the_backend_cannot_serve_is_a_client_error(payload: dict, status: int, message: str) -> None:
    with pytest.raises(UpstreamStatusError, match=message) as caught:
        serve(payload)

    assert caught.value.status == status


def test_the_configured_top_p_and_top_k_are_accepted() -> None:
    assert serve({"messages": HI, "top_p": 1.0, "top_k": 0, "n": 1})["choices"][0]["finish_reason"] == "stop"


def test_an_empty_completion_is_a_server_error() -> None:
    engine = FakeMLXEngine(rollout=replace(DEFAULT_ROLLOUT, output_tokens=(), rollout_log_probs=(), text=""))

    with pytest.raises(UpstreamStatusError, match="no response tokens") as caught:
        serve({"messages": HI}, engine)

    assert caught.value.status == 502


@pytest.mark.parametrize(
    ("text", "prompt_tail", "message"),
    [
        (
            "pondering</think>\nThe answer is 36.",
            THINKING_TAIL,
            {"role": "assistant", "content": "The answer is 36.", "reasoning_content": "pondering"},
        ),
        (
            "Okay, first I should read the file",
            THINKING_TAIL,
            {"role": "assistant", "content": "", "reasoning_content": "Okay, first I should read the file"},
        ),
        (
            "The answer is 36.",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            {"role": "assistant", "content": "The answer is 36."},
        ),
    ],
    ids=["closed-reasoning", "truncated-reasoning", "thinking-disabled"],
)
def test_reasoning_moves_to_reasoning_content_and_the_tensors_stay_whole(
    text: str, prompt_tail: str, message: dict
) -> None:
    response = serve({"messages": HI}, engine_with(text, prompt_tail=prompt_tail))

    assert response["choices"][0]["message"] == message
    assert response["training"]["tokens"] == [11, 12, 13, 21, 22]


def test_a_call_in_the_templates_syntax_becomes_tool_calls() -> None:
    response = serve({"messages": HI, "tools": READ_FILE}, engine_with(f"I'll read it.\n{XML_CALL}"))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I'll read it."
    [call] = choice["message"]["tool_calls"]
    assert call["function"] == {"name": "read_file", "arguments": '{"path": "README.md"}'}


def test_every_call_in_a_reply_is_parsed_in_order() -> None:
    second = XML_CALL.replace("README.md", "NOTES.md")

    response = serve({"messages": HI, "tools": READ_FILE}, engine_with(f"{XML_CALL}\n{second}"))

    calls = response["choices"][0]["message"]["tool_calls"]
    assert [json.loads(call["function"]["arguments"])["path"] for call in calls] == ["README.md", "NOTES.md"]
    assert response["choices"][0]["message"]["content"] is None


@pytest.mark.parametrize(
    ("payload", "engine_options"),
    [
        ({"messages": HI}, {}),
        ({"messages": HI, "tools": READ_FILE, "tool_choice": "none"}, {}),
        ({"messages": HI, "tools": READ_FILE}, {"tool_calling": False}),
    ],
    ids=["no-tools", "tool-choice-none", "template-without-call-syntax"],
)
def test_call_markup_stays_text_when_no_call_syntax_applies(payload: dict, engine_options: dict) -> None:
    response = serve(payload, engine_with(XML_CALL, **engine_options))

    assert response["choices"][0]["message"] == {"role": "assistant", "content": XML_CALL}
    assert response["choices"][0]["finish_reason"] == "stop"


def test_a_call_that_never_closes_is_a_server_error() -> None:
    engine = engine_with("<tool_call>\n<function=read_file>\n<parameter=path>\nREAD")

    with pytest.raises(UpstreamStatusError, match="parser could not parse") as caught:
        serve({"messages": HI, "tools": READ_FILE}, engine)

    assert caught.value.status == 502


def test_a_call_opened_inside_thinking_ends_the_thinking() -> None:
    engine = engine_with(f"I should look first.\n{XML_CALL}", prompt_tail=THINKING_TAIL)

    message = serve({"messages": HI, "tools": READ_FILE}, engine)["choices"][0]["message"]

    assert message["reasoning_content"] == "I should look first."
    assert message["tool_calls"][0]["function"]["name"] == "read_file"


@pytest.mark.parametrize(
    ("deltas", "shown"),
    [
        (("I'll read it.\n<tool_", "call>\n<function=read_file>", "</tool_call>"), ["I'll read it.", "", ""]),
        (("a < b ", "and <to", "p!\n"), ["a < b", " and", " <top!"]),
    ],
    ids=["call", "no-call"],
)
def test_the_stream_parser_shows_text_up_to_the_call_marker(deltas: tuple[str, ...], shown: list[str]) -> None:
    parser = MLXToolCallParser(FakeMLXEngine(), ("<tool_call>", "</tool_call>"), READ_FILE)

    assert [parser.parse_stream_chunk(delta) for delta in deltas] == [(text, []) for text in shown]


# -- Streaming


def stream(payload: dict, engine: FakeMLXEngine, *, take: int | None = None):
    """Stream one completion; return the frames and the stream. ``take`` closes it after that many frames."""
    handler = MLXServingRuntime(engine).inference_handler

    async def run():
        response = await handler.inference_stream(ARTIFACT, CHAT, {**payload, "stream": True})
        frames = []
        async for frame in response.chunks:
            frames.append(frame)
            if take is not None and len(frames) >= take:
                break
        await response.close()
        return frames, response

    return asyncio.run(run())


def events(frames: list[bytes]) -> list:
    """Every JSON chunk on the wire in order; ``"[DONE]"`` for the terminator."""
    parsed = []
    for frame in frames:
        for line in frame.decode().split("\n"):
            if line.startswith("data: "):
                data = line[len("data: ") :]
                parsed.append(data if data == "[DONE]" else json.loads(data))
    return parsed


def deltas(stream_events: list, key: str) -> str:
    return "".join(event["choices"][0]["delta"].get(key, "") for event in stream_events if isinstance(event, dict))


def test_a_stream_shows_the_text_as_it_is_sampled_and_records_the_whole_sample() -> None:
    engine = FakeMLXEngine()

    frames, response = stream({"messages": HI}, engine)

    wire = events(frames)
    assert wire[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert deltas(wire, "content") == "an answer"
    assert engine.streamed == ["an a", "nswer"]
    assert wire[-2]["choices"][0]["finish_reason"] == "stop"
    assert wire[-2]["choices"][0]["meta_info"] == response.record_response["choices"][0]["meta_info"]
    assert wire[-1] == "[DONE]"
    assert response.record_response_pending is True
    assert response.record_response["choices"][0]["message"] == {"role": "assistant", "content": "an answer"}
    assert response.record_response["training"]["tokens"] == [11, 12, 13, 21, 22]
    assert response.record_response["id"] == wire[0]["id"]


def test_a_stream_splits_reasoning_from_content_as_it_goes() -> None:
    engine = engine_with(
        "pondering</think>\nThe answer is 36.",
        prompt_tail=THINKING_TAIL,
        pieces=["ponder", "ing</thi", "nk>\nThe answer", " is 36."],
    )

    frames, response = stream({"messages": HI}, engine)

    assert deltas(events(frames), "reasoning_content") == "pondering"
    assert deltas(events(frames), "content") == "The answer is 36."
    assert response.record_response["choices"][0]["message"]["reasoning_content"] == "pondering"


def test_a_streamed_tool_call_is_held_and_arrives_as_tool_calls() -> None:
    engine = engine_with(
        f"I'll read it.\n{XML_CALL}",
        pieces=[
            "I'll read",
            " it.\n<tool_",
            "call>\n<function=read_file>\n",
            "<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
        ],
    )

    frames, response = stream({"messages": HI, "tools": READ_FILE}, engine)

    wire = events(frames)
    assert "<tool_call>" not in b"".join(frames).decode()
    assert deltas(wire, "content") == "I'll read it."
    [call] = [
        call
        for event in wire
        if isinstance(event, dict)
        for call in event["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert call["function"] == {"name": "read_file", "arguments": '{"path": "README.md"}'}
    assert call["id"] == response.record_response["choices"][0]["message"]["tool_calls"][0]["id"]
    assert wire[-2]["choices"][0]["finish_reason"] == "tool_calls"


@dataclass
class WaitingEngine(FakeMLXEngine):
    """Emits one piece, then waits until the listener cancels."""

    cancelled: threading.Event = field(default_factory=threading.Event)

    def generate(self, prompt_tokens, *, max_tokens=None, temperature=None, listener=None):
        if listener is None:
            return super().generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature)
        listener.emit("an ")
        for _ in range(500):
            if listener.cancelled():
                self.cancelled.set()
                return replace(self.rollout, finish_reason="cancelled")
            threading.Event().wait(0.01)
        return self.rollout


def test_closing_a_stream_early_cancels_the_generation_and_frees_the_engine() -> None:
    engine = WaitingEngine()
    handler = MLXServingRuntime(engine).inference_handler

    async def run():
        response = await handler.inference_stream(ARTIFACT, CHAT, {"messages": HI, "stream": True})
        frames = [await anext(response.chunks), await anext(response.chunks)]
        await response.close()
        follow_up = await handler.inference(ARTIFACT, CHAT, {"messages": HI})
        return frames, response, follow_up

    frames, response, follow_up = asyncio.run(run())

    assert deltas(events(frames), "content") == "an "
    assert engine.cancelled.is_set()
    assert response.record_response is None
    assert follow_up["choices"][0]["finish_reason"] == "stop"


def test_a_stream_of_no_response_tokens_fails_before_the_terminator() -> None:
    engine = FakeMLXEngine(
        rollout=replace(DEFAULT_ROLLOUT, output_tokens=(), rollout_log_probs=(), text=""), pieces=[]
    )
    handler = MLXServingRuntime(engine).inference_handler

    async def run():
        response = await handler.inference_stream(ARTIFACT, CHAT, {"messages": HI, "stream": True})
        frames = []
        with pytest.raises(UpstreamStatusError, match="no response tokens"):
            async for frame in response.chunks:
                frames.append(frame)
        return frames, response

    frames, response = asyncio.run(run())

    assert b"[DONE]" not in b"".join(frames)
    assert response.record_response is None
