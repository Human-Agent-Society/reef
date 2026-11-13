"""The vLLM chat facade answers from the exact samples of ``/inference/v1/generate``."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from tests.support.tokenizers import FakeTokenizer

from reef.core.errors import inference_aborted
from reef.inference.interfaces import InferenceArtifact, full_weight_name
from reef.inference.vllm.chat import VLLMGenerateClient, VLLMInferenceHandler, VLLMToolCallParser


class FixedParserClient(VLLMGenerateClient):
    """The vLLM client with one injected tool parser; remembers what the handler asked for."""

    def __init__(self, parser) -> None:
        self.parser = parser
        self.inputs = {}

    def tool_parser(self, tools, parser_name, tokenizer):
        self.inputs.update(tools=tools, parser_name=parser_name, tokenizer=tokenizer)
        return self.parser


def live_artifact() -> InferenceArtifact:
    return InferenceArtifact("artifact:engine:6")


def generate_response(*, finish_reason="stop"):
    content = [
        {"token": "token_id:20", "logprob": -0.25, "bytes": None},
        {"token": "token_id:21", "logprob": -0.5, "bytes": None},
    ]
    return {
        "request_id": "generate-tokens-1",
        "model": "served-model",
        "choices": [
            {"index": 0, "token_ids": [20, 21], "logprobs": {"content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


async def serve(handler_body, run):
    """Run ``run(base_url, requests)`` against a fake engine whose generate route answers ``handler_body``."""
    received = []

    async def generate(request):
        assert request.path == "/inference/v1/generate"
        received.append(await request.json())
        body = handler_body(received[-1]) if callable(handler_body) else handler_body
        return web.json_response(body)

    app = web.Application()
    app.router.add_post("/inference/v1/generate", generate)
    server = TestServer(app)
    await server.start_server()
    try:
        return await run(str(server.make_url("")).rstrip("/"), received)
    finally:
        await server.close()


@pytest.mark.unit
def test_pinned_vllm_chat_sends_the_selected_weight_name_and_the_base_model() -> None:
    artifact = InferenceArtifact("artifact-A", served_model=full_weight_name("/remote/model", "engine:6"))

    async def run():
        async def generate(request):
            assert request.headers["x-reef-served-model"] == artifact.served_model
            assert "model" not in await request.json()
            return web.json_response(generate_response())

        app = web.Application()
        app.router.add_post("/inference/v1/generate", generate)
        async with TestServer(app) as server:
            handler = VLLMInferenceHandler(
                str(server.make_url("")).rstrip("/"), model_path="model", tokenizer=FakeTokenizer()
            )
            payload = handler.address_model({"messages": [{"role": "user", "content": "hi"}]}, artifact.served_model)
            assert payload["model"] == "model"
            assert (await handler.inference(artifact, "/v1/chat/completions", payload))["choices"]

    asyncio.run(run())


@pytest.mark.unit
def test_vllm_facade_answers_from_engine_ids_without_retokenizing() -> None:
    async def run(base_url, received):
        tokenizer = FakeTokenizer()
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=tokenizer)
        response = await backend.inference(
            live_artifact(),
            "/v1/chat/completions",
            {
                "model": "served-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.7,
                "max_completion_tokens": 2,
                "logprobs": True,
                "top_logprobs": 1,
            },
        )
        assert received == [
            {
                "token_ids": [10, 11],
                "sampling_params": {"max_tokens": 2, "temperature": 0.7, "logprobs": 1},
                "stream": False,
            }
        ]
        assert tokenizer.rendered_messages == ([{"role": "user", "content": "hello"}], {})
        choice = response["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "<20><21>"}
        assert choice["finish_reason"] == "stop"
        assert choice["meta_info"] == {
            "request_id": "generate-tokens-1",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }
        assert [item["token"] for item in choice["logprobs"]["content"]] == ["<20>", "<21>"]
        assert "training" not in response, "serving returns no training tensors"

    body = generate_response()
    asyncio.run(serve(body, run))
    captured = VLLMGenerateClient().parse(body)
    assert captured.output_ids == [20, 21]
    assert captured.rollout_log_probs == [-0.25, -0.5]


@pytest.mark.unit
def test_vllm_facade_rejects_a_response_without_one_logprob_per_token() -> None:
    body = generate_response()
    body["choices"][0]["logprobs"]["content"].pop()

    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        with pytest.raises(ValueError, match="one log probability per sampled token"):
            await backend.inference(
                live_artifact(),
                "/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 2},
            )

    asyncio.run(serve(body, run))


@pytest.mark.unit
def test_vllm_sampling_params_and_adapter_selection_use_vllm_names() -> None:
    client = VLLMGenerateClient()
    sampling = client.sampling_params(
        {
            "max_tokens": 7,
            "temperature": 0.3,
            "seed": 11,
            "stop": ["\n"],
            "vllm_sampling_params": {"min_tokens": 2},
        },
        {"max_tokens": 512, "top_k": 0, "temperature": 0.6},
    )
    assert sampling == {"max_tokens": 7, "temperature": 0.3, "seed": 11, "stop": ["\n"], "top_k": 0, "min_tokens": 2}

    payload = client.payload(
        {"lora_path": "scenario-a:engine:9", "rid": "req-1", "top_logprobs": 3},
        [1, 2],
        sampling,
        stream=False,
    )
    assert payload == {
        "token_ids": [1, 2],
        "sampling_params": {**sampling, "logprobs": 3},
        "stream": False,
        "model": "scenario-a:engine:9",
        "request_id": "req-1",
    }
    unaddressed = client.payload({}, [1], sampling, stream=False)
    assert "model" not in unaddressed
    assert unaddressed["sampling_params"]["logprobs"] == 0
    with pytest.raises(ValueError, match="max_completion_tokens"):
        client.sampling_params({"max_tokens": 0}, {})


@pytest.mark.parametrize(
    "request_body, defaults",
    [
        ({"max_tokens": 4, "top_p": 0.9}, {}),
        ({"max_tokens": 4}, {"top_k": 20}),
        ({"max_tokens": 4, "vllm_sampling_params": {"min_p": 0.1}}, {}),
    ],
)
def test_vllm_sampling_params_reject_truncation_the_trainer_cannot_reproduce(request_body, defaults) -> None:
    with pytest.raises(ValueError, match="after top-p, top-k and min-p"):
        VLLMGenerateClient().sampling_params(request_body, defaults)
    accepted = VLLMGenerateClient().sampling_params(
        {"max_tokens": 4, "top_p": 1.0, "top_k": -1, "min_p": 0}, {"temperature": 1.0}
    )
    assert accepted == {"max_tokens": 4, "top_p": 1.0, "top_k": -1, "min_p": 0, "temperature": 1.0}


@pytest.mark.unit
def test_vllm_facade_streams_the_buffered_turn_as_openai_frames() -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        stream = await backend.inference_stream(
            live_artifact(),
            "/v1/chat/completions",
            {"model": "served-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2},
        )
        assert received[0]["stream"] is False
        frames = [frame async for frame in stream.chunks]
        events = [json.loads(frame[len(b"data: ") :]) for frame in frames if frame != b"data: [DONE]\n\n"]
        assert frames[-1] == b"data: [DONE]\n\n"
        assert [event["choices"][0]["delta"] for event in events] == [
            {"role": "assistant"},
            {"content": "<20><21>"},
            {},
        ]
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
        assert {event["id"] for event in events} == {stream.record_response["id"]}
        assert "training" not in stream.record_response
        assert stream.headers["Content-Type"].startswith("text/event-stream")

    asyncio.run(serve(generate_response(), run))


@pytest.mark.unit
def test_vllm_facade_streams_anthropic_messages_from_the_buffered_turn() -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        stream = await backend.inference_stream(
            live_artifact(),
            "/v1/messages",
            {"model": "served-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2},
        )
        frames = b"".join([frame async for frame in stream.chunks]).decode()
        kinds = [line[len("event: ") :] for line in frames.splitlines() if line.startswith("event: ")]
        assert kinds == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert stream.record_response["content"] == [{"type": "text", "text": "<20><21>"}]
        assert "training" not in stream.record_response

    asyncio.run(serve(generate_response(), run))


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_vllm_facade_streams_an_engine_abort_that_the_service_detects(path) -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        stream = await backend.inference_stream(
            live_artifact(),
            path,
            {"model": "served-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2},
        )
        [frame async for frame in stream.chunks]
        assert inference_aborted(stream.record_response)

    asyncio.run(serve(generate_response(finish_reason="abort"), run))


@pytest.mark.unit
@pytest.mark.parametrize("finish_reason", ["stop", "abort"])
def test_vllm_tool_call_parser_adapts_vllm_extraction(finish_reason: str) -> None:
    class FakeVLLMParser:
        def __init__(self) -> None:
            self.calls = 0

        def extract_tool_calls(self, text, request):
            self.calls += 1
            return SimpleNamespace(
                tools_called=True,
                tool_calls=[SimpleNamespace(function=SimpleNamespace(name="read", arguments='{"path":"README.md"}'))],
                content="",
            )

    fake = FakeVLLMParser()
    adapter = VLLMToolCallParser(fake, request=None)
    assert adapter.has_tool_call("<tool>...</tool>") is True
    remaining, calls = adapter.parse_non_stream("<tool>...</tool>")
    assert remaining == "" and [(call.name, call.parameters) for call in calls] == [("read", '{"path":"README.md"}')]
    assert fake.calls == 1

    async def run(base_url, received):
        factory = FixedParserClient(adapter)
        backend = VLLMInferenceHandler(
            base_url,
            model_path="model",
            tokenizer=FakeTokenizer(),
            tool_call_parser="hermes",
            client=factory,
        )
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        response = await backend.inference(
            live_artifact(),
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "read it"}], "tools": tools, "max_tokens": 2},
        )
        assert factory.inputs["tools"] == tools and factory.inputs["parser_name"] == "hermes"
        message = response["choices"][0]["message"]
        assert message["content"] is None
        assert [call["function"] for call in message["tool_calls"]] == [
            {"name": "read", "arguments": '{"path":"README.md"}'}
        ]
        expected_finish_reason = "abort" if finish_reason == "abort" else "tool_calls"
        assert response["choices"][0]["finish_reason"] == expected_finish_reason
        assert "training" not in response

        stream = await backend.inference_stream(
            live_artifact(),
            "/v1/messages",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [{"name": "read", "input_schema": {"type": "object"}}],
                "max_tokens": 2,
                "stream": True,
            },
        )
        frames = [frame async for frame in stream.chunks]
        assert b"message_stop" in frames[-1]
        assert "training" not in stream.record_response
        assert stream.record_response["stop_reason"] == ("end_turn" if finish_reason == "abort" else "tool_use")
        assert inference_aborted(stream.record_response) is (finish_reason == "abort")
        assert stream.record_response["content"][0]["type"] == "tool_use"

    asyncio.run(serve(generate_response(finish_reason=finish_reason), run))


@pytest.mark.unit
def test_vllm_facade_requires_a_tool_parser_for_tool_requests() -> None:
    backend = VLLMInferenceHandler("http://unused", model_path="model", tokenizer=FakeTokenizer())
    with pytest.raises(ValueError, match="tool_call_parser"):
        backend._chat_call(
            "/v1/chat/completions",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
            },
        )
