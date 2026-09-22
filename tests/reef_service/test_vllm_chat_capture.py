"""The vLLM chat facade records exact samples from ``/inference/v1/generate``."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.artifact import Artifact, LiveWeightArtifactRef
from reef.inference.vllm.chat import VLLMGenerateClient, VLLMInferenceHandler, VLLMToolCallParser


class FakeTokenizer:
    def __init__(self) -> None:
        self.rendered_messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert add_generation_prompt is True
        if not tokenize:
            return "<user></user><assistant>"
        self.rendered_messages = (messages, kwargs)
        return [10, 11]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in token_ids)


class FixedParserClient(VLLMGenerateClient):
    """The vLLM client with one injected tool parser; remembers what the handler asked for."""

    def __init__(self, parser) -> None:
        self.parser = parser
        self.inputs = {}

    def tool_parser(self, tools, parser_name, tokenizer):
        self.inputs.update(tools=tools, parser_name=parser_name, tokenizer=tokenizer)
        return self.parser


def _local_artifact(tmp_path) -> Artifact:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return Artifact.local(checkpoint)


def _live_artifact() -> Artifact:
    return Artifact(
        LiveWeightArtifactRef(
            content_id="live-test",
            release_id="live:engine:6",
            parent_release_id="checkpoint",
            runtime_load_id="engine:6",
        ),
        None,
    )


def _generate_response(*, stamps=None, finish_reason="stop", top_logprobs=None):
    content = [
        {"token": "token_id:20", "logprob": -0.25, "bytes": None},
        {"token": "token_id:21", "logprob": -0.5, "bytes": None},
    ]
    if top_logprobs is not None:
        for entry, candidates in zip(content, top_logprobs, strict=True):
            entry["top_logprobs"] = candidates
    response = {
        "request_id": "generate-tokens-1",
        "model": "served-model",
        "choices": [
            {"index": 0, "token_ids": [20, 21], "logprobs": {"content": content}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }
    if stamps is not None:
        response["kv_transfer_params"] = {"reef_token_runtime_load_ids": stamps}
    return response


async def _serve(handler_body, run):
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
def test_vllm_facade_records_engine_ids_and_connector_stamps_without_retokenizing() -> None:
    async def run(base_url, received):
        tokenizer = FakeTokenizer()
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=tokenizer)
        response = await backend.inference(
            _live_artifact(),
            "/v1/chat/completions",
            {
                "model": "served-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_completion_tokens": 2,
                "logprobs": True,
                "top_logprobs": 1,
            },
        )
        assert received == [
            {
                "token_ids": [10, 11],
                "sampling_params": {"max_tokens": 2, "temperature": 0.7, "top_p": 0.9, "logprobs": 1},
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
        assert response["training"] == {
            "tokens": [10, 11, 20, 21],
            "loss_mask": [1, 1],
            "rollout_log_probs": [-0.25, -0.5],
            "prompt_length": 2,
            "response_length": 2,
            "runtime_load_id": None,
            "runtime_load_spans": [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 2, "runtime_load_id": "engine:7"},
            ],
        }

    asyncio.run(_serve(_generate_response(stamps=["engine:6", "engine:7"]), run))


@pytest.mark.unit
def test_vllm_facade_serves_a_local_release_without_stamps_under_its_release_version(tmp_path) -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        artifact = _local_artifact(tmp_path)
        response = await backend.inference(
            artifact,
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 2},
        )
        assert received[0]["sampling_params"] == {"max_tokens": 2, "logprobs": 0}
        assert response["training"]["runtime_load_id"] == artifact.ref.release_id
        assert response["training"]["runtime_load_spans"] == [
            {"start": 0, "end": 2, "runtime_load_id": artifact.ref.release_id}
        ]

    asyncio.run(_serve(_generate_response(), run))


@pytest.mark.unit
def test_vllm_facade_rejects_a_live_release_without_per_token_versions() -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        with pytest.raises(ValueError, match="per-token runtime load IDs"):
            await backend.inference(
                _live_artifact(),
                "/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 2},
            )

    asyncio.run(_serve(_generate_response(), run))


@pytest.mark.unit
def test_vllm_facade_rejects_a_response_without_one_logprob_per_token() -> None:
    body = _generate_response(stamps=["engine:6", "engine:6"])
    body["choices"][0]["logprobs"]["content"].pop()

    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        with pytest.raises(ValueError, match="one log probability per sampled token"):
            await backend.inference(
                _live_artifact(),
                "/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 2},
            )

    asyncio.run(_serve(body, run))


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
        {"max_tokens": 512, "top_k": 20, "temperature": 0.6},
    )
    assert sampling == {"max_tokens": 7, "temperature": 0.3, "seed": 11, "stop": ["\n"], "top_k": 20, "min_tokens": 2}

    payload = client.payload(
        {"lora_path": "scenario-a:engine:9", "rid": "req-1", "top_logprobs": 1},
        [1, 2],
        sampling,
        capture_topk=3,
        stream=False,
    )
    assert payload == {
        "token_ids": [1, 2],
        "sampling_params": {**sampling, "logprobs": 3},
        "stream": False,
        "model": "scenario-a:engine:9",
        "request_id": "req-1",
    }
    assert "model" not in client.payload({}, [1], sampling, capture_topk=0, stream=False)
    with pytest.raises(ValueError, match="max_completion_tokens"):
        client.sampling_params({"max_tokens": 0}, {})


@pytest.mark.unit
def test_vllm_topk_capture_reads_token_ids_from_top_logprobs() -> None:
    complete = _generate_response(
        stamps=["engine:6", "engine:6"],
        top_logprobs=[
            [{"token": "token_id:20", "logprob": -0.25}, {"token": "token_id:5", "logprob": -1.5}],
            [{"token": "token_id:21", "logprob": -0.5}, {"token": "token_id:9", "logprob": -2.0}],
        ],
    )
    partial = _generate_response(
        stamps=["engine:6", "engine:6"],
        top_logprobs=[[{"token": "token_id:20", "logprob": -0.25}], [{"token": "token_id:21", "logprob": -0.5}]],
    )
    client = VLLMGenerateClient()
    captured = client.parse(_live_artifact(), complete, capture_topk=2)
    assert captured.topk_indices == [[20, 5], [21, 9]]
    assert captured.topk_log_probs == [[-0.25, -1.5], [-0.5, -2.0]]
    truncated = client.parse(_live_artifact(), partial, capture_topk=2)
    assert truncated.topk_indices is None and truncated.topk_log_probs is None


@pytest.mark.unit
def test_vllm_facade_streams_the_buffered_turn_as_openai_frames() -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        stream = await backend.inference_stream(
            _live_artifact(),
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
        assert stream.record_response["training"]["tokens"] == [10, 11, 20, 21]
        assert stream.headers["Content-Type"].startswith("text/event-stream")

    asyncio.run(_serve(_generate_response(stamps=["engine:6", "engine:6"]), run))


@pytest.mark.unit
def test_vllm_facade_streams_anthropic_messages_from_the_buffered_turn() -> None:
    async def run(base_url, received):
        backend = VLLMInferenceHandler(base_url, model_path="model", tokenizer=FakeTokenizer())
        stream = await backend.inference_stream(
            _live_artifact(),
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
        assert stream.record_response["training"]["request_messages"] == [{"role": "user", "content": "hello"}]

    asyncio.run(_serve(_generate_response(stamps=["engine:6", "engine:6"]), run))


@pytest.mark.unit
def test_vllm_tool_call_parser_adapts_vllm_extraction_without_changing_training_tokens() -> None:
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
            _live_artifact(),
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "read it"}], "tools": tools, "max_tokens": 2},
        )
        assert factory.inputs["tools"] == tools and factory.inputs["parser_name"] == "hermes"
        message = response["choices"][0]["message"]
        assert message["content"] is None
        assert [call["function"] for call in message["tool_calls"]] == [
            {"name": "read", "arguments": '{"path":"README.md"}'}
        ]
        assert response["choices"][0]["finish_reason"] == "tool_calls"
        assert response["training"]["tokens"] == [10, 11, 20, 21]

    asyncio.run(_serve(_generate_response(stamps=["engine:6", "engine:6"]), run))


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
