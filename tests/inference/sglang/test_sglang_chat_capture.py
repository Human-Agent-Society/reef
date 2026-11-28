from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
from aiohttp import ClientPayloadError, web
from aiohttp.test_utils import TestClient, TestServer
from tests.support import tokenizers

from recipes.openclawrl.turns import main_turn_message, turn_request_messages
from reef.api.inference_service import build_inference_app
from reef.api.streaming import stream_record
from reef.artifact import Artifact
from reef.core.errors import inference_aborted
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.inference.chat import remaining_tool_arguments, sse_json_events
from reef.inference.interfaces import FixedEngineClients, InferenceArtifact, full_weight_name
from reef.inference.sglang.chat import NativeStreamCapture, SGLangGenerateClient, SGLangInferenceHandler
from reef.storage.sql.coordination import memory_coordination
from reef.storage.sql.sqlite import SQLiteScenarioStorage


class RecordingClient(SGLangGenerateClient):
    """The SGLang client with a fake tool parser per request; remembers what the handler asked for."""

    def __init__(self, parser_type, inputs=None) -> None:
        self._parser_type = parser_type
        self._inputs = inputs

    def tool_parser(self, tools, parser_name, tokenizer):
        if self._inputs is not None:
            self._inputs.update(tools=tools, parser_name=parser_name)
        return self._parser_type()


class FakeTokenizer(tokenizers.FakeTokenizer):
    """The shared fake, which decodes the first token only and keeps special tokens, as these tests read it."""

    def decode(self, token_ids, *, skip_special_tokens=False):
        assert skip_special_tokens is False
        return f"<{token_ids[0]}>"


class HybridThinkingTokenizer(FakeTokenizer):
    """A Qwen3.5-style hybrid template: <think> opens by default and closes at once for enable_thinking=False."""

    vocabulary = {10: "<user>", 11: "<assistant>", 12: "<think>", 13: "\n", 14: "</think>"}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        prompt = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )
        if kwargs.get("enable_thinking", True):
            return [*prompt, 12, 13]
        return [*prompt, 12, 13, 13, 14, 13, 13]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.vocabulary[token_id] for token_id in token_ids)


@pytest.mark.unit
def test_reasoning_check_reads_the_rendered_prompt() -> None:
    tokenizer = HybridThinkingTokenizer()
    thinking = tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=True)
    answering = tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=True, enable_thinking=False)

    backend = SGLangInferenceHandler("http://unused", model_path="model", tokenizer=tokenizer)
    assert backend.reasoning_is_pre_opened(thinking) is True
    assert backend.reasoning_is_pre_opened(answering) is False

    pinned = SGLangInferenceHandler("http://unused", model_path="model", tokenizer=tokenizer, force_reasoning=True)
    assert pinned.reasoning_is_pre_opened(answering) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "force_reasoning", "message"),
    [
        ("No tags here.", False, {"role": "assistant", "content": "No tags here."}),
        ("<think>still thinking", False, {"role": "assistant", "content": "", "reasoning_content": "still thinking"}),
        (
            "<think>a</think>\n\nb</think>c",
            False,
            {"role": "assistant", "content": "b</think>c", "reasoning_content": "a"},
        ),
        (
            "plan the call\n<tool_call>{}</tool_call>",
            True,
            {"role": "assistant", "content": "<tool_call>{}</tool_call>", "reasoning_content": "plan the call"},
        ),
    ],
)
def test_reasoning_split_follows_sglang_qwen3_parser(text, force_reasoning, message) -> None:
    """A leading <think> starts reasoning; the first </think> or <tool_call> ends it."""
    split, _ = SGLangInferenceHandler._assistant_message(text, None, force_reasoning=force_reasoning)
    assert split == message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("template_kwargs", "message"),
    [
        ({}, {"role": "assistant", "content": "", "reasoning_content": "The answer is 5."}),
        ({"enable_thinking": False}, {"role": "assistant", "content": "The answer is 5."}),
    ],
)
def test_chat_facade_splits_a_tag_free_reply_by_the_rendered_prompt(tmp_path, template_kwargs, message) -> None:
    """A tag-free sample is truncated reasoning after an open <think>, and the answer after a closed one."""

    async def run() -> None:
        async def generate(request):
            return web.json_response(
                {
                    "text": "The answer is 5.",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.25, 20], [-0.5, 21]],
                        "weight_version": "wv-9",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=HybridThinkingTokenizer(),
            )
            response = await backend.inference(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "what is 2+3?"}],
                    "chat_template_kwargs": template_kwargs,
                    "max_completion_tokens": 2,
                },
            )
        finally:
            await server.close()

        assert response["choices"][0]["message"] == message
        assert "prompt_token_ids" not in response and "token_ids" not in response["choices"][0]

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        (None, {"role": "assistant", "content": "The answer is 5."}),
        ({"enable_thinking": True}, {"role": "assistant", "content": "", "reasoning_content": "The answer is 5."}),
    ],
)
def test_handler_chat_template_kwargs_apply_when_the_request_omits_them(tmp_path, request_kwargs, message) -> None:
    """The handler's default renders non-thinking prompts; a request's own key wins."""

    async def run() -> None:
        async def generate(request):
            return web.json_response(
                {
                    "text": "The answer is 5.",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.25, 20], [-0.5, 21]],
                        "weight_version": "wv-9",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        payload = {
            "model": "served-model",
            "messages": [{"role": "user", "content": "what is 2+3?"}],
            "max_completion_tokens": 2,
        }
        if request_kwargs is not None:
            payload["chat_template_kwargs"] = request_kwargs
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=HybridThinkingTokenizer(),
                chat_template_kwargs={"enable_thinking": False},
            )
            response = await backend.inference(local_artifact(tmp_path), "/v1/chat/completions", payload)
        finally:
            await server.close()

        assert response["choices"][0]["message"] == message

    asyncio.run(run())


@pytest.mark.unit
def test_handler_chat_template_kwargs_must_not_carry_tools() -> None:
    with pytest.raises(ValueError, match="tools is reserved"):
        SGLangInferenceHandler("http://engine", model_path="model", chat_template_kwargs={"tools": []})


def local_artifact(tmp_path) -> InferenceArtifact:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return InferenceArtifact.from_artifact(Artifact.local(checkpoint))


def live_artifact() -> InferenceArtifact:
    return InferenceArtifact("artifact:engine:6")


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["transport", "trailing-data"])
def test_native_terminal_marker_does_not_hide_a_late_failure(failure) -> None:
    async def chunks():
        yield b'data: {"text":"answer"}\n\ndata: [DONE]\n\n'
        if failure == "transport":
            raise ClientPayloadError("truncated body")
        yield b'data: {"error":{"message":"late failure"}}\n\n'

    async def run():
        with pytest.raises((ClientPayloadError, ValueError)):
            [event async for event in sse_json_events(chunks())]

    asyncio.run(run())


@pytest.mark.unit
def test_native_stream_capture_requires_a_finish_reason() -> None:
    capture = NativeStreamCapture()
    capture.accept({"text": "partial", "meta_info": {"completion_tokens": 1, "output_token_logprobs": [[-0.5, 20]]}})
    with pytest.raises(ValueError, match="without a finish reason"):
        capture.response()


@pytest.mark.unit
def test_native_stream_capture_rejects_cumulative_chunks() -> None:
    capture = NativeStreamCapture()
    capture.accept(
        {
            "text": "a",
            "output_ids": [20],
            "meta_info": {
                "completion_tokens": 1,
                "output_token_logprobs": [[-0.5, 20]],
            },
        }
    )

    with pytest.raises(ValueError, match="--incremental-streaming-output"):
        capture.accept(
            {
                "text": "answer",
                "output_ids": [20, 21],
                "meta_info": {
                    "completion_tokens": 2,
                    "output_token_logprobs": [[-0.5, 20], [-0.25, 21]],
                },
            }
        )


@pytest.mark.unit
def test_native_stream_capture_joins_many_disjoint_text_chunks() -> None:
    capture = NativeStreamCapture()
    capture.text = "prefix-"
    chunks = [f"chunk-{index};" for index in range(2048)]
    for index, text in enumerate(chunks, start=1):
        capture.accept(
            {
                "text": text,
                "meta_info": {
                    "finish_reason": {"type": "stop"} if index == len(chunks) else None,
                    "completion_tokens": index,
                    "output_token_logprobs": [[-0.5, 20]],
                },
            }
        )

    assert capture.text == "prefix-" + "".join(chunks)
    assert capture.response()["text"] == capture.text


@pytest.mark.unit
def test_chat_facade_forwards_only_an_explicit_lora_path() -> None:
    backend = SGLangInferenceHandler(
        "http://unused",
        model_path="model",
        tokenizer=FakeTokenizer(),
    )

    without_adapter = backend.client.payload({}, [10, 11], {}, stream=False)
    with_adapter = backend.client.payload({"lora_path": "reef_lora"}, [10, 11], {}, stream=False)

    assert "lora_path" not in without_adapter
    assert with_adapter["lora_path"] == "reef_lora"


@pytest.mark.unit
def test_chat_facade_answers_from_engine_native_ids_without_retokenizing(tmp_path) -> None:
    async def run() -> None:
        async def generate(request):
            assert request.path == "/generate"
            assert await request.json() == {
                "input_ids": [10, 11],
                "sampling_params": {
                    "max_new_tokens": 2,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                "return_logprob": True,
                "stream": False,
                "top_logprobs_num": 1,
            }
            return web.json_response(
                {
                    "text": "decoded answer",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.25, 20], [-0.5, 21]],
                        "weight_version": "wv-7",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            tokenizer = FakeTokenizer()
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=tokenizer,
            )
            response = await backend.inference(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_completion_tokens": 2,
                    "logprobs": True,
                    "top_logprobs": 1,
                    "return_token_ids": True,
                },
            )
        finally:
            await server.close()

        assert tokenizer.rendered_messages == ([{"role": "user", "content": "hello"}], {})
        assert response["choices"][0]["message"]["content"] == "decoded answer"
        assert "output_token_logprobs" not in response["choices"][0]["meta_info"]
        logprobs = response["choices"][0]["logprobs"]["content"]
        assert [(item["token"], item["logprob"]) for item in logprobs] == [("<20>", -0.25), ("<21>", -0.5)]
        assert (response["prompt_token_ids"], response["choices"][0]["token_ids"]) == ([10, 11], [20, 21])
        assert "training" not in response, "serving returns no training tensors"

    asyncio.run(run())


@pytest.mark.unit
def test_pinned_chat_sends_the_selected_weight_name_and_the_base_model() -> None:
    body = {
        "text": "answer",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "completion_tokens": 3,
            "output_token_logprobs": [[-0.1, 20], [-0.2, 21], [-0.3, 22]],
        },
    }
    artifact = InferenceArtifact("artifact-A", served_model=full_weight_name("/remote/model", "engine:6"))

    async def run():
        async def generate(request):
            assert request.headers["x-reef-served-model"] == artifact.served_model
            assert "lora_path" not in await request.json()
            return web.json_response(body)

        app = web.Application()
        app.router.add_post("/generate", generate)
        async with TestServer(app) as server:
            handler = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"), model_path="model", tokenizer=FakeTokenizer()
            )
            payload = handler.address_model({"messages": [{"role": "user", "content": "hi"}]}, artifact.served_model)
            assert payload["model"] == "model"
            assert (await handler.inference(artifact, "/v1/chat/completions", payload))["choices"]

    asyncio.run(run())


@pytest.mark.unit
def test_chat_facade_reports_the_engine_weight_version_as_the_runtime_load_id(tmp_path) -> None:
    body = {
        "text": "answer",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "completion_tokens": 3,
            "output_token_logprobs": [[-0.1, 20], [-0.2, 21], [-0.3, 22]],
            "weight_version": "engine:8",
        },
    }
    artifact = local_artifact(tmp_path)

    async def run() -> None:
        async def generate(request):
            native = await request.json()
            assert native["stream"] is False
            assert "stream_interval" not in native["sampling_params"]
            return web.json_response(body)

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
            )
            response = await backend.inference(
                artifact,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 3,
                },
            )
        finally:
            await server.close()

        assert "training" not in response
        assert response["choices"][0]["meta_info"]["runtime_load_id"] == "engine:8"

    asyncio.run(run())
    captured = SGLangGenerateClient().parse(body)
    assert captured.rollout_log_probs == [-0.1, -0.2, -0.3]


@pytest.mark.unit
def test_chat_facade_parses_tool_calls(tmp_path) -> None:
    async def run() -> None:
        async def generate(request):
            assert (await request.json())["input_ids"] == [10, 11]
            return web.json_response(
                {
                    "text": '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, 20], [-0.2, 21]],
                        "weight_version": "wv-tools",
                    },
                }
            )

        class FakeToolParser:
            @staticmethod
            def has_tool_call(text):
                return text.startswith("<tool_call>")

            @staticmethod
            def parse_non_stream(text):
                del text
                return "", [
                    SimpleNamespace(
                        name="read_file",
                        parameters='{"path":"README.md"}',
                        tool_index=0,
                    )
                ]

        parser_inputs = {}

        parser_factory = RecordingClient(FakeToolParser, parser_inputs)

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            tokenizer = FakeTokenizer()
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=tokenizer,
                tool_call_parser="qwen25",
                client=parser_factory,
            )
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
            response = await backend.inference(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "read README"}],
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_completion_tokens": 2,
                },
            )
        finally:
            await server.close()

        assert parser_inputs == {"tools": tools, "parser_name": "qwen25"}
        assert tokenizer.rendered_messages == (
            [{"role": "user", "content": "read README"}],
            {"tools": tools},
        )
        choice = response["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"][0]["function"] == {
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
        assert "training" not in response

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("asks_for_tokens", [False, True])
def test_anthropic_facade_normalizes_messages_and_returns_no_training_block(tmp_path, asks_for_tokens) -> None:
    """Reef's extension of Anthropic Messages: ``logprobs`` and ``return_token_ids`` add the sampled tokens' log
    probabilities and token IDs at the top level of the message."""

    async def run() -> None:
        async def generate(request):
            native = await request.json()
            assert native == {
                "input_ids": [10, 11],
                "sampling_params": {
                    "max_new_tokens": 2,
                    "temperature": 0.4,
                    "stop": ["DONE"],
                },
                "return_logprob": True,
                "stream": False,
            }
            return web.json_response(
                {
                    "text": "working\n</think>\nFinished.",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.2, 20], [-0.4, 21]],
                        "weight_version": "wv-anthropic",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            tokenizer = FakeTokenizer()
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=tokenizer,
            )
            response = await backend.inference(
                local_artifact(tmp_path),
                "/v1/messages",
                {
                    "model": "served-model",
                    "system": [{"type": "text", "text": "Be concise."}],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "text", "text": "there"},
                            ],
                        }
                    ],
                    "max_tokens": 2,
                    "temperature": 0.4,
                    "stop_sequences": ["DONE"],
                    **({"logprobs": True, "return_token_ids": True} if asks_for_tokens else {}),
                },
            )
        finally:
            await server.close()

        tokens = {
            "logprobs": {
                "content": [
                    {"token": "<20>", "bytes": list(b"<20>"), "logprob": -0.2, "top_logprobs": [ANY]},
                    {"token": "<21>", "bytes": list(b"<21>"), "logprob": -0.4, "top_logprobs": [ANY]},
                ]
            },
            "token_ids": [20, 21],
            "prompt_token_ids": [10, 11],
        }
        assert {key: response[key] for key in tokens if key in response} == (tokens if asks_for_tokens else {})
        canonical_messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello\nthere"},
        ]
        assert tokenizer.rendered_messages == (canonical_messages, {})
        assert response["type"] == "message"
        assert response["role"] == "assistant"
        assert response["stop_reason"] == "end_turn"
        assert response["usage"] == {"input_tokens": 2, "output_tokens": 2}
        assert response["content"] == [
            {"type": "thinking", "thinking": "working", "signature": ""},
            {"type": "text", "text": "Finished."},
        ]
        assert "training" not in response
        assert "meta_info" not in response, "only an engine abort adds meta_info"

    asyncio.run(run())


@pytest.mark.unit
def test_anthropic_count_tokens_uses_the_exact_template_without_exposing_training(tmp_path) -> None:
    tokenizer = FakeTokenizer()
    backend = SGLangInferenceHandler(
        "http://unused",
        model_path="model",
        tokenizer=tokenizer,
    )

    response = asyncio.run(
        backend.inference(
            local_artifact(tmp_path),
            "/v1/messages/count_tokens",
            {
                "model": "served-model",
                "system": "Be concise.",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert tokenizer.rendered_messages == (
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ],
        {},
    )
    assert response == {"input_tokens": 2}


@pytest.mark.unit
def test_anthropic_facade_converts_tool_history_and_sampled_tool_use(tmp_path) -> None:
    async def run() -> None:
        async def generate(request):
            assert (await request.json())["input_ids"] == [10, 11]
            return web.json_response(
                {
                    "text": '<tool_call>{"name":"write_file","arguments":{"path":"out.txt"}}</tool_call>',
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, 20]],
                        "weight_version": "wv-tools",
                    },
                }
            )

        class FakeToolParser:
            @staticmethod
            def has_tool_call(text):
                return text.startswith("<tool_call>")

            @staticmethod
            def parse_non_stream(text):
                del text
                return "", [SimpleNamespace(name="write_file", parameters={"path": "out.txt"})]

        parser_inputs = {}

        parser_factory = RecordingClient(FakeToolParser, parser_inputs)

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            tokenizer = FakeTokenizer()
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=tokenizer,
                tool_call_parser="qwen25",
                client=parser_factory,
            )
            response = await backend.inference(
                local_artifact(tmp_path),
                "/v1/messages",
                {
                    "model": "served-model",
                    "messages": [
                        {"role": "user", "content": "Read the file"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_read",
                                    "name": "read_file",
                                    "input": {"path": "in.txt"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_read",
                                    "content": [{"type": "text", "text": "contents"}],
                                },
                                {"type": "text", "text": "Now write it"},
                            ],
                        },
                    ],
                    "tools": [
                        {
                            "name": "write_file",
                            "description": "Write a file",
                            "input_schema": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        }
                    ],
                    "tool_choice": {"type": "any"},
                    "max_tokens": 1,
                },
            )
        finally:
            await server.close()

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                    "description": "Write a file",
                },
            }
        ]
        assert parser_inputs == {"tools": openai_tools, "parser_name": "qwen25"}
        rendered_messages, rendered_kwargs = tokenizer.rendered_messages
        assert rendered_messages[1]["tool_calls"][0]["function"]["arguments"] == {"path": "in.txt"}
        assert rendered_messages[2] == {"role": "tool", "tool_call_id": "toolu_read", "content": "contents"}
        assert rendered_messages[3] == {"role": "user", "content": "Now write it"}
        assert rendered_kwargs == {"tools": openai_tools}
        assert response["stop_reason"] == "tool_use"
        assert response["content"][0]["type"] == "tool_use"
        assert response["content"][0]["name"] == "write_file"
        assert response["content"][0]["input"] == {"path": "out.txt"}
        assert "training" not in response

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("finished", [True, False], ids=["finished", "missing-finish"])
def test_pinned_stream_ends_with_its_terminal_event_only_after_a_finish_reason(finished) -> None:
    artifact = InferenceArtifact("artifact-A", served_model=full_weight_name("/remote/model", "engine:6"))
    # Each sampled token has its own log probability, so the client's entries show their alignment.
    sampled = [("<20>", -0.25), ("<21>", -0.75)]

    async def run() -> None:
        async def generate(request):
            assert request.headers["x-reef-served-model"] == artifact.served_model
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for index, text in enumerate(("first", "later")):
                event = {
                    "text": text,
                    "output_ids": [20 + index],
                    "meta_info": {
                        "completion_tokens": index + 1,
                        "output_token_logprobs": [[sampled[index][1], 20 + index]],
                        "finish_reason": {"type": "stop"} if index == 1 and finished else None,
                    },
                }
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        async with TestServer(app) as server:
            handler = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"), model_path="model", tokenizer=FakeTokenizer()
            )
            stream = await handler.inference_stream(
                artifact,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_completion_tokens": 2,
                    "stream": True,
                    "logprobs": True,
                },
            )
            body = bytearray()
            try:
                with nullcontext() if finished else pytest.raises(ValueError, match="without a finish reason"):
                    async for chunk in stream.chunks:
                        body.extend(chunk)
            finally:
                await stream.close()
            assert b'"content": "first"' in body
            assert b'"content": "later"' in body
            assert body.endswith(b"data: [DONE]\n\n") is finished
            assert (stream.record_response is not None) is finished
            streamed = [
                (entry["token"], entry["logprob"])
                for line in bytes(body).split(b"\n")
                if line.startswith(b"data: {")
                for choice in json.loads(line.removeprefix(b"data: "))["choices"]
                for entry in (choice.get("logprobs") or {}).get("content", [])
            ]
            assert streamed == sampled
            if finished:
                recorded = stream.record_response["choices"][0]["logprobs"]["content"]
                assert [(entry["token"], entry["logprob"]) for entry in recorded] == sampled

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("include_usage", [False, True])
def test_streaming_chat_keeps_the_exact_response_for_recording(tmp_path, include_usage) -> None:
    async def run() -> None:
        text_parts = [f"chunk-{index};" for index in range(2048)]

        async def generate(request):
            native = await request.json()
            assert native["stream"] is True
            assert native["sampling_params"]["stream_interval"] == 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for token_count, text in enumerate(text_parts, start=1):
                output_id = token_count + 19
                event = {
                    "text": text,
                    "output_ids": [output_id],
                    "meta_info": {
                        "finish_reason": {"type": "stop"} if token_count == len(text_parts) else None,
                        "completion_tokens": token_count,
                        "output_token_logprobs": [[-0.5, output_id]],
                        "weight_version": "engine:6",
                    },
                }
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
            )
            stream = await backend.inference_stream(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": len(text_parts),
                    "logprobs": True,
                    "return_token_ids": True,
                    "stream_options": {"include_usage": include_usage},
                    "stream": True,
                },
            )
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        assert text_parts[0].encode() in body
        assert b'"logprobs": {"content":' in body
        assert body.endswith(b"data: [DONE]\n\n")
        chunks = [json.loads(frame[6:]) for frame in body.split(b"\n\n") if frame.startswith(b"data: {")]
        last_chunk = next(chunk for chunk in reversed(chunks) if chunk["choices"])
        assert (last_chunk["prompt_token_ids"], last_chunk["choices"][0]["token_ids"]) == (
            [10, 11],
            list(range(20, 20 + len(text_parts))),
        )
        usage_chunks = [chunk for chunk in chunks if chunk["choices"] == []]
        if include_usage:
            assert [chunk["usage"] for chunk in usage_chunks] == [
                {"prompt_tokens": 2, "completion_tokens": len(text_parts), "total_tokens": len(text_parts) + 2}
            ]
            assert all(chunk["usage"] is None for chunk in chunks if chunk["choices"])
        else:
            assert not usage_chunks
            assert all("usage" not in chunk for chunk in chunks)
        recorded = stream_record(stream, body, complete=True)
        assert recorded["choices"][0]["message"]["content"] == "".join(text_parts)
        assert [item["logprob"] for item in recorded["choices"][0]["logprobs"]["content"]] == [-0.5] * len(text_parts)
        assert (recorded["prompt_token_ids"], recorded["choices"][0]["token_ids"]) == (
            [10, 11],
            list(range(20, 20 + len(text_parts))),
        )
        assert "training" not in recorded
        assert recorded["stream_delivery"]["complete"] is True

    asyncio.run(run())


@pytest.mark.unit
def test_streaming_anthropic_messages_emits_native_sse_and_records_the_message(tmp_path) -> None:
    async def run() -> None:
        release_final = asyncio.Event()

        async def generate(request):
            native = await request.json()
            assert native["stream"] is True
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            first = {
                "text": "ans",
                "output_ids": [20],
                "meta_info": {
                    "finish_reason": None,
                    "completion_tokens": 1,
                    "output_token_logprobs": [[-0.5, 20]],
                    "weight_version": "engine:6",
                },
            }
            await response.write(f"data: {json.dumps(first)}\n\n".encode())
            await release_final.wait()
            final = {
                "text": "wer",
                "output_ids": [21],
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "completion_tokens": 2,
                    "output_token_logprobs": [[-0.25, 21]],
                    "weight_version": "engine:8",
                },
            }
            await response.write(f"data: {json.dumps(final)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
            )
            stream = await backend.inference_stream(
                live_artifact(),
                "/v1/messages",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 2,
                    "stream": True,
                },
            )
            iterator = stream.chunks.__aiter__()
            chunks = [await anext(iterator), await anext(iterator), await anext(iterator)]
            # message_start and the first real content delta arrive while the
            # upstream generation is still blocked before its final token.
            assert b'"type": "text_delta", "text": "ans"' in chunks[-1]
            assert stream.record_response is None
            release_final.set()
            chunks.extend([chunk async for chunk in iterator])
            body = b"".join(chunks)
        finally:
            await server.close()

        text = body.decode()
        assert "event: message_start" in text
        assert '"type": "text_delta", "text": "ans"' in text
        assert '"type": "text_delta", "text": "wer"' in text
        assert '"stop_reason": "max_tokens"' in text
        assert text.endswith('event: message_stop\ndata: {"type": "message_stop"}\n\n')
        assert "[DONE]" not in text
        recorded = stream_record(stream, body, complete=True)
        assert recorded["type"] == "message"
        assert recorded["content"] == [{"type": "text", "text": "answer"}]
        assert "training" not in recorded
        assert recorded["stream_delivery"]["complete"] is True

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_streaming_records_an_engine_abort_that_the_service_detects(path) -> None:
    async def run() -> None:
        async def generate(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            event = {
                "text": "ans",
                "output_ids": [20],
                "meta_info": {
                    "finish_reason": {"type": "abort", "message": "Aborted"},
                    "completion_tokens": 1,
                    "output_token_logprobs": [[-0.5, 20]],
                    "weight_version": "engine:6",
                },
            }
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"), model_path="model", tokenizer=FakeTokenizer()
            )
            stream = await backend.inference_stream(
                live_artifact(),
                path,
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 2,
                    "stream": True,
                },
            )
            [chunk async for chunk in stream.chunks]
        finally:
            await server.close()

        assert inference_aborted(stream.record_response)

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("pre_opened", [False, True])
def test_streaming_anthropic_reassembles_incremental_sglang_thinking_chunks(pre_opened: bool) -> None:
    async def run() -> None:
        async def generate(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            events = [
                {
                    "text": "\nwork  </thi",
                    "output_ids": [20],
                    "meta_info": {
                        "finish_reason": None,
                        "completion_tokens": 1,
                        "output_token_logprobs": [[-0.5, 20]],
                    },
                },
                {
                    "text": "nk>\n",
                    "output_ids": [21],
                    "meta_info": {
                        "finish_reason": None,
                        "completion_tokens": 2,
                        "output_token_logprobs": [[-0.25, 21]],
                    },
                },
                {
                    "text": "\nanswer",
                    "output_ids": [22],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 3,
                        "output_token_logprobs": [[-0.25, 22]],
                    },
                },
            ]
            for event in events:
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
                force_reasoning=pre_opened,
            )
            stream = await backend.inference_stream(
                live_artifact(),
                "/v1/messages",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 3,
                    "stream": True,
                },
            )
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        deltas = [event["delta"] for event in events if event["type"] == "content_block_delta"]
        thinking = "".join(delta["thinking"] for delta in deltas if delta["type"] == "thinking_delta")
        answer = "".join(delta["text"] for delta in deltas if delta["type"] == "text_delta")
        recorded = stream_record(stream, body, complete=True)
        if pre_opened:
            assert (thinking, answer) == ("work", "answer")
            assert recorded["content"] == [
                {"type": "thinking", "thinking": "work", "signature": ""},
                {"type": "text", "text": "answer"},
            ]
        else:
            assert (thinking, answer) == ("", "\nwork  </think>\n\nanswer")
            assert recorded["content"] == [{"type": "text", "text": answer}]
        assert "training" not in recorded

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "arguments", "failure", "repeated_think_tags", "completed_arguments"),
    [
        *(
            pytest.param(path, arguments, failure, repeated, None, id=f"{name}-{path}")
            for path in ("/v1/chat/completions", "/v1/messages")
            for name, arguments, failure, repeated in (
                ("equivalent", '{"path":"out.txt"}', None, False),
                ("prefix", '{"path": "out', None, False),
                ("spaced-prefix", '{"path":   "out', None, False),
                ("divergent-arguments", '{"path":"other.txt"}', "streamed tool-call arguments", False),
                ("divergent-text", '{"path":"out.txt"}', "streamed text", False),
                ("repeated-think-tags", '{"path":"out.txt"}', None, True),
                ("negative-index", '{"path":"out.txt"}', "streamed tool-call index", False),
            )
        ),
        pytest.param("/v1/chat/completions", '{"path":"README', None, False, '{"path":"README', id="raw-match"),
        pytest.param("/v1/chat/completions", '{"path":"RE', None, False, '{"path":"README', id="raw-prefix"),
        pytest.param("/v1/chat/completions", None, None, False, '{"path":"README', id="raw-buffered-only"),
        pytest.param(
            "/v1/chat/completions",
            '{"path":"else',
            "completed tool-call arguments",
            False,
            '{"path":"README',
            id="raw-divergence",
        ),
        pytest.param(
            "/v1/messages",
            '{"path":"README',
            "completed tool-call arguments are not valid JSON",
            False,
            '{"path":"README',
            id="anthropic-invalid-json",
        ),
        pytest.param("/v1/messages", "[]", "must be a JSON object", False, "[]", id="anthropic-non-object"),
    ],
)
def test_streamed_tool_turns_agree_with_the_buffered_result(
    tmp_path, path, arguments, failure, repeated_think_tags, completed_arguments
) -> None:
    raw_tool_call = '<tool_call>{"name":"write_file","arguments":{"path":"out.txt"}}</tool_call>'
    if failure == "streamed text":
        chunks = ("Before", raw_tool_call, "After")
    elif repeated_think_tags:
        chunks = ("\n ", "<think>", "\t<thi", "nk>work</think>", raw_tool_call)
    else:
        chunks = (raw_tool_call,)

    class FakeStreamingToolParser:
        """Present one tool call to the stream and buffered paths under the fixture's fault cases.

        ``raw_tool_call`` is the one tool chunk. ``arguments`` is the streamed fragment, which may differ
        from the buffered object. ``failure`` is None for an accepted case or the expected error text.

        ``has_tool_call`` checks for a tool marker. ``parse_stream_chunk`` passes ordinary text through.
        For the tool chunk it emits the fragment at index zero, or minus one for the invalid-index case.
        ``parse_non_stream`` keeps only text before the marker and returns the complete tool call.
        """

        @staticmethod
        def has_tool_call(text):
            return "<tool_call>" in text

        @staticmethod
        def parse_stream_chunk(text):
            if text != raw_tool_call:
                return text, []
            if arguments is None:
                return "", []
            return "", [
                SimpleNamespace(
                    name="write_file",
                    parameters=arguments,
                    tool_index=-1 if failure == "streamed tool-call index" else 0,
                )
            ]

        @staticmethod
        def parse_non_stream(text):
            before = text.partition("<tool_call>")[0]
            final = {"path": "out.txt"} if completed_arguments is None else completed_arguments
            return before, [SimpleNamespace(name="write_file", parameters=final)]

    async def run() -> None:
        async def generate(request):
            native = await request.json()

            def event(text, token_ids, total, *, finished):
                return {
                    "text": text,
                    "output_ids": token_ids,
                    "meta_info": {
                        "finish_reason": {"type": "stop"} if finished else None,
                        "completion_tokens": total,
                        "output_token_logprobs": [[-0.5, token_id] for token_id in token_ids],
                        "weight_version": "wv-tools",
                    },
                }

            if not native["stream"]:
                return web.json_response(
                    event("".join(chunks), list(range(20, 20 + len(chunks))), len(chunks), finished=True)
                )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for index, chunk in enumerate(chunks):
                item = event(chunk, [20 + index], index + 1, finished=index + 1 == len(chunks))
                await response.write(f"data: {json.dumps(item)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
                tool_call_parser="qwen25",
                client=RecordingClient(FakeStreamingToolParser),
            )
            artifact = local_artifact(tmp_path)
            anthropic = path == "/v1/messages"
            schema = {"type": "object", "properties": {"path": {"type": "string"}}}
            if anthropic:
                tool = {"name": "write_file", "input_schema": schema}
            else:
                tool = {"type": "function", "function": {"name": "write_file", "parameters": schema}}
            payload = {
                "model": "served-model",
                "messages": [{"role": "user", "content": "write it"}],
                "tools": [tool],
                "max_tokens": len(chunks),
            }
            if failure is not None:
                scenario = "tools"
                storage = SQLiteScenarioStorage(tmp_path / "records")
                coordination = memory_coordination()
                lease = coordination.control_lease.acquire("control")
                coordination.served_models.write(
                    ScenarioModel(scenario, ACTIVE),
                    ServedArtifact("artifact", str(server.make_url("/")), "served-model"),
                    control_lease_number=lease.number,
                )
                replica = build_inference_app(
                    coordination.served_models, storage, FixedEngineClients(backend), tokens=None
                )
                async with TestClient(TestServer(replica)) as client:
                    response = await client.post(
                        path, json={**payload, "stream": True}, headers={"x-reef-scenario": scenario}
                    )
                    assert response.status == 200
                    with pytest.raises(ClientPayloadError):
                        await response.read()
                store = storage.open(scenario)
                try:
                    (record,) = store.records.replay(scenario)
                    failed = record.payload["response"]
                    assert failed["complete"] is False
                    assert failure in failed["error"]
                    assert "[DONE]" not in failed["body"] and "message_stop" not in failed["body"]
                finally:
                    store.close()
                return
            buffered = await backend.inference(artifact, path, payload)
            stream = await backend.inference_stream(artifact, path, {**payload, "stream": True})
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        assert b"<tool_call>" not in body
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: {")]
        recorded = stream_record(stream, body, complete=True)
        if anthropic:
            (start,) = [
                event
                for event in events
                if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
            ]
            assert start["content_block"]["id"] == recorded["content"][-1]["id"]
            fragments = [
                event["delta"]["partial_json"]
                for event in events
                if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta"
            ]
            if repeated_think_tags:
                (thinking,) = [
                    event["delta"]["thinking"]
                    for event in events
                    if event["type"] == "content_block_delta" and event["delta"]["type"] == "thinking_delta"
                ]
                assert thinking == recorded["content"][0]["thinking"] == buffered["content"][0]["thinking"] == "work"
            assert events[-1]["type"] == "message_stop"
            assert recorded["stop_reason"] == buffered["stop_reason"] == "tool_use"
            assert (
                json.loads("".join(fragments)) == recorded["content"][-1]["input"] == buffered["content"][-1]["input"]
            )
        else:
            deltas = [
                delta
                for event in events
                for choice in event["choices"]
                for delta in choice["delta"].get("tool_calls", [])
            ]
            (announced,) = [delta for delta in deltas if "id" in delta]
            recorded_call = recorded["choices"][0]["message"]["tool_calls"][0]
            buffered_call = buffered["choices"][0]["message"]["tool_calls"][0]
            if repeated_think_tags:
                (thinking,) = [
                    choice["delta"]["reasoning_content"]
                    for event in events
                    for choice in event["choices"]
                    if "reasoning_content" in choice["delta"]
                ]
                assert thinking == recorded["choices"][0]["message"]["reasoning_content"] == "work"
            assert announced["id"] == recorded_call["id"]
            fragments = [delta["function"]["arguments"] for delta in deltas]
            assert body.endswith(b"data: [DONE]\n\n")
            emitted = "".join(fragments)
            if completed_arguments is None:
                assert json.loads(emitted) == json.loads(recorded_call["function"]["arguments"])
                assert json.loads(emitted) == json.loads(buffered_call["function"]["arguments"])
            else:
                assert emitted == recorded_call["function"]["arguments"] == buffered_call["function"]["arguments"]
                assert emitted == completed_arguments
        if arguments is not None:
            assert fragments[0] == arguments

    asyncio.run(run())


@pytest.mark.unit
def test_tool_argument_json_comparison_ignores_layout_without_coercing_types() -> None:
    assert (
        remaining_tool_arguments('{"path":"out.txt","enabled":true}', '{ "enabled": true, "path": "out.txt" }') == ""
    )
    with pytest.raises(ValueError, match="arguments differ"):
        remaining_tool_arguments('{"enabled":true}', '{"enabled":1}')


@pytest.mark.unit
def test_streaming_openai_tool_arguments_are_not_duplicated_when_json_spacing_differs(tmp_path) -> None:
    raw_tool_call = '<tool_call>{"name":"write_file","arguments":{"path":"out.txt"}}</tool_call>'

    class FakeStreamingToolParser:
        @staticmethod
        def has_tool_call(text):
            return text.startswith("<tool_call>")

        @staticmethod
        def parse_stream_chunk(text):
            del text
            return "", [
                SimpleNamespace(
                    name="write_file",
                    parameters='{"path": "out.txt"}',
                    tool_index=0,
                )
            ]

        @staticmethod
        def parse_non_stream(text):
            del text
            return "", [SimpleNamespace(name="write_file", parameters='{"path":"out.txt"}')]

    async def run() -> None:
        async def generate(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            event = {
                "text": raw_tool_call,
                "output_ids": [20],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "completion_tokens": 1,
                    "output_token_logprobs": [[-0.5, 20]],
                    "weight_version": "wv-tools",
                },
            }
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
                tool_call_parser="qwen25",
                client=RecordingClient(FakeStreamingToolParser),
            )
            stream = await backend.inference_stream(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "write it"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                },
                            },
                        }
                    ],
                    "max_completion_tokens": 1,
                    "stream": True,
                },
            )
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        payloads = [json.loads(frame[6:]) for frame in body.split(b"\n\n") if frame.startswith(b"data: {")]
        assert all("usage" not in payload for payload in payloads)
        deltas = [
            tool_call
            for payload in payloads
            for choice in payload["choices"]
            for tool_call in choice["delta"].get("tool_calls", [])
        ]
        assert [tool_call["function"].get("arguments") for tool_call in deltas] == ['{"path": "out.txt"}']
        recorded = stream_record(stream, body, complete=True)
        assert recorded["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"path":"out.txt"}'

    asyncio.run(run())


@pytest.mark.unit
def test_openclaw_reads_canonical_anthropic_exchange_from_private_training_block() -> None:
    canonical_messages = [{"role": "user", "content": "hello"}]
    canonical_message = {"role": "assistant", "content": "answer"}
    payload = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        "response": {
            "type": "message",
            "content": [{"type": "text", "text": "answer"}],
            "training": {
                "request_messages": canonical_messages,
                "request_tools": [{"type": "function", "function": {"name": "read"}}],
                "response_message": canonical_message,
            },
        },
    }

    assert main_turn_message(payload) == canonical_message
    assert turn_request_messages(payload) == canonical_messages


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n": 2}, "n=1"),
        ({"tools": []}, "non-empty list"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            "text-only message content",
        ),
        ({"max_completion_tokens": 0}, "positive integer"),
    ],
)
def test_chat_facade_rejects_requests_it_cannot_capture_exactly(tmp_path, overrides, message) -> None:
    async def run() -> None:
        backend = SGLangInferenceHandler(
            "http://unused",
            model_path="model",
            tokenizer=FakeTokenizer(),
        )
        with pytest.raises(ValueError, match=message):
            await backend.inference(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hello"}], **overrides},
            )

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("split_reasoning", [True, False])
def test_chat_facade_splits_reasoning_out_of_visible_content(tmp_path, split_reasoning: bool) -> None:
    """Thinking text must ride reasoning_content, never the visible reply.

    The chat template auto-opens <think>, so sampled text carries only the
    closing tag; everything before the LAST </think> is chain-of-thought.
    OpenAI-shaped consumers (agents, judges, user simulators) read content
    verbatim — leaking reasoning there made every downstream judgment grade
    the monologue instead of the reply. A subclass can preserve raw tags.
    """

    class ConfiguredHandler(SGLangInferenceHandler):
        SPLIT_REASONING = split_reasoning

    async def run() -> None:
        async def generate(request):
            return web.json_response(
                {
                    "text": "Okay, let me think. 2+3 is 5.\n</think>\n\nThe answer is 5.",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.25, 20], [-0.5, 21], [-0.75, 22]],
                        "weight_version": "wv-9",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = ConfiguredHandler(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
            )
            response = await backend.inference(
                local_artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "what is 2+3?"}],
                    "max_completion_tokens": 3,
                },
            )
        finally:
            await server.close()

        message = response["choices"][0]["message"]
        if split_reasoning:
            assert message["content"] == "The answer is 5."
            assert message["reasoning_content"] == "Okay, let me think. 2+3 is 5."
        else:
            assert message == {
                "role": "assistant",
                "content": "Okay, let me think. 2+3 is 5.\n</think>\n\nThe answer is 5.",
            }
        assert response["usage"]["completion_tokens"] == 3
        assert "training" not in response

    asyncio.run(run())
