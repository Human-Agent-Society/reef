from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import io
import json
import logging
import os
import sys
import threading
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from recipes.openclawrl.turns import main_turn_message, turn_request_messages
from reef.artifact import Artifact, ArtifactRef
from reef.core.images import PreparedImage
from reef.inference.chat import CapturedGeneration, ChatCall, ReasoningStreamSplitter, RenderedPrompt
from reef.inference.sglang.chat import NativeStreamCapture, SGLangGenerateClient, SGLangInferenceHandler
from reef.runtime.interfaces import inference_aborted
from reef.service.request_service import client_inference_response
from reef.service.streaming import stream_record


class RecordingClient(SGLangGenerateClient):
    """The SGLang client with a fake tool parser per request; remembers what the handler asked for."""

    def __init__(self, parser_type, inputs=None) -> None:
        self._parser_type = parser_type
        self._inputs = inputs

    def tool_parser(self, tools, parser_name, tokenizer):
        if self._inputs is not None:
            self._inputs.update(tools=tools, parser_name=parser_name)
        return self._parser_type()


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
                _artifact(tmp_path),
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
            response = await backend.inference(_artifact(tmp_path), "/v1/chat/completions", payload)
        finally:
            await server.close()

        assert response["choices"][0]["message"] == message

    asyncio.run(run())


@pytest.mark.unit
def test_handler_chat_template_kwargs_must_not_carry_tools() -> None:
    with pytest.raises(ValueError, match="tools is reserved"):
        SGLangInferenceHandler("http://engine", model_path="model", chat_template_kwargs={"tools": []})


def _artifact(tmp_path) -> Artifact:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return Artifact.local(checkpoint)


def _live_artifact() -> Artifact:
    return Artifact(
        ArtifactRef(content_id="live-test", release_id="live:engine:6", parent_release_id="checkpoint"),
        None,
    )


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
def test_chat_facade_forwards_only_an_explicit_lora_path() -> None:
    backend = SGLangInferenceHandler(
        "http://unused",
        model_path="model",
        tokenizer=FakeTokenizer(),
    )

    without_adapter = backend.client.payload({}, [10, 11], {}, capture_topk=0, stream=False)
    with_adapter = backend.client.payload({"lora_path": "reef_lora"}, [10, 11], {}, capture_topk=0, stream=False)

    assert "lora_path" not in without_adapter
    assert with_adapter["lora_path"] == "reef_lora"


@pytest.mark.unit
def test_chat_facade_records_engine_native_ids_without_retokenizing(tmp_path) -> None:
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
                _artifact(tmp_path),
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
        finally:
            await server.close()

        assert tokenizer.rendered_messages == ([{"role": "user", "content": "hello"}], {})
        assert response["choices"][0]["message"]["content"] == "decoded answer"
        assert "output_token_logprobs" not in response["choices"][0]["meta_info"]
        assert [item["token"] for item in response["choices"][0]["logprobs"]["content"]] == ["<20>", "<21>"]
        assert response["training"] == {
            "tokens": [10, 11, 20, 21],
            "loss_mask": [1, 1],
            "rollout_log_probs": [-0.25, -0.5],
            "prompt_length": 2,
            "response_length": 2,
            "runtime_load_id": "wv-7",
            "runtime_load_spans": [
                {"start": 0, "end": 2, "runtime_load_id": "wv-7"},
            ],
        }

    asyncio.run(run())


@pytest.mark.unit
def test_chat_facade_records_exact_versions_across_an_in_place_update(tmp_path) -> None:
    async def run() -> None:
        async def generate(request):
            native = await request.json()
            assert native["stream"] is False
            assert "stream_interval" not in native["sampling_params"]
            return web.json_response(
                {
                    "text": "answer",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 3,
                        "output_token_logprobs": [[-0.1, 20], [-0.2, 21], [-0.3, 22]],
                        # Tokenizer metadata can move after the final decode;
                        # the scheduler-owned list remains exact.
                        "weight_version": "engine:8",
                        "_reef_token_runtime_load_ids": ["engine:6", "engine:7", "engine:7"],
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
                tokenizer=FakeTokenizer(),
            )
            response = await backend.inference(
                _artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 3,
                },
            )
        finally:
            await server.close()

        assert response["training"]["rollout_log_probs"] == [-0.1, -0.2, -0.3]
        assert response["training"]["runtime_load_id"] is None
        assert response["training"]["runtime_load_spans"] == [
            {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
            {"start": 1, "end": 3, "runtime_load_id": "engine:7"},
        ]
        assert response["choices"][0]["meta_info"]["runtime_load_id"] == "engine:7"

    asyncio.run(run())


@pytest.mark.unit
def test_nonstream_fallback_uses_the_scheduler_stamped_final_version() -> None:
    async def run() -> None:
        async def generate(request):
            return web.json_response(
                {
                    "text": "answer",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, 20]],
                        # The tokenizer may move first while draining output.
                        "weight_version": "engine:7",
                        "_reef_token_runtime_load_ids": ["engine:6"],
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
                tokenizer=FakeTokenizer(),
            )
            response = await backend.inference(
                _live_artifact(),
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 1,
                },
            )
        finally:
            await server.close()

        assert response["choices"][0]["meta_info"]["runtime_load_id"] == "engine:6"
        assert response["training"]["runtime_load_spans"] == [{"start": 0, "end": 1, "runtime_load_id": "engine:6"}]

    asyncio.run(run())


@pytest.mark.unit
def test_live_chat_rejects_tokenizer_only_runtime_load_ids() -> None:
    async def run() -> None:
        async def generate(request):
            return web.json_response(
                {
                    "text": "answer",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 1,
                        "output_token_logprobs": [[-0.1, 20]],
                        "weight_version": "engine:7",
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
                tokenizer=FakeTokenizer(),
            )
            with pytest.raises(ValueError, match="scheduler-stamped token runtime load IDs"):
                await backend.inference(
                    _live_artifact(),
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_completion_tokens": 1,
                    },
                )
        finally:
            await server.close()

    asyncio.run(run())


@pytest.mark.unit
def test_chat_facade_parses_tool_calls_without_changing_training_tokens(tmp_path) -> None:
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
                _artifact(tmp_path),
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
        assert response["training"]["tokens"] == [10, 11, 20, 21]
        assert response["training"]["rollout_log_probs"] == [-0.1, -0.2]

    asyncio.run(run())


@pytest.mark.unit
def test_anthropic_facade_normalizes_messages_and_keeps_private_training_transcript(tmp_path) -> None:
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
                _artifact(tmp_path),
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
                },
            )
        finally:
            await server.close()

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
        assert response["training"]["tokens"] == [10, 11, 20, 21]
        assert response["training"]["request_messages"] == canonical_messages
        assert response["training"]["response_message"] == {
            "role": "assistant",
            "content": "Finished.",
            "reasoning_content": "working",
        }
        assert response["training"]["finish_reason"] == "stop"

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
            _artifact(tmp_path),
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
    assert response["input_tokens"] == 2
    assert response["training"]["runtime_load_id"].startswith("local:")
    assert client_inference_response(response) == {"input_tokens": 2}


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
                _artifact(tmp_path),
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
        assert response["training"]["request_tools"] == openai_tools

    asyncio.run(run())


@pytest.mark.unit
def test_streaming_chat_keeps_exact_training_response_for_recording(tmp_path) -> None:
    async def run() -> None:
        async def generate(request):
            native = await request.json()
            assert native["stream"] is True
            assert native["sampling_params"]["stream_interval"] == 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            event = {
                "text": "answer",
                "output_ids": [20],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "completion_tokens": 1,
                    "output_token_logprobs": [[-0.5, 20]],
                    "weight_version": "engine:6",
                    "_reef_token_runtime_load_ids": ["engine:6"],
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
                _artifact(tmp_path),
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 1,
                    "logprobs": True,
                    "stream": True,
                },
            )
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        assert b'"content": "answer"' in body
        assert b'"logprobs": {"content":' in body
        assert body.endswith(b"data: [DONE]\n\n")
        recorded = stream_record(stream, body, complete=True)
        assert recorded["training"]["tokens"] == [10, 11, 20]
        assert recorded["training"]["rollout_log_probs"] == [-0.5]
        assert recorded["stream_delivery"]["complete"] is True

    asyncio.run(run())


@pytest.mark.unit
def test_streaming_anthropic_messages_emits_native_sse_and_records_training(tmp_path) -> None:
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
                    "_reef_token_runtime_load_ids": ["engine:6"],
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
                    "_reef_token_runtime_load_ids": ["engine:7"],
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
                _live_artifact(),
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
        assert recorded["training"]["tokens"] == [10, 11, 20, 21]
        assert recorded["training"]["runtime_load_spans"] == [
            {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
            {"start": 1, "end": 2, "runtime_load_id": "engine:7"},
        ]
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
                    "_reef_token_runtime_load_ids": ["engine:6"],
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
                _live_artifact(),
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
def test_streaming_anthropic_reassembles_incremental_sglang_thinking_chunks() -> None:
    async def run() -> None:
        async def generate(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            events = [
                {
                    "text": "work</thi",
                    "output_ids": [20],
                    "meta_info": {
                        "finish_reason": None,
                        "completion_tokens": 1,
                        "output_token_logprobs": [[-0.5, 20]],
                        "_reef_token_runtime_load_ids": ["engine:6"],
                    },
                },
                {
                    "text": "nk>\nanswer",
                    "output_ids": [21],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 2,
                        "output_token_logprobs": [[-0.25, 21]],
                        "_reef_token_runtime_load_ids": ["engine:7"],
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
                force_reasoning=True,
            )
            stream = await backend.inference_stream(
                _live_artifact(),
                "/v1/messages",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 2,
                    "stream": True,
                },
            )
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        text = body.decode()
        assert '"type": "thinking_delta", "thinking": "work"' in text
        assert '"type": "text_delta", "text": "answer"' in text
        assert "</think>" not in text
        recorded = stream_record(stream, body, complete=True)
        assert recorded["training"]["tokens"] == [10, 11, 20, 21]
        assert recorded["training"]["response_message"] == {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "work",
        }

    asyncio.run(run())


@pytest.mark.unit
def test_streaming_anthropic_emits_incremental_tool_use_without_raw_markers(tmp_path) -> None:
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
                    parameters='{"path":"out.txt"}',
                    tool_index=0,
                )
            ]

        @staticmethod
        def parse_non_stream(text):
            del text
            return "", [SimpleNamespace(name="write_file", parameters={"path": "out.txt"})]

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
                _artifact(tmp_path),
                "/v1/messages",
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "write it"}],
                    "tools": [
                        {
                            "name": "write_file",
                            "input_schema": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        }
                    ],
                    "max_tokens": 1,
                    "stream": True,
                },
            )
            body = b"".join([chunk async for chunk in stream.chunks])
        finally:
            await server.close()

        text = body.decode()
        assert '"type": "tool_use"' in text
        assert '"type": "input_json_delta", "partial_json": "{\\"path\\":\\"out.txt\\"}"' in text
        assert "<tool_call>" not in text
        recorded = stream_record(stream, body, complete=True)
        assert recorded["stop_reason"] == "tool_use"
        assert recorded["content"] == [
            {
                "type": "tool_use",
                "id": recorded["content"][0]["id"],
                "name": "write_file",
                "input": {"path": "out.txt"},
            }
        ]
        assert recorded["training"]["tokens"] == [10, 11, 20]

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
            "require image_url.url",
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
                _artifact(tmp_path),
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
    the monologue instead of the reply. A subclass can preserve raw tags;
    training tokens stay untouched in either mode.
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
                _artifact(tmp_path),
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
        # The exact sampled ids still train, reasoning included.
        assert response["training"]["tokens"] == [10, 11, 20, 21, 22]
        assert response["training"]["response_length"] == 3

    asyncio.run(run())


# -- Image capture ------------------------------------------------------------

VISION_START, VISION_END, VISION_PAD, IMAGE_PAD, VIDEO_PAD = 248053, 248054, 248055, 248056, 248057
VISION_IDS = {
    "<|vision_start|>": VISION_START,
    "<|vision_end|>": VISION_END,
    "<|vision_pad|>": VISION_PAD,
    "<|image_pad|>": IMAGE_PAD,
    "<|video_pad|>": VIDEO_PAD,
}
#: A 640x480 image fits the default pixel limit: grid [1, 30, 40], 30 * 40 / 4 = 300 image tokens.
IMAGE_SIZE, IMAGE_GRID, IMAGE_TOKENS = (640, 480), [1, 30, 40], 300


class ImageTokenizer(FakeTokenizer):
    """Renders a text message as 10, a text part as 11 (or the vision id it names) and an image part as one placeholder."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return VISION_IDS.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        if not tokenize:
            return "<user></user><assistant>"
        self.rendered_messages = (messages, kwargs)
        ids = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                ids.append(VISION_IDS.get(content, 10))
                continue
            for part in content:
                if part["type"] == "image":
                    ids.extend([VISION_START, IMAGE_PAD, VISION_END])
                else:
                    ids.append(VISION_IDS.get(part["text"], 11))
        return [*ids, 12]


def image_model(tmp_path, model_type="qwen3_5", **processor_overrides):
    """A model directory with Qwen3.5's image processor settings and no weights."""
    model = tmp_path / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    processor = {"patch_size": 16, "merge_size": 2, "size": {"shortest_edge": 65536, "longest_edge": 16777216}}
    (model / "preprocessor_config.json").write_text(json.dumps({**processor, **processor_overrides}))
    return str(model)


def broken_png():
    """A PNG whose second IDAT chunk has a bad type; Pillow raises SyntaxError while it decodes."""
    Image = require_image_modules()
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 30, 30)).save(buffer, format="PNG")
    png = buffer.getvalue()
    start = png.index(b"IDAT") - 4
    length = int.from_bytes(png[start : start + 4], "big")
    data = png[start + 8 : start + 8 + length]
    half = len(data) // 2

    def chunk(kind, body):
        return len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")

    return png[:start] + chunk(b"IDAT", data[:half]) + chunk(b"I$AT", data[half:]) + png[start + 12 + length :]


def png_with_late_chunk(kind, body):
    """A valid PNG with one more chunk, with a correct CRC, after IDAT; Pillow reads it in ``load_end``."""
    png = base64.b64decode(png_base64((64, 48)))
    end = png.index(b"IEND") - 4
    chunk = len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")
    return base64.b64encode(png[:end] + chunk + png[end:]).decode()


def require_image_modules():
    """Skip without Pillow or transformers. With transformers, a missing ``smart_resize`` module fails."""
    from reef.core.images import SMART_RESIZE_MODULE

    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("transformers")
    importlib.import_module(SMART_RESIZE_MODULE)
    return Image


def png_base64(size=IMAGE_SIZE, mode="RGB", image_format="PNG"):
    """A red image; a GIF has a second, blue frame."""
    Image = require_image_modules()
    buffer = io.BytesIO()
    red = {"RGB": (200, 30, 30), "RGBA": (200, 30, 30, 128), "CMYK": (0, 220, 220, 30)}[mode]
    image = Image.new(mode, size, color=red)
    if image_format == "GIF":
        image.save(buffer, format="GIF", save_all=True, append_images=[Image.new(mode, size, color=(30, 30, 200))])
    else:
        image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode()


def openai_image(data):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def anthropic_image(data):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


#: The rendered prompt of one user message "look" plus one image, before and after the engine expands the image.
RENDERED = [11, VISION_START, IMAGE_PAD, VISION_END, 12]
EXPANDED = [11, VISION_START, *[IMAGE_PAD] * IMAGE_TOKENS, VISION_END, 12]


def image_requests(data):
    """One image request per supported place: path, body and the rendered ids before expansion."""
    return {
        "openai-user": (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, openai_image(data)]}]},
            RENDERED,
        ),
        "openai-tool": (
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "tool", "tool_call_id": "c1", "content": [openai_image(data)]},
                ]
            },
            [10, VISION_START, IMAGE_PAD, VISION_END, 12],
        ),
        "anthropic-user": (
            "/v1/messages",
            {
                "model": "m",
                "max_tokens": 2,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, anthropic_image(data)]}],
            },
            RENDERED,
        ),
        "anthropic-tool-result": (
            "/v1/messages",
            {
                "model": "m",
                "max_tokens": 2,
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "shot", "input": {}}]},
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": [anthropic_image(data)]}],
                    },
                ],
            },
            [10, 10, VISION_START, IMAGE_PAD, VISION_END, 12],
        ),
    }


def expanded(rendered):
    index = rendered.index(IMAGE_PAD)
    return [*rendered[:index], *[IMAGE_PAD] * IMAGE_TOKENS, *rendered[index + 1 :]]


async def generate_once(tmp_path, path, body, engine_response, *, stream=False, **handler_config):
    """Serve one image request against a fake SGLang; return the native request and the handler's response."""
    seen = {}

    async def generate(request):
        seen["native"] = await request.json()
        if not stream:
            return web.json_response(engine_response)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in engine_response:
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
            model_path=image_model(tmp_path),
            tokenizer=ImageTokenizer(),
            **handler_config,
        )
        if not stream:
            return await backend.inference(_artifact(tmp_path), path, body), seen
        handler_stream = await backend.inference_stream(_artifact(tmp_path), path, {**body, "stream": True})
        frames = b"".join([chunk async for chunk in handler_stream.chunks])
        return (frames, stream_record(handler_stream, frames, complete=True)), seen
    finally:
        await server.close()


def engine_response(prompt_ids, output_ids=(20, 21)):
    response = {
        "text": "a red square",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[-0.5, token] for token in output_ids],
            "weight_version": "wv-1",
        },
    }
    if prompt_ids is not None:
        response["prompt_token_ids"] = list(prompt_ids)
    return response


@pytest.mark.unit
@pytest.mark.parametrize("place", ["openai-user", "openai-tool", "anthropic-user", "anthropic-tool-result"])
def test_image_request_records_the_engine_prompt_ids(tmp_path, place) -> None:
    data = png_base64()
    path, body, rendered = image_requests(data)[place]

    response, seen = asyncio.run(generate_once(tmp_path, path, body, engine_response(expanded(rendered))))

    native = seen["native"]
    assert native["input_ids"] == rendered
    assert native["return_prompt_token_ids"] is True
    [image_url] = native["image_data"]
    png = base64.b64decode(image_url.removeprefix("data:image/png;base64,"))
    # The client's PNG already fits its grid, so Reef sends it re-encoded at the same size.
    assert pytest.importorskip("PIL.Image").open(io.BytesIO(png)).size == IMAGE_SIZE
    # Reef does not ban the vision special tokens.
    assert "logit_bias" not in native["sampling_params"]
    training = response["training"]
    assert training["tokens"] == [*expanded(rendered), 20, 21]
    assert training["prompt_length"] == len(rendered) - 1 + IMAGE_TOKENS
    assert training["images"] == [
        {
            "image_hash": hashlib.sha256(png).hexdigest(),
            "token_offset": rendered.index(IMAGE_PAD),
            "token_length": IMAGE_TOKENS,
            "grid": IMAGE_GRID,
            "orig_size": list(IMAGE_SIZE),
        }
    ]
    if path == "/v1/messages":
        assert response["usage"]["input_tokens"] == training["prompt_length"]
        # The record keeps the client's image block unchanged, not the prepared PNG.
        contents = [message["content"] for message in training["request_messages"]]
        assert any(isinstance(content, list) and anthropic_image(data) in content for content in contents)
    else:
        assert response["usage"]["prompt_tokens"] == training["prompt_length"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prompt_ids", "output_ids", "reason"),
    [
        (None, (20, 21), "no prompt_token_ids"),
        (RENDERED, (20, 21), "differ from the 304 ids Reef predicted"),
        *[(EXPANDED, (20, token), "vision special token") for token in VISION_IDS.values()],
    ],
)
def test_image_request_is_served_but_unrecorded_when_the_engine_ids_cannot_train(
    tmp_path, caplog, prompt_ids, output_ids, reason
) -> None:
    path, body, _ = image_requests(png_base64())["openai-user"]

    with caplog.at_level(logging.WARNING, logger="reef.inference.chat"):
        response, _ = asyncio.run(generate_once(tmp_path, path, body, engine_response(prompt_ids, output_ids)))

    assert response["choices"][0]["message"]["content"] == "a red square"
    assert reason in response["training"].pop("unrecorded")
    assert any("keeps no record" in record.getMessage() and reason in record.getMessage() for record in caplog.records)
    assert response["training"] == {
        "runtime_load_id": "wv-1",
        "runtime_load_spans": [{"start": 0, "end": len(output_ids), "runtime_load_id": "wv-1"}],
    }


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("turn", ["first", "later-user", "later-tool", "history"])
def test_image_request_that_names_a_rollout_is_refused_before_generation(tmp_path, turn, stream) -> None:
    # The rollout's previous call would hold the expanded image span, and the next call would send it
    # as text ids without the image, so a request that names a rollout carries no image.
    image = openai_image(png_base64())
    first = {"user": "rollout-1", "messages": [{"role": "user", "content": "go"}]}
    later = {**first, "messages": [*first["messages"], {"role": "assistant", "content": "ok"}]}
    requests = {
        "first": {**first, "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, image]}]},
        "later-user": {**later, "messages": [*later["messages"], {"role": "user", "content": [image]}]},
        "later-tool": {
            **later,
            "messages": [*later["messages"], {"role": "tool", "tool_call_id": "c1", "content": [image]}],
        },
        # An image in an earlier message, which the client sent before it named the rollout.
        "history": {
            **first,
            "messages": [
                {"role": "user", "content": [image]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "go"},
            ],
        },
    }
    prompts = []

    async def run():
        async def generate(request):
            prompts.append((await request.json())["input_ids"])
            return web.json_response(engine_response(None))

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            handler = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"),
                model_path=image_model(tmp_path),
                tokenizer=ImageTokenizer(),
                rollout_continuation=True,
            )
            artifact = _artifact(tmp_path)
            if turn.startswith("later"):
                await handler.inference(artifact, "/v1/chat/completions", first)
            kept = handler._rollout_prefixes.get("rollout-1")
            with pytest.raises(ValueError, match="rollout continuation does not support image content"):
                if stream:
                    await handler.inference_stream(
                        artifact, "/v1/chat/completions", {**requests[turn], "stream": True}
                    )
                else:
                    await handler.inference(artifact, "/v1/chat/completions", requests[turn])
            # The refused request leaves the rollout as it was: absent, or the first call's tokens.
            assert handler._rollout_prefixes.get("rollout-1") == kept
            return kept
        finally:
            await server.close()

    kept = asyncio.run(run())
    assert len(prompts) == (1 if turn.startswith("later") else 0)
    if turn.startswith("later"):
        assert kept.tokens == (10, 12, 20, 21)


@pytest.mark.unit
def test_image_request_without_a_rollout_is_captured_when_continuation_is_on(tmp_path) -> None:
    path, body, rendered = image_requests(png_base64())["openai-user"]

    response, seen = asyncio.run(
        generate_once(tmp_path, path, body, engine_response(expanded(rendered)), rollout_continuation=True)
    )

    assert seen["native"]["input_ids"] == rendered
    assert len(seen["native"]["image_data"]) == 1
    assert response["training"]["tokens"] == [*expanded(rendered), 20, 21]
    assert len(response["training"]["images"]) == 1


@pytest.mark.unit
def test_an_unrecorded_call_keeps_the_rollout_previous_call(tmp_path) -> None:
    # The refusal above keeps image calls out of rollouts; this checks the handler does not
    # read tokens that an unrecorded result does not have.
    handler = SGLangInferenceHandler(
        "http://engine.invalid",
        model_path=image_model(tmp_path),
        tokenizer=ImageTokenizer(),
        rollout_continuation=True,
    )
    previous = {"user": "rollout-1", "messages": [{"role": "user", "content": "go"}]}
    handler._remember_rollout("rollout-1", [10, 12, 20], previous)
    request = {
        **previous,
        "messages": [
            *previous["messages"],
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [openai_image(png_base64())]},
        ],
    }
    image = PreparedImage(b"png", "hash", (1, 30, 40), IMAGE_SIZE, IMAGE_TOKENS)
    call = ChatCall(
        anthropic=False,
        payload=request,
        request=request,
        rendered=RenderedPrompt(RENDERED, (image,), EXPANDED),
        sampling_params={},
        tool_parser=None,
    )
    captured = CapturedGeneration([20, 21], [-0.5, -0.5], ["wv-1", "wv-1"], "stop", "a red square", {})

    response = handler._response_from_capture(call, captured, None)

    assert "no prompt_token_ids" in response["training"]["unrecorded"]
    assert "tokens" not in response["training"]
    assert handler._rollout_prefixes["rollout-1"].tokens == (10, 12, 20)
    assert handler._rollout_prefixes["rollout-1"].request == previous


@pytest.mark.unit
def test_image_request_passes_the_client_logit_bias_unchanged(tmp_path) -> None:
    path, body, _ = image_requests(png_base64())["openai-user"]
    bias = {"0248056": 5, "7": -1.5}

    _, seen = asyncio.run(generate_once(tmp_path, path, {**body, "logit_bias": bias}, engine_response(EXPANDED)))

    assert seen["native"]["sampling_params"]["logit_bias"] == bias


@pytest.mark.unit
def test_text_only_request_records_a_sampled_vision_special_token(tmp_path) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    response, seen = asyncio.run(
        generate_once(tmp_path, "/v1/chat/completions", body, engine_response(None, (20, IMAGE_PAD)))
    )

    assert "unrecorded" not in response["training"]
    assert response["training"]["tokens"] == [*seen["native"]["input_ids"], 20, IMAGE_PAD]


def stream_events(prompt_ids_per_event):
    return [
        {
            "text": text,
            "output_ids": [token],
            "prompt_token_ids": prompt_ids,
            "meta_info": {
                "finish_reason": {"type": "stop"} if index else None,
                "completion_tokens": index + 1,
                "output_token_logprobs": [[-0.5, token]],
                "weight_version": "wv-1",
            },
        }
        for index, (text, token, prompt_ids) in enumerate(
            zip(("a red", " square"), (20, 21), prompt_ids_per_event, strict=True)
        )
    ]


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("agree", [True, False])
def test_streamed_image_request_records_the_engine_prompt_ids_when_every_event_agrees(tmp_path, path, agree) -> None:
    place = "openai-user" if path == "/v1/chat/completions" else "anthropic-user"
    _, body, _ = image_requests(png_base64())[place]
    events = stream_events([EXPANDED, EXPANDED if agree else RENDERED])

    (frames, recorded), seen = asyncio.run(generate_once(tmp_path, path, body, events, stream=True))

    assert seen["native"]["stream"] is True
    assert b"square" in frames
    if path == "/v1/messages":
        # message_start goes out before the engine answers, so it reports the predicted length.
        assert f'"input_tokens": {len(EXPANDED)}'.encode() in frames
    if agree:
        assert recorded["training"]["tokens"] == [*EXPANDED, 20, 21]
        assert "unrecorded" not in recorded["training"]
    else:
        assert "disagree" in recorded["training"]["unrecorded"]
        assert "tokens" not in recorded["training"]


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_streamed_image_request_is_unrecorded_when_the_output_holds_a_vision_special_token(tmp_path, path) -> None:
    place = "openai-user" if path == "/v1/chat/completions" else "anthropic-user"
    _, body, _ = image_requests(png_base64())[place]
    events = stream_events([EXPANDED, EXPANDED])
    events[1]["output_ids"] = [VISION_IDS["<|vision_end|>"]]
    events[1]["meta_info"]["output_token_logprobs"] = [[-0.5, VISION_IDS["<|vision_end|>"]]]

    (frames, recorded), _ = asyncio.run(generate_once(tmp_path, path, body, events, stream=True))

    assert b"square" in frames
    assert "vision special token" in recorded["training"]["unrecorded"]
    assert "tokens" not in recorded["training"]


@pytest.mark.unit
def test_anthropic_count_tokens_counts_the_expanded_image(tmp_path) -> None:
    _, body, _ = image_requests(png_base64())["anthropic-user"]
    backend = SGLangInferenceHandler("http://unused", model_path=image_model(tmp_path), tokenizer=ImageTokenizer())

    counted = asyncio.run(backend.inference(_artifact(tmp_path), "/v1/messages/count_tokens", body))

    assert counted["input_tokens"] == len(EXPANDED)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("messages", "extra", "message"),
    [
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}],
            {},
            "only base64 data URLs",
        ),
        ([{"role": "system", "content": ["IMAGE"]}, {"role": "user", "content": "hi"}], {}, "user and tool messages"),
        ([{"role": "user", "content": ["IMAGE", "IMAGE"]}], {}, "only 1 image per request"),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%"}}]}],
            {},
            "not valid base64",
        ),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
            {},
            "cannot be decoded",
        ),
        ([{"role": "user", "content": [{"type": "text", "text": "<|image_pad|>"}, "IMAGE"]}], {}, "vision special"),
        ([{"role": "user", "content": [{"type": "text", "text": "text only"}]}], {}, "text-only message content"),
        ([{"role": "user", "content": ["WIDE"]}], {}, "aspect ratio must be smaller than 200"),
        ([{"role": "user", "content": ["BROKEN"]}], {}, "cannot be decoded"),
        ([{"role": "user", "content": ["BMP"]}], {}, "cannot be decoded as PNG, JPEG, GIF, WEBP"),
        ([{"role": "user", "content": ["TRUNCATED"]}], {}, "cannot be decoded"),
        ([{"role": "user", "content": ["LATE_GAMA"]}], {}, "cannot be decoded"),
        ([{"role": "user", "content": ["LATE_CHRM"]}], {}, "cannot be decoded"),
        ([{"role": "user", "content": ["LATE_TRNS"]}], {}, "cannot be decoded"),
        ([{"role": "user", "content": ["LATE_ICCP"]}], {}, "cannot be decoded"),
        # A list without an image part keeps the text-only message.
        ([{"role": "user", "content": [{"type": "input_audio", "data": "x"}]}], {}, "text-only message content"),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}],
            {},
            "cannot be decoded",
        ),
        (
            [{"role": "user", "content": [{"type": "input_audio", "data": "x"}, "IMAGE"]}],
            {},
            "accepts only text parts",
        ),
    ],
)
def test_image_request_that_cannot_be_captured_exactly_is_refused(tmp_path, messages, extra, message) -> None:
    png = png_base64()
    images = {
        "IMAGE": openai_image(png),
        "WIDE": openai_image(png_base64((2010, 10))),
        "BROKEN": openai_image(base64.b64encode(broken_png()).decode()),
        "BMP": openai_image(png_base64(image_format="BMP")),
        "TRUNCATED": openai_image(base64.b64encode(base64.b64decode(png)[:60]).decode()),
        # Short chunks after IDAT: Pillow raises struct.error or IndexError while it converts the image.
        "LATE_GAMA": openai_image(png_with_late_chunk(b"gAMA", b"")),
        "LATE_CHRM": openai_image(png_with_late_chunk(b"cHRM", b"\x00")),
        "LATE_TRNS": openai_image(png_with_late_chunk(b"tRNS", b"\x00\x00\x00")),
        "LATE_ICCP": openai_image(png_with_late_chunk(b"iCCP", b"")),
    }
    messages = [
        (
            {
                **entry,
                "content": [images.get(part, part) if isinstance(part, str) else part for part in entry["content"]],
            }
            if isinstance(entry["content"], list)
            else entry
        )
        for entry in messages
    ]
    backend = SGLangInferenceHandler("http://unused", model_path=image_model(tmp_path), tokenizer=ImageTokenizer())

    with pytest.raises(ValueError, match=message):
        asyncio.run(backend.inference(_artifact(tmp_path), "/v1/chat/completions", {"messages": messages, **extra}))


@pytest.mark.unit
def test_image_request_is_refused_for_a_model_without_image_support(tmp_path) -> None:
    backend = SGLangInferenceHandler(
        "http://unused", model_path=image_model(tmp_path, model_type="qwen3"), tokenizer=ImageTokenizer()
    )
    body = image_requests(png_base64())["openai-user"][1]

    with pytest.raises(ValueError, match=r"supports Qwen3\.5 only"):
        asyncio.run(backend.inference(_artifact(tmp_path), "/v1/chat/completions", body))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("size", "mode", "image_format"),
    [
        ((1920, 1080), "RGB", "PNG"),
        ((640, 480), "RGBA", "PNG"),
        ((640, 480), "CMYK", "JPEG"),
        ((640, 480), "RGB", "GIF"),
    ],
)
def test_prepare_image_resizes_within_the_pixel_limit_and_encodes_rgb_png(tmp_path, size, mode, image_format) -> None:
    Image = require_image_modules()
    from reef.core.images import prepare_image
    from reef.inference.images import image_config_from_model

    config = image_config_from_model(image_model(tmp_path), ImageTokenizer(), max_pixels=1024 * 1024)
    raw = base64.b64decode(png_base64(size, mode, image_format))

    prepared = prepare_image(raw, config)

    with Image.open(io.BytesIO(prepared.png)) as png:
        assert png.format == "PNG"
        assert png.mode == "RGB"
        assert png.size == (prepared.grid[2] * 16, prepared.grid[1] * 16)
        # Every format keeps the red of the first frame.
        red, green, blue = png.getpixel((0, 0))
        assert red > 150 and green < 80 and blue < 80
    assert prepared.orig_size == size
    assert prepared.grid[1] * prepared.grid[2] * 256 <= 1024 * 1024
    assert prepared.token_length == prepared.grid[1] * prepared.grid[2] // 4
    assert prepare_image(raw, config).image_hash == prepared.image_hash == hashlib.sha256(prepared.png).hexdigest()


@pytest.mark.unit
@pytest.mark.skipif(not os.environ.get("REEF_TEST_QWEN3_5_PATH"), reason="needs Qwen3.5 tokenizer and processor files")
def test_expanded_image_prompt_matches_the_hf_processor() -> None:
    """Contract test (#749): Reef's rendering, expanded, equals the HF processor's input ids and grid."""
    Image = pytest.importorskip("PIL.Image")
    transformers = pytest.importorskip("transformers")
    from reef.core.images import expanded_prompt_ids, prepare_image
    from reef.inference.images import image_config_from_model

    model_path = os.environ["REEF_TEST_QWEN3_5_PATH"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    processor = transformers.AutoProcessor.from_pretrained(model_path)
    backend = SGLangInferenceHandler("http://unused", model_path=model_path, tokenizer=tokenizer)
    config = image_config_from_model(model_path, tokenizer, max_pixels=1024 * 1024)
    for size in ((640, 480), (1920, 1080), (333, 517)):
        data = png_base64(size)
        for place, (path, body, _) in image_requests(data).items():
            request = backend._anthropic_request(body) if path == "/v1/messages" else body
            rendered = asyncio.run(backend.prepare_prompt(request))
            prompt_ids, images, expected = rendered.prompt_ids, rendered.images, rendered.expected_prompt_ids
            template_messages = backend._template_messages(request["messages"])
            text = tokenizer.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)
            pixels = [Image.open(io.BytesIO(image.png)) for image in images]
            processed = processor(text=[text], images=pixels, return_tensors="np")
            unresized = processor(text=[text], images=pixels, return_tensors="np", do_resize=False)
            assert expected == processed["input_ids"][0].tolist(), (size, place)
            assert [list(image.grid) for image in images] == processed["image_grid_thw"].tolist(), (size, place)
            # The processor's own resize is a no-op on the prepared PNG.
            assert (processed["pixel_values"] == unresized["pixel_values"]).all(), (size, place)
            assert expanded_prompt_ids(prompt_ids, images, config) == expected
            assert prepare_image(base64.b64decode(data), config) == images[0]


@pytest.mark.unit
def test_image_config_reads_top_level_pixel_limits_and_refuses_a_limit_below_the_minimum(tmp_path) -> None:
    require_image_modules()
    from reef.inference.images import image_config_from_model

    model = image_model(tmp_path, min_pixels=4096, max_pixels=262144)
    config = image_config_from_model(model, ImageTokenizer(), max_pixels=1024 * 1024)
    assert (config.min_pixels, config.max_pixels) == (4096, 262144)

    # Near the minimum, resizing can round below it, so the limit must be at least twice the minimum.
    for limit in (40000, 70000, 131071):
        with pytest.raises(ValueError, match="at least twice the model processor's minimum of 65536"):
            image_config_from_model(image_model(tmp_path / "default"), ImageTokenizer(), max_pixels=limit)
    assert image_config_from_model(image_model(tmp_path / "default"), ImageTokenizer(), max_pixels=131072)
    # The check uses the lower of the two limits, and names the processor's when it is the lower one.
    with pytest.raises(ValueError, match="the model processor's max_pixels 100000 must be at least twice"):
        image_config_from_model(image_model(tmp_path / "low", max_pixels=100000), ImageTokenizer(), max_pixels=1 << 20)

    # As in the HF processor, a null top-level limit falls back to ``size``, and an integral float is accepted.
    model = image_model(tmp_path / "null", min_pixels=None, max_pixels=262144.0)
    config = image_config_from_model(model, ImageTokenizer(), max_pixels=1024 * 1024)
    assert (config.min_pixels, config.max_pixels) == (65536, 262144)


@pytest.mark.unit
def test_image_config_reads_the_image_processor_of_a_saved_processor(tmp_path) -> None:
    """``processor.save_pretrained`` writes only processor_config.json, with the image processor nested."""
    require_image_modules()
    from reef.inference.images import image_config_from_model

    model = Path(image_model(tmp_path))
    settings = json.loads((model / "preprocessor_config.json").read_text())
    (model / "preprocessor_config.json").unlink()
    (model / "processor_config.json").write_text(json.dumps({"image_processor": {**settings, "max_pixels": 262144}}))

    config = image_config_from_model(str(model), ImageTokenizer(), max_pixels=1024 * 1024)

    assert (config.patch_size, config.min_pixels, config.max_pixels) == (16, 65536, 262144)
    (model / "processor_config.json").unlink()
    with pytest.raises(ValueError, match=r"neither preprocessor_config\.json nor processor_config\.json"):
        image_config_from_model(str(model), ImageTokenizer(), max_pixels=1024 * 1024)


@pytest.mark.unit
def test_image_config_downloads_the_settings_of_a_hugging_face_model_id(tmp_path, monkeypatch) -> None:
    """With ``inference.remote`` and a hosted training backend, ``model_path`` stays a Hugging Face model ID."""
    require_image_modules()
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    from reef.inference.images import IMAGE_SETTINGS_FILES, image_config_from_model

    downloaded = image_model(tmp_path / "cache")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return downloaded

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    config = image_config_from_model("Qwen/Qwen3.5-0.8B", ImageTokenizer(), max_pixels=1024 * 1024)

    assert (config.patch_size, config.merge_size, config.min_pixels) == (16, 2, 65536)
    assert calls == [{"repo_id": "Qwen/Qwen3.5-0.8B", "allow_patterns": list(IMAGE_SETTINGS_FILES)}]
    # A local directory is read as it is, with no download.
    image_config_from_model(image_model(tmp_path / "local"), ImageTokenizer(), max_pixels=1024 * 1024)
    assert len(calls) == 1

    def offline_snapshot_download(**kwargs):
        raise LocalEntryNotFoundError("no cached snapshot, and the hub is not reachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline_snapshot_download)
    with pytest.raises(ValueError, match="is not a local directory, and Reef cannot download its image settings"):
        image_config_from_model("Qwen/Qwen3.5-0.8B", ImageTokenizer(), max_pixels=1024 * 1024)


@pytest.mark.unit
def test_image_larger_than_the_decode_limit_is_refused(tmp_path, monkeypatch) -> None:
    require_image_modules()
    from reef.core import images
    from reef.inference.images import image_config_from_model

    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_PIXELS", 640 * 480 - 1)
    config = image_config_from_model(image_model(tmp_path), ImageTokenizer(), max_pixels=1024 * 1024)

    with pytest.raises(ValueError, match="the image has 640x480 pixels"):
        images.prepare_image(base64.b64decode(png_base64()), config)


@pytest.mark.unit
def test_image_request_without_pillow_is_refused(tmp_path, monkeypatch) -> None:
    body = image_requests(base64.b64encode(b"not decoded").decode())["openai-user"][1]
    backend = SGLangInferenceHandler("http://unused", model_path=image_model(tmp_path), tokenizer=ImageTokenizer())
    monkeypatch.setitem(sys.modules, "PIL", None)

    with pytest.raises(ValueError, match="needs Pillow"):
        asyncio.run(backend.inference(_artifact(tmp_path), "/v1/chat/completions", body))


@pytest.mark.unit
def test_image_request_without_transformers_is_refused(tmp_path, monkeypatch) -> None:
    pytest.importorskip("PIL.Image")
    from reef.core.images import SMART_RESIZE_MODULE

    body = image_requests(base64.b64encode(b"not decoded").decode())["openai-user"][1]
    backend = SGLangInferenceHandler("http://unused", model_path=image_model(tmp_path), tokenizer=ImageTokenizer())
    monkeypatch.setitem(sys.modules, SMART_RESIZE_MODULE, None)

    with pytest.raises(ValueError, match=r"needs transformers 5\.5 or later"):
        asyncio.run(backend.inference(_artifact(tmp_path), "/v1/chat/completions", body))


@pytest.mark.unit
def test_image_preparation_runs_off_the_event_loop_thread(tmp_path, monkeypatch) -> None:
    from reef.core import images

    threads = []
    prepare = images.prepare_image

    def recording_prepare(raw, config):
        threads.append(threading.current_thread())
        return prepare(raw, config)

    monkeypatch.setattr(images, "prepare_image", recording_prepare)
    path, body, rendered = image_requests(png_base64())["anthropic-user"]

    asyncio.run(generate_once(tmp_path, path, body, engine_response(expanded(rendered))))

    assert threads and all(thread is not threading.main_thread() for thread in threads)


@pytest.mark.unit
def test_concurrent_first_requests_load_the_tokenizer_and_render_on_the_event_loop(tmp_path, monkeypatch) -> None:
    """Only image decoding runs in worker threads; the first tokenizer load and every render stay on the loop."""
    loads = []
    renders = []

    class RecordingTokenizer(ImageTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            renders.append(threading.current_thread())
            return super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
            )

    def from_pretrained(path, **options):
        loads.append(threading.current_thread())
        return RecordingTokenizer()

    # Import the real smart_resize first; the fake package below then only replaces AutoTokenizer.
    require_image_modules()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    _, image_body, rendered = image_requests(png_base64())["openai-user"]
    text_body = {"messages": [{"role": "user", "content": "hello"}]}

    async def run():
        async def generate(request):
            native = await request.json()
            return web.json_response(engine_response(expanded(rendered) if "image_data" in native else None))

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangInferenceHandler(str(server.make_url("")).rstrip("/"), model_path=image_model(tmp_path))
            artifact = _artifact(tmp_path)
            bodies = [image_body] * 6 + [text_body] * 2
            return await asyncio.gather(
                *(backend.inference(artifact, "/v1/chat/completions", body) for body in bodies)
            )
        finally:
            await server.close()

    responses = asyncio.run(run())

    assert [len(response["training"]["tokens"]) for response in responses] == [len(EXPANDED) + 2] * 6 + [4] * 2
    assert loads and all(thread is threading.main_thread() for thread in loads)
    assert renders and all(thread is threading.main_thread() for thread in renders)
