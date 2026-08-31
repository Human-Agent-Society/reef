from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from recipes.openclawrl.turns import main_turn_message, turn_request_messages
from reef.artifact import Artifact, LiveWeightArtifactRef
from reef.service.request_service import client_inference_response
from reef.service.streaming import stream_record
from reef.train.slime_backend.reef_adapters.sglang.chat import SGLangChatTrainingInferenceBackend, _NativeStreamCapture


class FakeTokenizer:
    def __init__(self) -> None:
        self.rendered_messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True and add_generation_prompt is True
        self.rendered_messages = (messages, kwargs)
        return [10, 11]

    def decode(self, token_ids, *, skip_special_tokens=False):
        assert skip_special_tokens is False
        return f"<{token_ids[0]}>"


def _artifact(tmp_path) -> Artifact:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return Artifact.local(checkpoint)


def _live_artifact() -> Artifact:
    return Artifact(
        LiveWeightArtifactRef(
            artifact_id="live-test",
            version="live:engine:6",
            parent_version="checkpoint",
            weight_version="engine:6",
        ),
        None,
    )


@pytest.mark.unit
def test_native_stream_capture_rejects_cumulative_chunks() -> None:
    capture = _NativeStreamCapture()
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
    backend = SGLangChatTrainingInferenceBackend(
        "http://unused",
        model_path="model",
        tokenizer=FakeTokenizer(),
    )

    without_adapter = backend._native_payload({}, [10, 11], {})
    with_adapter = backend._native_payload({"lora_path": "reef_lora"}, [10, 11], {})

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
            backend = SGLangChatTrainingInferenceBackend(
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
            "weight_version": "wv-7",
            "weight_version_spans": [
                {"start": 0, "end": 2, "weight_version": "wv-7"},
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
                        "_reef_token_weight_versions": ["engine:6", "engine:7", "engine:7"],
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangChatTrainingInferenceBackend(
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
        assert response["training"]["weight_version"] is None
        assert response["training"]["weight_version_spans"] == [
            {"start": 0, "end": 1, "weight_version": "engine:6"},
            {"start": 1, "end": 3, "weight_version": "engine:7"},
        ]
        assert response["choices"][0]["meta_info"]["weight_version"] == "engine:7"

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
                        "_reef_token_weight_versions": ["engine:6"],
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            backend = SGLangChatTrainingInferenceBackend(
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

        assert response["choices"][0]["meta_info"]["weight_version"] == "engine:6"
        assert response["training"]["weight_version_spans"] == [{"start": 0, "end": 1, "weight_version": "engine:6"}]

    asyncio.run(run())


@pytest.mark.unit
def test_live_chat_rejects_tokenizer_only_weight_versions() -> None:
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
            backend = SGLangChatTrainingInferenceBackend(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
            )
            with pytest.raises(ValueError, match="scheduler-stamped token weight versions"):
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

        def parser_factory(tools, parser_name):
            parser_inputs.update(tools=tools, parser_name=parser_name)
            return FakeToolParser()

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            tokenizer = FakeTokenizer()
            backend = SGLangChatTrainingInferenceBackend(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=tokenizer,
                tool_call_parser="qwen25",
                tool_parser_factory=parser_factory,
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
            backend = SGLangChatTrainingInferenceBackend(
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
    backend = SGLangChatTrainingInferenceBackend(
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
    assert response["training"]["weight_version"].startswith("local:")
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

        def parser_factory(tools, parser_name):
            parser_inputs.update(tools=tools, parser_name=parser_name)
            return FakeToolParser()

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            tokenizer = FakeTokenizer()
            backend = SGLangChatTrainingInferenceBackend(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=tokenizer,
                tool_call_parser="qwen25",
                tool_parser_factory=parser_factory,
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
                    "_reef_token_weight_versions": ["engine:6"],
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
            backend = SGLangChatTrainingInferenceBackend(
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
                    "_reef_token_weight_versions": ["engine:6"],
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
                    "_reef_token_weight_versions": ["engine:7"],
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
            backend = SGLangChatTrainingInferenceBackend(
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
        assert recorded["training"]["weight_version_spans"] == [
            {"start": 0, "end": 1, "weight_version": "engine:6"},
            {"start": 1, "end": 2, "weight_version": "engine:7"},
        ]
        assert recorded["stream_delivery"]["complete"] is True

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
                        "_reef_token_weight_versions": ["engine:6"],
                    },
                },
                {
                    "text": "nk>\nanswer",
                    "output_ids": [21],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 2,
                        "output_token_logprobs": [[-0.25, 21]],
                        "_reef_token_weight_versions": ["engine:7"],
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
            backend = SGLangChatTrainingInferenceBackend(
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
            backend = SGLangChatTrainingInferenceBackend(
                str(server.make_url("")).rstrip("/"),
                model_path="model",
                tokenizer=FakeTokenizer(),
                tool_call_parser="qwen25",
                tool_parser_factory=lambda tools, name: FakeStreamingToolParser(),
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
            "text-only message content",
        ),
        ({"max_completion_tokens": 0}, "positive integer"),
    ],
)
def test_chat_facade_rejects_requests_it_cannot_capture_exactly(tmp_path, overrides, message) -> None:
    async def run() -> None:
        backend = SGLangChatTrainingInferenceBackend(
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
def test_chat_facade_splits_reasoning_out_of_visible_content(tmp_path) -> None:
    """Thinking text must ride reasoning_content, never the visible reply.

    The chat template auto-opens <think>, so sampled text carries only the
    closing tag; everything before the LAST </think> is chain-of-thought.
    OpenAI-shaped consumers (agents, judges, user simulators) read content
    verbatim — leaking reasoning there made every downstream judgment grade
    the monologue instead of the reply. Training tokens stay untouched.
    """

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
            backend = SGLangChatTrainingInferenceBackend(
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
        assert message["content"] == "The answer is 5."
        assert message["reasoning_content"] == "Okay, let me think. 2+3 is 5."
        # The exact sampled ids still train, reasoning included.
        assert response["training"]["tokens"] == [10, 11, 20, 21, 22]
        assert response["training"]["response_length"] == 3

    asyncio.run(run())
