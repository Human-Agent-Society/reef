"""Rollout continuation: a later call's prompt is the previous call's exact tokens plus the rendered new messages."""

from __future__ import annotations

import asyncio
import copy
import json
import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.artifact import Artifact
from reef.core import AgentRecord, RequestType
from reef.core.chat_rendering import TokenizerChatTemplate, continued_calls, render_trajectory
from reef.core.trajectories import make_trajectory
from reef.inference import chat as chat_module
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.service.streaming import stream_record

IM_START, IM_END, NEWLINE = 998, 999, ord("\n")


class ChatMLTokenizer:
    """Character ids with ChatML's two special tokens, rendered the way Qwen's template renders a turn."""

    eos_token = "<|im_end|>"
    eos_token_id = IM_END
    specials = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens: list[int] = []
        index = 0
        while index < len(text):
            for special, token in self.specials.items():
                if text.startswith(special, index):
                    tokens.append(token)
                    index += len(special)
                    break
            else:
                tokens.append(ord(text[index]))
                index += 1
        return tokens

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        names = {token: special for special, token in self.specials.items()}
        return "".join(
            names.get(token, chr(token)) for token in token_ids if not (skip_special_tokens and token in names)
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


TOKENIZER = ChatMLTokenizer()


def first_request(rollout: str = "rollout-1") -> dict:
    return {
        "model": "qwen",
        "user": rollout,
        "messages": [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Find my order."}],
    }


def continued(request: dict, *new_messages: dict) -> dict:
    later = copy.deepcopy(request)
    later["messages"] += [{"role": "assistant", "content": "rewritten by the client"}, *new_messages]
    return later


class ScriptedEngine:
    """A fake SGLang ``/generate`` that answers each call with the next scripted output and keeps every prompt."""

    def __init__(self, outputs: list[tuple[list[int], str]]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[list[int]] = []
        self.server: TestServer | None = None

    async def generate(self, request):
        native = await request.json()
        self.prompts.append(list(native["input_ids"]))
        output_ids, finish = self.outputs.pop(0)
        payload = {
            "text": TOKENIZER.decode(output_ids, skip_special_tokens=True),
            "meta_info": {
                "finish_reason": {"type": finish},
                "completion_tokens": len(output_ids),
                "output_token_logprobs": [[-0.1, token] for token in output_ids],
                "weight_version": "weights-0",
            },
        }
        if not native.get("stream"):
            return web.json_response(payload)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        payload["output_ids"] = output_ids
        await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/generate", self.generate)
        self.server = TestServer(app)
        await self.server.start_server()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.server.close()

    def handler(self, **config) -> SGLangInferenceHandler:
        return SGLangInferenceHandler(
            str(self.server.make_url("")).rstrip("/"),
            model_path="qwen",
            tokenizer=TOKENIZER,
            rollout_continuation=True,
            **config,
        )


def artifact(tmp_path) -> Artifact:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    return Artifact.local(checkpoint)


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.mark.unit
def test_a_continuation_extends_the_sampled_tokens_with_only_the_new_messages(tmp_path) -> None:
    async def scenario():
        async with ScriptedEngine([([ord("a"), ord("b"), IM_END], "stop"), ([ord("c"), IM_END], "stop")]) as engine:
            handler = engine.handler()
            first = await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request())
            later = continued(first_request(), {"role": "tool", "content": "order A", "tool_call_id": "c1"})
            second = await handler.inference(artifact(tmp_path), "/v1/chat/completions", later)
        return engine, first, second

    engine, first, second = run(scenario())
    first_tokens = [*engine.prompts[0], ord("a"), ord("b"), IM_END]
    # The second prompt is the first call's prompt and sampled ids, then only the new context: the
    # end-of-turn was sampled, so the suffix starts after it.
    assert engine.prompts[1] == [
        *first_tokens,
        *TOKENIZER.encode("\n<|im_start|>tool\norder A<|im_end|>\n<|im_start|>assistant\n"),
    ]
    assert second["usage"]["prompt_tokens"] == len(engine.prompts[1])
    # The handler keeps the sampled ids to itself: serving returns no training tensors.
    assert "training" not in first and "training" not in second
    # Each response says how its prompt was built, for the training side's rendering.
    assert first["choices"][0]["meta_info"]["rollout_continuation"] is False
    assert second["choices"][0]["meta_info"]["rollout_continuation"] is True


def recorded(request: dict, response: dict, index: int) -> AgentRecord:
    """The inference record of one call, as the inference service stores it."""
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        agent_record_id=f"call-{index}",
        payload={**request, "response": response},
    )


@pytest.mark.unit
def test_training_renders_a_continued_rollout_as_the_engine_served_it(tmp_path) -> None:
    """Rendering the recorded calls gives the engine's last prompt, then the turn it sampled."""
    sampled = [[ord("a"), ord("b"), IM_END], [ord("c"), IM_END], [ord("d"), IM_END]]
    # Plain-text replies take user messages; a tool result follows a reply with a tool call, which this tokenizer
    # cannot sample. The user turns are the case that a template rewriting earlier turns breaks.
    tool = {"role": "user", "content": "order A"}
    user = {"role": "user", "content": "And the refund?"}

    async def scenario():
        async with ScriptedEngine([(ids, "stop") for ids in sampled]) as engine:
            handler = engine.handler()
            requests = [first_request()]
            responses = [await handler.inference(artifact(tmp_path), "/v1/chat/completions", requests[0])]
            for new_message in (tool, user):
                # The client resends the reply as it got it, then the new message.
                later = copy.deepcopy(requests[-1])
                later["messages"] += [responses[-1]["choices"][0]["message"], new_message]
                requests.append(later)
                responses.append(await handler.inference(artifact(tmp_path), "/v1/chat/completions", later))
        return engine, requests, responses

    engine, requests, responses = run(scenario())
    sample = make_trajectory(
        [recorded(request, response, i) for i, (request, response) in enumerate(zip(requests, responses, strict=True))]
    )
    assert continued_calls(sample.trajectory)
    rendered = render_trajectory(TokenizerChatTemplate(TOKENIZER), sample.trajectory)
    # The template closes the last reply with the newline that follows its end-of-turn; the engine never sampled it.
    assert rendered.tokens == [*engine.prompts[2], *sampled[2], NEWLINE]
    prompt_length = len(engine.prompts[0])
    trained = [token for token, flag in zip(rendered.tokens[prompt_length:], rendered.loss_mask, strict=True) if flag]
    assert trained == [*sampled[0], NEWLINE, *sampled[1], NEWLINE, *sampled[2], NEWLINE]


@pytest.mark.unit
def test_a_length_limited_turn_keeps_the_end_of_turn_as_masked_context(tmp_path) -> None:
    async def scenario():
        async with ScriptedEngine([([ord("a"), ord("b")], "length"), ([ord("c"), IM_END], "stop")]) as engine:
            handler = engine.handler()
            first = await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request())
            later = continued(first_request(), {"role": "user", "content": "Your reply was cut off."})
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", later)
        return engine, first

    engine, first = run(scenario())
    first_tokens = [*engine.prompts[0], ord("a"), ord("b")]
    assert first["choices"][0]["finish_reason"] == "length"
    # The handler supplies the end-of-turn that the engine did not sample, then the new context.
    assert engine.prompts[1][: len(first_tokens)] == first_tokens
    assert engine.prompts[1][len(first_tokens)] == IM_END


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda request: request["messages"][0].update(content="Different rules"), "previous messages"),
        (
            lambda request: request["messages"].append({"role": "assistant", "content": "two in a row"}),
            "only user and tool",
        ),
        (lambda request: request.update(model="other"), "cannot change model"),
    ],
)
def test_a_request_that_does_not_extend_its_rollout_is_refused_before_generation(tmp_path, mutate, message) -> None:
    async def scenario():
        async with ScriptedEngine([([ord("a"), IM_END], "stop")]) as engine:
            handler = engine.handler()
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request())
            later = continued(first_request(), {"role": "user", "content": "next"})
            mutate(later)
            with pytest.raises(ValueError, match=message):
                await handler.inference(artifact(tmp_path), "/v1/chat/completions", later)
        return engine

    engine = run(scenario())
    assert len(engine.prompts) == 1


@pytest.mark.unit
def test_a_new_rollout_and_a_request_without_user_render_from_the_template(tmp_path) -> None:
    async def scenario():
        async with ScriptedEngine([([ord("a"), IM_END], "stop")] * 3) as engine:
            handler = engine.handler()
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request("rollout-1"))
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request("rollout-2"))
            anonymous = continued(first_request(), {"role": "user", "content": "next"})
            anonymous.pop("user")
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", anonymous)
        return engine

    engine = run(scenario())
    rendered = TOKENIZER.apply_chat_template(first_request()["messages"], tokenize=True, add_generation_prompt=True)
    assert engine.prompts[0] == engine.prompts[1] == rendered
    # Without a user field the request is stateless: the whole conversation renders, rewritten turn included.
    assert engine.prompts[2] == TOKENIZER.apply_chat_template(
        continued(first_request(), {"role": "user", "content": "next"})["messages"],
        tokenize=True,
        add_generation_prompt=True,
    )


@pytest.mark.unit
def test_without_the_option_every_request_renders_from_the_template(tmp_path) -> None:
    async def scenario():
        async with ScriptedEngine([([ord("a"), IM_END], "stop")] * 2) as engine:
            handler = SGLangInferenceHandler(
                str(engine.server.make_url("")).rstrip("/"), model_path="qwen", tokenizer=TOKENIZER
            )
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request())
            later = continued(first_request(), {"role": "user", "content": "next"})
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", later)
        return engine, later

    engine, later = run(scenario())
    assert engine.prompts[1] == TOKENIZER.apply_chat_template(
        later["messages"], tokenize=True, add_generation_prompt=True
    )


@pytest.mark.unit
def test_the_oldest_rollout_is_forgotten_past_the_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chat_module, "ROLLOUT_PREFIX_LIMIT", 2)

    async def scenario():
        async with ScriptedEngine([([ord("a"), IM_END], "stop")] * 4) as engine:
            handler = engine.handler()
            for rollout in ("one", "two", "three"):
                await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request(rollout))
            # Rollout "one" was forgotten, so its continuation is a fresh render, not a prefix extension.
            later = continued(first_request("one"), {"role": "user", "content": "next"})
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", later)
        return engine, later

    engine, later = run(scenario())
    assert engine.prompts[3] == TOKENIZER.apply_chat_template(
        later["messages"], tokenize=True, add_generation_prompt=True
    )


@pytest.mark.unit
def test_a_streamed_continuation_uses_the_same_prefix(tmp_path) -> None:
    async def scenario():
        async with ScriptedEngine([([ord("a"), IM_END], "stop"), ([ord("c"), IM_END], "stop")]) as engine:
            handler = engine.handler()
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request())
            later = continued(first_request(), {"role": "user", "content": "next"})
            stream = await handler.inference_stream(
                artifact(tmp_path), "/v1/chat/completions", {**later, "stream": True}
            )
            body = b"".join([chunk async for chunk in stream.chunks])
            recorded = stream_record(stream, body, complete=True)
        return engine, recorded

    engine, recorded = run(scenario())
    first_tokens = [*engine.prompts[0], ord("a"), IM_END]
    assert engine.prompts[1][: len(first_tokens)] == first_tokens
    assert recorded["usage"]["prompt_tokens"] == len(engine.prompts[1])
    assert "training" not in recorded


@pytest.mark.integration
@pytest.mark.parametrize("truncated", [False, True])
def test_a_real_qwen3_template_keeps_every_sampled_token(tmp_path, truncated) -> None:
    path = os.environ.get("REEF_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("Set REEF_TEST_QWEN_TOKENIZER to a local Qwen3 tokenizer for this check")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    if truncated:
        sampled = "Let me check the order details"
        new_messages = [
            {"role": "assistant", "content": "", "reasoning_content": sampled},
            {"role": "user", "content": "Your reply was cut off. Act with one tool call."},
        ]
    else:
        sampled = 'Check order.\n</think>\n\n<tool_call>\n{"name": "lookup", "arguments": {"id": "A"}}\n</tool_call><|im_end|>'
        new_messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Check order.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"id": "A"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"order": "A"}'},
        ]
    outputs = [
        tokenizer.encode(sampled, add_special_tokens=False),
        tokenizer.encode("done<|im_end|>", add_special_tokens=False),
    ]
    prompts: list[list[int]] = []

    async def scenario():
        async def generate(request):
            native = await request.json()
            prompts.append(list(native["input_ids"]))
            output_ids = outputs[len(prompts) - 1]
            return web.json_response(
                {
                    "text": tokenizer.decode(output_ids),
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, token] for token in output_ids],
                        "weight_version": "weights-0",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            handler = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"), model_path=path, tokenizer=tokenizer, rollout_continuation=True
            )
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", first_request())
            later = copy.deepcopy(first_request())
            later["messages"] += new_messages
            await handler.inference(artifact(tmp_path), "/v1/chat/completions", later)
        finally:
            await server.close()

    run(scenario())
    # The second prompt keeps every token the first call sampled, so no sampled token is re-rendered.
    first_tokens = [*prompts[0], *outputs[0]]
    assert prompts[1][: len(first_tokens)] == first_tokens


@pytest.mark.integration
def test_a_real_qwen3_template_replays_a_continued_rollout_that_it_cannot_render_as_one_conversation(tmp_path) -> None:
    """Three calls: a reply with reasoning, a user message, a reply without reasoning, a user message, a reply.

    Rendered as one conversation, Qwen3's template drops the first reply's reasoning after the user message, and it
    renders the second reply's think block only while that reply is last. Rendered call by call, the tokens are the
    ones the engine served.
    """
    path = os.environ.get("REEF_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("Set REEF_TEST_QWEN_TOKENIZER to a local Qwen3 tokenizer for this check")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    sampled_text = [
        "<think>\nCheck the order first.\n</think>\n\nI will check the order.<|im_end|>",
        "<think>\n\n</think>\n\nThe order shipped yesterday.<|im_end|>",
        "<think>\nNothing else to do.\n</think>\n\nYou are welcome.<|im_end|>",
    ]
    outputs = [tokenizer.encode(text, add_special_tokens=False) for text in sampled_text]
    new_messages = [{"role": "user", "content": "Where is it now?"}, {"role": "user", "content": "Thanks."}]
    prompts: list[list[int]] = []

    async def scenario():
        async def generate(request):
            native = await request.json()
            prompts.append(list(native["input_ids"]))
            output_ids = outputs[len(prompts) - 1]
            # SGLang's text leaves out special tokens; the sampled end-of-turn is not part of the reply's content.
            return web.json_response(
                {
                    "text": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, token] for token in output_ids],
                        "weight_version": "weights-0",
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/generate", generate)
        server = TestServer(app)
        await server.start_server()
        try:
            handler = SGLangInferenceHandler(
                str(server.make_url("")).rstrip("/"), model_path=path, tokenizer=tokenizer, rollout_continuation=True
            )
            requests = [first_request()]
            responses = [await handler.inference(artifact(tmp_path), "/v1/chat/completions", requests[0])]
            for new_message in new_messages:
                later = copy.deepcopy(requests[-1])
                later["messages"] += [responses[-1]["choices"][0]["message"], new_message]
                requests.append(later)
                responses.append(await handler.inference(artifact(tmp_path), "/v1/chat/completions", later))
        finally:
            await server.close()
        return requests, responses

    requests, responses = run(scenario())
    assert responses[0]["choices"][0]["message"]["reasoning_content"] == "Check the order first."
    assert "reasoning_content" not in responses[1]["choices"][0]["message"]
    sample = make_trajectory(
        [recorded(request, response, i) for i, (request, response) in enumerate(zip(requests, responses, strict=True))]
    )
    template = TokenizerChatTemplate(tokenizer, {"enable_thinking": True})
    rendered = render_trajectory(template, sample.trajectory)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert rendered.tokens == [*prompts[2], *outputs[2], *newline]
    prompt_length = len(prompts[0])
    trained = [token for token, flag in zip(rendered.tokens[prompt_length:], rendered.loss_mask, strict=True) if flag]
    assert trained == [*outputs[0], *newline, *outputs[1], *newline, *outputs[2], *newline]
    # Without the continuation marks the same calls render as one conversation, which this template cannot.
    for record in sample.trajectory["extra"]["reef"]["records"][1:]:
        record["payload"]["response"]["choices"][0]["meta_info"]["rollout_continuation"] = False
    with pytest.raises(ValueError, match="prefix of the whole conversation"):
        render_trajectory(template, sample.trajectory)
