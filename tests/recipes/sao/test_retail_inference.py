"""Exact sampled prefixes must survive tool serialization and truncated turns."""

import asyncio
import copy
import os
from unittest.mock import AsyncMock

import pytest

from recipes.sao.examples.servelearnbench.inference import RetailInferenceHandler
from reef.core import AgentRecord, RequestType
from reef.record2task.samples import make_multi_turn_policy_trajectory


class ChatTokenizer:
    """Character tokenizer for handler lifecycle tests; EOS is the separate ID 999."""

    eos_token = "<|im_end|>"
    eos_token_id = 999

    def encode(self, text, *, add_special_tokens=False):
        parts = text.split(self.eos_token)
        tokens = []
        for i, part in enumerate(parts):
            if i:
                tokens.append(self.eos_token_id)
            tokens.extend(ord(char) for char in part)
        return tokens

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


def request():
    return {
        "model": "qwen",
        "user": "episode-1",
        "messages": [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Find my order."}],
    }


def install_native_outputs(monkeypatch, handler, outputs, versions=None):
    remaining = iter(zip(outputs, versions or ["v0"] * len(outputs), strict=True))

    async def complete(artifact, call):
        tokens, version = next(remaining)
        result = {
            "training": {
                "tokens": [*call.prompt_ids, *tokens],
                "loss_mask": [1] * len(tokens),
                "rollout_log_probs": [-0.1] * len(tokens),
                "runtime_load_id": version,
            }
        }
        return result, result

    mocked = AsyncMock(side_effect=complete)
    monkeypatch.setattr(handler, "_complete", mocked)
    return mocked


def assemble(responses):
    records = [
        AgentRecord.create(scenario="retail", request_type=RequestType.INFERENCE, payload={"response": response})
        for response in responses
    ]
    return make_multi_turn_policy_trajectory(records, 1.0, source_agent_record_id="report", realign_threshold=0)


def test_serial_history_preserves_actions_and_masks_only_new_context(monkeypatch):
    handler = RetailInferenceHandler("http://unused", model_path="qwen", tokenizer=ChatTokenizer())
    native = install_native_outputs(monkeypatch, handler, [[11, 12, 999], [13, 999]])
    first = request()
    second = copy.deepcopy(first)
    second["messages"] += [
        {"role": "assistant", "content": "rewritten tool syntax"},
        {"role": "tool", "content": "order A", "tool_call_id": "c1"},
    ]

    async def run():
        first_response = await handler.inference(None, "/v1/chat/completions", first)
        second_response = await handler.inference(None, "/v1/chat/completions", second)
        return first_response, second_response

    responses = asyncio.run(run())
    first_tokens = responses[0]["training"]["tokens"]
    assert responses[1]["training"]["tokens"][: len(first_tokens)] == first_tokens
    sample = assemble(responses)
    assert sample is not None
    assert sum(sample.training["loss_mask"]) == 5
    assert sample.training["loss_mask"][:3] == [1, 1, 1]
    assert sample.training["loss_mask"][-2:] == [1, 1]
    assert not any(sample.training["loss_mask"][3:-2])
    assert native.await_count == 2


def test_changed_history_fails_before_another_generation(monkeypatch):
    handler = RetailInferenceHandler("http://unused", model_path="qwen", tokenizer=ChatTokenizer())
    native = install_native_outputs(monkeypatch, handler, [[11, 999]])
    first = request()
    asyncio.run(handler.inference(None, "/v1/chat/completions", first))
    second = copy.deepcopy(first)
    second["messages"][0]["content"] = "Different rules"
    second["messages"] += [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    with pytest.raises(ValueError, match="extend the previous"):
        asyncio.run(handler.inference(None, "/v1/chat/completions", second))
    assert native.await_count == 1


def test_weight_change_is_allowed_only_at_episode_boundary(monkeypatch):
    handler = RetailInferenceHandler("http://unused", model_path="qwen", tokenizer=ChatTokenizer())
    install_native_outputs(monkeypatch, handler, [[11, 999], [12, 999], [13, 999]], ["v0", "v1", "v1"])
    first = request()
    asyncio.run(handler.inference(None, "/v1/chat/completions", first))
    second = copy.deepcopy(first)
    second["messages"] += [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    with pytest.raises(ValueError, match="weights changed"):
        asyncio.run(handler.inference(None, "/v1/chat/completions", second))
    fresh = {**first, "user": "episode-2"}
    response = asyncio.run(handler.inference(None, "/v1/chat/completions", fresh))
    assert response["training"]["runtime_load_id"] == "v1"


@pytest.mark.integration
@pytest.mark.parametrize("truncated", [False, True])
def test_real_qwen_template_keeps_every_sampled_token(monkeypatch, truncated):
    path = os.environ.get("REEF_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("Set REEF_TEST_QWEN_TOKENIZER to a local Qwen3 tokenizer for this integration check")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    handler = RetailInferenceHandler("http://unused", model_path=path, tokenizer=tokenizer)
    first = request()
    second = copy.deepcopy(first)
    if truncated:
        text = "Let me check the order details"
        second["messages"] += [
            {"role": "assistant", "content": "", "reasoning_content": text},
            {"role": "user", "content": "Your reply was cut off. Act with one tool call."},
        ]
    else:
        text = 'Check order.\n</think>\n\n<tool_call>\n{"name": "lookup", "arguments": {"id": "A"}}\n</tool_call><|im_end|>'
        second["messages"] += [
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
        tokenizer.encode(text, add_special_tokens=False),
        tokenizer.encode("done<|im_end|>", add_special_tokens=False),
    ]
    install_native_outputs(monkeypatch, handler, outputs)

    async def run():
        return [await handler.inference(None, "/v1/chat/completions", body) for body in (first, second)]

    responses = asyncio.run(run())
    sample = assemble(responses)
    assert sample is not None
    assert sum(sample.training["loss_mask"]) == sum(map(len, outputs))
    assert sample.training["tokens"][: len(responses[0]["training"]["tokens"])] == responses[0]["training"]["tokens"]
