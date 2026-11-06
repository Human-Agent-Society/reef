"""Provider payloads as ATIF (RFC #1054): every payload renders back exactly, and the ATIF objects follow the format."""

from __future__ import annotations

import pytest
from harbor.models.trajectories import Trajectory

from reef.core.atif_payloads import atif_call, canonical_json, folded_steps, patch_of, patched, provider_payload

CHAT = {
    "model": "m",
    "temperature": 0.2,
    "messages": [
        {"role": "developer", "content": "Be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read"},
                {"type": "image_url", "image_url": {"url": "a.png", "detail": "high"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "r1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.txt"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "r1", "content": "file", "name": "read"},
    ],
    "tools": [
        {"type": "function", "function": {"name": "read", "description": "Read", "parameters": {"type": "object"}}}
    ],
    "response": {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Done", "reasoning_content": "Read it", "refusal": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}},
        "training": {"tokens": [1, 2, 3], "loss_mask": [1]},
    },
}
ANTHROPIC = {
    "model": "claude",
    "max_tokens": 64,
    "system": [{"type": "text", "text": "Be precise", "cache_control": {"type": "ephemeral"}}],
    "tools": [
        {
            "name": "inspect",
            "description": "Inspect",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "web_search_20250305", "name": "web_search"},
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect", "cache_control": {"type": "ephemeral"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Hm", "signature": "sig"},
                {"type": "tool_use", "id": "call-1", "name": "inspect", "input": {"x": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "ok", "is_error": False},
                {"type": "text", "text": "<reminder>"},
            ],
        },
    ],
    "response": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "Good", "signature": "s2"}, {"type": "text", "text": "Done"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    },
}
RESPONSES = {
    "model": "m",
    "instructions": "Be precise",
    "tools": [{"type": "function", "name": "inspect", "parameters": {}, "strict": True}, {"type": "file_search"}],
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Think"}],
            "encrypted_content": "E",
        },
        {"type": "function_call", "call_id": "call-1", "name": "inspect", "arguments": '{"x":1}', "id": "fc_1"},
        {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    ],
    "response": {
        "id": "resp_1",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": "m1",
                "content": [{"type": "output_text", "text": "Done", "annotations": []}],
            }
        ],
    },
}


@pytest.mark.parametrize("payload", [CHAT, ANTHROPIC, RESPONSES], ids=["chat", "anthropic", "responses"])
def test_a_payload_renders_back_exactly_from_valid_atif(payload: dict) -> None:
    call = atif_call(payload)
    assert call is not None
    assert canonical_json(provider_payload(call)) == canonical_json(payload)
    steps = folded_steps([*call.entries, *call.reply])
    document = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "a", "version": "1", **(call.agent or {})},
        "steps": steps,
    }
    Trajectory.model_validate(document)
    assert steps[0]["source"] == "system"
    assert steps[-1]["source"] == "agent" and steps[-1]["message"] == "Done"
    calling = next(step for step in steps if step.get("tool_calls"))
    assert calling["observation"]["results"][0]["content"] == ("file" if payload is CHAT else "ok")


def test_private_fields_stay_in_extra_and_tools_take_the_function_schema() -> None:
    call = atif_call(ANTHROPIC)
    assert call is not None and call.agent is not None
    assert call.agent["tool_definitions"] == [
        {
            "type": "function",
            "function": {"name": "inspect", "parameters": {"type": "object"}, "description": "Inspect"},
        }
    ]
    assert call.agent["extra"]["reef"]["other_tools"] == [
        {"index": 1, "tool": {"type": "web_search_20250305", "name": "web_search"}}
    ]
    system, user, assistant, results = call.entries
    assert system[0]["message"] == [{"type": "text", "text": "Be precise"}]
    assert system[0]["extra"]["reef"]["patch"] == {"[]": {"0": {"~": {"cache_control": {"=": {"type": "ephemeral"}}}}}}
    assert user[0]["message"][1] == {
        "type": "image",
        "source": {"media_type": "image/png", "path": "data:image/png;base64,AAAA"},
    }
    assert assistant[0]["reasoning_content"] == "Hm" and assistant[0]["is_copied_context"] is True
    assert [element.get("source_call_id") for element in results] == ["call-1", None]
    assert call.reply[0][0]["metrics"] == {"prompt_tokens": 5, "completion_tokens": 3}


def test_a_payload_without_a_message_list_has_no_atif_call() -> None:
    assert atif_call({"prompt": "hi"}) is None
    assert atif_call({"input": "a string prompt"}) is None
    shapeless = {"messages": [{"role": "user", "content": "hi"}], "response": {"content": "done"}}
    call = atif_call(shapeless)
    assert call is not None and call.reply == ()
    assert provider_payload(call) == shapeless


@pytest.mark.parametrize(
    ("original", "rendered"),
    [
        ({"a": 1, "b": [1, {"c": None}]}, {"a": 1, "b": [1, {"c": 2}], "d": True}),
        ([1, 2, 3], [1, 2]),
        (True, 1),
        ({"x": "y"}, {"x": "y"}),
    ],
)
def test_a_patch_turns_the_rendered_value_into_the_original(original, rendered) -> None:
    patch = patch_of(original, rendered)
    assert canonical_json(patched(rendered, patch)) == canonical_json(original)
    assert (patch is None) == (canonical_json(original) == canonical_json(rendered))
