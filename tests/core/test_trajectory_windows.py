"""Captured SFT windows preserve each call's context and supervised targets."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

import pytest
from tests.support.chat_templates import CharTemplate

from reef.core.chat_rendering import render_trajectory
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import atif_messages, make_trajectory
from reef.core.trajectory_windows import captured_context_windows


def recorded(name: str, messages: list[dict[str, str]], answer: str, **fields: object) -> AgentRecord:
    """One captured text exchange with fixed time and caller-selected request fields."""
    return AgentRecord.create(
        agent_record_id=name,
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload={
            "model": "m",
            "messages": messages,
            "response": {"choices": [{"message": {"role": "assistant", "content": answer}}]},
            **fields,
        },
        created_at=1.0,
    )


def captured_trajectory(records: Sequence[AgentRecord]) -> TrajectorySample:
    """A legacy capture with complete provider payloads beside its ATIF steps."""
    return make_trajectory(records).with_metadata(
        records=[
            {
                "agent_record_id": record.agent_record_id,
                "created_at": record.created_at,
                "artifact_id": record.artifact_id,
                "payload": dict(record.payload),
            }
            for record in records
        ]
    )


def test_compacted_context_is_separate_and_copied_answers_never_train_twice() -> None:
    first_prompt = [{"role": "user", "content": "q"}]
    second_prompt = [*first_prompt, {"role": "assistant", "content": "A"}, {"role": "user", "content": "r"}]
    compacted_prompt = [
        {"role": "user", "content": "summary"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "s"},
    ]
    final_prompt = [*compacted_prompt, {"role": "assistant", "content": "C"}, {"role": "user", "content": "t"}]
    sample = replace(
        captured_trajectory(
            (
                recorded("a", first_prompt, "A"),
                recorded("b", second_prompt, "B"),
                recorded("c", compacted_prompt, "C"),
                recorded("d", final_prompt, "D"),
            )
        ).with_metadata(max_tokens=100, task={"id": "task"}),
        group_id="group",
    )
    original = deepcopy(sample)

    first, second = captured_context_windows(sample)

    assert sample == original
    assert [record["agent_record_id"] for record in first.metadata["records"]] == ["a", "b"]
    assert [record["agent_record_id"] for record in second.metadata["records"]] == ["c", "d"]
    assert first.trajectory["trajectory_id"] == first.metadata["source_agent_record_id"] == "b"
    assert second.trajectory["trajectory_id"] == second.metadata["source_agent_record_id"] == "d"
    assert atif_messages(second.trajectory)[:-3] == compacted_prompt
    for window, targets in ((first, "AB"), (second, "CD")):
        assert window.group_id == "group" and window.source_agent_record_ids == sample.source_agent_record_ids
        assert window.metadata["max_tokens"] == 100 and window.metadata["task"] == {"id": "task"}
        (rendered,) = render_trajectory(CharTemplate(), window.trajectory)
        response_tokens = rendered.tokens[-len(rendered.loss_mask) :]
        trained = [token for token, mask in zip(response_tokens, rendered.loss_mask, strict=True) if mask]
        assert trained == [100 + ord(character) for character in targets]


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "another"},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"system": "new instructions"},
        {"tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}]},
    ],
)
def test_model_template_system_or_tools_change_starts_an_independent_window(changed: dict[str, object]) -> None:
    prompt = [{"role": "user", "content": "q"}]
    continued = [*prompt, {"role": "assistant", "content": "A"}, {"role": "user", "content": "r"}]
    sample = captured_trajectory((recorded("a", prompt, "A"), recorded("b", continued, "B", **changed)))

    first, second = captured_context_windows(sample)

    assert len(first.metadata["records"]) == len(second.metadata["records"]) == 1
    agent_steps = [step for step in second.trajectory["steps"] if step["source"] == "agent"]
    assert [step.get("is_copied_context", False) for step in agent_steps] == [True, False]
    if "tools" in changed:
        assert "tool_definitions" not in first.trajectory["agent"]
        assert second.trajectory["agent"]["tool_definitions"] == changed["tools"]


def test_uncaptured_assistant_turn_starts_a_window_and_is_only_context() -> None:
    prompt = [{"role": "user", "content": "q"}]
    continued = [
        *prompt,
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "r"},
        {"role": "assistant", "content": "uncaptured"},
        {"role": "user", "content": "s"},
    ]
    sample = captured_trajectory((recorded("a", prompt, "A"), recorded("b", continued, "B")))

    _, second = captured_context_windows(sample)

    flags = [step.get("is_copied_context", False) for step in second.trajectory["steps"] if step["source"] == "agent"]
    assert flags == [True, True, False]


def test_existing_linear_plain_single_call_and_tokenized_samples_stay_unchanged() -> None:
    prompt = [{"role": "user", "content": "q"}]
    first = recorded(
        "a",
        prompt,
        "A",
        response={"choices": [{"message": {"role": "assistant", "content": "A", "tool_calls": None}}]},
    )
    second = recorded("b", [*prompt, {"role": "assistant", "content": "A"}], "B")
    linear = captured_trajectory((first, second))
    compacted = captured_trajectory((first, recorded("c", [{"role": "user", "content": "summary"}], "C")))
    plain = TrajectorySample({key: value for key, value in compacted.trajectory.items() if key != "extra"})
    for sample in (
        linear,
        make_trajectory((first, second)),
        captured_trajectory((first,)),
        plain,
        compacted.with_training(tokens=[1, 2], loss_mask=[1]),
    ):
        assert captured_context_windows(sample) == (sample,)
        assert captured_context_windows(sample)[0] is sample


@pytest.mark.parametrize("field", ["previous_response_id", "conversation"])
def test_server_held_context_cannot_be_claimed_as_a_complete_window(field: str) -> None:
    sample = captured_trajectory(
        (
            recorded("a", [{"role": "user", "content": "q"}], "A"),
            recorded("b", [{"role": "user", "content": "r"}], "B", **{field: "previous"}),
        )
    )
    with pytest.raises(ValueError, match="server-held"):
        captured_context_windows(sample)


@pytest.mark.parametrize("change", [{"message": "edited target"}, {"is_copied_context": True}])
def test_edited_targets_or_masks_are_never_overwritten_from_raw_records(change: dict[str, object]) -> None:
    sample = captured_trajectory(
        (
            recorded("a", [{"role": "user", "content": "q"}], "A"),
            recorded("b", [{"role": "user", "content": "s"}], "B"),
        )
    )
    steps = [*sample.trajectory["steps"][:-1], {**sample.trajectory["steps"][-1], **change}]
    edited = replace(sample, trajectory={**sample.trajectory, "steps": steps})
    with pytest.raises(ValueError, match="supervised targets"):
        captured_context_windows(edited)


@pytest.mark.parametrize("provider", ["anthropic", "responses"])
def test_provider_native_requests_and_responses_keep_the_actual_context(provider: str) -> None:
    records = []
    for name, question, answer in (("a", "q", "A"), ("b", "summary", "B")):
        if provider == "anthropic":
            payload = {
                "system": "instructions",
                "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
                "response": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            }
        else:
            payload = {
                "instructions": "instructions",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": question}]}],
                "response": {
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": answer}]}
                    ]
                },
            }
        records.append(
            recorded(
                name,
                payload.get("messages", []),
                answer,
                **{key: value for key, value in payload.items() if key != "messages"},
            )
        )
        if provider == "responses":
            records[-1] = replace(
                records[-1], payload={key: value for key, value in records[-1].payload.items() if key != "messages"}
            )
    sample = captured_trajectory(records)
    original = deepcopy(sample)

    first, second = captured_context_windows(sample)

    assert sample == original
    for window, question, answer in ((first, "q", "A"), (second, "summary", "B")):
        assert atif_messages(window.trajectory) == [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"messages": ["not a message"]}, "message objects"),
        ({"messages": [{"role": "user", "content": [{"type": "audio", "data": "opaque"}]}]}, "content block"),
        ({"response": {"choices": []}}, "complete request and response"),
        (
            {
                "response": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "opaque"}],
                }
            },
            "content block",
        ),
        (
            {"response": {"output": [{"type": "reasoning", "encrypted_content": "opaque"}]}},
            "unsupported provider message",
        ),
        ({"tools": [{"type": "web_search"}]}, "function tool definitions"),
    ],
)
def test_incomplete_or_unrepresentable_captured_content_is_refused(payload: dict[str, object], error: str) -> None:
    sample = captured_trajectory(
        (
            recorded("a", [{"role": "user", "content": "q"}], "A"),
            recorded(
                "b",
                payload.get("messages", [{"role": "user", "content": "summary"}]),
                "B",
                **{key: value for key, value in payload.items() if key != "messages"},
            ),
        )
    )
    with pytest.raises(ValueError, match=error):
        captured_context_windows(sample)


def test_training_overrides_cannot_replace_the_captured_provider_context() -> None:
    second = recorded("b", [{"role": "user", "content": "summary"}], "B")
    response = {**second.payload["response"], "training": {"request_messages": [{"role": "user", "content": "other"}]}}
    sample = captured_trajectory(
        (
            recorded("a", [{"role": "user", "content": "q"}], "A"),
            replace(second, payload={**second.payload, "response": response}),
        )
    )
    with pytest.raises(ValueError, match="training overrides"):
        captured_context_windows(sample)


def test_edited_tool_definitions_are_never_overwritten_from_raw_records() -> None:
    sample = captured_trajectory(
        (
            recorded("a", [{"role": "user", "content": "q"}], "A"),
            recorded("b", [{"role": "user", "content": "summary"}], "B"),
        )
    )
    agent = {
        **sample.trajectory["agent"],
        "tool_definitions": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }
    edited = replace(sample, trajectory={**sample.trajectory, "agent": agent})
    with pytest.raises(ValueError, match="tool definitions"):
        captured_context_windows(edited)


def test_equivalent_atif_messages_validate_while_raw_contexts_and_response_metrics_stay_separate() -> None:
    first = recorded(
        "a",
        [{"role": "user", "content": "q"}],
        "A",
        response={
            "choices": [{"message": {"role": "assistant", "content": "A"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )
    second = recorded(
        "b",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            {"role": "user", "content": "r"},
        ],
        "B",
    )
    sample = captured_trajectory((first, second))
    original = deepcopy(sample)

    before, after = captured_context_windows(sample)

    assert sample == original
    assert before.trajectory["steps"][-1]["metrics"] == sample.trajectory["steps"][1]["metrics"]
    assert before.trajectory["steps"][-1]["metrics"]["completion_tokens"] == 1
    assert [
        step.get("is_copied_context", False) for step in after.trajectory["steps"] if step["source"] == "agent"
    ] == [
        True,
        False,
    ]
