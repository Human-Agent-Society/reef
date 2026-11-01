"""Rendering an ATIF trajectory into the token IDs and the loss mask that a training component trains."""

from __future__ import annotations

from typing import Any

import pytest
from tests.support.chat_templates import CharTemplate

from reef.core.chat_rendering import RenderedTrajectory, TeacherRequest, render_trajectory, rendered_job_data
from reef.core.job_data import JobData, TrajectorySample
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import make_trajectory


def step(source: str, message: str, **fields: Any) -> dict[str, Any]:
    return {"source": source, "message": message, **fields}


def trajectory(*steps: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    agent = {"name": "a", **({"tool_definitions": tools} if tools else {})}
    return {"schema_version": "ATIF-v1.7", "agent": agent, "steps": list(steps)}


def test_every_agent_turn_trains_and_the_turns_between_are_masked() -> None:
    rendered = render_trajectory(
        CharTemplate(), trajectory(step("user", "q"), step("agent", "ab"), step("user", "r"), step("agent", "c"))
    )
    # user q | assistant ab | user r | assistant c.
    assert rendered.tokens == [2, 213, 9, 197, 198, 2, 214, 9, 199]
    # The mask starts after the first generation prompt; each turn trains the tokens after its role token.
    assert rendered.loss_mask == [1, 1, 0, 0, 0, 1]


def test_copied_context_trains_nothing_and_the_steps_after_the_last_trained_turn_are_left_out() -> None:
    rendered = render_trajectory(
        CharTemplate(),
        trajectory(
            step("user", "q"),
            step("agent", "a", is_copied_context=True),
            step("user", "r"),
            step("agent", "b"),
            step("user", "later"),
        ),
    )
    assert rendered.tokens == [2, 213, 9, 197, 2, 214, 9, 198]
    assert rendered.loss_mask == [1]


def test_the_agent_tools_render_with_every_part() -> None:
    template = CharTemplate()
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    rendered = render_trajectory(template, trajectory(step("user", "q"), step("agent", "a"), tools=tools))
    assert rendered.tokens[0] == 8 and all(seen == tools for seen in template.tools)


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        ((step("user", "q"), step("agent", "a", is_copied_context=True)), "agent steps to train"),
        ((step("agent", "a"),), "before its first trained agent step"),
        ((step("user", "q"), step("agent", "a", is_copied_context="yes")), "must be a boolean"),
    ],
)
def test_a_trajectory_that_cannot_train_is_refused(steps, error) -> None:
    with pytest.raises(ValueError, match=error):
        render_trajectory(CharTemplate(), trajectory(*steps))


def test_a_template_that_drops_earlier_turns_cannot_train_them_all() -> None:
    with pytest.raises(ValueError, match="prefix of the whole conversation"):
        render_trajectory(
            CharTemplate(drop_history=True),
            trajectory(step("user", "q"), step("agent", "a"), step("user", "r"), step("agent", "b")),
        )


def test_a_rendered_trajectory_has_a_prompt_and_trained_tokens() -> None:
    with pytest.raises(ValueError, match="prompt and one or more trained tokens"):
        RenderedTrajectory([1, 2], [1, 1])


def teacher_sample(teacher: TeacherRequest | None, *steps: dict[str, Any]) -> TrajectorySample:
    sample = TrajectorySample(trajectory(*(steps or (step("user", "q"), step("agent", "ab")))))
    return sample if teacher is None else sample.with_metadata(teacher=teacher.as_json())


def test_a_teacher_reads_its_request_then_the_students_reply_tokens() -> None:
    demonstration = TeacherRequest([{"role": "user", "content": "q!"}], max_prompt_tokens=3)
    own = TeacherRequest()
    data, left_out = rendered_job_data(
        JobData("d", (teacher_sample(demonstration), teacher_sample(own), teacher_sample(None))), CharTemplate()
    )
    assert left_out == ()
    with_request, with_own, plain = data.samples
    # The request "q!" renders as [2, 213, 133, 9], cut to 3 tokens, then the student's reply tokens.
    assert with_request.training["teacher_tokens"] == [2, 213, 133, 197, 198]
    assert with_own.training["teacher_tokens"] == with_own.training["tokens"] == [2, 213, 9, 197, 198]
    assert "teacher_tokens" not in plain.training


def test_a_teacher_sequence_over_its_limit_and_an_unrenderable_trajectory_are_left_out() -> None:
    samples = (
        teacher_sample(TeacherRequest(max_tokens=4)),
        teacher_sample(None, step("agent", "a")),
        teacher_sample(None),
    )
    data, left_out = rendered_job_data(JobData("d", samples), CharTemplate())
    assert left_out == (0, 1) and len(data.samples) == 1


def test_a_sample_renders_with_the_template_arguments_of_its_recorded_request() -> None:
    record = AgentRecord.create(
        agent_record_id="call",
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "user", "content": "q"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "response": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
        },
    )
    template = CharTemplate()
    _, left_out = rendered_job_data(JobData("d", (make_trajectory((record,), 1.0),)), template)
    assert left_out == () and template.arguments == {"enable_thinking": False}
