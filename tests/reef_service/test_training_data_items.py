"""ATIF-only training data: serialization, consumer fidelity, and mixed batching."""

import json
import pickle
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from reef_service._trajectories import policy_trajectory, recorded_trajectory

from reef.core import AgentRecord, RequestType, RuntimeLoadSpan
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import (
    make_trajectory,
    recorded_payload,
    recorded_payloads,
    source_record_id,
    trajectory_reward,
)
from reef.runtime.interfaces import InferenceStream
from reef.service.streaming import stream_record
from reef.train.algos import StepScheduling
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import TaskItem, TrainingBatch, TrajectoryItem, trajectories, trajectory_groups


def captured_trajectory(record_id="inference-1", reward=1.0):
    return policy_trajectory(
        record_id,
        (10, 11, 12, 13),
        (1, 0, 1),
        (-0.2, -0.3, -0.4),
        reward,
        runtime_load_id="weights-1",
        action_mask=(1, 0, 1),
        rollout_created_at=123.0,
        turn_count=2,
        topk_indices=((11, 20), (12, 21), (13, 22)),
        topk_log_probs=((-0.2, -1.2), (-0.3, -1.3), (-0.4, -1.4)),
        extras={"teacher_cands": [{"hint": "", "teacher_tokens": [10, 11, 12, 13]}]},
        runtime_load_spans=(RuntimeLoadSpan(0, 3, "weights-1"),),
    )


def atif_item():
    return TrajectoryItem(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "episode-1",
            "agent": {"name": "test", "version": "1.0"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Solve the task."},
                {"step_id": 2, "source": "agent", "message": "The answer is 42."},
            ],
            "extra": {"reef": {"reward": 1.0, "feedback": "correct"}, "external": {"keep": True}},
        }
    )


def test_mixed_items_survive_worker_serialization():
    batch = TrainingBatch("mixed", (captured_trajectory(), TaskItem(Path("tasks/example")), atif_item()))
    assert pickle.loads(pickle.dumps(batch)) == batch
    assert isinstance(batch.items[1], TaskItem)


# The cookbook recipes' step schedules, as their ``training_spec()`` binds them.
SCHEDULING = {
    "sao": StepScheduling(unit="sample"),
    "openclawrl": StepScheduling(unit="sample"),
    "tttd": StepScheduling(unit="sample", batch_size="actual"),
}


@pytest.mark.parametrize("algorithm", ["sao", "openclawrl", "tttd"])
def test_algorithms_consume_json_roundtripped_atif_with_identical_payloads(algorithm):
    items = tuple(replace(captured_trajectory(str(index), float(index)), group_id="group") for index in range(2))
    original = TrainingBatch("batch", items)
    loaded = TrainingBatch(
        "batch", tuple(replace(item, trajectory=json.loads(json.dumps(item.trajectory))) for item in items)
    )
    assert prepare_slime_step(loaded, algorithm, {}, SCHEDULING[algorithm]) == prepare_slime_step(
        original, algorithm, {}, SCHEDULING[algorithm]
    )
    assert loaded.items[0].training["runtime_load_spans"] == [{"start": 0, "end": 3, "runtime_load_id": "weights-1"}]


def test_harness_projection_reads_external_atif_without_reef_records():
    item = atif_item()
    payload = recorded_payload(item)
    assert payload["messages"] == [{"role": "user", "content": "Solve the task."}]
    assert payload["response"]["choices"][0]["message"] == {"role": "assistant", "content": "The answer is 42."}
    assert item.metadata["feedback"] == "correct"
    assert source_record_id(item) == "episode-1"


def test_original_provider_payloads_and_feedback_survive_json_roundtrip():
    first = {"messages": [{"role": "user", "content": "hi"}], "provider_specific": {"opaque": [1, 2]}}
    last = {"messages": [{"role": "user", "content": "next"}], "response": {"content": "done"}}
    item = recorded_trajectory("record", last, 0.0, {"detail": "feedback"}, (first, last))
    loaded = TrajectoryItem(json.loads(json.dumps(item.trajectory)))
    assert recorded_payloads(loaded) == (first, last)
    assert recorded_payload(loaded) == last
    assert loaded.metadata["feedback"] == {"detail": "feedback"}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_completed_stream_messages_survive_multiturn_atif_conversion(provider: str) -> None:
    records = []
    messages = []
    for question, answer in (("First question", "First answer"), ("Next question", "Next answer")):
        if provider == "openai":
            event = {"choices": [{"index": 0, "delta": {"content": answer}}]}
            terminal = "[DONE]"
        else:
            event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": answer}}
            terminal = json.dumps({"type": "message_stop"})
        body = f"data: {json.dumps(event)}\n\ndata: {terminal}\n\n".encode()

        async def _chunks(body: bytes = body) -> AsyncIterator[bytes]:
            yield body

        stream = InferenceStream(status=200, headers={"content-type": "text/event-stream"}, chunks=_chunks())
        response = stream_record(stream, body, complete=True)
        messages.append({"role": "user", "content": question})
        records.append(
            AgentRecord.create(
                scenario="stream",
                request_type=RequestType.INFERENCE,
                payload={"messages": list(messages), "response": response},
            )
        )
        messages.append({"role": "assistant", "content": answer})

    item = make_trajectory(records, reward=1.0)
    loaded = TrajectoryItem(json.loads(json.dumps(item.trajectory)))

    assert [(step["source"], step["message"]) for step in loaded.trajectory["steps"]] == [
        ("user", "First question"),
        ("agent", "First answer"),
        ("user", "Next question"),
        ("agent", "Next answer"),
    ]
    assert recorded_payloads(loaded) == tuple(record.payload for record in records)
    assert item.source_agent_record_ids == tuple(record.agent_record_id for record in records)
    assert trajectory_reward(loaded) == 1.0
    assert loaded.training == {}


def test_training_response_message_takes_precedence_over_stream_summary() -> None:
    item = recorded_trajectory(
        "stream",
        {
            "response": {
                "message": {"role": "assistant", "content": "Stream summary"},
                "training": {"response_message": {"role": "assistant", "content": "Captured answer"}},
            }
        },
        1.0,
    )

    assert item.trajectory["steps"][0]["message"] == "Captured answer"


@pytest.mark.parametrize("algorithm", ["sao", "openclawrl", "tttd"])
def test_existing_algorithms_reject_tasks_without_consuming_or_changing_state(algorithm):
    batch = TrainingBatch("mixed", (replace(captured_trajectory(), group_id="a"), TaskItem(Path("tasks/example"))))
    state = {"steps": 3}
    with pytest.raises(TypeError, match="unsupported item 1"):
        prepare_slime_step(batch, algorithm, state, SCHEDULING[algorithm])
    assert state == {"steps": 3}
    assert isinstance(batch.items[1], TaskItem)


def test_policy_backend_reports_missing_training_data_without_inventing_tokens():
    item = atif_item()
    prepared = prepare_slime_step(TrainingBatch("plain", (item,)), "sao", {}, SCHEDULING["sao"])
    from reef.train.slime_backend.data_builder import to_slime_rollout_data

    with pytest.raises(ValueError, match="training tensors"):
        to_slime_rollout_data(prepared.payload)
    assert item.training == {}


def test_groups_preserve_row_and_advantage_order():
    items = tuple(
        replace(captured_trajectory(str(index), reward), group_id="z" if index < 2 else "a")
        for index, reward in enumerate((0.0, 2.0, 3.0, 1.0))
    )
    batch = TrainingBatch("groups", items)
    assert trajectories(batch) == items
    assert trajectory_groups(batch) == (items[:2], items[2:])
    from recipes.tttd.objective import TttdObjective

    objective = TttdObjective()
    expected = tuple(
        value
        for group in (items[:2], items[2:])
        for value in objective.adaptive_entropic_advantages([trajectory_reward(item) for item in group])[0]
    )
    assert objective.prepare(batch, {}).advantages == expected


def test_noncontiguous_groups_fail_before_misaligned_training():
    batch = TrainingBatch(
        "groups",
        tuple(replace(captured_trajectory(str(index)), group_id=group) for index, group in enumerate(("a", "b", "a"))),
    )
    with pytest.raises(ValueError, match="contiguous"):
        trajectory_groups(batch)


def test_non_atif_payloads_are_not_accepted_as_trajectories():
    with pytest.raises(TypeError, match="ATIF"):
        TrajectoryItem(object())
    with pytest.raises(TypeError, match="TrajectoryItem or TaskItem"):
        TrainingBatch("invalid", ({"tokens": [1]},))


def test_request_only_batch_preserves_manual_training_contract():
    request = TrainingRequest("Improve the harness", "session", "release", id="request")
    batch = TrainingBatch("manual", request=request)
    assert trajectories(batch) == ()
    assert replace(batch, batch_id="next").request is request


def test_report_processor_reserves_and_acknowledges_mixed_data():
    from reef.train.processors.reported import ReportedFeedbackProcessor
    from reef.train.types import ProcessorContext

    class MixedProcessor(ReportedFeedbackProcessor):
        exclusive_sources = True

        def make_sample(self, context):
            source = context.inferences[0]
            return (
                TaskItem(Path(source.payload["task_path"]))
                if "task_path" in source.payload
                else make_trajectory(context.inferences)
            )

        def make_batch(self, items, batch_number):
            return TrainingBatch(str(batch_number), items)

    processor = MixedProcessor(ProcessorContext("mixed", {"batch_size": 2}))
    for index, payload in enumerate(
        ({"messages": [{"role": "user", "content": "hi"}]}, {"task_path": "tasks/example"})
    ):
        processor.ingest(
            AgentRecord.create(
                scenario="mixed",
                request_type=RequestType.INFERENCE,
                agent_record_id=f"inference-{index}",
                payload=payload,
            )
        )
        processor.ingest(
            AgentRecord.create(
                scenario="mixed",
                request_type=RequestType.REPORT,
                agent_record_id=f"report-{index}",
                payload={"references": [f"inference-{index}"], "score": 1.0},
            )
        )
    batch = processor.build_batch()
    assert processor.build_batch() is batch
    assert isinstance(batch.items[0], TrajectoryItem)
    assert isinstance(batch.items[1], TaskItem)
    ids = frozenset({"inference-0", "report-0", "inference-1", "report-1"})
    assert processor.retention_decision().protected_agent_record_ids == ids
    assert processor.acknowledge(batch.batch_id) == ids
    assert not processor.ready()


@pytest.mark.parametrize("algorithm", ["sao", "openclawrl"])
def test_independent_algorithms_ignore_group_metadata(algorithm):
    batch = TrainingBatch(
        "grouped", tuple(replace(captured_trajectory(str(index)), group_id="a") for index in range(2))
    )
    result = prepare_slime_step(batch, algorithm, {}, SCHEDULING[algorithm])
    assert result.payload["rollout_ids"] == [0, 1]
    assert result.payload["source_rows"] == [0, 1]


def test_extension_updates_preserve_unknown_metadata_and_leave_input_unchanged():
    original = atif_item()
    updated = original.with_training(tokens=[1, 2]).with_metadata(reward=0.5)
    assert updated.trajectory["extra"]["external"] == {"keep": True}
    assert original.training == {}
    assert original.metadata["reward"] == 1.0


def test_atif_multiturn_tools_preserve_history_observations_and_provider_fields():
    user = {"role": "user", "content": "Read the file"}
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "read-1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a.txt"}'}}
        ],
    }
    result = {"role": "tool", "tool_call_id": "read-1", "content": "file contents"}
    first = {"messages": [user], "response": {"choices": [{"message": call}]}, "opaque": {"x": 7}}
    second = {
        "messages": [user, call, result],
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "Done", "reasoning_content": "File read"}}]
        },
    }
    item = recorded_trajectory("record", second, 1.0, trajectory=(first, second))
    steps = item.trajectory["steps"]
    assert [step["step_id"] for step in steps] == [1, 2, 3]
    assert [step["source"] for step in steps] == ["user", "agent", "agent"]
    assert steps[1]["tool_calls"] == [
        {"tool_call_id": "read-1", "function_name": "read", "arguments": {"path": "a.txt"}}
    ]
    assert steps[1]["observation"]["results"][0]["content"] == "file contents"
    assert steps[1]["observation"]["results"][0]["source_call_id"] == "read-1"
    assert steps[2]["reasoning_content"] == "File read"
    assert "is_copied_context" not in steps[1]
    assert recorded_payloads(TrajectoryItem(json.loads(json.dumps(item.trajectory)))) == (first, second)


@pytest.mark.parametrize("provider", ["anthropic", "responses"])
def test_atif_provider_tool_calls_and_multimodal_context(provider):
    if provider == "anthropic":
        payload = {
            "system": "Be precise",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call-1", "name": "inspect", "input": {"x": 1}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "ok"}]},
            ],
            "response": {
                "content": [{"type": "thinking", "thinking": "Looks good"}, {"type": "text", "text": "Done"}]
            },
        }
    else:
        payload = {
            "instructions": "Be precise",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                    ],
                },
                {"type": "function_call", "call_id": "call-1", "name": "inspect", "arguments": '{"x":1}'},
                {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
            ],
            "response": {
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]}
                ]
            },
        }
    item = recorded_trajectory("record", payload, 1.0)
    steps = item.trajectory["steps"]
    assert [step["source"] for step in steps] == ["system", "user", "agent", "agent"]
    assert steps[1]["message"][1] == {
        "type": "image",
        "source": {"media_type": "image/png", "path": "data:image/png;base64,AAAA"},
    }
    assert steps[2]["is_copied_context"] is True
    assert steps[2]["tool_calls"][0]["arguments"] == {"x": 1}
    assert steps[2]["observation"]["results"][0]["content"] == "ok"
    assert steps[3]["message"] == "Done"
    assert recorded_payload(item) == payload


def test_external_atif_projection_preserves_images_reasoning_and_tool_results():
    document = dict(atif_item().trajectory)
    document["steps"] = [
        {
            "step_id": 1,
            "source": "user",
            "message": [
                {"type": "text", "text": "Inspect"},
                {"type": "image", "source": {"media_type": "image/png", "path": "data:image/png;base64,AAAA"}},
            ],
        },
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "reasoning_content": "Read it first",
            "tool_calls": [{"tool_call_id": "read-1", "function_name": "read", "arguments": {"path": "a"}}],
            "observation": {"results": [{"source_call_id": "read-1", "content": "ok"}]},
        },
        {"step_id": 3, "source": "agent", "message": "Done"},
    ]
    payload = recorded_payload(TrajectoryItem(document))
    assert payload["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    assert payload["messages"][1]["reasoning_content"] == "Read it first"
    assert payload["messages"][1]["tool_calls"][0]["function"] == {"name": "read", "arguments": '{"path": "a"}'}
    assert payload["messages"][2] == {"role": "tool", "tool_call_id": "read-1", "content": "ok"}
    assert payload["response"]["choices"][0]["message"]["content"] == "Done"
