"""An assembled sample carries the task its report names, so a recipe can group samples by task."""

from __future__ import annotations

from reef.core import AgentRecord, RequestType
from reef.core.trajectories import make_trajectory
from reef.train.processors.reported import ReportContext, SampleAssembly
from reef.train.types import TrajectoryItem

TASK = {"name": "openenv-00012-003-deduction", "path": "/tasks/openenv-00012-003-deduction", "digest": "ab" * 32}


def inference(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="play", request_type=RequestType.INFERENCE, payload={}, agent_record_id=record_id
    )


def report(*references: str, metadata: dict[str, object] | None = None) -> AgentRecord:
    payload: dict[str, object] = {"score": 1.0, "feedback": "verifier reward 1.0", "references": list(references)}
    if metadata is not None:
        payload["metadata"] = metadata
    return AgentRecord.create(
        scenario="play", request_type=RequestType.REPORT, payload=payload, agent_record_id="rep-1"
    )


def plain_sample(record: AgentRecord, score: float) -> TrajectoryItem:
    return make_trajectory((record,), score)


def assembled(metadata: dict[str, object] | None) -> TrajectoryItem:
    assembly = SampleAssembly(make_sample=plain_sample)
    record = inference("rec-1")
    context = ReportContext(report=report("rec-1", metadata=metadata), score=1.0, inferences=(record,))
    return assembly.build(context, 1.0)


def test_a_report_that_names_its_task_puts_it_on_the_sample() -> None:
    sample = assembled({"task": TASK, "episode": {"id": "e1", "labels": {"arm": "hint"}}})
    assert sample.metadata["task"] == TASK
    assert sample.metadata["report_agent_record_id"] == "rep-1" and sample.metadata["references"] == ["rec-1"]


def test_a_report_without_a_task_leaves_the_sample_as_before() -> None:
    for metadata in (
        None,
        {},
        {"task": "not a mapping"},
        {"task": {}},
        {"task": {"name": ""}},
        {"task": {"path": "/tasks/x"}},
        {"episode": {"id": "e1"}},
        {"harbor": {"trial_id": "t1"}},
    ):
        sample = assembled(metadata)
        assert "task" not in sample.metadata and sample.metadata["feedback"] == "verifier reward 1.0"


def test_a_harbor_report_names_its_task_the_way_the_shipped_harness_writes_it() -> None:
    sample = assembled({"harbor": {"trial_id": "run42:7", "task_name": "hello-world"}})
    assert sample.metadata["task"] == {"name": "hello-world"}
    both = assembled({"task": TASK, "harbor": {"trial_id": "run42:7", "task_name": "other"}})
    assert both.metadata["task"] == TASK, "an explicit task wins over the harbor trial's name"
