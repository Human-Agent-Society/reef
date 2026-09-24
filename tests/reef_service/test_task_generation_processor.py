"""Contracts for task generation hooks, without a background execution engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.train.processors import TaskGenerationProcessor
from reef.train.processors.task_generation import TaskGenerationRequest, TaskValidationResult
from reef.train.types import ProcessorContext, TaskItem, TrainingBatch

pytestmark = pytest.mark.unit


class GenerationOnlyProcessor(TaskGenerationProcessor):
    async def generate(self, request: TaskGenerationRequest) -> HarborTask:
        return HarborTask(
            name="generated-task",
            instruction=request.description,
            environment={"Dockerfile": (request.assets[0] / "Dockerfile").read_text()},
            tests={"test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"},
            source_agent_record_ids=tuple(record.agent_record_id for record in request.source_records),
        )


class ValidationOnlyProcessor(TaskGenerationProcessor):
    async def validate(self, task_path: Path) -> TaskValidationResult:
        task = read_harbor_task(task_path)
        if "solve.sh" not in task.solution:
            return TaskValidationResult(errors=("reference solution is missing",))
        return TaskValidationResult()


class ExampleTaskProcessor(GenerationOnlyProcessor):
    async def validate(self, task_path: Path) -> TaskValidationResult:
        read_harbor_task(task_path)
        return TaskValidationResult()


@pytest.fixture
def source_record() -> AgentRecord:
    return AgentRecord.create(scenario="tasks", request_type=RequestType.INFERENCE, payload={})


@pytest.mark.parametrize(
    "processor_class", [TaskGenerationProcessor, GenerationOnlyProcessor, ValidationOnlyProcessor]
)
def test_both_hooks_are_required(processor_class: type[TaskGenerationProcessor]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        processor_class(ProcessorContext("tasks"))


def test_async_hooks_preserve_sources_and_use_existing_task_types(tmp_path: Path, source_record: AgentRecord) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "Dockerfile").write_text("FROM python:3.12-slim\n")
    request = TaskGenerationRequest((source_record,), "Write a greeting to /workspace/answer.txt.", (assets,))
    processor = ExampleTaskProcessor(ProcessorContext("tasks"))

    task = asyncio.run(processor.generate(request))
    task_path = write_harbor_task(task, tmp_path / "candidates")
    result = asyncio.run(processor.validate(task_path))
    assert result.is_valid
    item = TaskItem(task_path, source_agent_record_ids=task.source_agent_record_ids)
    batch = TrainingBatch("tasks:1", (item,))
    assert batch.items[0].source_agent_record_ids == (source_record.agent_record_id,)
    assert read_harbor_task(task_path).instruction == request.description

    # Hook calls do not yet schedule work, consume records, or create a ready batch.
    processor.ingest(source_record)
    assert not processor.ready()
    assert processor.releasable_record_ids().isdisjoint(frozenset({source_record.agent_record_id}))


@pytest.mark.parametrize("errors", [(), ("oracle failed",), ("oracle failed", "nop agent passed")])
def test_validation_distinguishes_accepted_and_rejected_tasks(errors: tuple[str, ...]) -> None:
    result = TaskValidationResult(errors)
    assert result.is_valid is (not errors)
    assert result.errors == errors


def test_request_rejects_duplicate_and_cross_scenario_sources_and_takes_none(source_record: AgentRecord) -> None:
    other = AgentRecord.create(scenario="other", request_type=RequestType.INFERENCE, payload={})
    for records, message in (
        ((source_record, source_record), "distinct"),
        ((source_record, other), "one scenario"),
    ):
        with pytest.raises(ValueError, match=message):
            TaskGenerationRequest(records, "Generate a task.")
    # A designer prompted with a target alone generates from the description; the request carries no sources.
    assert TaskGenerationRequest((), "Generate a task.").source_records == ()


def test_request_and_rejection_require_explanations(source_record: AgentRecord) -> None:
    with pytest.raises(ValueError, match="description"):
        TaskGenerationRequest((source_record,), " ")
    with pytest.raises(ValueError, match="non-empty strings"):
        TaskValidationResult(errors=(" ",))
