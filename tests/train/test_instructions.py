"""Training instructions in the instructions queue (RFC #1054): filed once, given to trainers, settled by commits."""

from __future__ import annotations

import pytest
from tests.support.runtime_stubs import RecordingComponent
from tests.support.samples_service import make_samples

from reef.core.training_request import normalize_training_request_payload
from reef.sample2job import Sample2Job
from reef.storage.records import RecordConflict
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.instructions import InstructionQueue
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import Trainer
from reef.train.types import JobData


def payload(text: str) -> dict:
    return normalize_training_request_payload({"text": text, "session": "s-1", "release_id": "r-1"})


def test_an_instruction_is_filed_once_and_another_payload_under_its_id_conflicts() -> None:
    instructions = InstructionQueue(memory_coordination().work_queue).of("s")
    first = instructions.file("one", payload("one"))
    assert instructions.file("one", payload("one")) == first
    with pytest.raises(RecordConflict, match="different content"):
        instructions.file("one", payload("other"))
    assert [request.id for request in instructions.queued()] == ["one"]
    assert instructions.get("one") == first


def test_settling_finishes_the_consumed_instructions_oldest_first_and_leaves_other_scenarios() -> None:
    queue = InstructionQueue(memory_coordination().work_queue)
    instructions, other = queue.of("s"), queue.of("t")
    for name in ("one", "two", "three"):
        instructions.file(name, payload(name))
    other.file("elsewhere", payload("elsewhere"))
    # "three" is consumed but "two" is not: settling stops at the oldest that no commit consumed.
    instructions.settle({"one", "three"})
    assert [request.id for request in instructions.queued()] == ["two", "three"]
    assert instructions.get("one").final and not instructions.get("three").final
    instructions.settle({"two", "three"})
    assert instructions.queued() == ()
    assert [request.id for request in other.queued()] == ["elsewhere"]


def test_a_trainer_gets_the_queued_instructions_and_a_commit_settles_the_one_it_consumed() -> None:
    class InstructionJobData(Sample2Job):
        supported_training_modes = frozenset({"manual"})
        reads_instructions = True

        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    records = SQLiteRecordStore()
    component = RecordingComponent()
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=lambda context: TrainingPipeline(context, sample2job=InstructionJobData(context, None, 1)),
        job_loop=JobLoop(component, scenario="s"),
        training_mode="manual",
    )
    instructions = InstructionQueue(memory_coordination().work_queue).of("s")
    try:
        instructions.file("one", payload("one"))
        instructions.file("two", payload("two"))
        instructions.give([trainer])
        assert trainer.pending_instructions() == 2
        make_samples(trainer)
        # The commit that submits a job consumes its instruction; the job's end is the next commit.
        trainer.commit(trainer.prepare_commit(trainer.run_once()))
        trainer.commit(trainer.prepare_commit(trainer.run_once()))
        # The next give finishes the consumed instruction's item and keeps the other queued.
        instructions.give([trainer])
        assert instructions.get("one").final and [request.id for request in instructions.queued()] == ["two"]
        trainer.commit(trainer.prepare_commit(trainer.run_once()))
        trainer.commit(trainer.prepare_commit(trainer.run_once()))
        instructions.give([trainer])
        assert [batch.request.id for batch in component.batches] == ["one", "two"] and instructions.queued() == ()
        assert trainer.pending_instructions() == 0
    finally:
        trainer.close()
        records.close()
