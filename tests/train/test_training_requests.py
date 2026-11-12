"""Training requests in the training request queue (RFC #1054): filed once, given to trainers, settled by commits."""

from __future__ import annotations

import pytest
from tests.support.records import memory_records
from tests.support.runtime_stubs import STUB_ROOT, RecordingComponent
from tests.support.samples_service import make_samples

from reef.core.training_request import normalize_training_request_payload
from reef.pipeline_context import PipelineContext
from reef.sample2job import Sample2Job
from reef.storage.records import RecordConflict
from reef.storage.sql_coordination import memory_coordination
from reef.storage.work_queue import NewWork
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import Trainer
from reef.train.training_requests import TRAINING_REQUESTS_QUEUE, TrainingRequestQueue, training_request_kind
from reef.train.types import JobData


def payload(text: str) -> dict:
    return normalize_training_request_payload({"text": text, "session": "s-1", "artifact_id": "r-1"})


def test_a_request_is_filed_once_and_another_payload_under_its_id_conflicts() -> None:
    training_requests = TrainingRequestQueue(memory_coordination().work_queue).of("s")
    first = training_requests.file("one", payload("one"))
    assert training_requests.file("one", payload("one")) == first
    with pytest.raises(RecordConflict, match="different content"):
        training_requests.file("one", payload("other"))
    assert [request.id for request in training_requests.queued()] == ["one"]
    assert training_requests.get("one") == first


def test_settling_finishes_the_consumed_requests_oldest_first_and_leaves_other_scenarios() -> None:
    queue = TrainingRequestQueue(memory_coordination().work_queue)
    training_requests, other = queue.of("s"), queue.of("t")
    for name in ("one", "two", "three"):
        training_requests.file(name, payload(name))
    other.file("elsewhere", payload("elsewhere"))
    # "three" is consumed but "two" is not: settling stops at the oldest that no commit consumed.
    training_requests.settle({"one", "three"})
    assert [request.id for request in training_requests.queued()] == ["two", "three"]
    assert training_requests.get("one").final and not training_requests.get("three").final
    training_requests.settle({"two", "three"})
    assert training_requests.queued() == ()
    assert [request.id for request in other.queued()] == ["elsewhere"]


def test_a_trainer_gets_the_queued_requests_and_a_commit_settles_the_one_it_consumed() -> None:
    class RequestJobData(Sample2Job):
        supported_training_modes = frozenset({"manual"})
        reads_requests = True

        def make_job_data(self, samples, data_id):
            return JobData(data_id, samples)

    records = memory_records("s")
    component = RecordingComponent()
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=lambda context: TrainingPipeline(context, sample2job=RequestJobData(context, None, 1)),
        job_loop=JobLoop(component, scenario="s"),
        training_mode="manual",
        root=STUB_ROOT,
    )
    training_requests = TrainingRequestQueue(memory_coordination().work_queue).of("s")
    try:
        training_requests.file("one", payload("one"))
        training_requests.file("two", payload("two"))
        training_requests.give([trainer])
        assert len(trainer.pipeline.training_requests) == 2
        make_samples(trainer)
        # The commit that submits a job consumes its request; the job's end is the next commit.
        trainer.settle_commit(trainer.prepare_commit(trainer.advance_job()))
        trainer.settle_commit(trainer.prepare_commit(trainer.advance_job()))
        # The next give finishes the consumed request's item and keeps the other queued.
        training_requests.give([trainer])
        assert training_requests.get("one").final and [request.id for request in training_requests.queued()] == ["two"]
        trainer.settle_commit(trainer.prepare_commit(trainer.advance_job()))
        trainer.settle_commit(trainer.prepare_commit(trainer.advance_job()))
        training_requests.give([trainer])
        assert [batch.request_id for batch in component.batches] == ["one", "two"] and training_requests.queued() == ()
        assert trainer.pipeline.training_requests == {}
    finally:
        trainer.close()
        records.close()


def test_a_new_pipeline_reads_a_stored_request_failure_and_a_consumed_request_leaves_with_it() -> None:
    requests = TrainingRequestQueue(memory_coordination().work_queue).of("s")
    requests.file("poison", payload("poison"))
    requests.file("fine", payload("fine"))
    requests.mark_failed("poison", "RuntimeError: poison method")
    assert requests.failures() == {"poison": "RuntimeError: poison method"}

    restarted = TrainingPipeline(PipelineContext("s"))
    restarted.queue_requests(requests.queued(), requests.failures())
    assert restarted.request_failure("poison") == "RuntimeError: poison method"
    assert restarted.request_failure("fine") is None

    requests.settle(["poison"])
    assert requests.failures() == {}
    restarted.queue_requests(requests.queued(), requests.failures())
    assert restarted.request_failure("poison") is None


def test_retrying_a_request_written_before_artifact_identity_keeps_its_item():
    queue = memory_coordination().work_queue
    legacy = {"text": "one", "session": "s-1", "artifact_id": "r-1", "requires": []}
    original = queue.add(
        NewWork(TRAINING_REQUESTS_QUEUE, "one", "s", training_request_kind("s"), legacy, serial_key="s")
    )
    training_requests = TrainingRequestQueue(queue).of("s")
    assert training_requests.file("one", payload("one")) == original
    assert training_requests.queued()[0].artifact_id == "r-1"
    with pytest.raises(RecordConflict, match="different content"):
        training_requests.file("one", payload("changed"))
