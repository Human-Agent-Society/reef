"""The evaluation service's worker: rollout items, candidate endpoints and failures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from reef.core.evaluation import PolicyModel
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    CandidateEndpoint,
    CandidateEndpoints,
    EvaluationRollout,
    EvaluationRolloutWorker,
    rollout_item_id,
)
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue, memory_coordination
from reef.storage.work_queue import NewWork


def rollout(repeat: int = 0) -> EvaluationRollout:
    return EvaluationRollout(
        "s",
        "e1",
        "artifact-1",
        "weights",
        2,
        "add",
        "/tasks/add",
        repeat,
        PolicyModel("m", "a"),
        {"name": "x"},
        "docker",
    )


class UnreachableEndpoints(CandidateEndpoints):
    @contextmanager
    def open(self, rollout: EvaluationRollout) -> Iterator[CandidateEndpoint]:
        raise ConnectionError("the candidate cannot be served")
        yield CandidateEndpoint("http://unused", None)


@pytest.mark.unit
def test_a_rollout_item_round_trips_and_names_its_place_in_the_evaluation():
    value = rollout(repeat=1)
    queue = memory_coordination().work_queue
    item = queue.add(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    assert item.id == rollout_item_id("e1", 2, 1) == "evaluation_rollout:e1:00002:00001"
    assert EvaluationRollout.from_item(item) == value


@pytest.mark.unit
def test_a_rollout_that_cannot_reach_its_candidate_is_retried_then_fails():
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 2})
    value = rollout()
    queue.add(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    worker = EvaluationRolloutWorker(queue, UnreachableEndpoints(), worker_id="evaluation-1")

    assert worker.run_once() is True
    assert queue.get(EVALUATIONS_QUEUE, value.item_id()).state == "queued"
    assert worker.run_once() is True
    failed = queue.get(EVALUATIONS_QUEUE, value.item_id())
    assert failed.state == "failed" and "cannot be served" in failed.error
    assert worker.run_once() is False
