"""The evaluation service's worker: rollout items, candidate endpoints and failures."""

from __future__ import annotations

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
from reef.storage.work_queue import MemoryWorkQueue, NewWork


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
    def endpoint(self, rollout: EvaluationRollout) -> CandidateEndpoint:
        raise ConnectionError("the inference service is down")


@pytest.mark.unit
def test_a_rollout_item_round_trips_and_names_its_place_in_the_evaluation():
    value = rollout(repeat=1)
    queue = MemoryWorkQueue()
    item = queue.add(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    assert item.id == rollout_item_id("e1", 2, 1) == "evaluation_rollout:e1:00002:00001"
    assert EvaluationRollout.from_item(item) == value


@pytest.mark.unit
def test_a_rollout_calls_the_candidate_at_its_unrecorded_route_on_the_inference_service():
    endpoints = CandidateEndpoints("http://inference:8000/", "rollout-token")
    assert endpoints.endpoint(rollout()) == CandidateEndpoint(
        "http://inference:8000/reef/scenarios/s/artifacts/artifact-1/evaluation", "rollout-token"
    )
    with pytest.raises(ValueError, match="inference service's URL"):
        CandidateEndpoints("", None)


@pytest.mark.unit
def test_a_rollout_that_cannot_reach_its_candidate_is_retried_then_fails():
    queue = MemoryWorkQueue(max_attempts={EVALUATIONS_QUEUE: 2})
    value = rollout()
    queue.add(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    worker = EvaluationRolloutWorker(queue, UnreachableEndpoints("http://unused", None), worker_id="evaluation-1")

    assert worker.run_once() is True
    assert queue.get(EVALUATIONS_QUEUE, value.item_id()).state == "queued"
    assert worker.run_once() is True
    failed = queue.get(EVALUATIONS_QUEUE, value.item_id())
    assert failed.state == "failed" and "inference service is down" in failed.error
    assert worker.run_once() is False
