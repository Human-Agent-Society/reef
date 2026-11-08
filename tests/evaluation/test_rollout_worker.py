"""The evaluation service's worker: rollout items, candidate endpoints and failures."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from reef_client.serve import CapturedTurn

from reef.core.evaluation import EvaluationCall, PolicyModel
from reef.core.served_models import ServedArtifact
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    CandidateEndpoint,
    CandidateEndpoints,
    EvaluationRollout,
    EvaluationRolloutWorker,
    RolloutCalls,
    evaluation_call,
    reply_summary,
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
        repeat,
        PolicyModel("m", "a"),
        {"name": "x"},
        "docker",
        {"weights": "artifact-1"},
    )


class ServedWorker(EvaluationRolloutWorker):
    """A worker whose candidate needs no start."""

    def served_candidate(self, rollout: EvaluationRollout, lease_lost: threading.Event) -> ServedArtifact | None:
        return None


class UnreachableEndpoints(CandidateEndpoints):
    """Endpoints that cannot serve the candidate; they keep each call, and refuse the call at ``refused_index``."""

    def __init__(self, refused_index: int | None = None) -> None:
        self.calls: list[EvaluationCall] = []
        self.refused_index = refused_index

    def task_directory(self, rollout: EvaluationRollout) -> Path:
        return Path("/tasks") / rollout.task_name

    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        raise ConnectionError("the candidate cannot be served")
        yield CandidateEndpoint("http://unused", None)

    def add_call(self, rollout: EvaluationRollout, call: EvaluationCall) -> None:
        if call.summary.index == self.refused_index:
            raise OSError("the database is gone")
        self.calls.append(call)


def turn(path: str, status: int, response: dict[str, object] | None) -> CapturedTurn:
    return CapturedTurn("", path, status, {"messages": []}, response, None, False, 2.0, at=100.0)


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
    worker = ServedWorker(queue, UnreachableEndpoints(), worker_id="evaluation-1")

    assert worker.run_once() is True
    assert queue.get(EVALUATIONS_QUEUE, value.item_id()).state == "queued"
    assert worker.run_once() is True
    failed = queue.get(EVALUATIONS_QUEUE, value.item_id())
    assert failed.state == "failed" and "cannot be served" in failed.error
    assert worker.run_once() is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "reply", "expected"),
    [
        (
            "/v1/chat/completions",
            {
                "choices": [{"message": {"tool_calls": [{"id": "a"}, {"id": "b"}]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 2100, "completion_tokens": 340},
            },
            (2100, 340, "tool_calls", 2),
        ),
        (
            "/v1/responses",
            {
                "status": "completed",
                "output": [{"type": "reasoning"}, {"type": "function_call"}, {"type": "message"}],
                "usage": {"input_tokens": 50, "output_tokens": 7},
            },
            (50, 7, "completed", 1),
        ),
        (
            "/v1/responses",
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
            (None, None, "max_output_tokens", 0),
        ),
        (
            "/v1/messages?beta=true",
            {
                "content": [{"type": "text"}, {"type": "tool_use"}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 3, "cache_read_input_tokens": 40, "output_tokens": 9},
            },
            (43, 9, "tool_use", 1),
        ),
        ("/v1/chat/completions", {"choices": "none"}, (None, None, None, 0)),
        ("/v1/embeddings", {"usage": {"prompt_tokens": 4}}, (None, None, None, 0)),
    ],
)
def test_a_reply_summary_reads_each_model_api_dialect(path, reply, expected):
    assert reply_summary(path, reply) == expected


@pytest.mark.unit
def test_a_captured_call_becomes_an_evaluation_call_and_a_failed_one_keeps_its_error_body():
    answered = evaluation_call(
        turn("/v1/chat/completions", 200, {"choices": [{"finish_reason": "stop"}]}), rollout(repeat=1), 2, 5
    )
    assert answered.summary.to_json() == {
        "attempt": 2,
        "index": 5,
        "path": "/v1/chat/completions",
        "status": 200,
        "started_at": 98.0,
        "duration_s": 2.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": "stop",
        "tool_calls": 0,
    }
    assert (answered.summary.task_index, answered.summary.repeat) == (2, 1)
    error = {"error": {"message": "the engine stopped"}}
    failed = evaluation_call(turn("/v1/chat/completions", 502, error), rollout(), 1, 0)
    assert failed.summary.finish_reason is None and failed.summary.tool_calls == 0
    assert failed.request == {"messages": []} and failed.response == error
    trained = evaluation_call(
        turn("/v1/chat/completions", 200, {"choices": [], "training": {"ids": [1]}}), rollout(), 1, 0
    )
    assert trained.response == {"choices": []}


@pytest.mark.unit
def test_rollout_calls_number_the_calls_in_order_and_a_refused_call_does_not_stop_the_rollout(caplog):
    endpoints = UnreachableEndpoints(refused_index=1)
    calls = RolloutCalls(endpoints, rollout(), 3)
    for status in (200, 500, 200):
        calls.call_captured(turn("/v1/chat/completions", status, None))
    calls.close()
    assert [(call.summary.attempt, call.summary.index, call.summary.status) for call in endpoints.calls] == [
        (3, 0, 200),
        (3, 2, 200),
    ]
    assert "could not keep call 1 of rollout evaluation_rollout:e1:00002:00000" in caplog.text
