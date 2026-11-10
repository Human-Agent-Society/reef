"""The evaluation service's worker: rollout items, candidate endpoints and failures."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from reef_client.serve import CapturedTurn

from reef.core.errors import ReefError
from reef.core.evaluation import EvaluationCall, EvaluationTrajectory, PolicyModel
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
    run_while_held,
    verifier_output,
)
from reef.evaluation.trials import EvaluationTrial, TrialObserver
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue, memory_coordination
from reef.storage.work_queue import NewWork, WorkLeaseLost


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
        self.trajectories: list[EvaluationTrajectory] = []
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

    def add_trajectory(self, rollout: EvaluationRollout, trajectory: EvaluationTrajectory) -> None:
        self.trajectories.append(trajectory)


class OpenEndpoints(UnreachableEndpoints):
    """Endpoints that open an address that no agent calls, for tasks under ``root``."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def task_directory(self, rollout: EvaluationRollout) -> Path:
        task = self.root / rollout.task_name
        task.mkdir(parents=True, exist_ok=True)
        (task / "task.toml").write_text("")
        return task

    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        yield CandidateEndpoint("http://127.0.0.1:9", None)


class NoSandboxLab(TaskLab):
    """A lab whose trial writes its exception to its trial directory under ``root`` and ends with no call."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        trial = self.root / key
        trial.mkdir(parents=True)
        (trial / "exception.txt").write_text("SandboxError: the image did not build")
        return RolloutRow(rewards={}, error="the sandbox did not start", trial_uri=trial.as_uri())

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


class ReadingObserver(TrialObserver):
    """Keeps each trial and the exception text in its directory; raises after it keeps them when ``failing``."""

    def __init__(self, failing: bool = False) -> None:
        self.trials: list[tuple[EvaluationTrial, str]] = []
        self.failing = failing

    def observe(self, trial: EvaluationTrial) -> None:
        assert trial.directory is not None
        self.trials.append((trial, (trial.directory / "exception.txt").read_text()))
        if self.failing:
            raise OSError("the log store is gone")


@pytest.mark.parametrize("failing", [False, True])
def test_the_trial_observer_reads_a_failed_trial_and_its_failure_changes_nothing(tmp_path, caplog, failing):
    observer = ReadingObserver(failing)
    worker = ServedWorker(
        memory_coordination().work_queue,
        OpenEndpoints(tmp_path / "tasks"),
        worker_id="w",
        lab=NoSandboxLab(tmp_path / "trials"),
        trial_observer=observer,
    )
    with pytest.raises(ReefError, match="the rollout reached no model: the sandbox did not start"):
        worker.score(rollout(), threading.Event(), 2)
    [(trial, exception)] = observer.trials
    assert trial == EvaluationTrial("s", "e1", 2, "add", 0, 2, trial.directory, "the sandbox did not start")
    assert exception == "SandboxError: the image did not build"
    failed = "the trial observer failed on rollout evaluation_rollout:e1:00002:00000"
    assert (failed in caplog.text) == failing


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
        "caller": "agent",
        "label": None,
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


ATIF = {"schema_version": "ATIF-v1.4", "agent": {"name": "judge"}, "steps": [{"source": "user"}]}


def judge_call(**fields: object) -> str:
    reply = {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7}}
    return json.dumps({"request": {"messages": []}, "response": reply, "started_at": 1, "duration_s": 0.5, **fields})


@pytest.mark.unit
def test_the_verifier_output_keeps_valid_calls_and_trajectories_and_skips_the_rest(tmp_path, monkeypatch):
    judge = tmp_path / "verifier" / "judge"
    judge.mkdir(parents=True)
    lines = [
        judge_call(label="judge:fidelity"),
        "",
        judge_call(status=503, response={"error": "busy"}),
        "[1, 2]",
        judge_call(started_at="noon"),
        judge_call(path=7),
        judge_call(label=""),
        judge_call(),
    ]
    (judge / "calls.jsonl").write_text("\n".join(lines))
    (judge / "trajectory-b.json").write_text(json.dumps(ATIF))
    (judge / "trajectory-a.json").write_text('{"schema_version": "ATIF-v1.4"}')
    (judge / "trajectory-c.json").symlink_to(tmp_path / "secret.json")
    monkeypatch.setattr("reef.evaluation.rollout_worker.MAX_VERIFIER_CALLS", 2)

    calls, trajectories, skipped = verifier_output(tmp_path, rollout(), 2)

    assert [
        (call.summary.index, call.summary.status, call.summary.label, call.summary.prompt_tokens) for call in calls
    ] == [(0, 200, "judge:fidelity", 7), (1, 503, None, None)]
    assert {call.summary.caller for call in calls} == {"verifier"}
    assert [(trajectory.index, trajectory.name) for trajectory in trajectories] == [(0, "trajectory-b.json")]
    # Four bad lines, one call over the limit, a file that is no ATIF document and a symbolic link.
    assert skipped == 7
    assert verifier_output(None, rollout(), 1) == ([], [], 0)
    assert verifier_output(tmp_path / "missing", rollout(), 1) == ([], [], 0)


@pytest.mark.unit
def test_the_verifier_output_skips_a_file_over_the_byte_limit(tmp_path, monkeypatch):
    judge = tmp_path / "verifier" / "judge"
    judge.mkdir(parents=True)
    (judge / "calls.jsonl").write_text(judge_call())
    (judge / "trajectory.json").write_text(json.dumps(ATIF))
    monkeypatch.setattr("reef.evaluation.rollout_worker.MAX_VERIFIER_BYTES", len(judge_call()) + 10)
    calls, trajectories, skipped = verifier_output(tmp_path, rollout(), 1)
    assert (len(calls), trajectories, skipped) == (1, [], 1)


@pytest.mark.unit
def test_a_verifier_output_that_cannot_be_kept_is_logged_and_changes_nothing(tmp_path, caplog):
    judge = tmp_path / "verifier" / "judge"
    judge.mkdir(parents=True)
    (judge / "calls.jsonl").write_text("\n".join([judge_call(), "{}", judge_call()]))
    endpoints = UnreachableEndpoints(refused_index=0)
    worker = ServedWorker(memory_coordination().work_queue, endpoints, worker_id="w")
    worker.keep_verifier_output(rollout(), 1, tmp_path)
    assert [call.summary.index for call in endpoints.calls] == [1]
    assert "skipped 1 verifier lines or files of rollout evaluation_rollout:e1:00002:00000" in caplog.text
    assert "could not keep verifier call 0 of rollout evaluation_rollout:e1:00002:00000" in caplog.text


@pytest.mark.unit
def test_a_running_trial_is_cancelled_once_the_worker_loses_the_rollout(monkeypatch):
    monkeypatch.setattr("reef.evaluation.rollout_worker.LEASE_CHECK_S", 0.01)
    cancelled = threading.Event()

    class EndlessRunner:
        async def run_rollouts_async(self, requests, slots):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    lease_lost = threading.Event()
    threading.Timer(0.1, lease_lost.set).start()
    with pytest.raises(WorkLeaseLost):
        asyncio.run(run_while_held(EndlessRunner(), "request", lease_lost))
    assert cancelled.is_set()
