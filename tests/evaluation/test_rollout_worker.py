"""The evaluation service's worker: rollout items, candidate endpoints and failures."""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from reef_client.serve import CapturedTurn
from tests.support.polling import wait_until

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError, UpstreamLimit
from reef.core.evaluation import FAILED, EvaluationCall, EvaluationTrajectory, PolicyModel, RolloutScore
from reef.core.served_models import ServedArtifact
from reef.core.snapshot import SnapshotRef
from reef.evaluation import artifact_endpoint
from reef.evaluation.artifact_endpoint import CandidateArtifacts
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    TASK_TEMPLATE_KIND,
    CandidateEndpoint,
    CandidateEndpoints,
    EvaluationRollout,
    EvaluationRolloutWorker,
    LocalCandidateEndpoints,
    ModelCallRefusal,
    RolloutCalls,
    TaskTemplateWorker,
    TreeHarness,
    check_finish_timeout,
    evaluation_call,
    reply_summary,
    rollout_item_id,
    run_while_held,
    template_item_id,
    verifier_output,
)
from reef.evaluation.task_templates import DockerfileTemplateBuilder, TaskTemplateBuilder
from reef.evaluation.trials import EvaluationTrial, TrialObserver
from reef.execution.sandbox.agentenv.build import AgentEnvBuildError
from reef.execution.workers import ItemFailedError, ServiceWorkers, WorkerStoppedError
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.inference.http import HttpInferenceHandler
from reef.inference.interfaces import InferenceHandler, InferenceRuntime
from reef.storage.coordination.served_models import SERVE_ARTIFACT_KIND, SERVING_QUEUE
from reef.storage.coordination.work_queue import NewWork, WorkLeaseLostError, WorkQueue
from reef.storage.sql.coordination import SQLiteCoordinationDatabase, SQLWorkQueue, memory_coordination


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


def test_each_rollout_opens_its_candidate_with_its_own_session_key(monkeypatch) -> None:
    from reef.evaluation import rollout_worker

    opened = []

    class RecordedEndpoint:
        def __init__(self, *args, session_key: str | None, **kwargs) -> None:
            opened.append(session_key)
            self.url = "http://127.0.0.1:1"

        def __enter__(self) -> RecordedEndpoint:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(rollout_worker, "ArtifactEndpoint", RecordedEndpoint)
    artifacts = Mock(spec=CandidateArtifacts)
    artifacts.candidate.return_value = (Mock(), Mock())
    endpoints = LocalCandidateEndpoints(artifacts)
    for value in (rollout(0), rollout(1), rollout(0)):
        with endpoints.open(value, None):
            pass
    # The repeats of one task are other conversations; an attempt again of one rollout is the same one.
    assert opened == ["evaluation:e1:2:0", "evaluation:e1:2:1", "evaluation:e1:2:0"]


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
    """Task-root endpoints for worker tests, with an address supplied by ``open``.

    Values have a ``Path`` root and the inherited call and trajectory lists, initially empty, with no refused call index.
    ``task_directory`` creates ``root / rollout.task_name``, writes an empty ``task.toml``, and returns that directory.
    File errors propagate. Call and trajectory recording follow ``UnreachableEndpoints``.
    The default ``open`` ignores its inputs and yields ``http://127.0.0.1:9`` with no token or server.
    Subclasses can supply a live address for tests whose agents make calls.
    """

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


class SandboxErrorLab(NoSandboxLab):
    """A lab whose trial of a task in ``tasks`` (of every task when it is None) ends as Harbor records an error of the
    class ``exception_type`` that ended the trial, with the error ``error``. A trial of another task ends as
    ``NoSandboxLab``'s, with no result file."""

    def __init__(self, root: Path, exception_type: str, error: str, tasks: frozenset[str] | None = None) -> None:
        super().__init__(root)
        self.exception_type = exception_type
        self.error = error
        self.tasks = tasks

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        row = await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)
        if self.tasks is not None and Path(task_path).name not in self.tasks:
            return row
        result = {"exception_info": {"exception_type": self.exception_type, "exception_message": self.error}}
        (self.root / key / "result.json").write_text(json.dumps(result))
        return replace(row, error=self.error)


@pytest.mark.unit
def test_a_rollout_whose_sandbox_found_no_capacity_waits_and_its_worker_runs_other_rollouts(tmp_path):
    queue = memory_coordination().work_queue
    # A large task finds no capacity, and a small task after it can start.
    large = replace(rollout(), task_index=0, task_name="large")
    small = replace(rollout(), task_index=1, task_name="small")
    for value in (large, small):
        queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    lab = SandboxErrorLab(tmp_path / "t", "SandboxCapacityError", "no node has 12 vCPUs free", frozenset({"large"}))
    worker = ServedWorker(queue, OpenEndpoints(tmp_path / "tasks"), worker_id="w", lab=lab)
    taken_at = time.time()
    assert worker.run_once()
    waiting = queue.read(EVALUATIONS_QUEUE, large.item_id())
    assert (waiting.state, waiting.error) == ("queued", "waiting for sandbox capacity: no node has 12 vCPUs free")
    assert waiting.available_at >= taken_at + worker.first_deferral_seconds
    assert waiting.deferred_attempt_count == waiting.stopped_attempt_count == waiting.attempt_count == 1
    # The same worker starts the small rollout at once: only the refused rollout waits.
    assert worker.run_once()
    other = queue.read(EVALUATIONS_QUEUE, small.item_id())
    assert (other.attempt_count, other.stopped_attempt_count) == (1, 0)
    assert other.error == "ReefError: the rollout reached no model: the sandbox did not start"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exception_type", "state", "error", "given_back"),
    [
        # A refusal for want of capacity at a sandbox's start, in Reef's names: E2B's and Modal's, and AgentENV
        # Compose's. The rollout waits, and the attempt does not count.
        ("SandboxCapacityError", "queued", "waiting for sandbox capacity: {reason}", 1),
        ("TransientStartError", "queued", "waiting for sandbox capacity: {reason}", 1),
        # A sandbox request that the service refused as invalid: the rollout fails at once, with the reason.
        ("SandboxRequestError", "failed", "the sandbox service refused the sandbox: {reason}", 0),
        # Modal's own error, which comes from after the sandbox's creation, such as from a command of the agent: an
        # ordinary failed attempt, retried as any other.
        ("ResourceExhaustedError", "queued", "ReefError: the rollout reached no model: {reason}", 0),
    ],
    ids=["capacity", "compose-capacity", "invalid-request", "other-provider-error"],
)
def test_the_error_that_ended_a_trial_decides_whether_its_rollout_waits_fails_or_retries(
    tmp_path, exception_type, state, error, given_back
):
    queue = memory_coordination().work_queue
    value = rollout()
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    reason = "the service's reason"
    lab = SandboxErrorLab(tmp_path / "t", exception_type, reason)
    worker = ServedWorker(queue, OpenEndpoints(tmp_path / "tasks"), worker_id="w", lab=lab)
    assert worker.run_once()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.result, item.error) == (state, None, error.format(reason=reason))
    assert (item.attempt_count, item.stopped_attempt_count, item.deferred_attempt_count) == (1, given_back, given_back)


class AnsweringEndpoints(OpenEndpoints):
    """Endpoints for the failed-score test's model call, with the state and task operations of ``OpenEndpoints``.

    Values add no instance state. ``open`` ignores its inputs and starts an HTTP server on an unused loopback port.
    It yields that server's URL with no token; ``Handler`` defines its responses.
    Context exit stops and closes the server, including when the caller raises. Server setup errors propagate.
    """

    @contextmanager
    def open(self, rollout: EvaluationRollout, served: ServedArtifact | None) -> Iterator[CandidateEndpoint]:
        class Handler(BaseHTTPRequestHandler):
            """One request handler, constructed by this endpoint's server with only ``BaseHTTPRequestHandler`` state.

            ``do_POST`` accepts a body with a nonnegative integer ``Content-Length``, or an empty body without that header.
            It reads that many bytes and returns HTTP 200 with JSON body ``{"choices": []}``.
            The response has ``Content-Type: application/json`` and the body's byte length as ``Content-Length``.
            Invalid length conversion and I/O errors propagate. ``log_message`` ignores its arguments and writes nothing.
            Other request handling follows ``BaseHTTPRequestHandler``.
            """

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = b'{"choices": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield CandidateEndpoint(f"http://127.0.0.1:{server.server_port}", None)
        finally:
            server.shutdown()
            server.server_close()


class CalledThenFailedLab(TaskLab):
    """A stateless ``TaskLab`` for the worker test of a failure after a model call.

    ``run`` requires ``agent["kwargs"]`` to be a mapping with an ``api_base`` string naming a reachable HTTP base URL.
    It ignores ``task_path``, ``key``, ``tags``, ``overrides``, and all other agent entries.
    It POSTs ``{"model": "m", "messages": []}`` as JSON to ``{api_base}/chat/completions``, with a 10-second timeout.
    After reading the response, it returns a ``RolloutRow`` with empty rewards and no trial URI.
    Its error is ``VerifierTimeoutError: the verifier ran past 600 s``.
    Input and request errors propagate; the lab keeps no state or trial files.
    ``build_environment`` ignores both inputs and returns ``""`` to simulate a successful build without starting an environment.
    """

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        api_base = cast(Mapping[str, str], agent["kwargs"])["api_base"]
        request = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=b'{"model": "m", "messages": []}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        return RolloutRow(rewards={}, error="VerifierTimeoutError: the verifier ran past 600 s", trial_uri=None)

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


@pytest.mark.unit
def test_a_rollout_that_called_the_model_and_failed_finishes_with_its_reason(tmp_path):
    queue = memory_coordination().work_queue
    value = replace(rollout(), agent={"name": "x", "kwargs": {"api_base": "{base_url}/v1"}})
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    worker = ServedWorker(queue, AnsweringEndpoints(tmp_path / "tasks"), worker_id="w", lab=CalledThenFailedLab())
    assert worker.run_once() is True
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.error) == ("finished", None)
    assert item.result == {
        "score": FAILED,
        "rewards": None,
        "error": "VerifierTimeoutError: the verifier ran past 600 s",
    }


class CalledThenVerifierSandboxErrorLab(CalledThenFailedLab):
    """A lab whose agent calls the model, and whose separate verifier's sandbox then ends the trial with an error of
    the class ``exception_type`` and the error ``error``, as Harbor records it in the trial's ``result.json`` under
    ``root``."""

    def __init__(self, root: Path, exception_type: str, error: str) -> None:
        self.root = root
        self.exception_type = exception_type
        self.error = error

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)
        trial = self.root / key
        trial.mkdir(parents=True)
        result = {"exception_info": {"exception_type": self.exception_type, "exception_message": self.error}}
        (trial / "result.json").write_text(json.dumps(result))
        return RolloutRow(rewards={}, error=self.error, trial_uri=trial.as_uri())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exception_type", "state", "error", "given_back"),
    [
        # It waits for capacity and runs again.
        ("SandboxCapacityError", "queued", "waiting for sandbox capacity: {reason}", 1),
        # The verifier's sandbox request fails the same way again, so the rollout fails at once.
        ("SandboxRequestError", "failed", "the sandbox service refused the sandbox: {reason}", 0),
    ],
    ids=["no-capacity", "invalid-request"],
)
def test_a_rollout_whose_verifier_sandbox_was_refused_gets_no_score(
    tmp_path, exception_type, state, error, given_back
):
    queue = memory_coordination().work_queue
    value = replace(rollout(), agent={"name": "x", "kwargs": {"api_base": "{base_url}/v1"}})
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    reason = "the service's reason"
    lab = CalledThenVerifierSandboxErrorLab(tmp_path / "trials", exception_type, reason)
    worker = ServedWorker(queue, AnsweringEndpoints(tmp_path / "tasks"), worker_id="w", lab=lab)
    assert worker.run_once() is True
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    # The agent called the model, but the rollout gets no score of 0.
    assert (item.state, item.result, item.error) == (state, None, error.format(reason=reason))
    assert (item.attempt_count, item.stopped_attempt_count, item.deferred_attempt_count) == (1, given_back, given_back)


def turn(path: str, status: int, response: dict[str, object] | None) -> CapturedTurn:
    return CapturedTurn("", path, status, {"messages": []}, response, None, False, 2.0, at=100.0)


@pytest.mark.unit
def test_a_rollout_item_round_trips_and_names_its_place_in_the_evaluation():
    value = rollout(repeat=1)
    queue = memory_coordination().work_queue
    item = queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    assert item.id == rollout_item_id("e1", 2, 1) == "evaluation_rollout:e1:00002:00001"
    assert EvaluationRollout.from_item(item) == value


@pytest.mark.unit
def test_a_rollout_that_cannot_reach_its_candidate_is_retried_then_fails():
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 2})
    value = rollout()
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    worker = ServedWorker(queue, UnreachableEndpoints(), worker_id="evaluation-1")

    assert worker.run_once() is True
    assert queue.read(EVALUATIONS_QUEUE, value.item_id()).state == "queued"
    assert worker.run_once() is True
    failed = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert failed.state == "failed" and "cannot be served" in failed.error
    assert worker.run_once() is False


@pytest.mark.unit
def test_the_rollouts_of_one_candidate_ask_inference_to_serve_it_one_at_a_time():
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None))
    worker = EvaluationRolloutWorker(queue, UnreachableEndpoints(), worker_id="evaluation-1")
    served = ServedArtifact("artifact-1", "http://engine", "sampler:artifact-1")
    lease_lost = threading.Event()
    with ThreadPoolExecutor(2) as pool:
        waits = [pool.submit(worker.served_candidate, rollout(repeat), lease_lost) for repeat in (0, 1)]
        wait_until(lambda: len(queue.items(SERVING_QUEUE)) == 2)
        first = queue.take(SERVING_QUEUE, "inference", [SERVE_ARTIFACT_KIND])
        # The second item waits for the first, so inference has a free slot for other work.
        assert first is not None and queue.take(SERVING_QUEUE, "inference", [SERVE_ARTIFACT_KIND]) is None
        queue.finish(SERVING_QUEUE, first.id, "inference", {"served": served.to_json()})
        second = queue.take(SERVING_QUEUE, "inference", [SERVE_ARTIFACT_KIND])
        assert second is not None
        queue.finish(SERVING_QUEUE, second.id, "inference", {"served": served.to_json()})
        assert [wait.result(5) for wait in waits] == [served, served]


@pytest.mark.unit
def test_a_rollout_that_stops_waiting_for_its_candidate_cancels_its_serve_item():
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None))
    worker = EvaluationRolloutWorker(queue, UnreachableEndpoints(), worker_id="evaluation-1")
    lease_lost = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        wait = pool.submit(worker.served_candidate, rollout(), lease_lost)
        wait_until(lambda: len(queue.items(SERVING_QUEUE)) == 1)
        lease_lost.set()
        with pytest.raises(WorkLeaseLostError):
            wait.result(5)
    (item,) = queue.items(SERVING_QUEUE)
    assert item.state == "failed" and "stopped waiting" in item.error
    assert queue.take(SERVING_QUEUE, "inference", [SERVE_ARTIFACT_KIND]) is None


class ServingRefusedWorker(EvaluationRolloutWorker):
    """A worker that fails the test when it asks to serve a candidate."""

    def served_candidate(self, rollout: EvaluationRollout, lease_lost: threading.Event) -> ServedArtifact | None:
        raise AssertionError("a rollout that cannot run serves no candidate")


class ToolTaskEndpoints(OpenEndpoints):
    """Endpoints whose task asks its agent for an SSE MCP server."""

    def task_directory(self, rollout: EvaluationRollout) -> Path:
        task = super().task_directory(rollout)
        (task / "task.toml").write_text(
            '[[environment.mcp_servers]]\nname = "search"\ntransport = "sse"\nurl = "http://search:8000/sse"\n'
        )
        return task


@pytest.mark.unit
def test_a_tree_rollout_whose_task_needs_what_the_agent_lacks_fails_for_good_before_serving(tmp_path):
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 3})
    value = replace(rollout(), agent=None, tree=TreeHarness("codex", "openai-responses", {}))
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    worker = ServingRefusedWorker(queue, ToolTaskEndpoints(tmp_path), worker_id="evaluation-1")

    assert worker.run_once() is True
    failed = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert failed.state == "failed" and failed.attempt_count == 1
    assert failed.error == "task 'add' needs mcp_servers, which adapter 'codex' does not give its agent"


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
    # A captured body that is no JSON object is kept as None, with the call's status.
    for status in (200, 429):
        kept = evaluation_call(turn("/v1/chat/completions", status, []), rollout(), 1, 0)
        assert (kept.summary.status, kept.summary.finish_reason, kept.response) == (status, None, None)


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


RATE, PAYMENT, ACCESS = UpstreamLimit.RATE, UpstreamLimit.PAYMENT, UpstreamLimit.ACCESS
ANSWER = (200, {"choices": [{"message": {"role": "assistant", "content": "5"}, "finish_reason": "stop"}]}, {})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "response", "refusal"),
    [
        (429, {"error": {"code": "rate_limit", "retry_after_seconds": 30}}, ModelCallRefusal(RATE, 429, 30)),
        (402, {"error": {"code": "payment_required"}}, ModelCallRefusal(PAYMENT, 402, None)),
        # A capture of an older endpoint names no code, and its outer 402 still names a payment refusal.
        (402, {"error": {"message": "inference provider returned 402"}}, ModelCallRefusal(PAYMENT, 402, None)),
        # A wait that is no number of seconds, and a body that is no object, name no wait.
        (402, {"error": {"retry_after_seconds": True}}, ModelCallRefusal(PAYMENT, 402, None)),
        (429, [], ModelCallRefusal(RATE, 429, None)),
        # A 401, and a 403 that names no other limit, refuse access.
        (401, {"error": {"message": "invalid key"}}, ModelCallRefusal(ACCESS, 401, None)),
        (403, {"error": {"code": "access_denied"}}, ModelCallRefusal(ACCESS, 403, None)),
        # A 200 answer is no refusal, whatever its body names; nor is another failure.
        (200, {"choices": [], "error": {"code": "payment_required"}}, None),
        (502, {"error": {"message": "the engine stopped"}}, None),
    ],
)
def test_rollout_calls_read_each_refusal_and_a_later_answer_clears_only_a_rate_limit(status, response, refusal):
    calls = RolloutCalls(UnreachableEndpoints(), rollout(), 1)
    calls.call_captured(turn("/v1/chat/completions", status, response))
    assert calls.last_call_refusal == refusal
    calls.call_captured(turn("/v1/chat/completions", 200, ANSWER[1]))
    calls.close()
    kept = None if refusal is None or refusal.limit is RATE else refusal
    assert (calls.all_refusals.last, calls.last_call_refusal) == (kept, None)


# A gateway's wrapping of the provider's 402: its code is text, and the provider's number is in its message.
WRAPPED_402 = (
    402,
    {"error": {"message": 'Provider - {"error":{"code":402}}', "type": None, "param": None, "code": "402"}},
    {},
)
IN_FLIGHT = (
    402,
    {"error": {"code": 402, "metadata": {"limit_source": "openrouter_in_flight_budget"}}},
    {"Retry-After": "1"},
)
RATE_LIMITED_ERROR = re.compile(
    r"the model provider refused \d+ of the rollout's model calls, the last one with HTTP 429 \(rate_limit\)\. "
    r"Retry after (\d+) s\."
)
PAYMENT_ERROR = (
    "the model provider refused 1 of the rollout's model calls, the last one with HTTP 402 (payment_required). "
    "Check the provider balance or key limit, or reduce the request size before retrying."
)
ACCESS_ERROR = (
    "the model provider refused 1 of the rollout's model calls, the last one with HTTP {status} (access_denied). "
    "Check the provider key and its permissions before retrying."
)


@contextmanager
def scripted_provider(replies: list[tuple[int, dict, dict[str, str]]]) -> Iterator[tuple[str, list[dict]]]:
    """A model provider on loopback that answers the calls in order with ``replies``, and with the last one after
    them. Each reply is a status, a JSON body and headers. It yields its URL and the list of each call's JSON body.
    A test can change ``replies`` in place, as a provider whose account gets credits again."""
    bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body, headers = replies[min(len(bodies), len(replies)) - 1]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", bodies
    finally:
        server.shutdown()
        server.server_close()


class ProviderRuntime(InferenceRuntime):
    """A runtime that answers each call of any artifact from the provider at ``base_url``, through Reef's HTTP
    handler, as a hosted sampler does. It loads nothing and serves no route."""

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url=base_url)
        self.handler = HttpInferenceHandler(base_url, timeout_s=10, error_label="inference provider")

    @property
    def inference_handler(self) -> InferenceHandler:
        return self.handler

    @property
    def serves_any_artifact(self) -> bool:
        return True

    def activate_checkpoint(self, artifact, *, scenario=None) -> str:
        return "load-0"

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        return None


class CallingLab(TaskLab):
    """A test ``TaskLab`` that sends a fixed schedule of chat calls to the rollout's endpoint and starts no sandbox.

    Values: ``calls`` is a positive int (number of requests per trial); ``reward`` is the float reward written by each
    trial; ``fails_after_refusal`` is a bool controlling whether a non-200 final status produces an agent error;
    ``tool_calls`` is None (no call carries ``tools``) or a tuple of bools with ``len(tool_calls) == calls`` where
    ``tool_calls[i]`` says whether step ``i`` carries a non-empty ``tools`` list; ``statuses`` records the HTTP status
    of every call across trials in call order; ``runs`` is the non-negative count of ``run`` calls.

    Purpose: rollout-worker tests pass it to ``provider_rollouts`` to simulate an agent's call sequence against a
    scripted provider. ``run`` requires ``agent["kwargs"]["api_base"]``, sends ``calls`` ``POST /chat/completions``
    requests (continuing after non-200 replies), increments ``runs``, and returns
    ``RolloutRow(rewards={"reward": reward}, error=..., trial_uri=None)`` with an ``AgentError`` string iff
    ``statuses[-1] != 200`` and ``fails_after_refusal`` is True, else ``error=None``. ``build_environment`` returns
    ``""``. Construction raises ``ValueError`` if ``calls < 1`` or ``tool_calls`` does not have one bool per step."""

    def __init__(
        self,
        calls: int,
        reward: float,
        *,
        fails_after_refusal: bool = True,
        tool_calls: Sequence[bool] | None = None,
    ) -> None:
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise ValueError("calls is a positive integer")
        schedule: tuple[bool, ...] | None = None
        if tool_calls is not None:
            schedule = tuple(tool_calls)
            if len(schedule) != calls or not all(isinstance(flag, bool) for flag in schedule):
                raise ValueError("tool_calls has one bool for each call")
        self.calls = calls
        self.reward = reward
        self.fails_after_refusal = fails_after_refusal
        self.tool_calls = schedule
        self.statuses: list[int] = []
        self.runs = 0

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.runs += 1
        api_base = cast(Mapping[str, str], agent["kwargs"])["api_base"]
        for step in range(self.calls):
            body: dict[str, object] = {"model": "m", "messages": [{"role": "user", "content": f"step {step}"}]}
            if self.tool_calls is not None and self.tool_calls[step]:
                body["tools"] = [{"type": "function", "function": {"name": "bash"}}]
            request = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.statuses.append(response.status)
            except urllib.error.HTTPError as exc:
                self.statuses.append(exc.code)
        last = self.statuses[-1]
        error = f"AgentError: the model call answered {last}" if last != 200 and self.fails_after_refusal else None
        return RolloutRow(rewards={"reward": self.reward}, error=error, trial_uri=None)

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


def provider_rollouts(
    tmp_path: Path, provider_url: str, lab: TaskLab, repeats: int = 1
) -> tuple[WorkQueue, list[EvaluationRollout], ServedWorker, Mock]:
    """``repeats`` queued rollouts of one task, in queue order, whose candidate the provider at ``provider_url``
    answers through one runtime; the worker that runs them in ``lab`` through the evaluation's own endpoints and
    capture proxy; and the mock candidate store that keeps their calls."""
    task = tmp_path / "tasks" / "add"
    task.mkdir(parents=True)
    (task / "task.toml").write_text("")
    artifacts = Mock(spec=CandidateArtifacts)
    artifacts.candidate.return_value = (
        ProviderRuntime(provider_url),
        Artifact(SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact-1"),
    )
    artifacts.candidate_task_version.return_value = (task, None)
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 3})
    agent = {"name": "x", "kwargs": {"api_base": "{base_url}/v1"}}
    values = [replace(rollout(repeat), agent=agent) for repeat in range(repeats)]
    for value in values:
        queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    return queue, values, ServedWorker(queue, LocalCandidateEndpoints(artifacts), worker_id="w", lab=lab), artifacts


def kept_codes(artifacts: Mock) -> list[tuple[int, object]]:
    """The status and the error code of each call that the rollouts kept, in order."""
    codes: list[tuple[int, object]] = []
    for kept in artifacts.add_evaluation_call.call_args_list:
        call = cast(EvaluationCall, kept.args[3])
        error = (call.response or {}).get("error")
        codes.append((call.summary.status, error.get("code") if isinstance(error, Mapping) else None))
    return codes


@pytest.mark.unit
def test_a_failed_trial_after_a_402_fails_its_rollout_once_and_spares_the_other_rollouts(tmp_path):
    # The second call gets a gateway's 402; the other rollout's calls, and those of the retry, are answered.
    lab = CallingLab(2, reward=0.0)
    with scripted_provider([ANSWER, WRAPPED_402, ANSWER]) as (url, bodies):
        queue, (refused, other), worker, artifacts = provider_rollouts(tmp_path, url, lab, repeats=2)
        assert worker.run_once()
        failed = queue.read(EVALUATIONS_QUEUE, refused.item_id())
        # The agent got the outer 402; the rollout has no score, keeps no provider text, and does not run again.
        assert lab.statuses == [200, 402] and len(bodies) == 2
        assert (failed.state, failed.attempt_count, failed.result) == ("failed", 1, None)
        assert failed.error == PAYMENT_ERROR
        assert kept_codes(artifacts) == [(200, None), (402, "payment_required")]
        # The other rollout of the evaluation still runs, and a retry of the refused one runs it again.
        assert queue.read(EVALUATIONS_QUEUE, other.item_id()).state == "queued"
        lab.reward = 1.0
        assert worker.run_once()
        queue.retry(EVALUATIONS_QUEUE, refused.item_id())
        assert worker.run_once()
    for value, attempts in ((other, 1), (refused, 2)):
        item = queue.read(EVALUATIONS_QUEUE, value.item_id())
        assert (item.state, item.attempt_count, RolloutScore.from_json(item.result["score"]).value) == (
            "finished",
            attempts,
            1.0,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replies", "reward"),
    [([ANSWER, WRAPPED_402], 0.0), ([ANSWER, WRAPPED_402], 1.0), ([WRAPPED_402], 1.0)],
)
def test_a_refused_last_call_fails_the_rollout_even_without_an_agent_error(tmp_path, replies, reward):
    # Each trial exits with no agent error (as OpenCode does on HTTP 402/429), and its verifier writes a reward.
    lab = CallingLab(len(replies), reward=reward, fails_after_refusal=False)
    with scripted_provider(replies) as (url, bodies):
        queue, (value,), worker, _ = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    # The provider got each call once: no 402 is sent again.
    assert len(bodies) == len(replies)
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.attempt_count, item.result, item.error) == ("failed", 1, None, PAYMENT_ERROR)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replies", "error"),
    [
        # A later answer does not clear a refusal that reached the agent.
        ([ANSWER, WRAPPED_402, ANSWER], PAYMENT_ERROR),
        # A key revoked during the rollout: the agent stops on the 401.
        ([ANSWER, (401, {"error": {"message": "invalid key"}}, {})], ACCESS_ERROR.format(status=401)),
        # A 403 that names no spending cap refuses access.
        ([ANSWER, (403, {"error": {"message": "Forbidden"}}, {}), ANSWER], ACCESS_ERROR.format(status=403)),
    ],
)
def test_a_refusal_that_reached_the_agent_fails_the_rollout_once_whatever_calls_follow(tmp_path, replies, error):
    lab = CallingLab(len(replies), reward=0.0)
    with scripted_provider(replies) as (url, bodies):
        queue, (value,), worker, _ = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    # The provider got each call once: the endpoint sends no refusal of payment or access again.
    assert lab.statuses == [status for status, _, _ in replies] and len(bodies) == len(replies)
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.attempt_count, item.result, item.error) == ("failed", 1, None, error)


@pytest.mark.unit
def test_a_rate_limit_that_the_endpoint_sent_again_to_an_answer_keeps_the_score(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_FIRST_WAIT_SECONDS", 0.01)
    busy = (429, {"error": {"message": "busy"}}, {"Retry-After": "0"})
    lab = CallingLab(2, reward=0.0)
    with scripted_provider([ANSWER, busy, ANSWER]) as (url, bodies):
        queue, (value,), worker, _ = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    # The agent never saw the 429, so the verifier's 0 is a score of the model.
    assert lab.statuses == [200, 200] and len(bodies) == 3
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, RolloutScore.from_json(item.result["score"]).value) == ("finished", 0.0)


#: A 429 that the endpoint tries ``RATE_LIMIT_TRIES`` times, so that the agent gets it.
BUSY = [(429, {"error": {"message": "busy"}}, {"Retry-After": "0"})] * artifact_endpoint.RATE_LIMIT_TRIES


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replies", "statuses", "error"),
    [
        # The agent sends a rate-limited call again on its own, so a later answer means that it lost no step.
        ([ANSWER, *BUSY, ANSWER], [200, 429, 200], None),
        ([ANSWER, *BUSY], [200, 429], RATE_LIMITED_ERROR),
        # An agent does not send a refusal of payment or access again, so a later answer does not clear it.
        ([ANSWER, WRAPPED_402, ANSWER], [200, 402, 200], PAYMENT_ERROR),
        ([ANSWER, (401, {"error": {"message": "invalid key"}}, {}), ANSWER], [200, 401, 200], "access_denied"),
    ],
)
def test_a_later_answer_clears_a_rate_limit_that_reached_the_agent_but_no_other_refusal(
    tmp_path, monkeypatch, replies, statuses, error
):
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_FIRST_WAIT_SECONDS", 0.01)
    lab = CallingLab(len(statuses), reward=0.0)
    with scripted_provider(replies) as (url, bodies):
        queue, (value,), worker, _ = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    assert lab.statuses == statuses and len(bodies) == len(replies)
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    if error is None:
        assert (item.state, RolloutScore.from_json(item.result["score"]).value) == ("finished", 0.0)
        return
    assert (item.state, item.attempt_count, item.result) == ("failed", 1, None)
    if isinstance(error, re.Pattern):
        assert error.fullmatch(item.error) is not None
    else:
        assert error in item.error


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replies", "tool_calls", "fails_after_refusal", "expected_state", "expected_error"),
    [
        # Main-loop 402 masked by a trailing 200 OK auxiliary title-generation call (no tools).
        ([WRAPPED_402, ANSWER], [True, False], False, "failed", PAYMENT_ERROR),
        # Earlier main-loop turn succeeded, next main-loop turn hit 402, then trailing 200 OK title call.
        ([ANSWER, WRAPPED_402, ANSWER], [True, True, False], False, "failed", PAYMENT_ERROR),
        # Exhausted 429 on main-loop call masked by a trailing 200 OK auxiliary call.
        (
            [(429, {"error": {"message": "concurrency limit"}}, {"Retry-After": "600"}), ANSWER],
            [True, False],
            False,
            "failed",
            RATE_LIMITED_ERROR,
        ),
        # Auxiliary title call failed with 402 before or after a clean tool-bearing main loop (no trial error): kept.
        ([WRAPPED_402, ANSWER, ANSWER], [False, True, True], False, "finished", None),
        ([ANSWER, ANSWER, WRAPPED_402], [True, True, False], False, "finished", None),
        # Required tool-less step refused after a tool-bearing 200, failing the trial with result.error: rollout fails.
        ([ANSWER, WRAPPED_402], [True, False], True, "failed", PAYMENT_ERROR),
    ],
)
def test_auxiliary_calls_without_tools_do_not_mask_or_trigger_provider_refusal_failures(
    tmp_path, replies, tool_calls, fails_after_refusal, expected_state, expected_error
):
    lab = CallingLab(
        len(replies),
        reward=1.0 if expected_state == "finished" else 0.0,
        fails_after_refusal=fails_after_refusal,
        tool_calls=tool_calls,
    )
    with scripted_provider(replies) as (url, _):
        queue, (value,), worker, _ = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.attempt_count) == (expected_state, 1)
    if expected_state == "finished":
        assert item.error is None
        assert RolloutScore.from_json(item.result["score"]).value == 1.0
        return
    assert item.result is None
    if isinstance(expected_error, re.Pattern):
        assert expected_error.fullmatch(item.error or "") is not None
    else:
        assert item.error == expected_error


def timed_turn(
    status: int, started_at: float, duration_s: float, *, with_tools: bool | list[object] = False
) -> CapturedTurn:
    """A chat call that started at ``started_at`` and answered ``status`` after ``duration_s`` seconds."""
    request: dict[str, object] = {"messages": []}
    if isinstance(with_tools, list):
        request["tools"] = with_tools
    elif with_tools:
        request["tools"] = [{"type": "function", "function": {"name": "bash"}}]
    response: dict[str, object] | None = None
    if status == 402:
        response = {"error": {"code": "payment_required"}}
    elif status == 429:
        response = {"error": {"code": "rate_limit"}}
    return CapturedTurn(
        "", "/v1/chat/completions", status, request, response, None, False, duration_s, at=started_at + duration_s
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("turns", "refused", "expected_answered"),
    [
        # The agent sent each 429 again on its own, and a later main-loop call answered: 1 of 4, 4 of 9, 7 of 10.
        ([(200, True), (429, True), (200, True), (200, True)], None, 3),
        (
            [
                (429, True),
                (429, True),
                (200, True),
                (200, True),
                (429, True),
                (200, True),
                (429, True),
                (200, True),
                (200, True),
            ],
            None,
            5,
        ),
        (
            [
                (429, True),
                (429, True),
                (429, True),
                (429, True),
                (429, True),
                (429, True),
                (200, True),
                (200, True),
                (429, True),
                (200, True),
            ],
            None,
            3,
        ),
        # A trailing 200 OK auxiliary call (no tools or empty tools list) does not clear a 402 on the main loop.
        ([(402, True), (200, False)], (ModelCallRefusal(PAYMENT, 402, None), 1), 0),
        ([(200, True), (402, True), (200, False)], (ModelCallRefusal(PAYMENT, 402, None), 1), 1),
        ([(200, True), (402, True), (200, [])], (ModelCallRefusal(PAYMENT, 402, None), 1), 1),
        # A refused auxiliary call does not fail a main loop that the provider answered.
        ([(402, False), (200, True), (200, True)], None, 2),
        # A 429 with no later main-loop answer counts, also after an auxiliary answer.
        ([(200, True), (429, True)], (ModelCallRefusal(RATE, 429, None), 1), 1),
        ([(429, True), (200, False)], (ModelCallRefusal(RATE, 429, None), 1), 0),
        # An answer clears the 429s before it, but no payment refusal.
        ([(402, True), (429, True), (200, True)], (ModelCallRefusal(PAYMENT, 402, None), 1), 1),
        ([(429, True), (402, True), (429, True)], (ModelCallRefusal(RATE, 429, None), 3), 0),
    ],
)
def test_rollout_calls_count_lasting_refusals_and_unanswered_rate_limits_and_ignore_auxiliary_calls(
    turns, refused, expected_answered
):
    calls = RolloutCalls(UnreachableEndpoints(), rollout(), 1)
    for index, (status, with_tools) in enumerate(turns):
        calls.call_captured(timed_turn(status, 1000.0 + index, 0.2, with_tools=with_tools))
    calls.close()
    expected_refusal = None if refused is None else refused[0].reason(refused[1])
    assert (calls.refusal(trial_failed=False), calls.answered) == (expected_refusal, expected_answered)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("turns", "refused"),
    [
        # A failed trial counts a refused last auxiliary call.
        ([(200, True), (402, False)], (ModelCallRefusal(PAYMENT, 402, None), 1)),
        ([(200, True), (429, False)], (ModelCallRefusal(RATE, 429, None), 1)),
        # It does not count a refused auxiliary call that another call followed.
        ([(402, False), (200, True), (429, False), (200, True)], None),
        ([(402, False), (429, True), (200, True)], None),
        ([(200, False), (200, True)], None),
    ],
)
def test_a_failed_trial_counts_a_refused_last_call_when_no_tool_bearing_refusal_counts(turns, refused):
    calls = RolloutCalls(UnreachableEndpoints(), rollout(), 1)
    for index, (status, with_tools) in enumerate(turns):
        calls.call_captured(timed_turn(status, 1000.0 + index, 0.2, with_tools=with_tools))
    calls.close()
    assert calls.refusal(trial_failed=True) == (None if refused is None else refused[0].reason(refused[1]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("turns", "failure"),
    [
        # No failed call, and a refusal of a provider's limit is no failure of the service.
        ([(200, 0, 10), (429, 10, 1), (200, 11, 10)], None),
        # A short failure in a long run of calls leaves the score: 10 s of 1000 s.
        ([(200, 0, 500), (502, 500, 10), (200, 510, 490)], None),
        # So do a few short failures: 30 s of 1000 s.
        ([(200, 0, 300), (502, 300, 10), (200, 310, 300), (503, 610, 10), (500, 620, 10), (200, 630, 370)], None),
        # A call that waited out the request timeout took much of the time: 300 s of 1000 s.
        (
            [(200, 0, 400), (502, 400, 300), (200, 700, 300)],
            "the inference service failed 1 of 3 model calls, which took 300 s of 1000 s of model calls; "
            "the rollout has no score",
        ),
        # The agent stopped on a failed call, however short it was.
        (
            [(200, 0, 990), (503, 990, 1)],
            "the inference service failed 1 of 2 model calls, the last one among them, which took 1 s of 991 s of "
            "model calls; the rollout has no score",
        ),
    ],
)
def test_rollout_calls_name_a_service_failure_that_took_the_last_call_or_much_of_the_time(turns, failure):
    calls = RolloutCalls(UnreachableEndpoints(), rollout(), 1)
    for status, started_at, duration_s in turns:
        calls.call_captured(timed_turn(status, 1000.0 + started_at, duration_s))
    calls.close()
    assert calls.service_failure() == failure


@pytest.mark.unit
def test_a_rollout_whose_inference_service_failed_its_last_call_gets_no_score_and_runs_again(tmp_path):
    # The trial ends without an agent error, and its verifier writes 0, as a verifier does for work never done.
    lab = CallingLab(2, reward=0.0, fails_after_refusal=False)
    broken = (502, {"error": {"message": "the engine stopped"}}, {})
    with scripted_provider([ANSWER, broken]) as (url, _):
        queue, (value,), worker, _ = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
        item = queue.read(EVALUATIONS_QUEUE, value.item_id())
        # The attempt failed with no score, and the rollout waits for another attempt.
        assert lab.statuses == [200, 502]
        assert (item.state, item.attempt_count, item.result) == ("queued", 1, None)
        assert "the inference service failed 1 of 2 model calls, the last one among them" in (item.error or "")
        while worker.run_once():
            pass
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    # The service stayed down, so the rollout ends with no score once its attempts are used.
    assert (item.state, item.attempt_count, item.result) == ("failed", 3, None)
    assert lab.runs == 3


@pytest.mark.unit
def test_a_rate_limited_call_is_sent_again_in_the_same_sandbox_and_the_rollout_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_FIRST_WAIT_SECONDS", 0.01)
    busy = (429, {"error": {"message": "No deployments available for selected model"}}, {"Retry-After": "0"})
    # An answer whose body has an error code is still an answer.
    answer = (200, {**ANSWER[1], "error": {"code": "payment_required"}}, {})
    lab = CallingLab(1, reward=1.0)
    with scripted_provider([busy, IN_FLIGHT, answer]) as (url, bodies):
        queue, (value,), worker, artifacts = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.attempt_count, RolloutScore.from_json(item.result["score"]).value) == ("finished", 1, 1.0)
    # One sandbox: the provider got the same call three times, and the agent and the call log saw one answer.
    assert lab.runs == 1 and len(bodies) == 3 and bodies[0] == bodies[1] == bodies[2]
    assert lab.statuses == [200] and kept_codes(artifacts) == [(200, "payment_required")]


@pytest.mark.unit
def test_a_rate_limit_longer_than_a_call_can_wait_fails_the_rollout_once_and_sends_nothing_early(tmp_path):
    busy = (429, {"error": {"message": "concurrency limit"}}, {"Retry-After": "600"})
    lab = CallingLab(2, reward=0.0)
    with scripted_provider([busy]) as (url, bodies):
        queue, (first, second), worker, _ = provider_rollouts(tmp_path, url, lab, repeats=2)
        started = time.monotonic()
        assert worker.run_once()
        # The agent got the refusal at once, and its second call reached no provider: the route waits 600 s.
        assert lab.statuses == [429, 429] and len(bodies) == 1 and time.monotonic() - started < 30
        failed = queue.read(EVALUATIONS_QUEUE, first.item_id())
        assert (failed.state, failed.attempt_count, lab.runs) == ("failed", 1, 1)
        reason = RATE_LIMITED_ERROR.fullmatch(failed.error)
        assert reason is not None and 590 <= int(reason.group(1)) <= 600
        # The worker takes the next rollout at once, and it sends nothing to the provider before the wait ends.
        assert queue.read(EVALUATIONS_QUEUE, second.item_id()).state == "queued"
        assert worker.run_once()
    assert queue.read(EVALUATIONS_QUEUE, second.item_id()).state == "failed" and len(bodies) == 1
    assert queue.read(EVALUATIONS_QUEUE, first.item_id()).state == "failed", "no automatic retry"


@pytest.mark.unit
@pytest.mark.parametrize("spending_cap", [True, False])
def test_a_403_fails_the_rollout_once_as_a_spending_cap_or_a_refusal_of_access(tmp_path, spending_cap):
    metadata = {"metadata": {"limit_source": "openrouter_workspace_budget"}} if spending_cap else {}
    lab = CallingLab(1, reward=0.0)
    with scripted_provider([(403, {"error": {"code": 403, "message": "Forbidden", **metadata}}, {})]) as (url, bodies):
        queue, (value,), worker, artifacts = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert lab.statuses == [403] and len(bodies) == 1
    assert (item.state, item.attempt_count) == ("failed", 1)
    if spending_cap:
        assert item.error == (
            "the model provider refused 1 of the rollout's model calls, the last one with HTTP 403 "
            "(payment_required). Check the provider's spending cap, or wait until it resets, before retrying."
        )
        assert kept_codes(artifacts) == [(403, "payment_required")]
        return
    # Any other 403 refuses access, so another sandbox would meet the same refusal.
    assert (item.error, kept_codes(artifacts)) == (ACCESS_ERROR.format(status=403), [(403, "access_denied")])


#: OpenRouter's 403 for an input that its moderation flagged.
MODERATED_403 = {
    "error": {
        "code": 403,
        "message": "the input was flagged",
        "metadata": {"reasons": ["violence"], "flagged_input": "...", "provider_name": "p", "model_slug": "m"},
    }
}


@pytest.mark.unit
def test_a_moderation_403_refuses_no_access_and_fails_only_the_attempt(tmp_path):
    lab = CallingLab(1, reward=0.0)
    with scripted_provider([(403, MODERATED_403, {})]) as (url, bodies):
        queue, (value,), worker, artifacts = provider_rollouts(tmp_path, url, lab)
        assert worker.run_once()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    # The provider refused the content, not the key: the queue runs the rollout again, as after any failed attempt.
    assert lab.statuses == [403] and len(bodies) == 1
    assert (item.state, item.attempt_count, kept_codes(artifacts)) == ("queued", 1, [(403, None)])


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
@pytest.mark.parametrize(("event", "error"), [("lease_lost", WorkLeaseLostError), ("stopped", WorkerStoppedError)])
def test_a_running_trial_is_cancelled_once_the_worker_loses_the_rollout_or_stops(monkeypatch, event, error):
    monkeypatch.setattr("reef.evaluation.rollout_worker.LEASE_CHECK_S", 0.01)
    cancelled = threading.Event()

    async def endless_trial():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    events = {"lease_lost": threading.Event(), "stopped": threading.Event()}
    threading.Timer(0.1, events[event].set).start()
    with pytest.raises(error):
        asyncio.run(run_while_held(endless_trial(), events["lease_lost"], events["stopped"]))
    assert cancelled.is_set()


@pytest.mark.unit
@pytest.mark.parametrize("lease_lost", [False, True])
def test_work_that_ended_before_the_stop_keeps_its_result_unless_the_lease_is_lost(lease_lost):
    lost, stopped = threading.Event(), threading.Event()
    stopped.set()
    if lease_lost:
        lost.set()

    async def held() -> str:
        ended = asyncio.get_running_loop().create_future()
        ended.set_result("scored")
        return await run_while_held(ended, lost, stopped)

    if lease_lost:
        with pytest.raises(WorkLeaseLostError):
            asyncio.run(held())
    else:
        assert asyncio.run(held()) == "scored"


def e2b_rollout() -> EvaluationRollout:
    return replace(rollout(), environment="e2b")


class TemplateBuilds:
    """A stand-in for ``build_task_template`` of Reef's e2b sandbox: each call takes the next of ``outcomes``, an
    exception to raise or None for a ready template, and keeps the task directory in ``paths``."""

    def __init__(self, *outcomes: Exception | None) -> None:
        self.outcomes = list(outcomes)
        self.paths: list[Path] = []

    async def __call__(self, task_path: Path) -> str:
        self.paths.append(task_path)
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return "add__0123456789ab__aenv"


@pytest.mark.unit
def test_a_template_item_waits_again_for_a_template_that_found_no_free_builder(monkeypatch, tmp_path):
    pytest.importorskip("e2b")
    builds = TemplateBuilds(TimeoutError("the template was waiting after 600 s"), None)
    monkeypatch.setattr("reef.execution.sandbox.e2b.build_task_template", builds)
    queue = memory_coordination().work_queue
    endpoints = OpenEndpoints(tmp_path / "tasks")
    value = e2b_rollout()
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, "s", TASK_TEMPLATE_KIND, value.to_payload()))
    worker = TaskTemplateWorker(queue, endpoints, DockerfileTemplateBuilder(), worker_id="w")
    assert worker.run_once() is True
    assert queue.read(EVALUATIONS_QUEUE, item_id).state == "finished"
    assert builds.paths == [endpoints.task_directory(value)] * 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (TimeoutError("the template was waiting after 600 s"), "the template of task 'add' was not ready after 0 s"),
        (AgentEnvBuildError("aenv build of add failed with exit code 1"), "aenv build of add failed"),
    ],
)
def test_a_template_item_fails_as_retryable_when_its_template_is_not_ready(monkeypatch, tmp_path, outcome, message):
    pytest.importorskip("e2b")
    monkeypatch.setattr("reef.evaluation.rollout_worker.TEMPLATE_WAIT_S", 0.0)
    monkeypatch.setattr("reef.execution.sandbox.e2b.build_task_template", TemplateBuilds(outcome))
    queue = memory_coordination().work_queue
    value = e2b_rollout()
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, "s", TASK_TEMPLATE_KIND, value.to_payload()))
    worker = TaskTemplateWorker(queue, OpenEndpoints(tmp_path / "tasks"), DockerfileTemplateBuilder(), worker_id="w")
    assert worker.run_once() is True
    item = queue.read(EVALUATIONS_QUEUE, item_id)
    assert item.state == "queued" and message in str(item.error)


class RecordedBuilder(TaskTemplateBuilder):
    """A ``TaskTemplateBuilder`` that keeps the arguments of each build in ``builds`` and builds nothing."""

    def __init__(self) -> None:
        self.builds: list[tuple[str, str, Path]] = []

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        self.builds.append((scenario, task_name, task_directory))


@pytest.mark.unit
def test_a_template_item_builds_with_the_deployments_builder_and_not_from_the_dockerfile(monkeypatch, tmp_path):
    monkeypatch.setattr("reef.execution.sandbox.e2b.build_task_template", TemplateBuilds())
    queue = memory_coordination().work_queue
    endpoints = OpenEndpoints(tmp_path / "tasks")
    value = e2b_rollout()
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, "s", TASK_TEMPLATE_KIND, value.to_payload()))
    builder = RecordedBuilder()
    assert TaskTemplateWorker(queue, endpoints, builder, worker_id="w").run_once() is True
    assert queue.read(EVALUATIONS_QUEUE, item_id).state == "finished"
    assert builder.builds == [(value.scenario, value.task_name, endpoints.task_directory(value))]


class OwnedEndpoints(OpenEndpoints):
    """``OpenEndpoints`` whose scenario record stores the owner ``owner``, or none when it is None."""

    def __init__(self, root: Path, owner: str | None) -> None:
        super().__init__(root)
        self.owner = owner

    def task_owner(self, rollout: EvaluationRollout) -> str | None:
        return self.owner


class OwnerBuilder(TaskTemplateBuilder):
    """A builder that finds templates by owner: it keeps ``(owner, scenario, task_name)`` of each build in ``builds``
    and builds nothing. Its three-argument ``build`` fails the test, since the worker gives the owner."""

    def __init__(self) -> None:
        self.builds: list[tuple[str | None, str, str]] = []

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        raise AssertionError("the worker calls build_for_owner")

    async def build_for_owner(self, owner: str | None, scenario: str, task_name: str, task_directory: Path) -> None:
        self.builds.append((owner, scenario, task_name))


class OwnerDockerfileBuilder(DockerfileTemplateBuilder):
    """The Dockerfile builder with legacy owner-based registry selection."""

    async def build_for_owner(self, owner: str | None, scenario: str, task_name: str, task_directory: Path) -> None:
        pass


class CustomDockerfileBuilder(DockerfileTemplateBuilder):
    """The Dockerfile builder with a legacy name-based build."""

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        raise AssertionError("pinned work must skip the legacy build")


class InheritedDockerfileBuilder(DockerfileTemplateBuilder):
    """The Dockerfile builder with the default build hooks."""


@pytest.mark.unit
@pytest.mark.parametrize(
    ("queued_owner", "record_owner", "error"),
    [
        # An item that an older Reef queued names no owner: the record's owner counts.
        (None, "workspace-7", None),
        ("workspace-7", "workspace-7", None),
        # A scenario of the same name with another owner, or none, gets no template of this evaluation.
        ("workspace-7", "workspace-9", "has the owner 'workspace-9' now, not 'workspace-7'"),
        ("workspace-7", None, "has no owner now, not 'workspace-7'"),
    ],
)
def test_a_template_item_builds_for_the_owner_of_the_scenarios_record_and_never_for_another(
    tmp_path, queued_owner, record_owner, error
):
    queue = memory_coordination().work_queue
    # The scenario's name is a slug that says nothing of its owner.
    value = replace(e2b_rollout(), scenario="eval-run-3f9a", owner=queued_owner)
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, value.scenario, TASK_TEMPLATE_KIND, value.to_payload()))
    builder = OwnerBuilder()
    worker = TaskTemplateWorker(queue, OwnedEndpoints(tmp_path / "tasks", record_owner), builder, worker_id="w")
    assert worker.run_once() is True
    item = queue.read(EVALUATIONS_QUEUE, item_id)
    if error is None:
        assert item.state == "finished"
        assert builder.builds == [(record_owner, "eval-run-3f9a", "add")]
        return
    assert (item.state, item.attempt_count, builder.builds) == ("failed", 1, [])
    assert error in str(item.error)


@pytest.mark.unit
def test_a_template_item_gives_the_builder_its_pinned_task_digest(tmp_path):
    versions = []

    class VersionBuilder(OwnerDockerfileBuilder):
        async def build_for_version(self, owner, scenario, task_name, task_directory, *, task_digest):
            versions.append((owner, task_name, task_digest))

    class InheritedVersionBuilder(VersionBuilder):
        pass

    class VersionEndpoints(OwnedEndpoints):
        def task_version(self, rollout):
            return self.task_directory(rollout), rollout.task_digest

    queue = memory_coordination().work_queue
    owner = "workspace-7"
    digest = "1" * 64
    value = replace(e2b_rollout(), owner=owner, task_digest=digest)
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, value.scenario, TASK_TEMPLATE_KIND, value.to_payload()))
    builder = InheritedVersionBuilder()
    worker = TaskTemplateWorker(queue, VersionEndpoints(tmp_path / "tasks", owner), builder, worker_id="w")
    assert worker.run_once() is True
    assert queue.read(EVALUATIONS_QUEUE, item_id).state == "finished"
    assert versions == [(owner, value.task_name, digest)]


@pytest.mark.unit
@pytest.mark.parametrize("builder_type", [OwnerBuilder, OwnerDockerfileBuilder, CustomDockerfileBuilder])
def test_a_queued_pinned_template_never_uses_a_legacy_builder(tmp_path, builder_type):
    class VersionEndpoints(OwnedEndpoints):
        def task_version(self, rollout):
            return self.task_directory(rollout), rollout.task_digest

    queue = memory_coordination().work_queue
    owner = "workspace-7"
    value = replace(e2b_rollout(), owner=owner, task_digest="1" * 64)
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, value.scenario, TASK_TEMPLATE_KIND, value.to_payload()))
    builder = builder_type()
    worker = TaskTemplateWorker(queue, VersionEndpoints(tmp_path / "tasks", owner), builder, worker_id="w")
    assert worker.run_once() is True
    item = queue.read(EVALUATIONS_QUEUE, item_id)
    assert (item.state, item.attempt_count) == ("failed", 1)
    assert "does not support pinned task files" in str(item.error)


@pytest.mark.unit
@pytest.mark.parametrize("returns_mismatch", [False, True])
def test_a_pinned_rollout_refuses_name_only_or_mismatched_resolution(tmp_path, returns_mismatch):
    class WrongVersionEndpoints(OwnedEndpoints):
        def task_version(self, rollout):
            return self.task_directory(rollout), "0" * 64

    endpoint_type = WrongVersionEndpoints if returns_mismatch else OwnedEndpoints
    endpoints = endpoint_type(tmp_path, "workspace-7")
    worker = EvaluationRolloutWorker(memory_coordination().work_queue, endpoints, worker_id="w")
    value = replace(rollout(), task_digest="1" * 64)
    message = "returned different files" if returns_mismatch else "does not support pinned task files"
    with pytest.raises(ItemFailedError, match=message):
        worker.score(value, threading.Event(), 1)


@pytest.mark.unit
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    "resolver", ["base", "legacy", "inherited_legacy", "version", "explicit", "legacy_after_version"]
)
def test_template_and_trial_resolve_the_same_local_task(tmp_path, pinned, resolver):
    current, custom, explicit, admitted = (tmp_path / name for name in ("current", "custom", "explicit", "admitted"))
    for directory in (current, custom, explicit, admitted):
        directory.mkdir()
        (directory / "task.toml").write_text("")

    class Endpoints(LocalCandidateEndpoints):
        """Local endpoints whose trial opens without an inference server."""

        open = OpenEndpoints.open

    class NameBasedEndpoints(Endpoints):
        """Endpoints whose legacy resolver always selects the custom directory."""

        def task_directory(self, rollout):
            return custom

    class InheritedNameBasedEndpoints(NameBasedEndpoints):
        """Endpoints that inherit the custom legacy resolver."""

    class VersionEndpoints(NameBasedEndpoints):
        """Endpoints that delegate explicit version resolution to the built-in resolver."""

        def task_version(self, rollout):
            return super().task_version(rollout)

    class ExplicitVersionEndpoints(NameBasedEndpoints):
        """Endpoints that resolve unpinned versions to their own directory and known digest."""

        def task_version(self, rollout):
            if rollout.task_digest is None:
                return explicit, custom_digest
            return super().task_version(rollout)

    class NewNameBasedEndpoints(VersionEndpoints):
        """Endpoints that replace the legacy hook below an inherited version resolver."""

        def task_directory(self, rollout):
            return custom

    class Builder(RecordedBuilder):
        """A builder that records the directory and digest passed to preparation."""

        async def build_for_version(self, owner, scenario, task_name, task_directory, *, task_digest):
            prepared.append((task_directory, task_digest))

    class Lab(NoSandboxLab):
        """A lab that checks the trial uses the prepared directory, then simulates no sandbox."""

        async def run(self, task_path, agent, *, key, tags, overrides):
            [expected] = prepared
            assert Path(task_path) == expected[0]
            return await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)

    artifacts = Mock(spec=CandidateArtifacts)
    artifacts.task_owner.return_value = None
    artifacts.candidate_task.return_value = current
    current_digest, admitted_digest, custom_digest = "0" * 64, "1" * 64, "2" * 64
    artifacts.candidate_task_version.return_value = (current, current_digest)
    artifacts.candidate_task_for_digest.return_value = admitted
    endpoint_type = {
        "base": Endpoints,
        "legacy": NameBasedEndpoints,
        "inherited_legacy": InheritedNameBasedEndpoints,
        "version": VersionEndpoints,
        "explicit": ExplicitVersionEndpoints,
        "legacy_after_version": NewNameBasedEndpoints,
    }[resolver]
    endpoints = endpoint_type(artifacts)
    value = replace(e2b_rollout(), task_digest=admitted_digest if pinned else None)
    queue = memory_coordination().work_queue
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, value.scenario, TASK_TEMPLATE_KIND, value.to_payload()))
    prepared = []
    assert TaskTemplateWorker(queue, endpoints, Builder(), worker_id="template").run_once()
    template = queue.read(EVALUATIONS_QUEUE, item_id)
    worker = ServedWorker(queue, endpoints, worker_id="rollout", lab=Lab(tmp_path / "trials"))

    if pinned and resolver not in ("base", "version", "explicit"):
        assert template.state == "failed" and "must override task_version" in template.error
        assert not prepared
        with pytest.raises(ItemFailedError, match="must override task_version"):
            worker.score(value, threading.Event(), 1)
        return

    assert template.state == "finished"
    if pinned:
        assert prepared == [(admitted, admitted_digest)]
    elif resolver == "base":
        assert prepared == [(current, current_digest)]
    elif resolver == "explicit":
        assert prepared == [(explicit, custom_digest)]
    else:
        assert prepared == [(custom, None)]
    with pytest.raises(ReefError, match="the rollout reached no model: the sandbox did not start"):
        worker.score(replace(value, requires_prepared_templates=True), threading.Event(), 1)


def finished_template(queue: WorkQueue, value: EvaluationRollout, error: str | None) -> None:
    """Add the template item of ``value``'s task, and end it: finished when ``error`` is None, else failed for good
    with ``error``."""
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, value.scenario, TASK_TEMPLATE_KIND, value.to_payload()))
    assert queue.take(EVALUATIONS_QUEUE, "builder", [TASK_TEMPLATE_KIND]).id == item_id
    if error is None:
        queue.finish(EVALUATIONS_QUEUE, item_id, "builder", {})
    else:
        queue.fail(EVALUATIONS_QUEUE, item_id, "builder", error, retryable=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("template_error", "reason"),
    [
        ("the registry holds no image of task add", "the registry holds no image of task add"),
        (None, "the queue no longer holds its template item"),
    ],
)
def test_a_rollout_that_requires_prepared_templates_runs_no_trial_without_them(tmp_path, template_error, reason):
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 3})
    value = replace(e2b_rollout(), requires_prepared_templates=True)
    if template_error is not None:
        finished_template(queue, value, template_error)
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    # No candidate is served and the lab runs no trial, which would write under ``trials``.
    lab = NoSandboxLab(tmp_path / "trials")
    worker = ServingRefusedWorker(queue, UnreachableEndpoints(), worker_id="evaluation-1", lab=lab)
    assert worker.run_once() is True
    failed = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (failed.state, failed.attempt_count) == ("failed", 1)
    assert failed.error == f"the templates of task 'add' were not prepared: {reason}"
    assert not (tmp_path / "trials").exists()


class EnvironmentLab(CalledThenFailedLab):
    """A ``CalledThenFailedLab`` that keeps the Harbor environment config of each trial in ``environments``."""

    def __init__(self) -> None:
        self.environments: list[Mapping[str, object]] = []

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.environments.append(overrides["environment"])
        return await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)


@pytest.mark.unit
@pytest.mark.parametrize("requires", [True, False])
def test_the_trial_of_a_rollout_that_requires_prepared_templates_builds_no_template(tmp_path, requires):
    queue = memory_coordination().work_queue
    value = replace(
        e2b_rollout(),
        agent={"name": "x", "kwargs": {"api_base": "{base_url}/v1"}},
        requires_prepared_templates=requires,
    )
    finished_template(queue, value, None)
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    lab = EnvironmentLab()
    worker = ServedWorker(queue, AnsweringEndpoints(tmp_path / "tasks"), worker_id="w", lab=lab)
    assert worker.run_once() is True
    assert queue.read(EVALUATIONS_QUEUE, value.item_id()).state == "finished"
    [environment] = lab.environments
    # Only a rollout that requires prepared templates tells Reef's e2b class not to build one.
    assert environment["import_path"] == "reef.execution.sandbox.e2b:KillOnTimeoutE2BEnvironment"
    assert ("builds_templates" in environment.get("kwargs", {})) == requires
    if requires:
        assert environment["kwargs"]["builds_templates"] is False


@pytest.mark.unit
def test_only_a_rollout_on_e2b_requires_prepared_templates():
    with pytest.raises(ValueError, match="only a rollout on e2b requires prepared templates"):
        replace(rollout(), requires_prepared_templates=True)
    with pytest.raises(ValueError, match="owner is a non-empty string or None"):
        replace(e2b_rollout(), owner="")
    value = replace(e2b_rollout(), owner="workspace-7", requires_prepared_templates=True)
    item = memory_coordination().work_queue.append(
        NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload())
    )
    assert EvaluationRollout.from_item(item) == value


class HeldLab(CalledThenFailedLab):
    """A ``TaskLab`` whose trial of each task in ``tasks`` makes one model call, as ``CalledThenFailedLab``, then runs
    until ``released[task]`` is set, and ends with the reward 1.

    ``started[task]`` is set once the call has an answer. A cancelled trial sets ``cleaned_up[task]`` as it stops, as
    Harbor stops the sandbox of a cancelled trial. Each of the three maps holds one event for each task.
    """

    def __init__(self, *tasks: str) -> None:
        self.started = {task: threading.Event() for task in tasks}
        self.released = {task: threading.Event() for task in tasks}
        self.cleaned_up = {task: threading.Event() for task in tasks}

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        task = Path(task_path).name
        await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)
        self.started[task].set()
        try:
            while not self.released[task].is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cleaned_up[task].set()
            raise
        return RolloutRow(rewards={"reward": 1.0}, error=None, trial_uri=None)


def held_rollout(task_index: int, task_name: str) -> EvaluationRollout:
    """The rollout of task ``task_name`` at ``task_index``, whose agent calls the model through the rollout's
    endpoint, as ``HeldLab`` needs."""
    return replace(
        rollout(),
        task_index=task_index,
        task_name=task_name,
        agent={"name": "x", "kwargs": {"api_base": "{base_url}/v1"}},
    )


def add_rollout(queue: WorkQueue, value: EvaluationRollout) -> None:
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("on_stop", "finish_timeout_s", "state", "error"),
    [
        ("requeue", None, "queued", "the worker stopped"),
        ("finish", None, "finished", None),
        # A rollout that finishes before the timeout ends the stop, and the stop ends the timer.
        ("finish", 3600.0, "finished", None),
    ],
)
def test_a_stopping_worker_requeues_or_finishes_the_rollouts_that_it_runs(
    tmp_path, monkeypatch, on_stop, finish_timeout_s, state, error
):
    monkeypatch.setattr("reef.evaluation.rollout_worker.LEASE_CHECK_S", 0.01)
    queue = memory_coordination().work_queue
    value = held_rollout(2, "add")
    add_rollout(queue, value)
    lab = HeldLab("add")
    worker = ServedWorker(
        queue,
        AnsweringEndpoints(tmp_path / "tasks"),
        worker_id="w",
        lab=lab,
        slots=2,
        on_stop=on_stop,
        finish_timeout_s=finish_timeout_s,
    )
    assert worker.run_once() is True
    assert lab.started["add"].wait(10)
    worker.begin_stop()
    timer = worker.cancel_timer
    assert (timer is None) == (finish_timeout_s is None)
    # The rollout ends only when the lab lets it, so a worker that cancels it stops first.
    threading.Timer(0.3, lab.released["add"].set).start()
    worker.stop()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.error) == (state, error)
    assert lab.cleaned_up["add"].is_set() == (on_stop == "requeue")
    if on_stop == "finish":
        assert item.result == {"score": 1.0, "rewards": {"reward": 1.0}, "error": None}
        assert not worker.rollouts_cancelled.is_set()
    assert worker.cancel_timer is None and (timer is None or not timer.is_alive())


class GatedCutoffWorker(ServedWorker):
    """A ``ServedWorker`` whose cutoff waits, once the timer of ``finish_timeout_s`` calls it, until the test sets
    ``cutoff_allowed``. So the test decides what ends before the cutoff, with no race against the clock.

    ``cutoff_reached`` is set when the timer calls the cutoff, and ``cutoff_count`` counts the calls.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cutoff_reached = threading.Event()
        self.cutoff_allowed = threading.Event()
        self.cutoff_count = 0

    def cancel_rollouts(self) -> None:
        self.cutoff_count += 1
        self.cutoff_reached.set()
        self.cutoff_allowed.wait(10)
        super().cancel_rollouts()


@pytest.mark.unit
def test_a_stopping_worker_keeps_the_rollout_that_finishes_in_time_and_gives_back_the_one_that_runs_at_the_cutoff(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("reef.evaluation.rollout_worker.LEASE_CHECK_S", 0.01)
    # Each rollout has one attempt: a stop that counted as an attempt would leave it none.
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 1})
    in_time, at_cutoff, fresh = held_rollout(0, "in-time"), held_rollout(1, "at-cutoff"), held_rollout(2, "fresh")
    add_rollout(queue, in_time)
    add_rollout(queue, at_cutoff)
    lab = HeldLab("in-time", "at-cutoff", "fresh")
    endpoints = AnsweringEndpoints(tmp_path / "tasks")
    # Three slots: the worker has a free slot while it stops, and still takes no new rollout.
    worker = GatedCutoffWorker(
        queue, endpoints, worker_id="w", lab=lab, poll_seconds=0.01, slots=3, on_stop="finish", finish_timeout_s=0.05
    )
    workers = ServiceWorkers()
    workers.add(worker)
    workers.start()
    assert lab.started["in-time"].wait(10) and lab.started["at-cutoff"].wait(10)

    workers.begin_stop()
    timer = worker.cancel_timer
    # Another stop request keeps the cutoff of the first.
    workers.begin_stop()
    assert timer is not None and worker.cancel_timer is timer
    add_rollout(queue, fresh)
    assert worker.cutoff_reached.wait(10)
    lab.released["in-time"].set()
    wait_until(lambda: queue.read(EVALUATIONS_QUEUE, in_time.item_id()).state == "finished")
    worker.cutoff_allowed.set()
    workers.stop()

    finished = queue.read(EVALUATIONS_QUEUE, in_time.item_id())
    assert (finished.state, finished.attempt_count, finished.stopped_attempt_count) == ("finished", 1, 0)
    assert finished.result == {"score": 1.0, "rewards": {"reward": 1.0}, "error": None}
    given_back = queue.read(EVALUATIONS_QUEUE, at_cutoff.item_id())
    assert (given_back.state, given_back.error) == ("queued", "the worker stopped")
    assert (given_back.attempt_count, given_back.stopped_attempt_count) == (1, 1)
    assert queue.has_attempts_left(given_back)
    assert lab.cleaned_up["at-cutoff"].is_set() and not lab.cleaned_up["in-time"].is_set()
    assert queue.read(EVALUATIONS_QUEUE, fresh.item_id()).attempt_count == 0
    assert worker.cutoff_count == 1 and worker.cancel_timer is None and not timer.is_alive()
    # Another worker starts the rollout that was given back from its start, and finishes it.
    lab.released["at-cutoff"].set()
    assert ServedWorker(queue, endpoints, worker_id="w2", lab=lab).run_once() is True
    restarted = queue.read(EVALUATIONS_QUEUE, at_cutoff.item_id())
    assert (restarted.state, restarted.attempt_count, restarted.stopped_attempt_count) == ("finished", 2, 1)


@pytest.mark.unit
def test_a_rollout_that_the_worker_takes_as_its_stop_begins_goes_back_with_no_trial(tmp_path, monkeypatch):
    queue = SQLWorkQueue(SQLiteCoordinationDatabase(None), max_attempts={EVALUATIONS_QUEUE: 1})
    value = held_rollout(2, "add")
    add_rollout(queue, value)
    # No candidate is served and the lab runs no trial, which would write under ``trials``.
    lab = NoSandboxLab(tmp_path / "trials")
    worker = ServingRefusedWorker(
        queue, UnreachableEndpoints(), worker_id="w", lab=lab, on_stop="finish", finish_timeout_s=60.0
    )
    take = queue.take

    def take_as_the_stop_begins(*args, **kwargs):
        item = take(*args, **kwargs)
        worker.begin_stop()
        return item

    monkeypatch.setattr(queue, "take", take_as_the_stop_begins)
    assert worker.run_once() is False
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.error) == ("queued", "the worker stopped")
    assert (item.attempt_count, item.stopped_attempt_count) == (1, 1) and queue.has_attempts_left(item)
    assert not (tmp_path / "trials").exists()
    # A stopping worker takes no rollout at all.
    assert worker.run_once() is False
    assert queue.read(EVALUATIONS_QUEUE, value.item_id()).attempt_count == 1
    worker.stop()
    assert worker.cancel_timer is None


class SlowCancelBuilder(TaskTemplateBuilder):
    """A builder whose build runs until it is cancelled, and whose cancellation ends only when ``ended`` is set, as a
    build that waits for its builder to answer. ``started`` is set when the build starts; ``saw_end`` is whether
    ``ended`` was set within 10 s of the cancellation."""

    def __init__(self, ended: threading.Event) -> None:
        self.ended = ended
        self.started = threading.Event()
        self.saw_end = False

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.saw_end = await asyncio.to_thread(self.ended.wait, 10)
            raise


@pytest.mark.unit
def test_the_cutoff_of_the_rollouts_does_not_wait_for_the_stop_of_the_template_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("reef.evaluation.rollout_worker.LEASE_CHECK_S", 0.01)
    queue = memory_coordination().work_queue
    value = held_rollout(2, "add")
    add_rollout(queue, value)
    template = e2b_rollout()
    template_id = template_item_id(template.evaluation_id, template.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, template_id, "s", TASK_TEMPLATE_KIND, template.to_payload()))
    lab = HeldLab("add")
    endpoints = AnsweringEndpoints(tmp_path / "tasks")
    rollouts = ServedWorker(
        queue, endpoints, worker_id="w", lab=lab, poll_seconds=0.01, on_stop="finish", finish_timeout_s=0.05
    )
    # The template worker's stop waits until the rollout worker has cancelled its rollouts.
    builder = SlowCancelBuilder(rollouts.rollouts_cancelled)
    # As in a process of the evaluation service: the template worker is added last, so it stops first.
    workers = ServiceWorkers()
    workers.add(rollouts)
    workers.add(TaskTemplateWorker(queue, endpoints, builder, worker_id="t", poll_seconds=0.01))
    workers.start()
    assert lab.started["add"].wait(10) and builder.started.wait(10)

    workers.stop()

    assert builder.saw_end
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.stopped_attempt_count) == ("queued", 1) and lab.cleaned_up["add"].is_set()
    template_item = queue.read(EVALUATIONS_QUEUE, template_id)
    assert (template_item.state, template_item.stopped_attempt_count) == ("queued", 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("on_stop", "finish_timeout_s", "message"),
    [
        ("drain", None, "on_stop must be one of requeue, finish, not 'drain'"),
        ("requeue", 60.0, "finish_timeout_s needs on_stop: finish"),
        ("finish", 0.0, "finish_timeout_s is a number of seconds above 0"),
        ("finish", math.inf, "finish_timeout_s is a number of seconds above 0"),
        # Longer than a timer can wait, and an int too large for a float.
        ("finish", math.nextafter(threading.TIMEOUT_MAX, math.inf), "at most"),
        ("finish", 10**400, "at most"),
    ],
)
def test_a_worker_refuses_an_unknown_stop_and_a_timeout_that_it_cannot_keep(on_stop, finish_timeout_s, message):
    with pytest.raises(ValueError, match=message):
        ServedWorker(
            memory_coordination().work_queue,
            UnreachableEndpoints(),
            worker_id="w",
            on_stop=on_stop,
            finish_timeout_s=finish_timeout_s,
        )


@pytest.mark.unit
def test_the_longest_finish_timeout_is_a_wait_that_the_timer_can_keep():
    # ``threading.Timer`` waits on a timed lock, which takes ``TIMEOUT_MAX`` and refuses a much longer timeout.
    assert threading.Lock().acquire(timeout=threading.TIMEOUT_MAX)
    with pytest.raises(OverflowError):
        threading.Lock().acquire(timeout=2 * threading.TIMEOUT_MAX)
    check_finish_timeout("finish", threading.TIMEOUT_MAX)
