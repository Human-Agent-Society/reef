"""The evaluation service's worker: rollout items, candidate endpoints and failures."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from reef_client.serve import CapturedTurn

from reef.core.errors import ReefError
from reef.core.evaluation import FAILED, EvaluationCall, EvaluationTrajectory, PolicyModel
from reef.core.served_models import ServedArtifact
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    TASK_TEMPLATE_KIND,
    CandidateEndpoint,
    CandidateEndpoints,
    EvaluationRollout,
    EvaluationRolloutWorker,
    RolloutCalls,
    TaskTemplateWorker,
    TreeHarness,
    evaluation_call,
    reply_summary,
    rollout_item_id,
    run_while_held,
    template_item_id,
    verifier_output,
)
from reef.evaluation.task_templates import DockerfileTemplateBuilder, TaskTemplateBuilder
from reef.evaluation.trials import EvaluationTrial, TrialObserver
from reef.execution.sandbox.agentenv_build import AgentEnvBuildError
from reef.execution.workers import WorkerStoppedError
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue, memory_coordination
from reef.storage.work_queue import NewWork, WorkLeaseLostError, WorkQueue


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

    class VersionBuilder(OwnerBuilder):
        async def build_for_version(self, owner, scenario, task_name, task_directory, *, task_digest):
            versions.append((owner, task_name, task_digest))

    queue = memory_coordination().work_queue
    owner = "workspace-7"
    digest = "1" * 64
    value = replace(e2b_rollout(), owner=owner, task_digest=digest)
    item_id = template_item_id(value.evaluation_id, value.task_index)
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, value.scenario, TASK_TEMPLATE_KIND, value.to_payload()))
    builder = VersionBuilder()
    worker = TaskTemplateWorker(queue, OwnedEndpoints(tmp_path / "tasks", owner), builder, worker_id="w")
    assert worker.run_once() is True
    assert queue.read(EVALUATIONS_QUEUE, item_id).state == "finished"
    assert versions == [(owner, value.task_name, digest)]


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
    """A ``TaskLab`` whose trial makes one model call, as ``CalledThenFailedLab``, then runs until ``released`` is
    set, and ends with the reward 1. ``started`` is set once the call has an answer."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.released = threading.Event()

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)
        self.started.set()
        while not self.released.is_set():
            await asyncio.sleep(0.01)
        return RolloutRow(rewards={"reward": 1.0}, error=None, trial_uri=None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("on_stop", "state", "error"), [("requeue", "queued", "the worker stopped"), ("finish", "finished", None)]
)
def test_a_stopping_worker_requeues_or_finishes_the_rollouts_that_it_runs(
    tmp_path, monkeypatch, on_stop, state, error
):
    monkeypatch.setattr("reef.evaluation.rollout_worker.LEASE_CHECK_S", 0.01)
    queue = memory_coordination().work_queue
    value = replace(rollout(), agent={"name": "x", "kwargs": {"api_base": "{base_url}/v1"}})
    queue.append(NewWork(EVALUATIONS_QUEUE, value.item_id(), "s", EVALUATION_ROLLOUT_KIND, value.to_payload()))
    lab = HeldLab()
    worker = ServedWorker(
        queue, AnsweringEndpoints(tmp_path / "tasks"), worker_id="w", lab=lab, slots=2, on_stop=on_stop
    )
    assert worker.run_once() is True
    assert lab.started.wait(10)
    # The rollout ends only when the lab lets it, so a worker that cancels it stops first.
    threading.Timer(0.3, lab.released.set).start()
    worker.stop()
    item = queue.read(EVALUATIONS_QUEUE, value.item_id())
    assert (item.state, item.error) == (state, error)
    if on_stop == "finish":
        assert item.result == {"score": 1.0, "rewards": {"reward": 1.0}, "error": None}


@pytest.mark.unit
def test_a_worker_refuses_an_unknown_stop():
    with pytest.raises(ValueError, match="on_stop must be one of requeue, finish, not 'drain'"):
        ServedWorker(memory_coordination().work_queue, UnreachableEndpoints(), worker_id="w", on_stop="drain")
