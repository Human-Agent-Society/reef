"""Stand-ins for AutoEnvScaling's tests: sessions and sources, an author, a sandbox, a Reef and a generator.

The generator service itself is :class:`~tests.support.task_service_stubs.StandInGenerator`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from tests.support.task_service_stubs import DOCUMENT, StandInGenerator

from recipes.beta.autoenvscaling import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    AutoEnvScalingGenerator,
    CandidateChecks,
    Repository,
    SessionSource,
    SourceError,
    TaskAuthor,
    load_sources,
    workspace_reply,
)
from recipes.beta.autoenvscaling.author import AgentTools, ModelCalls, RecordingGateway
from reef.core.tasks import HarborTask, TaskGenerationRequest, TaskPlayResult
from reef.record2task.generator import GeneratorError, WrittenTask
from reef.sandbox.e2b import E2BExecutor, E2BSession
from reef.sandbox.process import EpisodeExecutor, EpisodeLaunchError, ProcessOutcome, SandboxProgram


def stand_in_e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``import e2b`` succeed where the package is missing.

    Building the agent author checks that the package imports; no test here opens a sandbox.
    """
    if importlib.util.find_spec("e2b") is None:
        monkeypatch.setitem(sys.modules, "e2b", types.ModuleType("e2b"))


def sources(count: int) -> list[dict[str, object]]:
    """A sources file's entries: one session per repository, all at the same base commit."""
    return [
        {
            "record_id": f"session-{number}",
            "scenario": "sessions",
            "repository": {"url": f"https://example.com/org/repo-{number}.git", "commit": "3f2c1ab", "path": "/app"},
        }
        for number in range(1, count + 1)
    ]


def source_of(entry: Mapping[str, Any]) -> SessionSource:
    return SessionSource(entry["record_id"], entry["scenario"], Repository(**entry["repository"]))


class NoChecks(CandidateChecks):
    def check(self, task_dir: Path) -> dict[str, object]:
        return {"ran": False}


class ScriptedSession(E2BSession):
    """A sandbox that runs no pi: launch writes a scripted session log; with ``stop``, it sets the stop signal (a
    cancellation mid run) and waits for the sandbox to be closed."""

    def __init__(self, events: list[dict[str, object]], *, stop: threading.Event | None = None) -> None:
        self.events = events
        self.stop = stop
        self.is_closed = threading.Event()

    def launch(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        timeout: float,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> ProcessOutcome:
        if self.stop is not None:
            self.stop.set()
            self.is_closed.wait(10)
            raise EpisodeLaunchError("the E2B sandbox could not run the process: it is gone")
        (root / "sessions").mkdir(parents=True, exist_ok=True)
        (root / "sessions" / "session.jsonl").write_text("".join(json.dumps(event) + "\n" for event in self.events))
        return ProcessOutcome(0, "", "")

    def pull(self, root: Path, relative: str = "") -> None:
        return None

    def close(self) -> None:
        self.is_closed.set()


@dataclass(frozen=True)
class ScriptedExecutor(E2BExecutor):
    """An executor whose sandbox is the test's session."""

    session: ScriptedSession | None = None

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[EpisodeExecutor, str]:
        return self, program.binary

    def open(self) -> E2BSession:
        if self.session is None:
            raise EpisodeLaunchError("no scripted session")
        return self.session


def run_agent(
    tmp_path: Path,
    session: ScriptedSession,
    calls: ModelCalls,
    stop: threading.Event,
    *,
    checks: CandidateChecks | None = None,
    has_behaviour_rule: bool = True,
) -> None:
    """The agent's run in ``tmp_path / "run"`` over a scripted sandbox, its calls through a gateway to no Reef."""
    author = AgentAuthor(
        reef_url="http://127.0.0.1:9",
        token=None,
        scenario="autoenvscaling",
        model="designer",
        api="openai",
        executor=ScriptedExecutor(session=session),
        cache_dir=tmp_path / "cache",
        timeout_seconds=60.0,
        max_model_calls=calls.budget,
        has_behaviour_rule=has_behaviour_rule,
    )
    tools = AgentTools(tmp_path, checks if checks is not None else NoChecks(1))
    gateway = RecordingGateway(author.upstream, calls, tools, headers={})
    gateway.start()
    try:
        author.run(tmp_path / "run", gateway, tools, calls, stop)
    finally:
        gateway.stop()


USER_REQUEST = "Please add a --dry-run flag to the deploy command. "
# A session whose rendering exceeds a prompt Designer's budget of 2000 characters: its tool result is cut to fit.
SESSION = {
    "messages": [
        {"role": "user", "content": USER_REQUEST},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"c": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "y" * 3000}]},
    ],
    "response": {"role": "assistant", "content": [{"type": "text", "text": "Added the flag."}]},
}


def serving_reef(reply: str) -> tuple[ThreadingHTTPServer, list[dict[str, Any]]]:
    """A stand-in Reef: it serves ``SESSION`` as every record and answers each model call with ``reply`` and a
    receipt, in the dialect of the route called; the calls it took are listed."""
    calls: list[dict[str, Any]] = []

    class Reef(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.answer(json.dumps({"payload": SESSION}).encode(), {})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {name.lower(): value for name, value in self.headers.items()}
            calls.append({"path": self.path, "headers": headers, **body})
            answer = (
                {
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": reply}]}
                    ]
                }
                if self.path == "/v1/responses"
                else {"choices": [{"message": {"role": "assistant", "content": reply}}]}
            )
            self.answer(json.dumps(answer).encode(), {"x-reef-agent-record-id": f"rec-{len(calls)}"})

        def answer(self, data: bytes, headers: Mapping[str, str]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Reef)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def fenced(document: Mapping[str, object]) -> str:
    """A prompt Designer's reply holding ``document``."""
    return "Here is the task.\n```json\n" + json.dumps(document) + "\n```"


def agent_files(tmp_path: Path, *, has_behaviour_rule: bool, checks: CandidateChecks) -> str:
    """Everything the agent's run wrote beside its workspace, its rules among them."""
    run_agent(
        tmp_path,
        ScriptedSession([]),
        ModelCalls(5),
        threading.Event(),
        checks=checks,
        has_behaviour_rule=has_behaviour_rule,
    )
    return "\n".join(path.read_text(errors="replace") for path in (tmp_path / "run").rglob("*") if path.is_file())


DESCRIPTION = "coding tasks rebuilt from recorded coding-agent sessions"
REQUEST = TaskGenerationRequest((), DESCRIPTION)
NAMES = ["harbor-00000-000-autoenvscaling", "harbor-00000-001-autoenvscaling", "harbor-00000-002-autoenvscaling"]


def write_workspace_task(task_dir: Path, port: int, *, has_tmux: bool = True) -> Path:
    """A task laid out file by file, the way the agent leaves it in ./task/."""
    document = json.loads(json.dumps(DOCUMENT).replace("8471", str(port)))
    if not has_tmux:
        document["environment"]["Dockerfile"] = document["environment"]["Dockerfile"].replace("tmux", "curl")
    files = {"instruction.md": document["instruction"], "hint.txt": document["hint"]}
    for label in ("environment", "tests", "solution"):
        files.update({f"{label}/{name}": text for name, text in document[label].items()})
    for name, text in files.items():
        (task_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (task_dir / name).write_text(text)
    return task_dir


class StandInAuthor(TaskAuthor):
    """Writes a task as the agent would, checks it once through the generator's checks, and returns its receipts.

    A source named in ``unusable`` holds no session; one named in ``unreachable`` is an outage.
    """

    def __init__(
        self,
        *,
        receipts: tuple[str, ...] = ("agent-1", "agent-2"),
        has_tmux: bool = True,
        unusable: Sequence[str] = (),
        unreachable: Sequence[str] = (),
    ) -> None:
        self.receipts = receipts
        self.has_tmux = has_tmux
        self.unusable = unusable
        self.unreachable = unreachable
        self.calls: list[dict[str, Any]] = []

    def author(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
    ) -> AuthoredTask:
        self.calls.append({"source": source.record_id, "tags": dict(tags), "work_dir": work_dir})
        if source.record_id in self.unusable:
            raise SourceError(f"the record {source.record_id} holds no session messages")
        if source.record_id in self.unreachable:
            raise AuthorError("the agent could not run: the sandbox did not start")
        # Each generation and index writes distinct content, so no task duplicates another.
        port = 8000 + 100 * int(tags["generation"]) + int(tags["index"])
        task_dir = write_workspace_task(work_dir / "workspace" / "task", port, has_tmux=self.has_tmux)
        self.calls[-1]["check"] = checks.check(task_dir)
        return AuthoredTask(workspace_reply(task_dir), self.receipts)


def sources_file(tmp_path: Path, count: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources(count)))
    return path


def autoenvscaling(
    tmp_path: Path,
    author: TaskAuthor,
    service: StandInGenerator | None = None,
    *,
    source_count: int = 2,
    **fields: Any,
) -> AutoEnvScalingGenerator:
    return AutoEnvScalingGenerator(
        scenario="autoenvscaling",
        service=service if service is not None else StandInGenerator(tmp_path / "tasks"),
        author=author,
        sources=load_sources(sources_file(tmp_path, source_count)),
        state_dir=tmp_path / "state",
        **{"count": 3, **fields},
    )


def played(generator: AutoEnvScalingGenerator, *training: tuple[float, ...]) -> tuple[TaskPlayResult, ...]:
    """Generate, validate, write and play every task the way the generated-task recipe does.

    The valid tasks take the training rewards in turn.
    """
    rewards = training or ((0.0, 1.0),)
    results = []
    for item in asyncio.run(generator.generate(REQUEST)):
        validation = asyncio.run(generator.validate(item.task))
        if not validation.is_valid:
            results.append(TaskPlayResult(item.task, error="; ".join(validation.errors)))
            continue
        asyncio.run(generator.service.write_task(item.task))
        results.append(TaskPlayResult(item.task, rewards[len(results) % len(rewards)]))
    return tuple(results)


def report(tmp_path: Path, generation: int = 0) -> dict[str, Any]:
    return json.loads((tmp_path / "state" / f"generation-{generation:05d}.json").read_text())


def cursor(tmp_path: Path) -> dict[str, int]:
    return json.loads((tmp_path / "state" / "cursor.json").read_text())


class RefusingGenerator(StandInGenerator):
    """A generator that refuses every write with the given error."""

    def __init__(self, root: Path, error: GeneratorError) -> None:
        super().__init__(root)
        self.error = error

    async def write_task(self, task: HarborTask) -> WrittenTask:
        raise self.error


def generating_config(tmp_path: Path, **fields: object) -> dict[str, object]:
    return {
        "generator_url": "http://127.0.0.1:8910",
        "state_dir": str(tmp_path / "state"),
        "sources_path": str(sources_file(tmp_path, 2)),
        "agent_model": "gpt-6-sol",
        "agent_reef_url": "http://127.0.0.1:8901",
        **fields,
    }
