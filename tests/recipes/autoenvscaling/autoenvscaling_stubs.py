"""Stand-ins for AutoEnvScaling's tests: sessions, a Designer, pi and Claude Code sandboxes, a Reef and a generator.

The generator service itself is :class:`~tests.support.task_service_stubs.StandInGenerator`.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from tests.support.task_service_stubs import DOCUMENT, StandInGenerator, generator_state

from recipes.beta.autoenvscaling import (
    AgentDesigner,
    AutoEnvScalingGenerator,
    CandidateChecks,
    DesignedTask,
    DesignerOutage,
    Repository,
    SessionSource,
    SourceError,
    TaskDesigner,
    VariantDesigner,
)
from recipes.beta.autoenvscaling import designer as designer_module
from recipes.beta.autoenvscaling import load_sources, task_check, workspace_reply
from recipes.beta.autoenvscaling.checks import PassRateBand
from recipes.beta.autoenvscaling.designer import AgentTools, ModelCalls, RecordingGateway, TaskVariant
from reef.core.tasks import HarborTask
from reef.execution.sandbox.e2b import E2BExecutor, E2BSession
from reef.execution.sandbox.process import (
    ProcessOutcome,
    ProgramExecutor,
    ProgramLaunchError,
    ProgramTimeout,
    SandboxProgram,
)
from reef.harness.harbor.rollouts import TaskRollout
from reef.recipe.generated_tasks.generator import GeneratedTaskResult, RejectedTask, RolledOutTask
from reef.recipe.generated_tasks.processor import StoredGeneratorState
from reef.record2sample.generator import GeneratorError, WrittenTask
from reef.record2sample.generator.inputs import TaskGenerationRequest


def stand_in_e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``import e2b`` succeed where the package is missing.

    Building the agent Designer checks that the package imports; no test here opens a sandbox.
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
    """A sandbox that runs no pi: launch writes a scripted session log and keeps the command it was given; with
    ``stop``, it sets the stop signal (a cancellation mid run) and waits for the sandbox to be closed."""

    def __init__(self, events: list[dict[str, object]], *, stop: threading.Event | None = None) -> None:
        self.events = events
        self.stop = stop
        self.is_closed = threading.Event()
        self.argv: list[str] = []

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
        self.argv = list(argv)
        if self.stop is not None:
            self.stop.set()
            self.is_closed.wait(10)
            raise ProgramLaunchError("the E2B sandbox could not run the process: it is gone")
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

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[ProgramExecutor, str]:
        return self, program.binary

    def open(self) -> E2BSession:
        if self.session is None:
            raise ProgramLaunchError("no scripted session")
        return self.session


def run_agent(
    tmp_path: Path,
    session: ScriptedSession,
    calls: ModelCalls,
    stop: threading.Event,
    *,
    checks: CandidateChecks | None = None,
    has_behaviour_rule: bool = True,
    variant: TaskVariant | None = None,
) -> None:
    """The agent's run in ``tmp_path / "run"`` over a scripted sandbox, its calls through a gateway to no Reef."""
    designer = AgentDesigner(
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
    gateway = RecordingGateway(designer.upstream, calls, tools, headers={})
    gateway.start()
    try:
        designer.run(
            tmp_path / "run",
            gateway,
            tools,
            calls,
            stop,
            extra_rules="" if variant is None else designer_module.VARIANT_INPUTS_TEXT,
            prompt=(
                designer_module.AGENT_PROMPT
                if variant is None
                else designer_module.VARIANT_PROMPT.format(direction=variant.direction)
            ),
        )
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


def agent_files(
    tmp_path: Path,
    *,
    has_behaviour_rule: bool,
    checks: CandidateChecks,
    session: ScriptedSession | None = None,
    variant: TaskVariant | None = None,
) -> str:
    """Everything the agent's run wrote beside its workspace, its rules among them."""
    run_agent(
        tmp_path,
        session if session is not None else ScriptedSession([]),
        ModelCalls(5),
        threading.Event(),
        checks=checks,
        has_behaviour_rule=has_behaviour_rule,
        variant=variant,
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


class StandInDesigner(VariantDesigner):
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

    def vary(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant,
    ) -> DesignedTask:
        return self.design(source, work_dir, checks, tags=tags, stop=stop, variant=variant)

    def design(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant | None = None,
    ) -> DesignedTask:
        self.calls.append({"source": source.record_id, "tags": dict(tags), "work_dir": work_dir, "variant": variant})
        if source.record_id in self.unusable:
            raise SourceError(f"the record {source.record_id} holds no session messages")
        if source.record_id in self.unreachable:
            raise DesignerOutage("the agent could not run: the sandbox did not start")
        # Each generation and index writes distinct content, so no task duplicates another.
        port = 8000 + 100 * int(tags["generation"]) + int(tags["index"])
        task_dir = write_workspace_task(work_dir / "workspace" / "task", port, has_tmux=self.has_tmux)
        self.calls[-1]["check"] = checks.check(task_dir)
        return DesignedTask(workspace_reply(task_dir), self.receipts)


def sources_file(tmp_path: Path, count: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources(count)))
    return path


def autoenvscaling(
    tmp_path: Path,
    designer: TaskDesigner,
    service: StandInGenerator | None = None,
    *,
    source_count: int = 2,
    **fields: Any,
) -> AutoEnvScalingGenerator:
    fields["band"] = PassRateBand(fields.pop("band_low", 0.25), fields.pop("band_high", 0.75))
    session_sources = load_sources(sources_file(tmp_path, source_count))
    return AutoEnvScalingGenerator(
        scenario="autoenvscaling",
        service=service if service is not None else StandInGenerator(tmp_path / "tasks"),
        designer=designer,
        sources=session_sources,
        state=state(tmp_path),
        temporary=tempfile.TemporaryDirectory(dir=tmp_path),
        **{"count": 3, **fields},
    )


def state(tmp_path: Path) -> StoredGeneratorState:
    """The state of the generator that ``autoenvscaling(tmp_path, ...)`` builds, opened again as a restart does."""
    return generator_state(tmp_path / "state", "autoenvscaling")


def state_document(tmp_path: Path, key: str) -> Any:
    """The JSON value of one document of the generator's state."""
    document = state(tmp_path).read(key)
    assert document is not None
    return json.loads(document)


def run_tasks(generator: AutoEnvScalingGenerator, *training: tuple[float, ...]) -> tuple[GeneratedTaskResult, ...]:
    """Generate, validate, write and run every task the way the generated-task recipe does.

    The valid tasks take the training rewards in turn.
    """
    rewards = training or ((0.0, 1.0),)
    results = []
    for item in asyncio.run(generator.generate(REQUEST)):
        validation = asyncio.run(generator.validate(item))
        if not validation.is_valid:
            results.append(RejectedTask(item, "; ".join(validation.errors)))
            continue
        asyncio.run(generator.service.write_task(item))
        results.append(RolledOutTask(item, rewards[len(results) % len(rewards)]))
    return tuple(results)


def report(tmp_path: Path, generation: int = 0) -> dict[str, Any]:
    document: dict[str, Any] = state_document(tmp_path, f"report/{generation:05d}")
    return document


def cursor(tmp_path: Path) -> dict[str, int]:
    document: dict[str, int] = state_document(tmp_path, "cursor")
    return document


class ScriptedRollouts(StandInGenerator):
    """A generator service whose rollouts take ``outcomes`` in turn: a reward, or None for a rollout that could not
    run (no reward, the error ``rollout_error``). With ``error``, each rollout request fails with it instead.

    Each request is recorded in ``rollout_requests``.
    """

    def __init__(
        self,
        root: Path,
        outcomes: Sequence[float | None],
        *,
        is_solvable: bool = True,
        error: str = "",
        rollout_error: str = "sandbox failed",
    ) -> None:
        super().__init__(root, is_solvable=is_solvable)
        self.outcomes = list(outcomes)
        self.error = error
        self.rollout_error = rollout_error

    async def run_rollouts(
        self,
        task_path: Path,
        *,
        scenario: str,
        arm: str,
        count: int,
        extra_instruction_files: Sequence[str],
        tags: Mapping[str, str],
        model: str | None = None,
    ) -> tuple[TaskRollout, ...]:
        self.rollout_requests.append(
            {"task": task_path.name, "arm": arm, "count": count, "tags": dict(tags), "model": model}
        )
        if self.error:
            raise GeneratorError(self.error)
        rewards = [self.outcomes.pop(0) for _ in range(count)]
        return tuple(
            TaskRollout(
                task_path=task_path,
                name=task_path.name,
                rollout_id=f"rollout-{len(self.rollout_requests)}-{number}",
                reward=reward,
                rewards={} if reward is None else {"reward": reward},
                error=self.rollout_error if reward is None else "",
                receipts=(),
                failed_calls=0,
                report_agent_record_ids=(),
                trial_uri=None,
            )
            for number, reward in enumerate(rewards)
        )

    @property
    def requested_counts(self) -> list[object]:
        return [request["count"] for request in self.rollout_requests]


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
        "sources_path": str(sources_file(tmp_path, 2)),
        "agent_model": "gpt-6-sol",
        "agent_reef_url": "http://127.0.0.1:8901",
        **fields,
    }


# The Claude Code OAuth token of the scripted Claude Code runs, and the session id of their transcript.
TOKEN = "sk-ant-oat01-reef-test-token-7f3a91"
SESSION_ID = "0b6f2c1e-5d4a-4e8b-9c3f-2a1d7e6b5c40"


def result_json(subtype: str = "success", is_error: bool = False, **fields: object) -> str:
    """What ``claude -p --output-format json`` prints at its end: a success carries its reply in ``result``, an
    error subtype its ``errors``."""
    document = {"type": "result", "subtype": subtype, "is_error": is_error, "num_turns": 3, "session_id": SESSION_ID}
    return json.dumps({**({"result": "Done."} if subtype == "success" else {}), **document, **fields})


def checked(arguments: Sequence[str], environ: Mapping[str, str]) -> str:
    """What the agent's ``python3 "$REEF_DESIGNER_CHECK" ...`` prints."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        task_check.main(list(arguments), environ)
    return output.getvalue()


class ScriptedClaudeSession(E2BSession):
    """A sandbox that runs no Claude Code. ``launch`` keeps what it was given, writes what Claude Code would (the
    task, a transcript named ``session_name``, its own state), runs the agent's ``task_check`` commands
    (``checks``) against the gateway, and prints ``stdout``, or times out with ``is_timeout``. ``leak`` goes into
    the transcript and ``task_leak`` into the task; ``link_leak`` goes into a workspace file that a link in the
    task's tests reads. With ``stop``, it sets the stop signal (a cancellation mid run) and waits for the sandbox
    to be closed. ``launch_error`` and ``close_error`` are what a failing launch and close raise."""

    def __init__(
        self,
        *,
        stdout: str = "",
        exit_code: int = 0,
        stderr: str = "",
        checks: Sequence[Sequence[str]] = (),
        leak: str = "",
        task_leak: str = "",
        link_leak: str = "",
        session_name: str = SESSION_ID,
        is_timeout: bool = False,
        launch_error: str = "",
        close_error: str = "",
        stop: threading.Event | None = None,
    ) -> None:
        self.env: dict[str, str] = {}
        self.stdout = stdout or result_json()
        self.exit_code = exit_code
        self.stderr = stderr
        self.checks = checks
        self.leak = leak
        self.task_leak = task_leak
        self.link_leak = link_leak
        self.session_name = session_name
        self.is_timeout = is_timeout
        self.launch_error = launch_error
        self.close_error = close_error
        self.stop = stop
        self.is_closed = threading.Event()
        self.argv: list[str] = []
        self.launch_env: dict[str, str] = {}
        self.writable: list[Path] = []
        self.readonly: list[Path] = []
        self.check_outputs: list[str] = []

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
        self.argv = list(argv)
        self.launch_env = dict(env)
        self.writable = list(writable_paths)
        self.readonly = list(readonly_paths)
        if self.stop is not None:
            self.stop.set()
            self.is_closed.wait(10)
            raise ProgramLaunchError("the E2B sandbox could not run the process: it is gone")
        if self.launch_error:
            raise ProgramLaunchError(self.launch_error)
        task_dir = write_workspace_task(workspace / "task", 8600)
        if self.task_leak:
            with (task_dir / "instruction.md").open("a") as stream:
                stream.write(f"\nThe key is {self.task_leak}.\n")
        if self.link_leak:
            (workspace / "notes.txt").write_text(self.link_leak)
            (task_dir / "tests" / "data.txt").symlink_to("../../notes.txt")
        self.check_outputs = [checked(arguments, env) for arguments in self.checks]
        projects = root / "claude" / "projects" / f"-home-user-reef-{root.name}-workspace"
        projects.mkdir(parents=True, exist_ok=True)
        events = [
            {"type": "user", "message": {"role": "user", "content": "Rebuild"}},
            {"type": "x", "note": self.leak},
        ]
        (projects / f"{self.session_name}.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        (root / "claude" / ".claude.json").write_text(json.dumps({"oauthAccount": {"emailAddress": "o@example.com"}}))
        (root / "claude" / "statsig").mkdir(exist_ok=True)
        (root / "claude" / "statsig" / "cache").write_text("gates")
        (root / "home" / ".cache").mkdir(parents=True, exist_ok=True)
        if self.is_timeout:
            raise ProgramTimeout("the agent ran out of time")
        return ProcessOutcome(self.exit_code, self.stdout, self.stderr)

    def pull(self, root: Path, relative: str = "") -> None:
        return None

    def close(self) -> None:
        self.is_closed.set()
        if self.close_error:
            raise ProgramLaunchError(self.close_error)


@dataclass(frozen=True)
class ClaudeExecutor(E2BExecutor):
    """An executor whose sandbox is the test's session."""

    session: E2BSession | None = None

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[ProgramExecutor, str]:
        return self, program.binary

    def open(self) -> E2BSession:
        if self.session is None:
            raise ProgramLaunchError("no scripted session")
        return self.session
