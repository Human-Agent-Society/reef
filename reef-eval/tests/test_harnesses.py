"""Offline contract tests for the reference harness adapters."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from examples.harnesses.coral.config import coral_config
from examples.harnesses.openevolve.config import openevolve_config

ROOT = Path(__file__).parent.parent
OPENEVOLVE = ROOT / "examples" / "harnesses" / "openevolve"
CORAL_GRADER_SRC = ROOT / "examples" / "harnesses" / "coral" / "grader" / "src"


class _Judge(BaseHTTPRequestHandler):
    submissions: list[dict] = []

    def do_POST(self):
        assert self.path == "/submit"
        length = int(self.headers["Content-Length"])
        solution = json.loads(self.rfile.read(length))
        self.submissions.append(solution)
        self._reply(
            {
                "score": 0.75,
                "best": 0.75,
                "remaining": 9,
                "reason": "valid",
            }
        )

    def do_GET(self):
        assert self.path == "/status"
        self._reply({"used": len(self.submissions), "remaining": None})

    def _reply(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _judge_server():
    _Judge.submissions = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Judge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "reef_eval_openevolve_evaluator", OPENEVOLVE / "evaluator.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openevolve_evaluator_submits_candidate_to_tide(monkeypatch):
    server = _judge_server()
    try:
        monkeypatch.setenv("JUDGE_URL", f"http://127.0.0.1:{server.server_port}")
        result = _load_evaluator().evaluate(str(OPENEVOLVE / "initial_program.py"))
    finally:
        server.shutdown()
        server.server_close()

    assert result == {"score": 0.75}
    assert len(_Judge.submissions) == 1
    assert len(_Judge.submissions[0]["circles"]) == 3


def test_coral_grader_client_uses_same_judge(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CORAL_GRADER_SRC))
    from reef_eval_coral_grader.judge import submit_file

    solution = tmp_path / "solution.json"
    solution.write_text('{"circles": []}')
    server = _judge_server()
    try:
        result = submit_file(solution, f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()

    assert result["score"] == 0.75
    assert _Judge.submissions == [{"circles": []}]


def test_codex_harness_reuses_version_pinned_harbor_agent(tmp_path):
    pytest.importorskip("harbor")
    from harbor.agents.installed.codex import Codex

    from examples.harnesses.codex.agent import CODEX_VERSION, CodexHarness

    assert issubclass(CodexHarness, Codex)
    harness = CodexHarness(logs_dir=tmp_path, model_name="openai/test-model")
    assert harness.name() == "codex"
    assert harness.version() == CODEX_VERSION

    with pytest.raises(ValueError, match=f"requires Codex {CODEX_VERSION}"):
        CodexHarness(
            logs_dir=tmp_path,
            model_name="openai/test-model",
            version="different-version",
        )


def _load_codex_finalize():
    spec = importlib.util.spec_from_file_location(
        "reef_eval_codex_finalize", ROOT / "examples" / "harnesses" / "finalize.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codex_finalize_submits_artifact_when_judge_saw_nothing(tmp_path, monkeypatch):
    server = _judge_server()
    artifact = tmp_path / "solution.json"
    artifact.write_text(json.dumps({"circles": [[0.5, 0.5, 0.5]]}))
    try:
        monkeypatch.setenv("JUDGE_URL", f"http://127.0.0.1:{server.server_port}")
        result = _load_codex_finalize().finalize(artifact)
    finally:
        server.shutdown()
        server.server_close()

    assert result["score"] == 0.75
    assert _Judge.submissions == [{"circles": [[0.5, 0.5, 0.5]]}]


def test_codex_finalize_stays_out_when_agent_already_submitted(tmp_path, monkeypatch):
    server = _judge_server()
    artifact = tmp_path / "solution.json"
    artifact.write_text(json.dumps({"circles": []}))
    try:
        monkeypatch.setenv("JUDGE_URL", f"http://127.0.0.1:{server.server_port}")
        _Judge.submissions.append({"circles": [[0.25, 0.25, 0.25]]})
        result = _load_codex_finalize().finalize(artifact)
    finally:
        server.shutdown()
        server.server_close()

    # Best-of semantics already cover the run: no extra submission is spent.
    assert result is None
    assert len(_Judge.submissions) == 1


def test_codex_finalize_skips_without_judge_or_artifact(tmp_path, monkeypatch):
    finalize = _load_codex_finalize().finalize
    monkeypatch.delenv("JUDGE_URL", raising=False)
    assert finalize(tmp_path / "solution.json") is None

    monkeypatch.setenv("JUDGE_URL", "http://127.0.0.1:1")
    assert finalize(tmp_path / "missing.json") is None


class _FakeEnvironment:
    def __init__(self, *, workdir: str = "/app", fail: bool = False):
        self.workdir = workdir
        self.fail = fail
        self.uploads: list[tuple] = []
        self.commands: list[str] = []

    async def upload_file(self, source_path, target_path):
        self.uploads.append((Path(source_path), target_path))

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if self.fail:
            raise RuntimeError("environment down")
        stdout = f"{self.workdir}\n" if command == "pwd" else ""
        return SimpleNamespace(stdout=stdout, return_code=0)


async def test_codex_fallback_submits_final_artifact_after_run(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.agents.installed.codex import Codex

    from examples.harnesses.codex.agent import CodexHarness

    async def fake_run(self, instruction, environment, context):
        return None

    monkeypatch.setattr(Codex, "run", fake_run)
    harness = CodexHarness(logs_dir=tmp_path, model_name="openai/test-model")
    environment = _FakeEnvironment()
    await harness.run("pack circles", environment, SimpleNamespace())

    assert [target for _, target in environment.uploads] == [
        "/tmp/reef_eval_finalize.py"
    ]
    assert environment.commands == [
        "pwd",
        "python3 /tmp/reef_eval_finalize.py /app/solution.json",
    ]


async def test_codex_fallback_survives_agent_crash_and_env_failure(
    tmp_path, monkeypatch
):
    pytest.importorskip("harbor")
    from harbor.agents.installed.codex import Codex

    from examples.harnesses.codex.agent import CodexHarness

    async def crashing_run(self, instruction, environment, context):
        raise TimeoutError("budget spent")

    async def fake_run_noop(self, instruction, environment, context):
        return None

    monkeypatch.setattr(Codex, "run", crashing_run)
    harness = CodexHarness(logs_dir=tmp_path, model_name="openai/test-model")

    # A timeout is a normal ending: the fallback still runs afterwards...
    environment = _FakeEnvironment()
    with pytest.raises(TimeoutError):
        await harness.run("pack circles", environment, SimpleNamespace())
    assert any("reef_eval_finalize" in command for command in environment.commands)

    # ...a broken environment never turns the fallback into a trial failure...
    monkeypatch.setattr(Codex, "run", fake_run_noop)
    broken = _FakeEnvironment(fail=True)
    await harness.run("pack circles", broken, SimpleNamespace())

    # ...and final_artifact=None disables the fallback entirely.
    harness_no_fallback = CodexHarness(
        logs_dir=tmp_path, model_name="openai/test-model", final_artifact=None
    )
    disabled_env = _FakeEnvironment()
    await harness_no_fallback.run("pack circles", disabled_env, SimpleNamespace())
    assert disabled_env.commands == []


async def test_coral_finalize_submits_shared_repo_solution(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from examples.harnesses.coral.agent import CoralHarness

    harness = CoralHarness(logs_dir=tmp_path, model_name="openai/test-model")
    submitted: list[str] = []

    async def fake_submit(environment, artifact):
        submitted.append(artifact)

    monkeypatch.setattr(harness, "_submit_final_artifact", fake_submit)
    await harness._finalize(SimpleNamespace())
    assert submitted == ["/opt/reef-eval-harness/coral/seed/solution.json"]


def _recording_harness(tmp_path, events: list[str], fail_at: str | None = None):
    """A concrete TideHarnessBase recording the SOP phases it runs, in order."""
    pytest.importorskip("harbor")
    from examples.harnesses.base import TideHarnessBase

    class Harness(TideHarnessBase):
        @staticmethod
        def name() -> str:
            return "sop-test"

        def version(self) -> str:
            return "test"

        async def setup(self, environment) -> None:
            pass

        async def _prepare(self, instruction, environment):
            events.append("prepare")
            return {"prepared": True}

        async def _launch(self, prepared, instruction, environment, context):
            assert prepared == {"prepared": True}
            events.append("launch")
            if fail_at == "launch":
                raise TimeoutError("budget spent")

        async def _finalize(self, environment):
            events.append("finalize")
            if fail_at == "finalize":
                raise RuntimeError("judge unreachable")

        async def _collect_usage(self, environment, context):
            events.append("collect_usage")
            if fail_at == "collect_usage":
                raise RuntimeError("usage unreadable")

    return Harness(logs_dir=tmp_path, model_name="openai/test-model")


async def test_sop_runs_phases_in_order(tmp_path):
    events: list[str] = []
    await _recording_harness(tmp_path, events).run(
        "instruction", SimpleNamespace(), SimpleNamespace()
    )
    assert events == ["prepare", "launch", "finalize", "collect_usage"]


async def test_sop_finalizes_and_meters_even_when_launch_stops(tmp_path):
    events: list[str] = []
    with pytest.raises(TimeoutError):
        await _recording_harness(tmp_path, events, fail_at="launch").run(
            "instruction", SimpleNamespace(), SimpleNamespace()
        )
    assert events == ["prepare", "launch", "finalize", "collect_usage"]


async def test_sop_finalize_and_usage_never_mask_the_run(tmp_path):
    for fail_at in ("finalize", "collect_usage"):
        events: list[str] = []
        await _recording_harness(tmp_path, events, fail_at=fail_at).run(
            "instruction", SimpleNamespace(), SimpleNamespace()
        )
        assert events == ["prepare", "launch", "finalize", "collect_usage"]


def test_generated_configs_keep_reef_eval_as_the_scorer():
    evolve = openevolve_config("test-model", "https://example.test/v1")
    assert evolve["llm"]["primary_model"] == "test-model"
    assert evolve["llm"]["api_base"] == "https://example.test/v1"
    assert evolve["evaluator"]["parallel_evaluations"] == 1

    coral = coral_config("pack circles", "test-model", agents=3)
    assert coral["grader"]["entrypoint"] == "reef_eval_coral_grader.grader:Grader"
    assert coral["agents"] == {
        "count": 3,
        "runtime": "codex",
        "model": "test-model",
        "max_turns": 200,
    }
    assert "coral eval" in coral["task"]["description"]


def test_coral_usage_aggregates_every_codex_turn(tmp_path):
    from examples.harnesses.coral.usage import usage_records

    logs = tmp_path / "results" / "run" / ".coral" / "public" / "logs"
    logs.mkdir(parents=True)
    (logs / "agent-1.0.log").write_text(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"one"}',
                '{"type":"turn.completed","usage":{"input_tokens":100,'
                '"cached_input_tokens":60,"output_tokens":20}}',
            ]
        )
    )
    (logs / "agent-2.0.log").write_text(
        '{"type":"turn.completed","usage":{"input_tokens":200,'
        '"cached_input_tokens":100,"output_tokens":40}}\n'
    )

    assert list(usage_records(tmp_path, "test-model")) == [
        {
            "model": "test-model",
            "input_tokens": 100,
            "cached_input_tokens": 60,
            "output_tokens": 20,
        },
        {
            "model": "test-model",
            "input_tokens": 200,
            "cached_input_tokens": 100,
            "output_tokens": 40,
        },
    ]


def test_base_harness_populates_context_tokens(tmp_path):
    pytest.importorskip("harbor")
    from harbor.agents.base import BaseAgent
    from harbor.models.agent.context import AgentContext

    from examples.harnesses.base import TideHarnessBase

    assert issubclass(TideHarnessBase, BaseAgent)

    class _UsageHarness(TideHarnessBase):
        @staticmethod
        def name() -> str:
            return "usage-test"

        def version(self) -> str:
            return "test"

        async def setup(self, environment) -> None:
            pass

        async def _launch(self, prepared, instruction, environment, context) -> None:
            pass

    harness = _UsageHarness(
        logs_dir=tmp_path,
        model_name="openai/gpt-5-mini",
    )
    context = AgentContext()
    harness._populate_usage(
        context,
        json.dumps(
            {
                "model": "gpt-5-mini",
                "input_tokens": 1_000,
                "cached_input_tokens": 600,
                "output_tokens": 200,
            }
        ),
    )

    assert harness._model_name() == "gpt-5-mini"
    assert context.n_input_tokens == 1_000
    assert context.n_cache_tokens == 600
    assert context.n_output_tokens == 200
    assert context.metadata["usage_records"] == 1


def test_openevolve_usage_tracking_records_sdk_usage(tmp_path, monkeypatch):
    pytest.importorskip("openevolve")
    usage_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("REEF_EVAL_USAGE_FILE", str(usage_path))
    spec = importlib.util.spec_from_file_location(
        "reef_eval_openevolve_usage", OPENEVOLVE / "usage.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_usage_tracking()

    module._record_usage(
        SimpleNamespace(
            model="gpt-5-mini",
            usage=SimpleNamespace(
                prompt_tokens=90,
                prompt_tokens_details=SimpleNamespace(cached_tokens=50),
                completion_tokens=20,
            ),
        ),
        "fallback-model",
    )
    assert json.loads(usage_path.read_text()) == {
        "model": "gpt-5-mini",
        "input_tokens": 90,
        "cached_input_tokens": 50,
        "output_tokens": 20,
    }


def test_openevolve_runner_installs_tracking_before_cli(monkeypatch):
    pytest.importorskip("openevolve")
    monkeypatch.syspath_prepend(str(OPENEVOLVE))
    spec = importlib.util.spec_from_file_location(
        "reef_eval_openevolve_runner", OPENEVOLVE / "runner.py"
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    events = []
    monkeypatch.setattr(
        runner, "install_usage_tracking", lambda: events.append("tracking")
    )
    fake_cli = ModuleType("openevolve.cli")
    fake_cli.main = lambda: events.append("cli") or 17
    monkeypatch.setitem(sys.modules, "openevolve.cli", fake_cli)

    assert runner.main() == 17
    assert events == ["tracking", "cli"]
