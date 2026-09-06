"""No paid calls: exercise isolation admission, failures and evidence recovery."""

import hashlib
import io
import json
import shlex
import tarfile
from types import SimpleNamespace

import pytest

from recipes.meta_harness.examples.terminal_bench.e2b_executor import E2BEpisodeExecutor
from recipes.meta_harness.examples.terminal_bench.isolated_adapter import finalize_render, isolated_descriptor
from reef.harness.adapters import get_adapter
from reef.harness.executor import EpisodeLaunchError, EpisodeTimeout, LocalExecutor
from reef.harness.render import RenderError, render_composition


class Stream(io.BytesIO):
    def __iter__(self):
        return iter([self.getvalue()])


class SandboxDouble:
    sandbox_id = "owned-test-sandbox"

    def __init__(self, *, failure=None, mismatch=False, bad_output=False):
        self.failure = failure
        self.mismatch = mismatch
        self.calls = []
        self.killed = False
        self.files = SimpleNamespace(read=self.read, write=self.write)
        self.commands = SimpleNamespace(run=self.run)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            item = tarfile.TarInfo("../escape" if bad_output else "terminus/sessions/trial.json")
            item.size = 2
            archive.addfile(item, io.BytesIO(b"{}"))
        self.evidence = stream.getvalue()

    def read(self, path, **kwargs):
        if path == "/opt/reef-runtime.json":
            return b"different" if self.mismatch else b"manifest"
        return Stream(self.evidence)

    def write(self, path, payload, **kwargs):
        assert "SECRET" not in path

    def run(self, command, **kwargs):
        from recipes.meta_harness.examples.terminal_bench.e2b_completion import READ_RECEIPT

        self.calls.append((command, kwargs))
        parts = shlex.split(command)
        if READ_RECEIPT in parts:
            return SimpleNamespace(exit_code=0, stdout=json.dumps(self.completion), stderr="")
        if "isolated_runner" in command:
            assert kwargs["user"] == "root"
            assert "SECRET" not in command
            assert kwargs["envs"]["E2B_API_KEY"] == "SECRET"
            payload = json.loads(parts[-1])
            code = getattr(self.failure, "exit_code", 0)
            self.completion = {
                "status": "complete",
                "receipt": {
                    "protocol": "e2b-protected-completion-v1",
                    "token": payload["token"],
                    "wrapper_pid": 2099,
                    "child_pid": 2100,
                    "command_sha256": payload["command_sha256"],
                    "input_sha256": payload["input_sha256"],
                    "started_at": "2026-09-05T09:00:00+00:00",
                    "finished_at": "2026-09-05T10:00:00+00:00",
                    "status": "complete",
                    "exit_code": code,
                    "child_return_code": code,
                },
                "stdout": getattr(self.failure, "stdout", "done"),
                "stderr": getattr(self.failure, "stderr", ""),
                "stdout_truncated": False,
                "stderr_truncated": False,
            }
            if self.failure:
                from e2b.sandbox.commands.command_handle import CommandExitException

                if not isinstance(self.failure, CommandExitException):
                    raise self.failure

                def wait():
                    raise self.failure

                return SimpleNamespace(pid=2099, wait=wait, disconnect=lambda: None)
            return SimpleNamespace(
                pid=2099, wait=lambda: SimpleNamespace(exit_code=0, stdout="done", stderr=""), disconnect=lambda: None
            )
        return SimpleNamespace(exit_code=0, stdout="done", stderr="")

    def kill(self, **kwargs):
        self.killed = True


def launch(monkeypatch, tmp_path, sandbox):
    from e2b import Sandbox

    monkeypatch.setattr(Sandbox, "create", lambda *args, **kwargs: sandbox)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "terminus/sessions").mkdir(parents=True)
    executor = E2BEpisodeExecutor("snapshot:v1", hashlib.sha256(b"manifest").hexdigest())
    return executor.launch(
        ["reef-terminus-e2b", "--task", "terminal-bench/example"],
        root=tmp_path,
        workspace=tmp_path / "workspace",
        env={"E2B_API_KEY": "SECRET"},
        timeout=600,
        writable_paths=[tmp_path / "terminus/sessions"],
    )


def test_nonzero_run_collects_evidence_and_kills_sandbox(monkeypatch, tmp_path):
    from e2b.sandbox.commands.command_handle import CommandExitException

    sandbox = SandboxDouble(failure=CommandExitException(stdout="out", stderr="err", exit_code=7, error="failure"))
    result = launch(monkeypatch, tmp_path, sandbox)
    assert (result.exit_code, result.stdout, result.stderr) == (7, "out", "err")
    assert (
        json.loads((tmp_path / "terminus/sessions/trial.json").read_text())["execution_completion"]["exit_code"] == 7
    )
    assert sandbox.killed


def test_timeout_kills_runner_and_never_reports_success(monkeypatch, tmp_path):
    from e2b.exceptions import TimeoutException

    sandbox = SandboxDouble(failure=TimeoutException("elapsed"))
    with pytest.raises(EpisodeTimeout):
        launch(monkeypatch, tmp_path, sandbox)
    assert sandbox.killed


def test_initial_start_failure_preserves_process_and_sandbox_diagnostics(monkeypatch, tmp_path):
    sandbox = SandboxDouble(failure=ConnectionError("secret request body"))
    with pytest.raises(EpisodeLaunchError) as raised:
        launch(monkeypatch, tmp_path, sandbox)
    diagnostic = raised.value.e2b_diagnostics
    assert diagnostic["sandbox_id"] == "owned-test-sandbox"
    assert diagnostic["phase"] == "command_start"
    assert diagnostic["command_pid"] is None
    assert diagnostic["error_type"] == "ConnectionError"
    assert diagnostic["cleanup_confirmed"] is True
    assert "secret" not in str(diagnostic)


def test_interrupted_archive_download_restarts_read_without_reexecuting_runner(monkeypatch, tmp_path):
    from recipes.meta_harness.examples.terminal_bench import e2b_executor

    sandbox = SandboxDouble()
    attempts = []
    original_read = sandbox.read

    class BrokenStream(Stream):
        def __iter__(self):
            yield b"partial tar header"
            raise ConnectionError("reset")

    def read(path, **kwargs):
        if path == "/tmp/reef-output.tar":
            attempts.append(path)
            if len(attempts) == 1:
                return BrokenStream()
        return original_read(path, **kwargs)

    sandbox.files.read = read
    monkeypatch.setattr(e2b_executor.time, "sleep", lambda _: None)
    assert launch(monkeypatch, tmp_path, sandbox).exit_code == 0
    assert len(attempts) == 2
    assert sum("isolated_runner" in command for command, _ in sandbox.calls) == 1
    assert (
        json.loads((tmp_path / "terminus/sessions/trial.json").read_text())["execution_completion"]["exit_code"] == 0
    )


def test_cleanup_failure_keeps_final_cost_and_collected_evidence(monkeypatch, tmp_path):
    sandbox = SandboxDouble()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        raw = json.dumps(
            {
                "task_name": "example",
                "finished_at": "2026-09-05T10:00:00Z",
                "agent_execution": {"started_at": "2026-09-05T09:00:00Z"},
                "agent_result": {"cost_usd": 0.25},
            }
        ).encode()
        item = tarfile.TarInfo("terminus/trials/a/result.json")
        item.size = len(raw)
        archive.addfile(item, io.BytesIO(raw))
    sandbox.evidence = stream.getvalue()

    def kill(**kwargs):
        raise ConnectionError("secret cleanup request")

    sandbox.kill = kill
    from e2b import Sandbox

    monkeypatch.setattr(Sandbox, "create", lambda *a, **k: sandbox)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "terminus/trials").mkdir(parents=True)
    (tmp_path / "terminus/sessions").mkdir()
    executor = E2BEpisodeExecutor("snapshot:v1", hashlib.sha256(b"manifest").hexdigest())
    with pytest.raises(EpisodeLaunchError, match="cleanup") as raised:
        executor.launch(
            ["reef-terminus-e2b", "--task", "terminal-bench/example"],
            root=tmp_path,
            workspace=tmp_path / "workspace",
            env={"E2B_API_KEY": "SECRET"},
            timeout=600,
            writable_paths=[tmp_path / "terminus/trials", tmp_path / "terminus/sessions"],
        )
    diagnostic = raised.value.e2b_diagnostics
    assert diagnostic["observed_cost_usd"] == 0.25
    assert diagnostic["command_end_received"] is True
    assert diagnostic["evidence_collected"] is True
    assert diagnostic["cleanup_confirmed"] is False
    assert (tmp_path / "terminus/trials/a/result.json").is_file()
    from recipes.meta_harness.examples.terminal_bench.e2b_transport import failure_evidence

    retained = failure_evidence(raised.value)
    assert retained["trajectory"][0]["outcome"]["cost_usd"] == 0.25
    assert retained["trajectory"][0]["execution_completion"]["exit_code"] == 0


@pytest.mark.parametrize("finished", [False, True])
def test_killed_runner_partial_evidence_reaches_normal_reader_without_inventing_usage(tmp_path, finished):
    from tests.reef_service.test_meta_harness_health import raw_trial

    from recipes.meta_harness.examples.terminal_bench.e2b_executor import preserve_missing_summary
    from reef.harness.trajectory import read_terminus_atif

    sessions, trials = tmp_path / "terminus/sessions", tmp_path / "terminus/trials"
    (trials / "a/agent").mkdir(parents=True)
    (trials / "a/agent/trajectory.json").write_text(
        json.dumps(
            {
                "agent": {"api_key": "never-copy-extra-metadata"},
                "steps": [{"source": "assistant", "message": "partial command API-SECRET"}],
            }
        )
    )
    (trials / "a/trial.log").write_text("last diagnostic API-SECRET")
    if finished:
        (trials / "a/result.json").write_text(json.dumps(raw_trial(reward=1, cost=0.25)))
    preserve_missing_summary(
        ["reef-terminus-e2b", "--task", "terminal-bench/example"],
        tmp_path,
        [sessions, trials],
        {"OPENAI_API_KEY": "API-SECRET"},
        {"sandbox_id": "owned", "command_pid": 2153},
        -1,
    )
    events = read_terminus_atif(sessions)
    verifier = events[0]
    assert verifier["type"] == "verifier" and verifier["outcome"]["valid"] is False
    assert verifier["outcome"]["phase"] == "missing_runner_summary"
    assert verifier["observed_cost_usd"] == (0.25 if finished else None)
    assert verifier["runner_exit_code"] == -1
    assert verifier["transport_diagnostics"]["command_pid"] == 2153
    assert any("partial command [REDACTED]" in str(event) for event in events)
    assert "API-SECRET" not in str(events) and "never-copy-extra-metadata" not in str(events)


@pytest.mark.parametrize("options", [{"mismatch": True}, {"bad_output": True}])
def test_invalid_runtime_or_outputs_fail_closed(monkeypatch, tmp_path, options):
    sandbox = SandboxDouble(**options)
    with pytest.raises(EpisodeLaunchError):
        launch(monkeypatch, tmp_path, sandbox)
    assert sandbox.killed
    if options.get("mismatch"):
        assert not sandbox.calls


def test_executable_candidate_is_compiled_but_never_imported_on_host(tmp_path):
    marker = tmp_path / "host-executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nclass Agent: pass\n"
    files = {"terminus/config.json": "{}", "terminus/context/candidate.py": source}
    assert finalize_render(files) == files
    assert not marker.exists()
    descriptor = get_adapter("terminus")
    with pytest.raises(ValueError, match="E2B"):
        isolated_descriptor(descriptor, LocalExecutor())
    with pytest.raises(RenderError, match="not supported safely"):
        render_composition([("code_extension", {"name": "candidate", "code": source})], descriptor)


def test_isolated_runner_refuses_host_invocation():
    from recipes.meta_harness.examples.terminal_bench.isolated_runner import require_isolation

    with pytest.raises(RuntimeError, match="prepared E2B"):
        require_isolation()


def test_remote_import_uses_content_identity_and_requires_terminus(monkeypatch, tmp_path):
    import sys

    from harbor.agents.terminus_2 import Terminus2

    from recipes.meta_harness.examples.terminal_bench import isolated_runner

    monkeypatch.setattr(isolated_runner, "require_isolation", lambda: None)
    code = "from harbor.agents.terminus_2 import Terminus2\nclass Agent(Terminus2): pass\n"
    tree = {"terminus/config.json": '{"model_name":"test"}', "terminus/context/os.py": code}
    path = tmp_path / "terminus/context/os.py"
    path.parent.mkdir(parents=True)
    path.write_text(code)
    agent = isolated_runner.make_agent(tmp_path, tree)
    assert "name" not in agent
    name = agent["import_path"].split(":")[0]
    assert name.startswith("reef_candidate_")
    # Real Harbor import resolution, without constructing an LLM or a trial.
    from harbor.agents.factory import _import_agent_class

    assert issubclass(_import_agent_class(agent["import_path"]), Terminus2)
    del sys.modules[name]
    code = "class Agent: pass\n"
    tree["terminus/context/os.py"] = code
    path.write_text(code)
    with pytest.raises(ValueError, match="subclass"):
        isolated_runner.make_agent(tmp_path, tree)
