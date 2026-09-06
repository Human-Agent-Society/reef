"""Lost command streams require protected completion or invalid retained evidence."""

import copy
import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from recipes.meta_harness.examples.terminal_bench.e2b_completion import RemoteCompletion
from recipes.meta_harness.examples.terminal_bench.e2b_transport import failure_evidence
from reef.harness.executor import EpisodeLaunchError

from .test_meta_harness_e2b import SandboxDouble, launch


def reader():
    calls = []
    sandbox = SimpleNamespace(commands=SimpleNamespace())
    completion = RemoteCompletion(
        sandbox, "/python", ["/python", "runner"], "/episode/workspace", "a" * 64, lambda: 60
    )
    value = {
        "status": "complete",
        "receipt": {
            "protocol": "e2b-protected-completion-v1",
            "token": completion.token,
            "wrapper_pid": 123,
            "child_pid": 124,
            "command_sha256": completion.command_sha256,
            "input_sha256": "a" * 64,
            "started_at": "2026-09-05T09:00:00+00:00",
            "finished_at": "2026-09-05T10:00:00+00:00",
            "status": "complete",
            "exit_code": 7,
            "child_return_code": 7,
        },
        "stdout": "out",
        "stderr": "err",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=json.dumps(value))

    sandbox.commands.run = run
    return completion, value, calls


def test_matching_complete_receipt_preserves_exit_and_caches_without_another_command():
    completion, _, calls = reader()
    result = completion.read_finished(123)
    assert (result.exit_code, result.stdout, result.stderr) == (7, "out", "err")
    assert completion.read_finished(123) is result and len(calls) == 1
    assert calls[0][1]["user"] == "root"
    assert completion.evidence(end_received=False)["command_end_received"] is False
    with pytest.raises(EpisodeLaunchError):
        completion.read_finished(125)


@pytest.mark.parametrize("state", ["running", "missing"])
def test_missing_or_running_receipt_is_not_completion(state):
    completion, value, _ = reader()
    value["status"] = state
    value["receipt"]["status"] = state
    assert completion.read_finished(123) is None
    with pytest.raises(EpisodeLaunchError):
        completion.evidence(end_received=False)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("protocol", "other"),
        ("token", "b" * 32),
        ("wrapper_pid", 124),
        ("wrapper_pid", True),
        ("child_pid", 0),
        ("command_sha256", "b" * 64),
        ("input_sha256", "b" * 64),
        ("exit_code", True),
        ("exit_code", 256),
        ("child_return_code", 0),
        ("started_at", "2026-09-05T09:00:00"),
        ("finished_at", "2026-09-05T08:00:00+00:00"),
    ],
)
def test_mismatched_or_incomplete_receipt_never_supplies_an_exit_code(field, replacement):
    completion, value, _ = reader()
    value["receipt"][field] = replacement
    with pytest.raises(EpisodeLaunchError):
        completion.read_finished(123)
    assert completion.result is None


class DroppedStream(SandboxDouble):
    def __init__(self, *, completed):
        super().__init__()
        self.completed = completed
        self.probes = 0
        self.connections = []
        self.commands.connect = self.connect

    def connect(self, pid, **kwargs):
        from e2b.exceptions import NotFoundException

        self.connections.append(pid)
        raise NotFoundException("the original process is no longer attachable")

    def run(self, command, **kwargs):
        result = super().run(command, **kwargs)
        if "isolated_runner" in command:

            def wait():
                raise ConnectionError("stream dropped")

            return SimpleNamespace(pid=2099, wait=wait, disconnect=lambda: None)
        if "completion receipt requires its owner" in command:
            self.probes += 1
            if not self.completed:
                result.stdout = json.dumps({"status": "missing"})
            elif self.probes == 1:
                value = copy.deepcopy(self.completion)
                value["status"] = value["receipt"]["status"] = "running"
                result.stdout = json.dumps(value)
        return result


def test_drop_then_not_found_recovers_exact_completed_runner_without_restarting(monkeypatch, tmp_path):
    from recipes.meta_harness.examples.terminal_bench import e2b_transport

    monkeypatch.setattr(e2b_transport.time, "sleep", lambda _: None)
    sandbox = DroppedStream(completed=True)
    result = launch(monkeypatch, tmp_path, sandbox)
    assert result.exit_code == 0 and result.stdout == "done"
    assert sandbox.connections == [2099]
    assert sum("isolated_runner" in command for command, _ in sandbox.calls) == 1
    receipt = json.loads((tmp_path / "terminus/sessions/trial.json").read_text())["execution_completion"]
    assert receipt["wrapper_pid"] == 2099 and receipt["status"] == "complete"
    assert receipt["command_end_received"] is False
    assert sandbox.killed


@pytest.mark.parametrize("state", ["missing", "unfinished", "finished"])
def test_unrecoverable_stream_retains_evidence_before_cleanup_without_inventing_bill(monkeypatch, tmp_path, state):
    import hashlib

    from e2b import Sandbox

    from recipes.meta_harness.examples.terminal_bench import e2b_transport
    from recipes.meta_harness.examples.terminal_bench.e2b_executor import E2BEpisodeExecutor

    from .test_meta_harness_health import raw_trial

    monkeypatch.setattr(e2b_transport.time, "sleep", lambda _: None)
    sandbox = DroppedStream(completed=False)
    stream = io.BytesIO()
    files = {
        "terminus/trials/a/agent/trajectory.json": json.dumps(
            {"steps": [{"source": "assistant", "message": "retained SECRET evidence"}]}
        ).encode()
    }
    if state != "missing":
        trial = raw_trial(reward=1, cost=0.25)
        if state == "finished":
            trial["finished_at"] = "2026-09-04T01:00:00+00:00"
        files["terminus/trials/a/result.json"] = json.dumps(trial).encode()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, raw in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    sandbox.evidence = stream.getvalue()

    def kill(**kwargs):
        assert (tmp_path / "terminus/trials/a/agent/trajectory.json").exists()
        sandbox.killed = True

    sandbox.kill = kill
    monkeypatch.setattr(Sandbox, "create", lambda *args, **kwargs: sandbox)
    (tmp_path / "workspace").mkdir()
    sessions, trials = tmp_path / "terminus/sessions", tmp_path / "terminus/trials"
    sessions.mkdir(parents=True)
    trials.mkdir()
    executor = E2BEpisodeExecutor("snapshot:v1", hashlib.sha256(b"manifest").hexdigest())
    with pytest.raises(EpisodeLaunchError) as caught:
        executor.launch(
            ["reef-terminus-e2b", "--task", "terminal-bench/example"],
            root=tmp_path,
            workspace=tmp_path / "workspace",
            env={"E2B_API_KEY": "SECRET"},
            timeout=600,
            writable_paths=[sessions, trials],
        )
    diagnostics = caught.value.e2b_diagnostics
    assert diagnostics["evidence_collected"] is True and diagnostics["cleanup_confirmed"] is True
    assert diagnostics["command_end_received"] is False
    assert diagnostics.get("observed_cost_usd") == (0.25 if state == "finished" else None)
    evidence = failure_evidence(caught.value)
    assert "retained [REDACTED] evidence" in str(evidence) and "SECRET" not in str(evidence)
    assert evidence["trajectory"][0]["outcome"]["valid"] is False
    assert sandbox.connections == [2099]
    assert sum("isolated_runner" in command for command, _ in sandbox.calls) == 1
