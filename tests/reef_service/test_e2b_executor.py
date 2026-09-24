"""Hosted episodes use the ordinary factory and preserve the runner's contracts."""

import json
import shlex
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.e2b import E2BExecutor, E2BSession, protected_command, remote_root, unpack
from reef.harness.episodes.executor import EpisodeLaunchError, SandboxUnavailable, build_executor
from reef.harness.episodes.run import run_episode


def test_factory_resolves_remote_binary_and_only_explicit_environment(monkeypatch):
    checked = []
    monkeypatch.setattr(E2BExecutor, "preflight", lambda self: checked.append(self))
    executor = build_executor(
        {
            "executor": "e2b",
            "adapter": "pi",
            "binary": "/opt/agent/pi",
            "sandbox": {"e2b_api_key_env": "TEST_SANDBOX_KEY", "env_from": ["MODEL_SETTING"], "forward_ports": [9000]},
        },
        environ={"TEST_SANDBOX_KEY": "test-key", "MODEL_SETTING": "enabled", "PRIVATE_HOST_VALUE": "hidden"},
    )
    assert isinstance(executor, E2BExecutor)
    assert executor.binary == "/opt/agent/pi" and executor.npm_package.endswith("@0.84.2")
    assert executor.env == {"MODEL_SETTING": "enabled"} and executor.forward_ports == (9000,)
    assert checked == [executor]
    assert "test-key" not in repr(executor)


@pytest.mark.parametrize(
    "section",
    [
        {"limits": {"processes": 10}},
        {"egress_hosts": ["example.org"]},
        {"forward_ports": [True]},
        {"forward_ports": [65536]},
        {"env_from": ["MISSING"]},
        {"e2b_api_key_env": False},
        {"unexpected": 1},
    ],
)
def test_unsupported_or_invalid_settings_fail_before_allocation(section):
    with pytest.raises(SandboxUnavailable):
        E2BExecutor.from_config(section, {"E2B_API_KEY": "test-key"})


def test_non_npm_agents_need_a_template_and_container_agents_keep_their_own_execution():
    with pytest.raises(SandboxUnavailable, match="explicit e2b_template"):
        E2BExecutor(api_key="test-key").for_adapter(get_adapter("native"))
    executor = E2BExecutor(api_key="test-key", template="custom").for_adapter(get_adapter("native"))
    assert executor.binary == "reef-native"
    with pytest.raises(SandboxUnavailable, match="own containers"):
        E2BExecutor(api_key="test-key", template="custom").for_adapter(get_adapter("terminus"))


def test_preflight_checks_remote_readiness_once_and_always_closes(monkeypatch):
    pytest.importorskip("e2b")
    session = Mock()
    opened = []

    def open_session(executor):
        opened.append(executor)
        return session

    monkeypatch.setattr(E2BExecutor, "open", open_session)
    executor = E2BExecutor(
        api_key="test-key", template="custom", binary="pi", owner="deployment", forward_ports=(9000,)
    )
    executor.preflight()
    executor.preflight()
    assert len(opened) == 1 and opened[0].owner == "" and opened[0].forward_ports == ()
    assert "command -v pi" in session.sandbox.commands.run.call_args.args[0]
    session.close.assert_called_once()


def test_unreachable_provider_refuses_startup(monkeypatch):
    pytest.importorskip("e2b")

    def fail(executor):
        raise EpisodeLaunchError("provider unavailable")

    monkeypatch.setattr(E2BExecutor, "open", fail)
    with pytest.raises(SandboxUnavailable, match="provider unavailable"):
        E2BExecutor(api_key="test-key", template="custom").preflight()


def test_remote_command_mounts_inputs_readonly_inside_writable_state(tmp_path):
    command = protected_command(
        "pi",
        {"HOME": remote_root(tmp_path)},
        tmp_path,
        tmp_path / "workspace",
        (tmp_path / "pi-agent",),
        (tmp_path / "pi-agent/models.json",),
    )
    args = shlex.split(command)
    state = f"{remote_root(tmp_path)}/pi-agent"
    config = f"{state}/models.json"
    assert args[:3] == ["exec", "bwrap", "--unshare-user"]
    assert args[args.index(state) - 1 : args.index(state) + 2] == ["--bind", state, state]
    assert args[args.index(config) - 1 : args.index(config) + 2] == ["--ro-bind", config, config]
    assert "--unshare-pid" in args and "--clearenv" in args


def test_failed_copy_back_cannot_be_scored_as_a_success(tmp_path, monkeypatch):
    pytest.importorskip("e2b")
    sandbox = Mock()
    sandbox.commands.run.return_value.wait.return_value = SimpleNamespace(exit_code=0, stdout="ok", stderr="")
    session = E2BSession(sandbox)
    monkeypatch.setattr(session, "push", lambda root: None)

    def fail(root):
        raise OSError("connection lost")

    monkeypatch.setattr(session, "pull", fail)
    try:
        with pytest.raises(EpisodeLaunchError, match=r"copy .* back"):
            session.launch(["pi"], root=tmp_path, workspace=tmp_path, env={}, timeout=10)
    finally:
        session.close()


def test_invalid_archive_does_not_destroy_the_local_episode(tmp_path):
    import tarfile

    root = tmp_path / "episode"
    root.mkdir()
    (root / "input").write_text("original")
    with pytest.raises(tarfile.ReadError):
        unpack(b"interrupted archive", root)
    assert (root / "input").read_text() == "original"


def test_cleanup_failure_is_reported_and_can_be_retried():
    sandbox = Mock()
    sandbox.kill.side_effect = [OSError("connection lost"), True]
    session = E2BSession(sandbox)
    with pytest.raises(EpisodeLaunchError, match="could not stop"):
        session.close()
    session.close()
    session.close()
    assert sandbox.kill.call_count == 2 and session.closed


def test_session_preserves_script_driver_commands(tmp_path, monkeypatch):
    pytest.importorskip("e2b")
    sandbox = Mock()
    sandbox.commands.run.return_value.wait.return_value = SimpleNamespace(exit_code=0, stdout="ok", stderr="")
    session = E2BSession(sandbox, binary="pi")
    monkeypatch.setattr(session, "push", lambda root: None)
    monkeypatch.setattr(session, "pull", lambda root: None)
    try:
        session.launch(["node", "trial.mjs"], root=tmp_path, workspace=tmp_path, env={}, timeout=10)
        command = shlex.split(sandbox.commands.run.call_args.args[0])
        assert command[-1] == "exec node trial.mjs"
    finally:
        session.close()


def test_ordinary_episode_reads_remote_trajectory_and_residue_and_closes(monkeypatch, tmp_path):
    pytest.importorskip("e2b")
    sandbox = Mock()
    sandbox.commands.run.return_value.wait.return_value = SimpleNamespace(exit_code=0, stdout="answer", stderr="")
    session = E2BSession(sandbox, binary="pi")
    monkeypatch.setattr(session, "push", lambda root: None)

    def pull(root):
        (root / "sessions/result.jsonl").write_text(
            json.dumps({"type": "message", "message": {"role": "assistant", "content": "answer"}}) + "\n"
        )
        (root / "unexpected.txt").write_text("residue")

    monkeypatch.setattr(session, "pull", pull)
    monkeypatch.setattr(E2BExecutor, "open", lambda self: session)
    result = run_episode(
        get_adapter("pi"),
        {"pi-agent/models.json": "{}"},
        "answer",
        executor=E2BExecutor(template="test"),
        keep_dir=tmp_path / "record",
    )
    assert result.exit_code == 0 and result.residue == ("unexpected.txt",)
    assert result.trajectory and (tmp_path / "record/result.jsonl").exists()
    assert "--ro-bind" in sandbox.commands.run.call_args.args[0]
    sandbox.kill.assert_called_once()
