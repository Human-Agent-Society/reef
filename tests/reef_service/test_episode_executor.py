"""The episode executor abstraction: local behavior, sandbox construction,
and the fail-fast contract when a required sandbox cannot isolate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import reef
from reef.core.errors import ReefError
from reef.harness.executor import (
    EpisodeLaunchError,
    EpisodeTimeout,
    LocalExecutor,
    SandboxExecutor,
    SandboxLimits,
    SandboxUnavailable,
    build_executor,
)


def _script(tmp_path: Path, body: str, name: str = "prog") -> Path:
    path = tmp_path / name
    path.write_text(f"#!/usr/bin/env python3\n{body}\n")
    path.chmod(0o755)
    return path


def test_local_executor_runs_in_the_workspace_and_returns_the_outcome(tmp_path: Path) -> None:
    root = tmp_path / "root"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    prog = _script(tmp_path, "import os; print(os.getcwd()); print('err', file=__import__('sys').stderr)")
    outcome = LocalExecutor().launch([sys.executable, str(prog)], root=root, workspace=workspace, env={}, timeout=10.0)
    assert outcome.exit_code == 0
    assert outcome.stdout.strip() == str(workspace)
    assert "err" in outcome.stderr


def test_local_executor_does_not_forward_parent_stdin(tmp_path: Path) -> None:
    root = tmp_path / "root"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    reader = _script(tmp_path, "import sys; print(sys.stdin.read())", "reader")
    driver = _script(
        tmp_path,
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "from reef.harness.executor import LocalExecutor",
                f"root = Path({str(root)!r})",
                f"outcome = LocalExecutor().launch([sys.executable, {str(reader)!r}], ",
                "    root=root, workspace=root / 'workspace', env={}, timeout=10.0)",
                "print(outcome.stdout, end='')",
            ]
        ),
    )
    # The driver runs from tmp_path, so it needs reef on its own path: the
    # test job type-checks and tests the source tree without installing it.
    env = {**os.environ, "PYTHONPATH": str(Path(reef.__file__).resolve().parents[1])}
    result = subprocess.run(
        [sys.executable, str(driver)],
        input="INJECTED_STDIN_MARKER",
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    assert result.stdout == "\n"


def test_local_executor_maps_a_missing_binary(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "workspace").mkdir(parents=True)
    with pytest.raises(EpisodeLaunchError):
        LocalExecutor().launch(
            [str(tmp_path / "does-not-exist")], root=root, workspace=root / "workspace", env={}, timeout=10.0
        )


def test_local_executor_kills_the_tree_on_timeout(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "workspace").mkdir(parents=True)
    prog = _script(tmp_path, "import time; time.sleep(30)")
    with pytest.raises(EpisodeTimeout):
        LocalExecutor().launch(
            [sys.executable, str(prog)], root=root, workspace=root / "workspace", env={}, timeout=0.5
        )


def test_build_executor_defaults_to_local() -> None:
    assert isinstance(build_executor(None), LocalExecutor)
    assert isinstance(build_executor({"executor": "local"}), LocalExecutor)


def test_build_executor_rejects_an_unknown_kind() -> None:
    with pytest.raises(ReefError, match="unknown episode executor"):
        build_executor({"executor": "vm"})


def test_sandbox_preflight_fails_fast_without_bubblewrap(monkeypatch) -> None:
    monkeypatch.setattr("reef.harness.executor.shutil.which", lambda name: None)
    with pytest.raises(SandboxUnavailable, match="bubblewrap"):
        SandboxExecutor().preflight()
    with pytest.raises(ReefError, match="bubblewrap"):
        build_executor({"executor": "sandbox"})


def _probe_result(monkeypatch, returncode: int, stderr: str = "") -> list[list[str]]:
    """Stand in for the nesting probe: record the argv and answer with the given exit."""
    seen: list[list[str]] = []

    def run(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode, "", stderr)

    monkeypatch.setattr("reef.harness.executor.subprocess.run", run)
    return seen


def test_sandbox_preflight_proves_the_host_can_nest_a_jail(monkeypatch) -> None:
    monkeypatch.setattr("reef.harness.executor.shutil.which", lambda name: "/usr/bin/bwrap")
    seen = _probe_result(monkeypatch, 1, "bwrap: No permissions to create a new namespace\n")
    with pytest.raises(SandboxUnavailable, match=r"cannot nest a bubblewrap jail.*No permissions"):
        SandboxExecutor().preflight()
    # An episode jail with a tool jail inside it around true: the shape every sandboxed tool call takes.
    argv = seen[0]
    episode, tool = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
    assert episode[0] == tool[0] == "/usr/bin/bwrap" and tool[-2:] == ["--", "/bin/true"]
    for flag in ("--unshare-user", "--unshare-pid", "--unshare-net", "--dev", "--clearenv"):
        assert flag in episode and flag in tool
    # The episode jail mounts a fresh /proc; the tool jail cannot nest that, so it binds the episode's read only.
    assert "--proc" in episode and "--proc" not in tool
    assert tool[tool.index("/proc") - 1 : tool.index("/proc") + 2] == ["--ro-bind", "/proc", "/proc"]
    for jail in (episode, tool):
        assert jail[jail.index("/usr") - 1 : jail.index("/usr") + 2] == ["--ro-bind", "/usr", "/usr"]
    _probe_result(monkeypatch, 0)
    SandboxExecutor().preflight()

    def refuse(argv, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr("reef.harness.executor.subprocess.run", refuse)
    with pytest.raises(SandboxUnavailable, match="cannot nest a bubblewrap jail on this host: boom"):
        SandboxExecutor().preflight()


def test_sandbox_argv_isolates_the_filesystem_env_and_network(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("reef.harness.executor.shutil.which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr("reef.harness.executor.os.environ", {"PATH": "/usr/bin"})
    root = tmp_path / "root"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    sandbox = SandboxExecutor()
    argv = sandbox._bwrap_argv([sys.executable, "-c", "pass"], root=root, workspace=workspace, env={"A": "b"})

    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv  # no egress hosts: network is denied
    assert "--die-with-parent" in argv
    assert "--clearenv" in argv
    # The root is read-only, its workspace writable.
    ro = [argv[i + 2] for i, tok in enumerate(argv) if tok == "--ro-bind" and argv[i + 1] == str(root)]
    rw = [argv[i + 2] for i, tok in enumerate(argv) if tok == "--bind" and argv[i + 1] == str(workspace)]
    assert ro == [str(root)]
    assert rw == [str(workspace)]
    # The overlay env is passed; host secrets are not (only PATH from inherited).
    assert "--setenv" in argv
    joined = " ".join(argv)
    assert "A b" in joined
    # The native loop reads this and holds each tool call to its declared capabilities.
    assert "--setenv REEF_NATIVE_ENFORCE bwrap" in joined


def test_sandbox_opens_runtime_state_but_rebinds_rendered_inputs_read_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("reef.harness.executor.shutil.which", lambda name: "/usr/bin/bwrap")
    root = tmp_path / "root"
    workspace = root / "workspace"
    state = root / "codex"
    config = state / "config.toml"
    workspace.mkdir(parents=True)
    state.mkdir()
    config.write_text("model = 'm'\n")

    argv = SandboxExecutor()._bwrap_argv(
        ["true"],
        root=root,
        workspace=workspace,
        env={},
        writable_paths=(state,),
        readonly_paths=(config,),
    )

    root_mount = argv.index("--ro-bind", argv.index(str(root)) - 1)
    state_mount = argv.index("--bind", root_mount + 1)
    config_mount = argv.index("--ro-bind", root_mount + 1)
    workspace_mount = argv.index("--bind", state_mount + 1)
    assert root_mount < state_mount < config_mount < workspace_mount
    assert argv[state_mount + 1 : state_mount + 3] == [str(state), str(state)]
    assert argv[config_mount + 1 : config_mount + 3] == [str(config), str(config)]


def test_sandbox_shares_the_network_when_egress_is_allowlisted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("reef.harness.executor.shutil.which", lambda name: "/usr/bin/bwrap")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = SandboxExecutor(egress_hosts=("127.0.0.1:8000",))
    argv = sandbox._bwrap_argv(["true"], root=tmp_path, workspace=workspace, env={})
    assert "--unshare-net" not in argv


def test_build_executor_reads_sandbox_limits(monkeypatch) -> None:
    monkeypatch.setattr("reef.harness.executor.shutil.which", lambda name: "/usr/bin/bwrap")
    _probe_result(monkeypatch, 0)
    executor = build_executor(
        {
            "executor": "sandbox",
            "sandbox": {"egress_hosts": ["127.0.0.1:8000"], "limits": {"cpu_seconds": 30, "memory_bytes": 1 << 30}},
        }
    )
    assert isinstance(executor, SandboxExecutor)
    assert executor.egress_hosts == ("127.0.0.1:8000",)
    assert executor.limits == SandboxLimits(cpu_seconds=30, memory_bytes=1 << 30)
