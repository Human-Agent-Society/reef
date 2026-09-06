"""Capability enforcement for native tools: the bwrap profile a declaration derives, the enforcer the
environment selects, what a call under it looks like to the loop, and, where bubblewrap is installed, what
the jail denies."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from reef.harness.episodes.executor import SandboxExecutor, SandboxUnavailable
from reef.harness.runners.native import ToolModule, _invoke, load_tools
from reef.harness.runners.native.enforce import (
    CHILD,
    ENFORCE_ENV,
    BwrapEnforcer,
    InProcessEnforcer,
    bwrap_argv,
    denied,
    select_enforcer,
)

PROBE = """\
import errno
import json
import os
import socket
import subprocess


def run(args, workdir):
    print("noise on stdout must not reach the reply")
    seen = {}
    try:
        with open(os.path.join(workdir, "probe.txt"), "w") as handle:
            handle.write("x")
        seen["write"] = "ok"
    except OSError as exc:
        seen["write"] = errno.errorcode.get(exc.errno, str(exc))
    try:
        subprocess.run(["sh", "-c", "true"], check=True)
        seen["exec"] = "ok"
    except (OSError, subprocess.CalledProcessError) as exc:
        seen["exec"] = type(exc).__name__
    # bwrap brings loopback up in an empty namespace, so the interfaces, not a connect, tell the namespaces apart.
    names = sorted(name for _, name in socket.if_nameindex())
    seen["network"] = "ok" if names != ["lo"] else "lo only"
    if args.get("raise"):
        raise RuntimeError("kaboom")
    return json.dumps(seen, sort_keys=True)
"""


def _fake_python(tmp_path: Path) -> dict[str, Path]:
    """A venv shaped interpreter: a prefix with pyvenv.cfg and lib over a base prefix with lib."""
    base = tmp_path / "base"
    (base / "lib").mkdir(parents=True)
    prefix = tmp_path / "venv"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    interpreter = prefix / "bin" / "python"
    interpreter.write_text("")
    return {"interpreter": interpreter, "prefix": prefix, "base_prefix": base}


def _mounts(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, token in enumerate(argv) if token == flag]


def _covered(mounts: list[str], path: Path) -> bool:
    return any(Path(mount) == path or Path(mount) in path.parents for mount in mounts)


def _tools(tmp_path: Path, capabilities: list[str], code: str = PROBE) -> dict[str, ToolModule]:
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "probe.py").write_text(
        f"{code}\nNAME = 'probe'\nDESCRIPTION = 'probe'\nPARAMETERS = {{}}\nCAPABILITIES = {capabilities!r}\n"
    )
    return load_tools(tools)


def _fake_bwrap(tmp_path: Path, monkeypatch, body: str) -> None:
    """A ``bwrap`` on PATH that stands in for the real one, so the child protocol runs without a jail."""
    fake = tmp_path / "fakebin"
    fake.mkdir(exist_ok=True)
    (fake / "bwrap").write_text(f"#!{sys.executable}\nimport os, sys\n{body}\n")
    (fake / "bwrap").chmod(0o755)
    if str(fake) not in os.environ.get("PATH", "").split(os.pathsep):
        monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ.get('PATH', '')}")


def test_denied_is_the_complement_of_the_declaration_over_what_bwrap_can_withhold() -> None:
    assert denied(()) == ["write", "exec", "network"]
    assert denied(("read",)) == ["write", "exec", "network"]
    assert denied(("read", "write")) == ["exec", "network"]
    assert denied(("exec", "write", "network")) == []


def test_the_profile_follows_the_declaration(tmp_path: Path) -> None:
    python = _fake_python(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    def profile(*capabilities: str) -> list[str]:
        return bwrap_argv(capabilities, workdir=work, tools_dir=tools, path="/usr/bin:/bin", **python)

    for capabilities in ((), ("read",), ("write",), ("exec",), ("network",), ("exec", "write", "network")):
        argv = profile(*capabilities)
        assert argv[0] == "bwrap" and "--die-with-parent" in argv and "--clearenv" in argv
        # A jail inside the episode's cannot mount a fresh /proc, so it binds the episode's read only.
        assert "--unshare-pid" in argv and "--proc" not in argv
        assert argv[argv.index("/proc") - 1 : argv.index("/proc") + 2] == ["--ro-bind", "/proc", "/proc"]
        assert ("--unshare-net" in argv) is ("network" not in capabilities)
        writable, readonly = _mounts(argv, "--bind"), _mounts(argv, "--ro-bind")
        assert (writable == [str(work)]) is ("write" in capabilities)
        assert (str(work) in readonly) is ("write" not in capabilities)
        assert ("/usr" in readonly) is ("exec" in capabilities)
        assert ("PATH" in argv) is ("exec" in capabilities)
        if "exec" in capabilities:
            assert argv[argv.index("PATH") - 1 : argv.index("PATH") + 2] == ["--setenv", "PATH", "/usr/bin:/bin"]
        # Whatever the declaration, the interpreter, its prefixes, the tools and the child script are bound.
        for path in (
            python["interpreter"],
            python["prefix"] / "lib",
            python["prefix"] / "pyvenv.cfg",
            python["base_prefix"] / "lib",
            tools,
            CHILD,
        ):
            assert _covered(readonly, path)
        assert argv[argv.index("--") + 1 :] == [str(python["interpreter"]), "-I", "-X", "utf8", str(CHILD)]
        assert argv[argv.index("--chdir") + 1] == str(work) and argv[argv.index("HOME") + 1] == str(work)
        # The workspace is the last mount, so nothing bound earlier shadows it.
        assert argv.index(str(work)) > max(argv.index(mount) for mount in readonly if mount != str(work))
    # Without exec no directory that holds a shell is bound, so Python's own /bin:/usr/bin fallback finds nothing
    # once PATH is unset; the interpreter is bound as one file, never through its bin directory.
    argv = profile("write", "network")
    assert not {"/bin", "/usr/bin", "/usr", "/opt", "/usr/local"} & set(argv)
    assert "PATH" not in argv
    assert not [mount for mount in _mounts(argv, "--ro-bind") if Path(mount).name in ("bin", "sbin")]
    assert _mounts(argv, "--ro-bind").count(str(python["interpreter"])) == 1


def test_the_environment_selects_the_enforcer(monkeypatch) -> None:
    assert isinstance(select_enforcer({}), InProcessEnforcer)
    assert isinstance(select_enforcer({ENFORCE_ENV: ""}), InProcessEnforcer)
    assert select_enforcer({ENFORCE_ENV: "none"}).mode == "none"
    monkeypatch.setattr("reef.harness.runners.native.enforce.shutil.which", lambda name: "/usr/bin/bwrap")
    assert isinstance(select_enforcer({ENFORCE_ENV: "bwrap"}), BwrapEnforcer)
    monkeypatch.setattr("reef.harness.runners.native.enforce.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match=r"bwrap.*not on PATH"):
        select_enforcer({ENFORCE_ENV: "bwrap"})
    with pytest.raises(ValueError, match=r"seccomp.*names no enforcer"):
        select_enforcer({ENFORCE_ENV: "seccomp"})
    tool = ToolModule("shout", "", {}, lambda args, workdir: "", ["read", "write"])
    assert InProcessEnforcer().describe(tool) == {"mode": "none", "denied": []}
    assert BwrapEnforcer().describe(tool) == {"mode": "bwrap", "denied": ["exec", "network"]}
    assert BwrapEnforcer().describe(None) == {"mode": "bwrap", "denied": []}


def test_a_sandboxed_call_runs_the_child_protocol_and_keeps_the_error_codes(tmp_path: Path, monkeypatch) -> None:
    # The stand in runs the command after "--" as is, so the child script and the reply path are the real ones.
    _fake_bwrap(tmp_path, monkeypatch, "argv = sys.argv[sys.argv.index('--') + 1:]\nos.execv(argv[0], argv)")
    tools = _tools(tmp_path, ["read"])
    work = tmp_path / "work"
    work.mkdir()
    ok = _invoke(tools, "probe", "{}", work, enforcer=BwrapEnforcer())
    assert ok["is_error"] is False, ok
    assert json.loads(ok["content"]) == {"exec": "ok", "network": "ok", "write": "ok"}
    assert ok["arguments"] == {} and ok["meta"]["truncated"] is False
    failed = _invoke(tools, "probe", '{"raise": true}', work, enforcer=BwrapEnforcer())
    assert failed["error"] == {"code": "TOOL_FAILED", "message": "RuntimeError: kaboom"}
    built = {"shout": ToolModule("shout", "", {}, lambda args, workdir: "", ["read"])}
    assert _invoke(built, "shout", "{}", work, enforcer=BwrapEnforcer())["error"] == {
        "code": "SANDBOX_FAILED",
        "message": "tool 'shout' has no module file to run in a child process",
    }
    _fake_bwrap(
        tmp_path, monkeypatch, "print('bwrap: No permissions to create a new namespace', file=sys.stderr)\nsys.exit(1)"
    )
    broken = _invoke(tools, "probe", "{}", work, enforcer=BwrapEnforcer())
    assert broken["error"]["code"] == "SANDBOX_FAILED"
    assert broken["error"]["message"] == "tool process exited 1: bwrap: No permissions to create a new namespace"


def require_nested_jail() -> None:
    """A live jail test runs where two bubblewrap jails nest; elsewhere it skips with the preflight's own reason."""
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap (bwrap) is not on PATH")
    try:
        SandboxExecutor().preflight()
    except SandboxUnavailable as exc:
        pytest.skip(str(exc))


def test_bwrap_denies_what_the_declaration_withholds(tmp_path: Path) -> None:
    require_nested_jail()
    work = tmp_path / "work"
    work.mkdir()
    bare = _invoke(_tools(tmp_path, []), "probe", "{}", work, enforcer=BwrapEnforcer())
    assert bare["is_error"] is False, bare
    assert json.loads(bare["content"]) == {"exec": "FileNotFoundError", "network": "lo only", "write": "EROFS"}
    assert not (work / "probe.txt").exists()
    full = _invoke(_tools(tmp_path, ["exec", "write", "network"]), "probe", "{}", work, enforcer=BwrapEnforcer())
    assert full["is_error"] is False, full
    assert json.loads(full["content"]) == {"exec": "ok", "network": "ok", "write": "ok"}
    assert (work / "probe.txt").read_text() == "x"
