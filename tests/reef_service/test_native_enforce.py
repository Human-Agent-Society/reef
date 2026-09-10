"""Capability enforcement for native tools: the bwrap profile a declaration derives, the enforcer the
environment selects, what a call under it looks like to the loop, and, where bubblewrap is installed, what
the jail denies."""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

from reef.harness.episodes.executor import SandboxExecutor, SandboxUnavailable
from reef.harness.runners.native import LoadError, ToolModule, _invoke, _ModuleRun, load_tools
from reef.harness.runners.native.enforce import (
    CHILD,
    ENFORCE_ENV,
    BwrapEnforcer,
    InProcessEnforcer,
    bwrap_argv,
    denied,
    select_enforcer,
)
from reef.harness.tree.render import render_native_module

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


def test_a_tool_module_is_read_at_load_and_imported_only_where_the_call_runs(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "imported"
    options = {
        "name": "probe",
        "description": "writes its pid when imported",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        "capabilities": ["write"],
        "code": (
            f"import os\nfrom pathlib import Path\n\nPath({str(marker)!r}).write_text(str(os.getpid()))\n\n\n"
            "def run(args, workdir):\n    return 'ran'\n"
        ),
    }
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    module = tools_dir / "probe.py"
    module.write_text(render_native_module("native_tool", options))
    work = tmp_path / "work"
    work.mkdir()
    tools = load_tools(tools_dir)
    # The load read the declaration the render wrote; the module's top level ran nowhere.
    assert not marker.exists()
    tool = tools["probe"]
    assert (tool.name, tool.description, tool.capabilities) == ("probe", options["description"], ("write",))
    assert tool.parameters == options["parameters"] and tool.path == module
    _fake_bwrap(tmp_path, monkeypatch, "argv = sys.argv[sys.argv.index('--') + 1:]\nos.execv(argv[0], argv)")
    assert _invoke(tools, "probe", "{}", work, enforcer=BwrapEnforcer())["content"] == "ran"
    assert int(marker.read_text()) != os.getpid()
    marker.unlink()
    # In process the module imports at the first call and once.
    assert _invoke(tools, "probe", "{}", work, enforcer=InProcessEnforcer())["content"] == "ran"
    assert int(marker.read_text()) == os.getpid()
    marker.unlink()
    assert _invoke(tools, "probe", "{}", work, enforcer=InProcessEnforcer())["content"] == "ran"
    assert not marker.exists()
    # A top level that raises is found where the call runs, under either enforcer, and fails that call only.
    boom = "raise RuntimeError('boom')\n\n\ndef run(args, workdir):\n    return 1\n"
    module.write_text(render_native_module("native_tool", {**options, "code": boom}))
    tools = load_tools(tools_dir)
    assert _invoke(tools, "probe", "{}", work, enforcer=BwrapEnforcer())["error"] == {
        "code": "TOOL_FAILED",
        "message": "RuntimeError: boom",
    }
    assert _invoke(tools, "probe", "{}", work, enforcer=InProcessEnforcer())["error"] == {
        "code": "TOOL_FAILED",
        "message": "LoadError: probe.py failed to import: RuntimeError: boom",
    }
    # What the static read refuses: no run at the top level, a declaration that is not a literal, no parse.
    for source, message in (
        (
            "def helper(args, workdir):\n    return 1\n\nNAME = 'probe'\n",
            r"^no top level statement of probe\.py binds run\(args, workdir\)$",
        ),
        ("def run(args, workdir):\n    return 1\n\nNAME = str(1)\n", r"tool probe\.py must declare NAME as a literal"),
        ("def run(args, workdir:\n    return 1\n", r"probe\.py failed to parse: SyntaxError: "),
    ):
        module.write_text(source)
        with pytest.raises(LoadError, match=message):
            load_tools(tools_dir)


def test_the_static_read_follows_compound_statements_at_module_scope(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    module = tools_dir / "probe.py"
    work = tmp_path / "work"
    work.mkdir()
    # A run bound inside a compound statement at module scope loads and runs: the binding the import makes.
    for source, answer in (
        (
            "try:\n    import reef_no_such_module\n\n    def run(args, workdir):\n        return 'fast'\n"
            "except ImportError:\n\n    def run(args, workdir):\n        return 'fallback'\n",
            "fallback",
        ),
        ("if 1 > 2:\n    run = None\nelse:\n\n    def run(args, workdir):\n        return 'else'\n", "else"),
        (
            "import contextlib\n\nwith contextlib.nullcontext():\n\n    def run(args, workdir):\n        return 'with'\n",
            "with",
        ),
        ("for _ in range(1):\n\n    def run(args, workdir):\n        return 'for'\n", "for"),
        ("while True:\n\n    def run(args, workdir):\n        return 'while'\n\n    break\n", "while"),
        ("match 1:\n    case 1:\n\n        def run(args, workdir):\n            return 'match'\n", "match"),
        ("try:\n    pass\nfinally:\n\n    def run(args, workdir):\n        return 'finally'\n", "finally"),
    ):
        module.write_text(f"{source}\nNAME = 'probe'\n")
        assert _invoke(load_tools(tools_dir), "probe", "{}", work, enforcer=InProcessEnforcer())["content"] == answer
    # The last binding in source order is the declaration, whichever branch would run.
    for source in (
        "def run(args, workdir):\n    return 1\n\nNAME = 'lit'\nif True:\n    NAME = 'cond'\n",
        "def run(args, workdir):\n    return 1\n\nNAME = 'lit'\nif False:\n    NAME = 'cond'\n",
    ):
        module.write_text(source)
        assert list(load_tools(tools_dir)) == ["cond"]
    # A function or class body is not module scope.
    for source in (
        "def outer():\n    def run(args, workdir):\n        return 1\n",
        "class Tool:\n    def run(self, args, workdir):\n        return 1\n",
        "async def outer():\n    def run(args, workdir):\n        return 1\n",
    ):
        module.write_text(source)
        with pytest.raises(LoadError, match=r"^no top level statement of probe\.py binds run\(args, workdir\)$"):
            load_tools(tools_dir)


def test_a_run_bound_to_something_not_callable_fails_the_first_call(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    module = tools_dir / "probe.py"
    work = tmp_path / "work"
    work.mkdir()
    for source, message in (
        ("run = 5\n\nNAME = 'probe'\n", "LoadError: tool probe.py binds run but it is not callable"),
        (
            "if 1 > 2:\n\n    def run(args, workdir):\n        return 1\n\n\nNAME = 'probe'\n",
            "LoadError: tool probe.py did not bind run(args, workdir) when imported",
        ),
    ):
        module.write_text(source)
        tools = load_tools(tools_dir)
        assert list(tools) == ["probe"]
        for _ in range(2):
            assert _invoke(tools, "probe", "{}", work, enforcer=InProcessEnforcer())["error"] == {
                "code": "TOOL_FAILED",
                "message": message,
            }
    # A top level that exits the interpreter is kept as the tool's error like any other exception: two calls, one run.
    count = tmp_path / "count"
    module.write_text(
        f"import sys\n\nwith open({str(count)!r}, 'a') as handle:\n    handle.write('x')\nsys.exit(3)\n\n\n"
        "def run(args, workdir):\n    return 1\n\n\nNAME = 'probe'\n"
    )
    tools = load_tools(tools_dir)
    for _ in range(2):
        assert _invoke(tools, "probe", "{}", work, enforcer=InProcessEnforcer())["error"] == {
            "code": "TOOL_FAILED",
            "message": "LoadError: probe.py failed to import: SystemExit: 3",
        }
    assert count.read_text() == "x"


def test_the_static_read_counts_every_binding_form_at_module_scope(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    module = tools_dir / "probe.py"
    work = tmp_path / "work"
    work.mkdir()
    impl = "def impl(args, workdir):\n    return 'bound'\n\n\n"
    # A run bound by an unpacking, for, with, match or walrus target loads and runs: the binding the import makes.
    for source in (
        "run, helper = impl, None\n",
        "helper, *rest, run = None, None, impl\n",
        "for run in (impl,):\n    pass\n",
        "import contextlib\n\nwith contextlib.nullcontext(impl) as run:\n    pass\n",
        "match (impl,):\n    case [run]:\n        pass\n",
        "(run := impl)\n",
        "if (run := impl):\n    pass\n",
    ):
        module.write_text(f"{impl}{source}\nNAME = 'probe'\n")
        assert _invoke(load_tools(tools_dir), "probe", "{}", work, enforcer=InProcessEnforcer())["content"] == "bound"
    # A form the read counts that the import does not keep, or keeps as something not callable, fails the first call.
    for source, message in (
        (
            "try:\n    raise ValueError('x')\nexcept ValueError as run:\n    pass\n",
            "LoadError: tool probe.py did not bind run(args, workdir) when imported",
        ),
        ("*run, helper = impl, None\n", "LoadError: tool probe.py binds run but it is not callable"),
        (
            "match {'k': 1}:\n    case {**run}:\n        pass\n",
            "LoadError: tool probe.py binds run but it is not callable",
        ),
    ):
        module.write_text(f"{impl}{source}\nNAME = 'probe'\n")
        assert _invoke(load_tools(tools_dir), "probe", "{}", work, enforcer=InProcessEnforcer())["error"] == {
            "code": "TOOL_FAILED",
            "message": message,
        }
    # A comprehension target, a lambda parameter or body, and a subscript target bind nothing at module scope.
    for source in (
        "helper = [run for run in (impl,)]\n",
        "helper = lambda run: run\n",
        "helper = lambda: (run := impl)\n",
        "helper = {}\nhelper['run'] = impl\n",
    ):
        module.write_text(f"{impl}{source}\nNAME = 'probe'\n")
        with pytest.raises(LoadError, match=r"^no top level statement of probe\.py binds run\(args, workdir\)$"):
            load_tools(tools_dir)


def test_a_source_the_parser_cannot_build_is_a_load_error(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    module = tools_dir / "probe.py"
    head = "def run(args, workdir):\n    return 1\n\n\nX = 1"
    # A flat operator chain is a tree as deep as it has terms; one the parser builds is read whole.
    module.write_text(head + " + 1" * 2000 + "\nNAME = 'probe'\n")
    assert list(load_tools(tools_dir)) == ["probe"]
    source = head + " + 1" * 100_000 + "\nNAME = 'probe'\n"
    try:
        ast.parse(source)
    except RecursionError:
        pass
    else:
        pytest.skip("this interpreter builds a 100000 term chain")
    module.write_text(source)
    with pytest.raises(LoadError, match=r"^probe\.py failed to parse: RecursionError: "):
        load_tools(tools_dir)


def test_concurrent_first_calls_import_once_and_a_raising_top_level_fails_every_call_alike(tmp_path: Path) -> None:
    log = tmp_path / "imports.log"
    module = tmp_path / "slow.py"
    module.write_text(
        f"import os\nimport time\n\nwith open({str(log)!r}, 'a') as handle:\n    handle.write(str(os.getpid()) + '\\n')\n"
        "time.sleep(0.2)\n\n\ndef run(args, workdir):\n    return 'ok'\n"
    )
    run = _ModuleRun(module)
    results: list[str] = []
    barrier = threading.Barrier(8)

    def call() -> None:
        barrier.wait()
        results.append(run({}, str(tmp_path)))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == ["ok"] * 8 and log.read_text().splitlines() == [str(os.getpid())]
    # A top level that raises runs once; every call fails with the first call's message.
    count = tmp_path / "count"
    boom = tmp_path / "boom.py"
    boom.write_text(
        f"with open({str(count)!r}, 'a') as handle:\n    handle.write('x')\nraise RuntimeError('boom')\n\n\n"
        "def run(args, workdir):\n    return 1\n"
    )
    failing = _ModuleRun(boom)
    messages = []
    for _ in range(3):
        with pytest.raises(LoadError) as raised:
            failing({}, str(tmp_path))
        messages.append(str(raised.value))
    assert messages == ["boom.py failed to import: RuntimeError: boom"] * 3 and count.read_text() == "x"


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
