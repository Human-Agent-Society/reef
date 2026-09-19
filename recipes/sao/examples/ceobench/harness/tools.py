"""The bash agent's six tools, executed in the task container.

The benchmark's agent has ``bash``, ``read_file``, ``write_file``,
``edit_file``, ``search_files``, and ``glob_files``, all confined to the
agent's workspace (``saas_bench.agents.bash_agent.tools`` at commit
d2b7b32e). :data:`TOOL_SCRIPT` is that executor, kept call for call: the same
bash environment (the checkout's interpreter first on ``PATH``, the
workspace as home and temp directory, the engine's port for the CLI), the
same output assembly (``[stderr]``, ``[exit code: N]``, ``(no output)``, the
30,000-character cut), the same timeout rules (``next-week`` past its limit
ends the run; any other command returns its partial output and an error),
and the same file semantics (numbered lines, unique-match edits, 100 files
and 200 matches per search, 200 names per glob, no path may leave the
workspace).

:class:`ContainerTools` runs that script in the task container through
Harbor's ``exec`` as the unprivileged agent user, one call per tool call,
with the call as JSON on stdin and the result as one JSON line back. The
benchmark sandboxes the same shell with ``bwrap`` where it can; here the
container is the sandbox and the user boundary keeps the engine's process
and source out of reach.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import tempfile
from collections.abc import Coroutine
from pathlib import Path
from typing import NamedTuple, TypeVar

from harbor.environments.base import BaseEnvironment

from .agent import Workspace
from .values import JsonValue

#: Where the script lands in the container, and the interpreter it runs with:
#: the pinned checkout's, so the agent's ``python`` is the one the benchmark's
#: executor puts first on ``PATH``.
SCRIPT_PATH = "/tmp/reef-ceobench-tools.py"
CHECKOUT_PYTHON = "/opt/ceobench/.venv/bin/python"
DEFAULT_BASH_TIMEOUT_S = 1200
#: A tool call may run as long as the bash limit, plus the exec's own overhead.
EXEC_MARGIN_S = 120.0
#: The benchmark ends the run when this command outlives its limit.
NEXT_WEEK_COMMAND = "./novamind-operation next-week"

TOOL_SCRIPT = r'''
"""The benchmark's tool executor, run in the agent's workspace. Reads one call as JSON on stdin."""
import fnmatch, json, os, re, signal, subprocess, sys
from pathlib import Path

OUTPUT_LIMIT = 30000
NEXT_WEEK = "./novamind-operation next-week"


def resolve_path(workspace, path_str):
    p = Path(path_str)
    resolved = p.resolve() if p.is_absolute() else (workspace / p).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return resolved


def run_bash(workspace, port, timeout, args):
    command = args.get("command", "")
    if not command:
        return {"result": "Error: No command provided"}
    ws = str(workspace)
    venv_bin = os.path.join(sys.prefix, "bin")
    path = ([venv_bin] if os.path.isdir(venv_bin) else []) + ["/usr/local/bin", "/usr/bin", "/bin"]
    env = {
        "PATH": ":".join(path),
        "HOME": ws,
        "TMPDIR": ws,
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TERM": os.environ.get("TERM", "xterm"),
        "NOVAMIND_API_PORT": str(port),
    }
    proc = subprocess.Popen(
        ["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ws, env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.kill()
        try:
            partial_stdout, partial_stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            partial_stdout, partial_stderr = "", ""
        if NEXT_WEEK in command:
            return {
                "timeout": f"next_week timed out after {timeout}s",
                "partial_stdout": partial_stdout or "",
                "partial_stderr": partial_stderr or "",
            }
        parts = []
        if partial_stdout:
            parts.append(partial_stdout)
        if partial_stderr:
            parts.append(f"[stderr]\n{partial_stderr}")
        parts.append(f"Error: Command timed out after {timeout} seconds")
        return {"result": "\n".join(parts)}
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if proc.returncode != 0:
        parts.append(f"[exit code: {proc.returncode}]")
    output = "\n".join(parts) if parts else "(no output)"
    if len(output) > OUTPUT_LIMIT:
        output = (
            output[:15000]
            + "\n\n... (output truncated — exceeded 30,000 character limit) ...\n\n"
            + output[-15000:]
        )
    if NEXT_WEEK in command and proc.returncode != 0 and ("step_week_timeout" in output or "step_day_timeout" in output):
        return {
            "timeout": "next_week engine-side timeout (step_week_timeout)",
            "partial_stdout": stdout or "",
            "partial_stderr": stderr or "",
        }
    return {"result": output}


def read_file(workspace, args):
    path = resolve_path(workspace, args["path"])
    if not path.exists():
        return f"Error: File not found: {args['path']}"
    if not path.is_file():
        return f"Error: Not a file: {args['path']}"
    lines = path.read_text().split("\n")
    start = max(0, args.get("offset", 1) - 1)
    limit = args.get("limit")
    lines = lines[start : start + limit] if limit else lines[start:]
    return "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(lines, start=start + 1))


def write_file(workspace, args):
    path = resolve_path(workspace, args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"])
    return f"File written: {args['path']} ({len(args['content'])} bytes)"


def edit_file(workspace, args):
    path = resolve_path(workspace, args["path"])
    if not path.exists():
        return f"Error: File not found: {args['path']}"
    content = path.read_text()
    count = content.count(args["old_string"])
    if count == 0:
        return f"Error: old_string not found in {args['path']}"
    if count > 1:
        return f"Error: old_string found {count} times in {args['path']} (must be unique)"
    path.write_text(content.replace(args["old_string"], args["new_string"], 1))
    return f"File edited: {args['path']}"


def search_files(workspace, args):
    search_path = args.get("path", ".")
    resolved = resolve_path(workspace, search_path)
    if not resolved.exists():
        return f"Error: Path not found: {search_path}"
    try:
        regex = re.compile(args["pattern"])
    except re.error as error:
        return f"Error: Invalid regex: {error}"
    files = [resolved] if resolved.is_file() else sorted(resolved.rglob(args.get("glob", "*")))
    matches = []
    for fpath in files[:100]:
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for i, line in enumerate(content.split("\n"), 1):
            if regex.search(line):
                matches.append(f"{fpath.relative_to(workspace)}:{i}: {line}")
                if len(matches) >= 200:
                    break
        if len(matches) >= 200:
            break
    return "\n".join(matches) if matches else "No matches found."


def glob_files(workspace, args):
    matches = sorted(workspace.glob(args["pattern"]))
    if not matches:
        return "No matching files."
    names = []
    for match in matches[:200]:
        try:
            names.append(str(match.relative_to(workspace)))
        except ValueError:
            names.append(str(match))
    return "\n".join(names)


def main():
    call = json.loads(sys.stdin.read())
    workspace = Path(call["workspace"])
    if call.get("op") == "memory":
        memory = workspace / "MEMORY.md"
        print(json.dumps({"memory": memory.read_text() if memory.exists() else None}))
        return
    name, args = call["name"], call.get("args") or {}
    if not isinstance(args, dict):
        print(json.dumps({"result": "Error: Tool arguments must be a JSON object"}))
        return
    handlers = {
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "search_files": search_files,
        "glob_files": glob_files,
    }
    try:
        if name == "bash":
            print(json.dumps(run_bash(workspace, call["port"], int(call["bash_timeout"]), args)))
        elif name in handlers:
            print(json.dumps({"result": handlers[name](workspace, args)}))
        else:
            print(json.dumps({"result": f"Error: Unknown tool '{name}'"}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"result": f"Error: {error}"}))


main()
'''


Result = TypeVar("Result")


class ToolOutcome(NamedTuple):
    """One tool call's output, or the engine timeout the benchmark ends a run on."""

    result: str
    timeout: str | None = None


class ContainerTools(Workspace):
    """The agent's tools and workspace in the task container, from the episode thread."""

    def __init__(
        self,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        *,
        workspace: str,
        port: int,
        user: str | None,
        bash_timeout_s: int = DEFAULT_BASH_TIMEOUT_S,
        python: str = CHECKOUT_PYTHON,
        logger: logging.Logger | None = None,
    ) -> None:
        self.environment = environment
        self.loop = loop
        self.workspace = workspace
        self.port = int(port)
        self.user = user
        self.bash_timeout_s = int(bash_timeout_s)
        self.python = python
        self.logger = logger or logging.getLogger(__name__)

    def await_result(self, coroutine: Coroutine[object, object, Result], timeout_s: float) -> Result:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=timeout_s)

    def install(self) -> None:
        """Put the executor script in the container, where the agent user can read it."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(TOOL_SCRIPT)
            local = Path(handle.name)
        try:
            self.await_result(self.environment.upload_file(local, SCRIPT_PATH), EXEC_MARGIN_S)
            self.await_result(self.environment.exec(f"chmod 644 {SCRIPT_PATH}"), EXEC_MARGIN_S)
        finally:
            local.unlink(missing_ok=True)

    def run(self, call: dict[str, JsonValue], timeout_s: float) -> dict[str, JsonValue]:
        encoded = base64.b64encode(json.dumps(call).encode("utf-8")).decode("ascii")
        command = f"printf %s {encoded} | base64 -d | {shlex.quote(self.python)} {SCRIPT_PATH}"
        result = self.await_result(
            self.environment.exec(command, user=self.user, timeout_sec=int(timeout_s)), timeout_s + 60.0
        )
        lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
        try:
            payload = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"tool {call.get('name') or call.get('op')} exited {result.return_code} without an answer: "
                f"{(result.stderr or result.stdout or '')[-2000:]}"
            )
        return payload

    def execute(self, name: str, arguments: dict) -> ToolOutcome:
        """Run one tool call as the benchmark's executor would; return its output for the agent."""
        call = {
            "name": name,
            "args": arguments,
            "workspace": self.workspace,
            "port": self.port,
            "bash_timeout": self.bash_timeout_s,
        }
        payload = self.run(call, self.bash_timeout_s + EXEC_MARGIN_S)
        if payload.get("timeout"):
            return ToolOutcome(result=str(payload.get("partial_stdout") or ""), timeout=str(payload["timeout"]))
        return ToolOutcome(result=str(payload.get("result", "")))

    def memory(self) -> str | None:
        payload = self.run({"op": "memory", "workspace": self.workspace}, EXEC_MARGIN_S)
        memory = payload.get("memory")
        return str(memory) if memory is not None else None
