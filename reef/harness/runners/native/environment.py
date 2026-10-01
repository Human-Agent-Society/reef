"""Native tool calls and team git in a task environment: the loop runs on this host, the calls run in the container.

``TaskEnvironmentEnforcer`` runs each tool call as the bwrap enforcer's child does, with the container's own
``python3``: the request goes up as a file under the support directory (``/reef``), the child ``sandboxed.py``
imports the tool module there and replies on stdout. The container needs ``python3`` and no Reef.
``EnvironmentCommandRunner`` runs the git commands of ``workspace: own`` stages there too, so the member clones
and Reef's git directory live in the container under the support directory, outside the task's workdir. Nothing
here imports Harbor: ``reef.harness.runners.native.harbor`` implements ``TaskEnvironment`` over a Harbor environment.
"""

from __future__ import annotations

import json
import shlex
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from reef.harness.runners.native.enforce import Enforcer, SandboxFailed, Tool, ToolFailed
from reef.harness.runners.native.workspaces import CommandOutcome, CommandRunner

#: Where the child, the tool modules, the requests and the team git state live in the container.
SUPPORT_PATH = PurePosixPath("/reef")
#: The longest one tool call may run in the container; a tool's own timeouts are shorter.
TOOL_CALL_TIMEOUT_SECONDS = 1800.0
#: The longest a directory or an upload of one file may take.
FILE_TIMEOUT_SECONDS = 120.0


class TaskEnvironmentError(Exception):
    """The environment could not run a command or take a file; the failure is the environment's, not the tool's."""


class TaskEnvironment(ABC):
    """What the runner needs of a task environment, called from the loop's worker threads."""

    @abstractmethod
    def exec(self, command: str, *, cwd: str | None, timeout_seconds: float) -> CommandOutcome:
        """Run ``command`` with ``sh`` semantics in ``cwd`` (None: the task's workdir) as the task's agent user."""

    @abstractmethod
    def upload_file(self, source_path: Path, target_path: str) -> None: ...


class TaskEnvironmentEnforcer(Enforcer):
    """Each call imports the tool's module afresh in the task container; the container is the boundary."""

    mode = "task-environment"

    def __init__(self, environment: TaskEnvironment, support_path: PurePosixPath = SUPPORT_PATH) -> None:
        self.environment = environment
        self.support_path = support_path

    def describe(self, tool: Tool | None) -> dict[str, Any]:
        return {"mode": self.mode, "denied": []}

    def run(self, tool: Tool, arguments: dict[str, Any], workdir: Path) -> Any:
        if tool.path is None:
            raise SandboxFailed(f"tool {tool.name!r} has no module file to run in the task environment")
        module_path = self.support_path / "tools" / tool.path.name
        request = {"path": str(module_path), "arguments": arguments, "workdir": str(workdir)}
        # A file, not the command line: Harbor's exec takes no stdin, and a large argument outgrows an argv.
        request_path = self.support_path / "requests" / f"{uuid.uuid4().hex}.json"
        child, quoted = shlex.quote(str(self.support_path / "sandboxed.py")), shlex.quote(str(request_path))
        try:
            self.upload_text(json.dumps(request, default=str), str(request_path))
            done = self.environment.exec(
                f"python3 {child} < {quoted}; code=$?; rm -f {quoted}; exit $code",
                cwd=str(workdir),
                timeout_seconds=TOOL_CALL_TIMEOUT_SECONDS,
            )
        except TaskEnvironmentError as exc:
            raise SandboxFailed(f"the task environment could not run the call: {exc}") from exc
        reply: Any = None
        if done.return_code == 0:
            try:
                reply = json.loads(done.stdout)
            except json.JSONDecodeError:
                reply = None
        if not isinstance(reply, dict):
            raise SandboxFailed(f"tool process exited {done.return_code}: {done.stderr.strip()[-600:]}")
        if not reply.get("ok"):
            raise ToolFailed(str(reply.get("error") or "the tool failed"))
        return reply.get("text", "")

    def write_output(self, output_path: Path, text: str) -> None:
        """The workdir is in the container, so the whole result is saved there."""
        try:
            made = self.environment.exec(
                f"mkdir -p {shlex.quote(str(output_path.parent))}", cwd=None, timeout_seconds=FILE_TIMEOUT_SECONDS
            )
            if made.return_code != 0:
                raise TaskEnvironmentError(f"mkdir exited {made.return_code}: {made.stderr.strip()[-600:]}")
            self.upload_text(text, str(output_path))
        except TaskEnvironmentError as exc:
            raise SandboxFailed(f"the task environment could not save the full result: {exc}") from exc

    def upload_text(self, text: str, target_path: str) -> None:
        """``text`` as a file readable by the task's agent user, whoever the upload writes it as."""
        with tempfile.TemporaryDirectory(prefix="reef-native-") as directory:
            source_path = Path(directory) / "upload"
            source_path.write_text(text, encoding="utf-8")
            source_path.chmod(0o644)
            self.environment.upload_file(source_path, target_path)


class EnvironmentCommandRunner(CommandRunner):
    """Team git in the task environment; a command the environment could not run is a nonzero outcome."""

    def __init__(self, environment: TaskEnvironment) -> None:
        self.environment = environment

    def run(self, argv: Sequence[str], *, cwd: str, timeout_seconds: float) -> CommandOutcome:
        try:
            return self.environment.exec(shlex.join(argv), cwd=cwd, timeout_seconds=timeout_seconds)
        except TaskEnvironmentError as exc:
            return CommandOutcome(1, "", f"{argv[0]}: {exc}")
