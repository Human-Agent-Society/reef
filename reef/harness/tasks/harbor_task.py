"""Run one run of a Harbor task directory with a prompt adapter's agent inside the task's container.

A prompt adapter's ``run.task`` command takes the task as text. For a
Harbor task directory, Reef runs one Harbor trial instead: Harbor starts the
task's container, :class:`AdapterAgent` installs the adapter's pinned agent
in it, copies in the rendered tree, and runs the adapter's command there with
the task's instruction as ``{task}``, in the task's working directory. The
task's verifier then scores the container. The agent's session files come
back, and the adapter's trajectory reader reads them.

The run's trajectory starts with one ``verifier`` event that holds the
task and its rewards, so
``reef.evaluation.harness.VerifierRewardScorer`` reads the reward of every
task run the same way. The adapter needs no section for this: it uses
the adapter's ``install``, ``dirs``, ``run.task`` and ``sessions``.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
import tempfile
import urllib.parse
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from reef.core.tasks import HarborTaskError, primary_reward, task_instruction
from reef.execution.sandbox.environment import run_sync
from reef.execution.sandbox.process import ProgramLaunchError
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tasks.rollouts import ReefEvalLab, TaskLab, TaskRolloutError
from reef.harness.tasks.run import TaskResult, TaskRunError, headless_env, write_rendered
from reef.harness.tasks.trajectory import reader_for

#: Where the rendered tree goes in the task's container: the run root of ``run.task``.
CONTAINER_ROOT = "/reef-run"
#: The file under the agent's log directory that holds the command's exit code and output.
COMMAND_FILE = "command.json"
#: The directory under the agent's log directory that receives the agent's session files.
SESSIONS_DIR = "sessions"
#: The install kinds an agent can have in a task's container.
CONTAINER_INSTALL_KINDS = frozenset({"npm"})


def can_run_in_task_container(descriptor: AdapterDescriptor) -> bool:
    """Whether ``descriptor``'s agent can play a Harbor task directory in the task's container.

    The agent needs an install that works in any glibc container (npm), and
    its command must take the task as text: an adapter whose ``{task}`` is
    the task directory runs the task through its own runner.
    """
    return (
        not descriptor.takes_task_directory
        and descriptor.install is not None
        and descriptor.install.kind in CONTAINER_INSTALL_KINDS
    )


class AdapterAgent(BaseInstalledAgent):
    """A Harbor agent that runs one Reef adapter's binary in a task's container.

    Values: ``package`` and ``version`` name the npm package to install;
    ``binary`` is the command the package installs; ``argv`` is the adapter's
    ``run.task`` argument template, with ``{task}`` slots; ``env`` is the
    run environment, already pointed at ``CONTAINER_ROOT``;
    ``writable_paths`` are directories under the root that the agent writes;
    ``sessions`` is the session directory under the root; ``files_dir`` is a
    local directory that holds the rendered tree as it lands under the root;
    ``timeout_s`` bounds the command.
    Purpose: :func:`run_harbor_task` names this class in the agent spec of a
    Harbor trial; Harbor builds it with these keyword arguments.
    """

    def __init__(
        self,
        logs_dir: Path,
        *,
        package: str,
        version: str,
        binary: str,
        argv: Sequence[str],
        env: Mapping[str, str],
        writable_paths: Sequence[str],
        sessions: str,
        files_dir: str,
        timeout_s: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, version=version, **kwargs)
        self.package = package
        self.binary = binary
        self.argv = tuple(argv)
        self.env = dict(env)
        self.writable_paths = tuple(writable_paths)
        self.sessions = sessions
        self.files_dir = Path(files_dir)
        self.timeout_s = timeout_s

    @staticmethod
    @override
    def name() -> str:
        return "reef-adapter"

    @override
    def get_version_command(self) -> str | None:
        return f". ~/.nvm/nvm.sh; {shlex.quote(self.binary)} --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # nvm needs curl; a task image without it gets it from apt, as Harbor's own npm agents do.
        await self.exec_as_root(
            environment,
            command="command -v curl >/dev/null || (apt-get update && apt-get install -y curl)",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                f"set -euo pipefail; {nvm_node_install_snippet()} && "
                f"npm install -g {shlex.quote(f'{self.package}@{self._version}')} && "
                f"command -v {shlex.quote(self.binary)}"
            ),
        )

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        directories = [CONTAINER_ROOT, *(f"{CONTAINER_ROOT}/{path}" for path in self.writable_paths)]
        await self.exec_as_root(environment, command=f"mkdir -p {' '.join(map(shlex.quote, directories))}")
        await environment.upload_dir(self.files_dir, CONTAINER_ROOT)
        # The agent may run as a user other than root, and it writes its sessions under the root.
        await self.exec_as_root(environment, command=f"chmod -R a+rwX {CONTAINER_ROOT}")
        arguments = " ".join(
            shlex.quote(token.replace("{root}", CONTAINER_ROOT).replace("{task}", instruction)) for token in self.argv
        )
        assignments = " ".join(shlex.quote(f"{name}={value}") for name, value in self.env.items())
        # nvm loads from the user's own home before the run environment moves HOME to the root.
        command = (
            f'. "$HOME/.nvm/nvm.sh" && agent="$(command -v {shlex.quote(self.binary)})" && '
            f'env {assignments} "$agent" {arguments} </dev/null'
        )
        result = await environment.exec(command=command, timeout_sec=int(self.timeout_s))
        (self.logs_dir / COMMAND_FILE).write_text(
            json.dumps({"exit_code": result.return_code, "stdout": result.stdout or "", "stderr": result.stderr or ""})
            + "\n",
            encoding="utf-8",
        )
        sessions = f"{CONTAINER_ROOT}/{self.sessions}"
        if await environment.is_dir(sessions):
            await environment.download_dir(sessions, self.logs_dir / SESSIONS_DIR)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Keep Harbor's context as it is: Reef reads the session files, not Harbor's token counts."""


def trial_directory(trial_uri: str | None) -> Path | None:
    """The local trial directory that Harbor's ``file://`` trial URI names, or ``None``."""
    if trial_uri is None or not trial_uri.startswith("file://"):
        return None
    path = Path(urllib.parse.unquote(urllib.parse.urlparse(trial_uri).path))
    return path if path.is_dir() else None


def run_harbor_task(
    descriptor: AdapterDescriptor,
    files: Mapping[str, str],
    task: str,
    *,
    timeout: float = 600.0,
    keep_dir: Path | None = None,
    lab: TaskLab | None = None,
) -> TaskResult:
    """Run one run of ``descriptor``'s agent on the Harbor task directory ``task``, in the task's container.

    ``files`` is the rendered tree; its model binding must name an address that the container reaches.
    ``lab`` runs the trial (reef-eval's Lab under a fresh directory by default, the seam a test replaces).
    The result's trajectory is one ``verifier`` event, then the adapter's reading of the agent's session files;
    ``exit_code``, ``stdout`` and ``stderr`` are the agent command's, and an agent that never ran reports exit
    code 1 with the trial's error. ``keep_dir`` receives a copy of the session files. ``TaskRunError`` when
    the adapter cannot run in a container or the trial cannot start.
    """
    if not can_run_in_task_container(descriptor):
        raise TaskRunError(
            f"adapter {descriptor.name!r} cannot play a Harbor task directory in the task's container: "
            "it needs an npm install, and a command that takes the task as text"
        )
    install = descriptor.install
    if install is None:
        raise TaskRunError(f"adapter {descriptor.name!r} declares no install")
    reader = reader_for(descriptor.trajectory_format)  # fail before the container starts
    task_path = Path(task)
    try:
        task_instruction(task_path)
    except HarborTaskError as exc:
        raise TaskRunError(f"{task!r} is not a Harbor task directory: {exc}") from exc
    # Harbor bind-mounts the trial's directories into the task's container. Docker on macOS runs in a VM that
    # shares the home directory with the host and, under colima, not the temp directory.
    parent = Path.home() / ".reef" / "runs" if sys.platform == "darwin" else None
    if parent is not None:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TaskRunError(f"cannot create the run directory {parent}: {exc}") from exc
    work = Path(tempfile.mkdtemp(prefix=f"reef-task-run-{descriptor.name}-", dir=parent))
    try:
        tree_dir = work / "tree"
        try:
            write_rendered(descriptor, files, tree_dir, CONTAINER_ROOT)
        except ProgramLaunchError as exc:
            raise TaskRunError(str(exc)) from exc
        tree_dir.mkdir(exist_ok=True)
        agent = {
            "import_path": f"{__name__}:{AdapterAgent.__name__}",
            "kwargs": {
                "package": install.package,
                "version": install.version,
                "binary": descriptor.binary,
                "argv": list(descriptor.argv),
                "env": headless_env(descriptor, Path(CONTAINER_ROOT)),
                "writable_paths": list(descriptor.writable_paths),
                "sessions": descriptor.trajectory_path,
                "files_dir": str(tree_dir),
                "timeout_s": timeout,
            },
        }
        runner = lab or ReefEvalLab(work / "lab")
        try:
            row = run_sync(
                runner.run(
                    task_path,
                    agent,
                    key=uuid.uuid4().hex,
                    tags={},
                    overrides={"environment": {"type": "docker"}},
                )
            )
        except TaskRolloutError as exc:
            raise TaskRunError(str(exc)) from exc
        trial = trial_directory(row.trial_uri)
        agent_dir = None if trial is None else trial / "agent"
        command = None if agent_dir is None else agent_dir / COMMAND_FILE
        sessions = None if agent_dir is None else agent_dir / SESSIONS_DIR
        if command is not None and command.is_file():
            ran = json.loads(command.read_text(encoding="utf-8"))
            exit_code, stdout, stderr = int(ran["exit_code"]), str(ran["stdout"]), str(ran["stderr"])
        else:
            exit_code, stdout, stderr = 1, "", row.error
        rewards = dict(row.rewards)
        verifier = {
            "type": "verifier",
            "task": task,
            "rewards": rewards,
            "reward": primary_reward(rewards),
            "failed": not rewards,
            "error": row.error,
        }
        session_events: tuple[dict[str, Any], ...] = ()
        if sessions is not None and sessions.is_dir():
            session_events = reader(sessions)
            if keep_dir is not None and any(sessions.iterdir()):
                shutil.copytree(sessions, keep_dir, symlinks=True)
        return TaskResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            trajectory=(verifier, *session_events),
            residue=(),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


__all__ = ["CONTAINER_ROOT", "AdapterAgent", "can_run_in_task_container", "run_harbor_task"]
