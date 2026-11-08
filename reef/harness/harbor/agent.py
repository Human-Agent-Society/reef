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

import asyncio
import base64
import contextlib
import json
import secrets
import shlex
import shutil
import sys
import tempfile
import urllib.parse
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, override

import aiohttp
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import TaskConfig

from reef.core.tasks import HarborTaskError, primary_reward, task_instruction
from reef.execution.sandbox import relay as sandbox_relay
from reef.execution.sandbox.environment import run_sync
from reef.execution.sandbox.process import ProgramLaunchError
from reef.execution.sandbox.remote import HOP_BY_HOP
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, InstallSpec
from reef.harness.harbor import bridge_client
from reef.harness.harbor.rollouts import ReefEvalLab, TaskLab, TaskRolloutError
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import TaskResult, TaskRunError, headless_env, write_rendered
from reef.harness.tasks.trajectory import reader_for
from reef.harness.tree import HarnessTree, render_tree

#: Where the rendered tree goes in the task's container: the run root of ``run.task``.
CONTAINER_ROOT = "/reef-run"
#: The file under the agent's log directory that holds the command's exit code and output.
COMMAND_FILE = "command.json"
#: The directory under the agent's log directory that receives the agent's session files.
SESSIONS_DIR = "sessions"
#: The install kinds an agent can have in a task's container.
CONTAINER_INSTALL_KINDS = frozenset({"npm", "uv"})
#: Where Astral's installer puts uv in a task's container, under the agent user's own home.
CONTAINER_UV = "$HOME/.local/bin/uv"
#: Where the bridge's relay lives in a task's container, and the loopback port on which it answers the bridge client.
RELAY_DIR = "/tmp/reef-relay"
RELAY_TUNNEL_PORT = 47811
#: The exchanges that the bridge keeps waiting at once: the model calls that it can answer at the same time.
BRIDGE_POLLERS = 4
#: How long the bridge waits for the relay to answer after it starts.
RELAY_READY_SECONDS = 60.0


def can_run_in_task_container(descriptor: AdapterDescriptor) -> bool:
    """Whether ``descriptor``'s agent can play a Harbor task directory in the task's container.

    The agent needs an install that works in any glibc container (npm or uv), and
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

    Values: ``package`` and ``version`` name the package to install, and
    ``install_kind`` its channel: ``npm``, or ``uv`` for uv itself;
    ``binary`` is the command the package installs; ``argv`` is the adapter's
    ``run.task`` argument template, with ``{task}`` slots; ``env`` is the
    run environment, already pointed at ``CONTAINER_ROOT``;
    ``writable_paths`` are directories under the root that the agent writes;
    ``sessions`` is the session directory under the root; ``files_dir`` is a
    local directory that holds the rendered tree as it lands under the root;
    ``timeout_s`` bounds the command. ``bridge_port`` is the loopback port of
    the rollout's capture proxy on this host when the container cannot reach
    it: the agent then bridges the container's ``127.0.0.1:<bridge_port>`` to
    it while the command runs (:meth:`bridged`).
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
        install_kind: str = "npm",
        bridge_port: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, version=version, **kwargs)
        if install_kind not in CONTAINER_INSTALL_KINDS:
            raise ValueError(f"install kind {install_kind!r} cannot install in a task's container")
        self.install_kind = install_kind
        self.bridge_port = bridge_port
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
        if self.install_kind == "uv":
            return f'"{CONTAINER_UV}" --version'
        return f". ~/.nvm/nvm.sh; {shlex.quote(self.binary)} --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # nvm and Astral's installer need curl; a task image without it gets it from apt, as Harbor's own npm
        # agents do.
        await self.exec_as_root(
            environment,
            command="command -v curl >/dev/null || (apt-get update && apt-get install -y curl)",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        if self.install_kind == "uv":
            command = uv_install_command(str(self._version))
        else:
            command = (
                f"set -euo pipefail; {nvm_node_install_snippet()} && "
                f"npm install -g {shlex.quote(f'{self.package}@{self._version}')} && "
                f"command -v {shlex.quote(self.binary)}"
            )
        await self.exec_as_agent(environment, command=command)

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
        # The agent resolves from the user's own home before the run environment moves HOME to the root. uv keeps
        # its cache and Pythons there too, so the program's dependencies install once per container.
        if self.install_kind == "uv":
            find_agent = (
                f'agent="{CONTAINER_UV}" && export UV_CACHE_DIR="$HOME/.cache/uv" '
                'UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"'
            )
        else:
            find_agent = f'. "$HOME/.nvm/nvm.sh" && agent="$(command -v {shlex.quote(self.binary)})"'
        command = f'{find_agent} && env {assignments} "$agent" {arguments} </dev/null'
        async with self.bridged(environment):
            result = await environment.exec(command=command, timeout_sec=int(self.timeout_s))
        (self.logs_dir / COMMAND_FILE).write_text(
            json.dumps({"exit_code": result.return_code, "stdout": result.stdout or "", "stderr": result.stderr or ""})
            + "\n",
            encoding="utf-8",
        )
        sessions = f"{CONTAINER_ROOT}/{self.sessions}"
        if await environment.is_dir(sessions):
            await environment.download_dir(sessions, self.logs_dir / SESSIONS_DIR)

    @contextlib.asynccontextmanager
    async def bridged(self, environment: BaseEnvironment) -> AsyncIterator[None]:
        """Bridge the container's ``127.0.0.1:<bridge_port>`` to the same port on this host for the time of the
        block, or do nothing without a ``bridge_port``.

        Reef's sandbox relay answers that port in the container and holds each call. ``BRIDGE_POLLERS`` loops here
        take the calls through Harbor's exec, with ``bridge_client``, answer each from this host's port, and give the
        answer back in the next exchange. Exec works in every Harbor environment, also for a task with several compose
        services. Raise ``ProgramLaunchError`` when the relay does not answer.
        """
        if self.bridge_port is None:
            yield
            return
        tunnel_port = RELAY_TUNNEL_PORT + 1 if self.bridge_port == RELAY_TUNNEL_PORT else RELAY_TUNNEL_PORT
        await self.exec_as_agent(environment, command=f"mkdir -p {RELAY_DIR}/replies && chmod 700 {RELAY_DIR}")
        await environment.upload_file(Path(sandbox_relay.__file__), f"{RELAY_DIR}/relay.py")
        await environment.upload_file(Path(bridge_client.__file__), f"{RELAY_DIR}/client.py")
        with tempfile.TemporaryDirectory(prefix="reef-relay-") as staging:
            secret_file = Path(staging) / "secret"
            secret_file.write_text(secrets.token_hex(16), encoding="utf-8")
            await environment.upload_file(secret_file, f"{RELAY_DIR}/secret")
        python = await self.relay_python(environment)
        await self.exec_as_agent(
            environment,
            command=(
                f"nohup {python} {RELAY_DIR}/relay.py --port {self.bridge_port} --tunnel-port {tunnel_port} "
                f"--tunnel-host 127.0.0.1 --secret-file {RELAY_DIR}/secret >{RELAY_DIR}/relay.log 2>&1 &"
            ),
        )
        client = f"{python} {RELAY_DIR}/client.py {tunnel_port} {RELAY_DIR}/secret"
        deadline = asyncio.get_running_loop().time() + RELAY_READY_SECONDS
        while (await environment.exec(command=f"{client} ready")).return_code != 0:
            if asyncio.get_running_loop().time() > deadline:
                raise ProgramLaunchError(f"the bridge's relay did not answer in {RELAY_READY_SECONDS:.0f} s")
            await asyncio.sleep(0.5)
        async with aiohttp.ClientSession(auto_decompress=False) as local:
            pollers = [
                asyncio.create_task(self.relay_calls(environment, client, local)) for _ in range(BRIDGE_POLLERS)
            ]
            try:
                yield
            finally:
                for poller in pollers:
                    poller.cancel()
                await asyncio.gather(*pollers, return_exceptions=True)

    async def relay_calls(self, environment: BaseEnvironment, client: str, local: aiohttp.ClientSession) -> None:
        """Answer the relay's calls one at a time, until cancelled: each exchange gives the last answer back and
        takes the next call. A failed exchange runs again with the same answer, which the relay takes once."""
        reply = ""
        while True:
            result = await environment.exec(command=f"{client} {reply}".rstrip(), timeout_sec=120)
            if result.return_code != 0:
                await asyncio.sleep(1)
                continue
            reply = ""
            if not (result.stdout or "").strip():
                continue
            request = json.loads(result.stdout)
            piece = await self.answer(local, request)
            path = f"{RELAY_DIR}/replies/{request['id']}.json"
            with tempfile.TemporaryDirectory(prefix="reef-relay-") as staging:
                staged = Path(staging) / "piece.json"
                staged.write_text(json.dumps(piece), encoding="utf-8")
                await environment.upload_file(staged, path)
            reply = f"{request['id']} {path}"

    async def answer(self, local: aiohttp.ClientSession, request: Mapping[str, Any]) -> dict[str, Any]:
        """The relay's one piece that answers ``request`` from this host's ``bridge_port``: the whole answer."""
        headers = {
            name: value for name, value in (request.get("headers") or {}).items() if name.lower() not in HOP_BY_HOP
        }
        body = base64.b64decode(request.get("body") or "")
        try:
            async with local.request(
                str(request.get("method") or "GET"),
                f"http://127.0.0.1:{self.bridge_port}{request.get('path') or '/'}",
                headers=headers,
                data=body or None,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=3600),
            ) as response:
                data = await response.read()
                answer = {
                    "status": response.status,
                    "headers": {
                        name: value for name, value in response.headers.items() if name.lower() not in HOP_BY_HOP
                    },
                }
        except (aiohttp.ClientError, TimeoutError) as exc:
            data = f"the Reef host did not answer: {exc}".encode()
            answer = {"status": 502, "headers": {"Content-Type": "text/plain"}}
        return {**answer, "data": base64.b64encode(data).decode(), "end": True, "sequence": 0}

    async def relay_python(self, environment: BaseEnvironment) -> str:
        """The command that runs the relay: the container's ``python3``, else a Python from uv, which the agent
        installs at the python adapter's pin unless it is the agent's own install."""
        if (await environment.exec(command="command -v python3")).return_code == 0:
            return "python3"
        if self.install_kind != "uv":
            install = get_adapter("python").install
            if install is None:
                raise ProgramLaunchError("the python adapter declares no uv install for the bridge's relay")
            await self.exec_as_agent(environment, command=uv_install_command(install.version))
        return f'"{CONTAINER_UV}" run --no-project --python 3.12'

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Keep Harbor's context as it is: Reef reads the session files, not Harbor's token counts."""


def uv_install_command(version: str) -> str:
    """The shell command that installs uv ``version`` with Astral's installer under the user's own home."""
    installer = f"https://astral.sh/uv/{shlex.quote(version)}/install.sh"
    return (
        f'set -euo pipefail; curl -LsSf {installer} | env UV_INSTALL_DIR="$HOME/.local/bin" '
        f'UV_NO_MODIFY_PATH=1 sh && "{CONTAINER_UV}" --version'
    )


class TreeAgent(AdapterAgent):
    """An ``AdapterAgent`` that renders one harness tree for its model when Harbor builds it.

    Values: ``adapter`` names the adapter, and ``tree_dir`` is the directory of the tree. ``model``, ``base_url``,
    ``api_key`` and ``api`` are the values of a ``ModelBinding``. ``files_dir`` receives the rendered tree. ``bridge``
    bridges the container's loopback to ``base_url``, a loopback URL of this host, as ``AdapterAgent``'s
    ``bridge_port``. The other values are ``AdapterAgent``'s.
    Purpose: ``tree_agent`` names this class. The runner fills the placeholders of the binding in the agent's
    configuration, and the tree stays in a directory so that the runner does not fill them in the tree's own text.
    """

    def __init__(
        self,
        logs_dir: Path,
        *,
        adapter: str,
        tree_dir: str,
        model: str,
        base_url: str,
        api_key: str,
        api: str,
        files_dir: str,
        bridge: bool = False,
        **kwargs: Any,
    ) -> None:
        descriptor = get_adapter(adapter)
        binding = ModelBinding(base_url, model, api_key, api)
        files = render_tree(HarnessTree.read(Path(tree_dir)), descriptor, binding.config_overlays(descriptor))
        write_rendered(descriptor, files, Path(files_dir), CONTAINER_ROOT)
        if bridge:
            kwargs["bridge_port"] = urllib.parse.urlparse(base_url).port
        super().__init__(logs_dir, files_dir=files_dir, **kwargs)


def adapter_agent_kwargs(
    descriptor: AdapterDescriptor, install: InstallSpec, files_dir: Path, timeout: float
) -> dict[str, object]:
    """The keyword arguments of an ``AdapterAgent`` for ``descriptor``, whose install is ``install``."""
    return {
        "install_kind": install.kind,
        "package": install.package,
        "version": install.version,
        "binary": descriptor.binary,
        "argv": list(descriptor.argv),
        "env": headless_env(descriptor, Path(CONTAINER_ROOT)),
        "writable_paths": list(descriptor.writable_paths),
        "sessions": descriptor.trajectory_path,
        "files_dir": str(files_dir),
        "timeout_s": timeout,
    }


def tree_agent(
    descriptor: AdapterDescriptor,
    *,
    api: str,
    tree_dir: Path,
    files_dir: Path,
    timeout: float,
    bridge: bool = False,
) -> dict[str, object]:
    """The Harbor agent configuration that runs the tree in ``tree_dir``, with the placeholders of the model binding,
    as ``TaskRolloutRunner``'s ``agent``. ``bridge`` bridges the container's loopback to the binding's loopback URL
    (``TreeAgent``). ``ValueError`` for an adapter that cannot run in a task's container."""
    install = descriptor.install
    if install is None or not can_run_in_task_container(descriptor):
        raise ValueError(f"adapter {descriptor.name!r} cannot play a Harbor task directory in the task's container")
    return {
        "import_path": f"{__name__}:{TreeAgent.__name__}",
        "kwargs": {
            **adapter_agent_kwargs(descriptor, install, files_dir, timeout),
            "adapter": descriptor.name,
            "tree_dir": str(tree_dir),
            "model": "{model}",
            "base_url": "{base_url}",
            "api_key": "{api_key}",
            "api": api,
            "bridge": bridge,
        },
    }


def task_agent_tools(task: Path) -> tuple[str, ...]:
    """The settings of the Harbor task directory ``task`` that ``AdapterAgent`` does not give its agent."""
    environment = TaskConfig.model_validate_toml((task / "task.toml").read_text(encoding="utf-8")).environment
    return (
        *(("mcp_servers",) if environment.mcp_servers else ()),
        *(("skills_dir",) if environment.skills_dir else ()),
    )


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
            "kwargs": adapter_agent_kwargs(descriptor, install, tree_dir, timeout),
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


__all__ = [
    "CONTAINER_ROOT",
    "AdapterAgent",
    "TreeAgent",
    "can_run_in_task_container",
    "run_harbor_task",
    "task_agent_tools",
    "tree_agent",
]
