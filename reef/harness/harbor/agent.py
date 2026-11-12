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
from typing import Any, ClassVar, override

import aiohttp
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig, TaskConfig

from reef.core.tasks import HarborTaskError, primary_reward, task_instruction
from reef.execution.sandbox import relay as sandbox_relay
from reef.execution.sandbox.environment import run_sync
from reef.execution.sandbox.process import ProgramLaunchError
from reef.execution.sandbox.remote import HOP_BY_HOP
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, InstallSpec
from reef.harness.harbor import bridge_client
from reef.harness.harbor.agent_bundle import BUNDLE_PREFIX, ContainerLibc, agent_bundle, agent_cache_directory
from reef.harness.harbor.node_runtime import (
    PLATFORM_COMMAND,
    node_archive,
    node_build_name,
    node_cache_directory,
    node_cpu,
)
from reef.harness.harbor.rollouts import COMMAND_FILE, ReefEvalLab, TaskLab, TaskRolloutError
from reef.harness.harbor.uv_runtime import uv_archive, uv_build_name, uv_cache_directory
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import TaskResult, TaskRunError, headless_env, write_rendered
from reef.harness.tasks.trajectory import reader_for
from reef.harness.tree import HarnessTree, render_tree

#: Where the rendered tree goes in the task's container: the run root of ``run.task``.
CONTAINER_ROOT = "/reef-run"
#: The directory under the agent's log directory that receives the agent's session files.
SESSIONS_DIR = "sessions"
#: The ATIF document of the run under the agent's log directory, and the pattern of it and its continuations.
TRAJECTORY_FILE = "trajectory.json"
TRAJECTORY_FILES = "trajectory*.json"
#: The install kinds an agent can have in a task's container.
CONTAINER_INSTALL_KINDS = frozenset({"npm", "uv", "git"})
#: Where Reef puts uv in a task's container, under the agent user's own home.
CONTAINER_UV = "$HOME/.local/bin/uv"
#: Where a git install keeps its checkout (``src``) and its venv (``venv``) in a task's container.
CONTAINER_GIT_AGENT = "$HOME/.reef-agent"
#: Where an npm agent lives in a task's container: ``npm`` is the npm prefix of the bundle that Reef uploads, and
#: ``node`` the Node that Reef uploads into a glibc image. It is outside the user's home, which a task can keep small.
CONTAINER_NPM_ROOT = "/opt/reef"
CONTAINER_NPM_PREFIX = f"{CONTAINER_NPM_ROOT}/{BUNDLE_PREFIX}"
#: The shell test that passes on a musl image, such as Alpine, as Harbor's own npm agents test it. musl's ldd
#: prints its version and exits 1, so its exit code is dropped: with pipefail it would fail the pipe.
MUSL_CHECK = "{ ldd --version 2>&1 || true; } | grep -qi musl || [ -f /etc/alpine-release ]"
#: Where a glibc image gets the Node that Reef uploads, and where the uploaded archives wait for their unpacking.
CONTAINER_NODE = f"{CONTAINER_NPM_ROOT}/node"
CONTAINER_NODE_ARCHIVE = "/tmp/reef-node.tar.gz"
CONTAINER_AGENT_ARCHIVE = "/tmp/reef-agent.tar.gz"
#: The shell command that succeeds when the container's ``python3`` can run the bridge's relay and its client.
RELAY_PYTHON_CHECK = "python3 -c 'import sys; sys.exit(sys.version_info < (3, 7))'"
#: The shell command that drops ``SSL_CERT_DIR`` when it names no directory. uv trusts no certificate then, and it
#: uses its own certificates without the variable. Modal sets the variable also for an image with no certificates.
UV_CERTIFICATES = '{ [ -d "${SSL_CERT_DIR:-/}" ] || unset SSL_CERT_DIR; }'
#: Where Reef uploads the uv archive in a task's container, until uv is installed.
CONTAINER_UV_ARCHIVE = "/tmp/reef-uv.tar.gz"
#: The shell command that puts an npm agent on the PATH: the agent's commands, then the uploaded Node. A musl image
#: has no uploaded Node, and the Node of its packages is on the PATH already.
NPM_AGENT_PATH = f'export PATH="{CONTAINER_NPM_PREFIX}/bin:{CONTAINER_NODE}/bin:$PATH"'
#: Where the bridge's relay lives in a task's container, and the loopback port on which it answers the bridge client.
RELAY_DIR = "/tmp/reef-relay"
RELAY_TUNNEL_PORT = 47811
#: The exchanges that the bridge keeps waiting at once: the model calls that it can answer at the same time.
BRIDGE_POLLERS = 4
#: How long the bridge waits for the relay to answer after it starts.
RELAY_READY_SECONDS = 60.0


def can_run_in_task_container(descriptor: AdapterDescriptor) -> bool:
    """Whether ``descriptor``'s agent can play a Harbor task directory in the task's container.

    The agent needs an install that works in any glibc container (npm, uv, or git), and
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
    ``install_kind`` its channel: ``npm``, ``uv`` for uv itself, or ``git``
    for a checkout of ``repository`` at ``ref``, installed editable into a venv;
    ``binary`` is the command the package installs; ``argv`` is the adapter's
    ``run.task`` argument template, with ``{task}`` slots; ``env`` is the
    run environment, already pointed at ``CONTAINER_ROOT``;
    ``writable_paths`` are directories under the root that the agent writes;
    ``sessions`` is the session directory under the root; ``files_dir`` is a
    local directory that holds the rendered tree as it lands under the root;
    ``timeout_s`` bounds the command. ``task_skills`` is the directory under
    the root where the agent reads a task's skills, or ``None`` when it reads
    none: a run copies the task's ``skills_dir``, which Harbor names, there
    before the tree, so the tree's file wins on one path. ``bridge_port`` is the loopback port of
    the rollout's capture proxy on this host when the container cannot reach
    it: the agent then bridges the container's ``127.0.0.1:<bridge_port>`` to
    it while the command runs (:meth:`bridged`).
    Purpose: :func:`run_harbor_task` names this class in the agent spec of a
    Harbor trial; Harbor builds it with these keyword arguments.
    """

    #: Harbor refuses a task with a ``skills_dir`` for an agent that does not declare this. ``run`` copies the
    #: task's skills for an adapter with ``task_skills``; ``task_agent_tools`` names the task for another adapter.
    capabilities: ClassVar[AgentCapabilities] = AgentCapabilities(skills=True)

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
        trajectory_format: str,
        files_dir: str,
        timeout_s: float,
        install_kind: str = "npm",
        repository: str = "",
        ref: str = "",
        task_skills: str | None = None,
        bridge_port: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, version=version, **kwargs)
        self.reader = reader_for(trajectory_format)
        if install_kind not in CONTAINER_INSTALL_KINDS:
            raise ValueError(f"install kind {install_kind!r} cannot install in a task's container")
        if install_kind == "git" and not (repository and ref):
            raise ValueError("a git install needs a repository and a ref")
        self.install_kind = install_kind
        self.repository = repository
        self.ref = ref
        self.task_skills = task_skills
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
        if self.install_kind == "git":
            return f'"{CONTAINER_GIT_AGENT}/venv/bin/{self.binary}" --version'
        return f"{NPM_AGENT_PATH}; {shlex.quote(self.binary)} --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.install_kind == "uv":
            await self.install_uv(environment, str(self._version))
        elif self.install_kind == "git":
            # A git install needs git. A task image without it gets it from its package manager, as Harbor's own
            # agents do.
            await self.ensure_system_dependencies(environment, ("git",))
            uv = get_adapter("python").install
            if uv is None:
                raise ProgramLaunchError("the python adapter declares no uv install for a git install")
            await self.install_uv(environment, uv.version)
            await self.exec_as_agent(environment, command=self.git_install_command())
        else:
            await self.install_npm_agent(environment)

    async def install_npm_agent(self, environment: BaseEnvironment) -> None:
        """Put the npm agent under ``CONTAINER_NPM_ROOT`` in the container, from a bundle that Reef built on its host.

        npm on the host installed the package for the container's CPU and C library (``agent_bundle``), so the
        container reaches no package registry: a task can break its network. The container unpacks the bundle and
        runs the package's install scripts offline with ``npm rebuild``. A glibc image also gets the Node that Reef
        uploads. The Node builds for musl need libraries that a bare musl image lacks, so a musl image uses the Node
        of its packages.
        """
        root = shlex.quote(CONTAINER_NPM_ROOT)
        if (await environment.exec(command=MUSL_CHECK)).return_code == 0:
            libc: ContainerLibc = "musl"
            cpu = node_cpu((await environment.exec(command="uname -m")).stdout or "")
            await self.ensure_system_dependencies(environment, ("nodejs", "npm"))
            unpack_node = f"mkdir -p {root}"
        else:
            libc = "glibc"
            platform = (await environment.exec(command=PLATFORM_COMMAND)).stdout or ""
            # The container needs no download tool for Node: a task can replace curl, or hold a glibc that is too
            # old for the official build.
            name = node_build_name(platform)
            cpu = node_cpu(platform.splitlines()[0])
            archive = await asyncio.to_thread(node_archive, name, node_cache_directory())
            await environment.upload_file(archive, CONTAINER_NODE_ARCHIVE)
            unpack_node = (
                f"mkdir -p {CONTAINER_NODE} && "
                f"tar -xzf {CONTAINER_NODE_ARCHIVE} -C {CONTAINER_NODE} --strip-components=1"
            )
        bundle = await asyncio.to_thread(
            agent_bundle, self.package, str(self._version), cpu, libc, agent_cache_directory()
        )
        await environment.upload_file(bundle, CONTAINER_AGENT_ARCHIVE)
        # Root unpacks under a directory that every user reads; the agent user may have a small home or none.
        await self.exec_as_root(
            environment,
            command=(
                f"set -euo pipefail; {unpack_node} && tar -xzf {CONTAINER_AGENT_ARCHIVE} -C {root} && "
                f"{NPM_AGENT_PATH} && node --version && "
                f"npm rebuild --global --prefix={CONTAINER_NPM_PREFIX} --no-audit --no-fund && "
                f"rm -f {CONTAINER_NODE_ARCHIVE} {CONTAINER_AGENT_ARCHIVE}"
            ),
        )
        await self.exec_as_agent(
            environment, command=f"set -euo pipefail; {NPM_AGENT_PATH} && command -v {shlex.quote(self.binary)}"
        )

    async def install_uv(self, environment: BaseEnvironment, version: str) -> None:
        """Put uv ``version`` at ``CONTAINER_UV`` in the container.

        Reef downloads the build of the container's machine on its own host and uploads it, so the container needs
        no download tool. The build is static: it runs on a glibc image and on a musl image.
        """
        machine = await environment.exec(command="uname -m")
        name = uv_build_name(version, machine.stdout or "")
        archive = await asyncio.to_thread(uv_archive, name, uv_cache_directory())
        await environment.upload_file(archive, CONTAINER_UV_ARCHIVE)
        await self.exec_as_agent(
            environment,
            command=(
                f'set -euo pipefail; mkdir -p "$HOME/.local/bin" && '
                f'tar -xzf {CONTAINER_UV_ARCHIVE} -C "$HOME/.local/bin" --strip-components=1 && '
                f'"{CONTAINER_UV}" --version'
            ),
        )
        await self.exec_as_root(environment, command=f"rm -f {CONTAINER_UV_ARCHIVE}")

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        directories = [CONTAINER_ROOT, *(f"{CONTAINER_ROOT}/{path}" for path in self.writable_paths)]
        await self.exec_as_root(environment, command=f"mkdir -p {' '.join(map(shlex.quote, directories))}")
        if self.skills_dir and self.task_skills is not None:
            # The task's skills land before the tree, so the tree's file wins on one path.
            skills = shlex.quote(f"{CONTAINER_ROOT}/{self.task_skills}")
            await self.exec_as_root(
                environment, command=f"mkdir -p {skills} && cp -R {shlex.quote(self.skills_dir)}/. {skills}/"
            )
        await environment.upload_dir(self.files_dir, CONTAINER_ROOT)
        # The agent may run as a user other than root, and it writes its sessions under the root.
        await self.exec_as_root(environment, command=f"chmod -R a+rwX {CONTAINER_ROOT}")
        arguments = " ".join(
            shlex.quote(token.replace("{root}", CONTAINER_ROOT).replace("{task}", instruction)) for token in self.argv
        )
        assignments = " ".join(shlex.quote(f"{name}={value}") for name, value in self.env.items())
        # The agent resolves from the user's own home before the run environment moves HOME to the root. uv keeps
        # its cache and Pythons there too, so the program's dependencies install once per container.
        if self.install_kind == "git":
            find_agent = f'agent="{CONTAINER_GIT_AGENT}/venv/bin/{self.binary}"'
        elif self.install_kind == "uv":
            find_agent = (
                f'{UV_CERTIFICATES} && agent="{CONTAINER_UV}" && export UV_CACHE_DIR="$HOME/.cache/uv" '
                'UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"'
            )
        else:
            find_agent = f'{NPM_AGENT_PATH} && agent="$(command -v {shlex.quote(self.binary)})"'
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
        # Only Reef writes the ATIF files that the verifier reads. Harbor's upload adds files but removes none.
        planted = shlex.quote(str(self.environment_logs_dir)) + "/" + TRAJECTORY_FILES
        await self.exec_as_root(environment, command=f"rm -f {planted}")

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
        """The command that runs the relay: the container's ``python3`` when it is Python 3.7 or later, which the
        relay needs, else the path of a Python 3.12 that uv installs. The agent installs uv at the python adapter's
        pin unless it is the agent's own install. ``RuntimeError`` when uv cannot install the Python."""
        if (await environment.exec(command=RELAY_PYTHON_CHECK)).return_code == 0:
            return "python3"
        if self.install_kind not in ("uv", "git"):
            install = get_adapter("python").install
            if install is None:
                raise ProgramLaunchError("the python adapter declares no uv install for the bridge's relay")
            await self.install_uv(environment, install.version)
        # One install, with its error, in place of a download at each call of the relay and of its client.
        found = await self.exec_as_agent(
            environment,
            command=(
                f'set -euo pipefail; {UV_CERTIFICATES}; "{CONTAINER_UV}" python install --quiet 3.12 && '
                f'"{CONTAINER_UV}" python find --no-project 3.12'
            ),
        )
        return shlex.quote((found.stdout or "").strip().splitlines()[-1])

    def git_install_command(self) -> str:
        """The shell command that installs the agent from a checkout at the pinned ref, as Reef does on a person's
        machine (``reef.harness.tasks.vendor_install``): editable into a venv, without the checkout's ``.git``. uv
        that ``install`` put in the container builds the venv with Python 3.12."""
        source, venv = f"{CONTAINER_GIT_AGENT}/src", f"{CONTAINER_GIT_AGENT}/venv"
        return (
            f'set -euo pipefail; {UV_CERTIFICATES}; rm -rf "{CONTAINER_GIT_AGENT}" && '
            f"git clone --quiet --depth 1 --branch {shlex.quote(self.ref)} {shlex.quote(self.repository)} "
            f'"{source}" && rm -rf "{source}/.git" && "{CONTAINER_UV}" venv --quiet --python 3.12 "{venv}" && '
            f'"{CONTAINER_UV}" pip install --quiet --python "{venv}/bin/python" -e "{source}" && '
            f'"{venv}/bin/{self.binary}" --version'
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Write the run's ATIF documents beside the session files, for the task's verifier (RFC #1290).

        Harbor uploads the log directory to the container before the verifier starts. Harbor's context stays as it
        is: Reef reads the session files, not Harbor's token counts. A session that does not convert writes no ATIF,
        and the run goes on.
        """
        for planted in self.logs_dir.glob(TRAJECTORY_FILES):
            planted.unlink()
        try:
            documents = self.reader.to_atif(self.logs_dir / SESSIONS_DIR)
        except Exception:
            self.logger.warning("the agent's sessions do not convert to ATIF", exc_info=True)
            return
        names = [TRAJECTORY_FILE, *(f"trajectory.cont-{index}.json" for index in range(1, len(documents)))]
        for index, document in enumerate(documents):
            if index + 1 < len(names):
                document = {**document, "continued_trajectory_ref": names[index + 1]}
            (self.logs_dir / names[index]).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


class TreeAgent(AdapterAgent):
    """An ``AdapterAgent`` that renders one harness tree for its model when Harbor builds it.

    Values: ``adapter`` names the adapter, and ``tree_dir`` is the directory of the tree. ``mcp_servers``, which
    Harbor passes, are the MCP servers of the task: the render writes them into the agent's config
    (``task_mcp_overlays``). ``model``, ``base_url``,
    ``api_key`` and ``api`` are the values of a ``ModelBinding``. ``files_dir`` receives the rendered tree. ``bridge``
    bridges the container's loopback to ``base_url``, a loopback URL of this host, as ``AdapterAgent``'s
    ``bridge_port``. The other values are ``AdapterAgent``'s.
    Purpose: ``tree_agent`` names this class. The runner fills the placeholders of the binding in the agent's
    configuration, and the tree stays in a directory so that the runner does not fill them in the tree's own text.
    """

    #: Harbor refuses a task with MCP servers for an agent that does not declare ``mcp_servers``. The render writes
    #: them for an adapter with ``task_mcp``; ``task_agent_tools`` names the task for another adapter. No adapter's
    #: agent gets a server's HTTP headers, so ``mcp_http_headers`` stays false.
    capabilities: ClassVar[AgentCapabilities] = AgentCapabilities(skills=True, mcp_servers=True)

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
        files = render_tree(
            HarnessTree.read(Path(tree_dir)),
            descriptor,
            binding.config_overlays(descriptor),
            task_overlays=task_mcp_overlays(descriptor, kwargs.get("mcp_servers") or ()),
        )
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
        "repository": install.repository,
        "ref": install.ref,
        "package": install.package,
        "version": install.version,
        "binary": descriptor.binary,
        "argv": list(descriptor.argv),
        "env": headless_env(descriptor, Path(CONTAINER_ROOT)),
        "writable_paths": list(descriptor.writable_paths),
        "sessions": descriptor.trajectory_path,
        "trajectory_format": descriptor.trajectory_format,
        "files_dir": str(files_dir),
        "timeout_s": timeout,
        "task_skills": descriptor.task_skills_path,
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


def task_mcp_overlays(
    descriptor: AdapterDescriptor, servers: Sequence[MCPServerConfig]
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """The ``(config target, data)`` pair that gives ``descriptor``'s agent the MCP ``servers`` of a task, as
    ``render_tree``'s ``task_overlays``, or nothing without servers.

    ``ValueError`` when the adapter declares no ``run.task.mcp``, or no entry for the transport of a server.
    """
    if not servers:
        return ()
    mcp = descriptor.task_mcp
    if mcp is None:
        raise ValueError(f"adapter {descriptor.name!r} does not give its agent the MCP servers of a task")
    entries: dict[str, Any] = {}
    for server in servers:
        if server.transport not in mcp.entries:
            raise ValueError(
                f"adapter {descriptor.name!r} cannot use the {server.transport} MCP server {server.name!r}"
            )
        entries[server.name] = mcp.entry(
            server.transport, command=server.command or "", args=server.args, url=server.url or ""
        )
    return ((mcp.target, {mcp.key: entries}),)


def task_agent_tools(descriptor: AdapterDescriptor, task: Path) -> tuple[str, ...]:
    """The settings of the Harbor task directory ``task`` that ``TreeAgent`` does not give ``descriptor``'s agent.

    ``mcp_servers`` for an adapter that names no ``task_mcp``, or none for the transport of one of the servers;
    ``mcp_servers headers`` for a server with HTTP headers, which no adapter's agent gets; and ``skills_dir`` for
    an adapter that names no ``task_skills_path``.
    """
    environment = TaskConfig.model_validate_toml((task / "task.toml").read_text(encoding="utf-8")).environment
    mcp, servers = descriptor.task_mcp, environment.mcp_servers
    return (
        *(
            ("mcp_servers",)
            if servers and (mcp is None or any(server.transport not in mcp.entries for server in servers))
            else ()
        ),
        *(("mcp_servers headers",) if any(server.headers for server in servers) else ()),
        *(("skills_dir",) if environment.skills_dir and descriptor.task_skills_path is None else ()),
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
            "it needs an npm, uv or git install, and a command that takes the task as text"
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
    "task_mcp_overlays",
    "tree_agent",
]
