"""A prompt adapter's agent plays a Harbor task directory in the task's container, scored by the task's verifier."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import shutil
import socket
import subprocess
import tarfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from reef.evaluation.harness.evaluator import container_binding
from reef.evaluation.harness.scoring import verifier_reward
from reef.harness.adapters import get_adapter
from reef.harness.harbor.agent import (
    CONTAINER_ROOT,
    MUSL_CHECK,
    NPM_AGENT_PATH,
    RELAY_PYTHON_CHECK,
    AdapterAgent,
    TreeAgent,
    adapter_agent_kwargs,
    can_run_in_task_container,
    run_harbor_task,
    task_agent_tools,
    task_mcp_overlays,
    tree_agent,
)
from reef.harness.harbor.agent_bundle import AgentBundleError, agent_bundle
from reef.harness.harbor.node_runtime import (
    NODE_BUILDS,
    PLATFORM_COMMAND,
    NodeBuild,
    NodeRuntimeError,
    host_node,
    node_archive,
    node_build_name,
    node_cpu,
)
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.harness.harbor.uv_runtime import UV_BUILDS, UV_VERSION, UvBuild, UvRuntimeError, uv_archive, uv_build_name
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import TaskRunError
from reef.harness.tasks.trajectory import failed_trial_error
from reef.harness.tree import HarnessTree, RenderError, render_tree

SESSION_EVENT = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "391"}]}}


class StandInLab(TaskLab):
    """A trial that writes what ``AdapterAgent`` leaves in the agent's log directory, or nothing when ``ran`` is
    false, and keeps the agent spec it got."""

    def __init__(self, trials: Path, rewards: Mapping[str, float], *, ran: bool = True, error: str = "") -> None:
        self.trials = trials
        self.rewards = rewards
        self.ran = ran
        self.error = error
        self.agents: list[Mapping[str, object]] = []

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.agents.append(agent)
        trial = self.trials / key
        if self.ran:
            sessions = trial / "agent" / "sessions"
            sessions.mkdir(parents=True)
            (sessions / "session.jsonl").write_text(json.dumps(SESSION_EVENT) + "\n")
            (trial / "agent" / "command.json").write_text(json.dumps({"exit_code": 0, "stdout": "", "stderr": ""}))
        else:
            trial.mkdir(parents=True)
        return RolloutRow(rewards=dict(self.rewards), error=self.error, trial_uri=trial.as_uri())

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


def harbor_task(root: Path) -> Path:
    task = root / "multiply"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Write 17 times 23 to /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return task


def test_the_agent_plays_the_task_in_its_container_and_the_verifier_reward_scores_it(tmp_path: Path) -> None:
    pi = get_adapter("pi")
    task = harbor_task(tmp_path)
    lab = StandInLab(tmp_path / "trials", {"reward": 1.0})
    result = run_harbor_task(pi, {"pi-agent/AGENTS.md": "Be exact.\n"}, str(task), keep_dir=tmp_path / "kept", lab=lab)

    assert result.exit_code == 0
    assert result.trajectory[0]["type"] == "verifier" and result.trajectory[1:] == (SESSION_EVENT,)
    assert verifier_reward(str(task), result) == 1.0
    assert (tmp_path / "kept" / "session.jsonl").is_file()
    (agent,) = lab.agents
    kwargs = agent["kwargs"]
    assert agent["import_path"] == "reef.harness.harbor.agent:AdapterAgent"
    assert (kwargs["package"], kwargs["version"]) == (pi.install.package, pi.install.version)
    assert kwargs["argv"] == list(pi.argv) and kwargs["sessions"] == pi.trajectory_path
    # The run's environment points the agent at the root in the container, where the tree lands.
    assert kwargs["env"]["PI_CODING_AGENT_DIR"] == f"{CONTAINER_ROOT}/pi-agent"
    assert not Path(kwargs["files_dir"]).exists()  # the staged tree goes with the run


def test_a_trial_that_never_ran_the_agent_is_a_failed_trial_with_its_error(tmp_path: Path) -> None:
    lab = StandInLab(tmp_path / "trials", {}, ran=False, error="the image did not build")
    result = run_harbor_task(get_adapter("pi"), {}, str(harbor_task(tmp_path)), lab=lab)

    assert result.exit_code == 1 and result.stderr == "the image did not build"
    assert failed_trial_error(result.trajectory) == "the image did not build"


def test_only_an_adapter_with_an_npm_uv_or_git_install_and_a_text_prompt_plays_a_task_in_a_container(
    tmp_path: Path,
) -> None:
    assert all(
        can_run_in_task_container(get_adapter(name))
        for name in ("claude", "pi", "opencode", "codex", "python", "hermes")
    )
    # An adapter whose own command takes the task directory runs it itself; native declares no install.
    assert not can_run_in_task_container(replace(get_adapter("pi"), takes_task_directory=True))
    assert not can_run_in_task_container(get_adapter("native"))
    with pytest.raises(TaskRunError, match="'native' cannot play a Harbor task directory"):
        run_harbor_task(get_adapter("native"), {}, str(harbor_task(tmp_path)))
    with pytest.raises(TaskRunError, match="is not a Harbor task directory"):
        run_harbor_task(get_adapter("pi"), {}, str(tmp_path / "missing"))


class RecordingEnvironment:
    """A task container that keeps each command an agent runs in it and each file it uploads, and holds no session
    directory. ``events`` is both, in order. The container lacks the commands ``missing``, installs packages with
    ``manager``, and is a musl image when ``musl``. ``platform`` is what its ``PLATFORM_COMMAND`` prints."""

    def __init__(
        self,
        *,
        missing: tuple[str, ...] = (),
        manager: str = "apt-get",
        musl: bool = False,
        platform: str = "x86_64\nglibc 2.39\n",
    ) -> None:
        self.missing = missing
        self.manager = manager
        self.musl = musl
        self.platform = platform
        self.commands: list[str] = []
        self.uploads: dict[str, str] = {}
        self.events: list[str] = []
        self.stream_handle = SimpleNamespace(provider="modal")

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.commands.append(command)
        self.events.append(command)
        if command.startswith("for manager in "):
            return SimpleNamespace(return_code=0, stdout=self.manager, stderr="")
        if command == PLATFORM_COMMAND:
            return SimpleNamespace(return_code=0, stdout=self.platform, stderr="")
        if "python find" in command:
            return SimpleNamespace(return_code=0, stdout="/home/agent/uv/cpython-3.12/bin/python3.12\n", stderr="")
        if command == "uname -m":
            return SimpleNamespace(return_code=0, stdout=self.platform.splitlines()[0] + "\n", stderr="")
        absent = any(command.startswith(f"command -v {name} >/dev/null") for name in self.missing)
        absent = absent or (command == RELAY_PYTHON_CHECK and "python3" in self.missing)
        failed = absent or (command == MUSL_CHECK and not self.musl)
        return SimpleNamespace(return_code=1 if failed else 0, stdout="", stderr="")

    async def upload_dir(self, source, target) -> None:
        pass

    async def upload_file(self, source, target) -> None:
        self.uploads[target] = Path(source).read_text()

    async def is_dir(self, path) -> bool:
        return False


@pytest.fixture
def uv_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the uv builds that an agent asked for. Each one is a stand-in archive, not a download."""
    builds: list[str] = []

    def archive(name: str, cache: Path) -> Path:
        builds.append(name)
        (tmp_path / f"uv-{name}.tar.gz").write_text(f"the {name} uv build")
        return tmp_path / f"uv-{name}.tar.gz"

    monkeypatch.setattr("reef.harness.harbor.agent.uv_archive", archive)
    return builds


def test_the_python_adapter_uploads_uv_and_runs_its_program_with_the_instruction(
    tmp_path: Path, uv_builds: list[str]
) -> None:
    python = get_adapter("python")
    (tmp_path / "logs").mkdir()
    agent = AdapterAgent(tmp_path / "logs", **adapter_agent_kwargs(python, python.install, tmp_path / "files", 60.0))
    # The image has no curl and is an arm64 musl image: Reef uploads the static build of the machine.
    environment = RecordingEnvironment(missing=("curl",), musl=True, platform="aarch64\n")
    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("Play the scene.", environment, AgentContext()))

    (install,) = (command for command in environment.commands if "tar -xzf /tmp/reef-uv.tar.gz" in command)
    run = environment.commands[-2]
    assert python.install.version == UV_VERSION and uv_builds == ["aarch64"]
    assert environment.uploads["/tmp/reef-uv.tar.gz"] == "the aarch64 uv build"
    assert '-C "$HOME/.local/bin" --strip-components=1 && "$HOME/.local/bin/uv" --version' in install
    assert environment.commands[environment.commands.index(install) + 1].endswith("rm -f /tmp/reef-uv.tar.gz")
    assert not any("curl" in command or "apt-get" in command for command in environment.commands)
    assert 'agent="$HOME/.local/bin/uv"' in run
    assert f"\"$agent\" run --script {CONTAINER_ROOT}/program/main.py 'Play the scene.'" in run
    assert f"REEF_PROGRAM_SESSION_DIR={CONTAINER_ROOT}/program/sessions" in run


def test_the_relay_runs_on_the_python_of_the_image_or_on_a_python_from_an_uploaded_uv(
    tmp_path: Path, uv_builds: list[str]
) -> None:
    pi = get_adapter("pi")
    agent = AdapterAgent(tmp_path, **adapter_agent_kwargs(pi, pi.install, tmp_path / "files", 60.0))
    assert asyncio.run(agent.relay_python(RecordingEnvironment())) == "python3" and uv_builds == []

    # An image with no python3, or with one before 3.7, cannot run the relay: Reef uploads uv for a Python.
    old = RecordingEnvironment(missing=("python3", "curl"))
    assert asyncio.run(agent.relay_python(old)) == "/home/agent/uv/cpython-3.12/bin/python3.12"
    assert uv_builds == ["x86_64"] and "/tmp/reef-uv.tar.gz" in old.uploads
    # uv installs the Python once, without a certificate directory that does not exist.
    install = old.commands[-1]
    assert '{ [ -d "${SSL_CERT_DIR:-/}" ] || unset SSL_CERT_DIR; }; "$HOME/.local/bin/uv" python install' in install


@pytest.mark.parametrize("name", ["opencode", "codex"])
def test_an_instruction_that_starts_with_a_dash_reaches_the_agent_after_the_end_of_its_options(
    tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Markdown list item is a common first line of a task. Without "--", opencode and codex read it as an option
    # and print their usage.
    monkeypatch.setattr("reef.harness.harbor.agent.node_archive", lambda name, cache: tmp_path / "node.tar.gz")
    (tmp_path / "node.tar.gz").write_text("node")
    adapter = get_adapter(name)
    (tmp_path / "logs").mkdir()
    agent = AdapterAgent(tmp_path / "logs", **adapter_agent_kwargs(adapter, adapter.install, tmp_path / "files", 60.0))
    environment = RecordingEnvironment()
    asyncio.run(agent.run("- You are given a state dictionary.", environment, AgentContext()))
    assert " -- '- You are given a state dictionary.' </dev/null" in environment.commands[-2]


def test_the_hermes_adapter_installs_its_pinned_checkout_editable_and_runs_its_binary(
    tmp_path: Path, uv_builds: list[str]
) -> None:
    hermes = get_adapter("hermes")
    (tmp_path / "logs").mkdir()
    agent = AdapterAgent(tmp_path / "logs", **adapter_agent_kwargs(hermes, hermes.install, tmp_path / "files", 60.0))
    environment = RecordingEnvironment(missing=("curl", "git"))
    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("Play the scene.", environment, AgentContext()))

    # The image gets git from its package manager, and uv from Reef: it needs no curl.
    (tools,) = (command for command in environment.commands if "apt-get install" in command)
    (install,) = (command for command in environment.commands if "git clone" in command)
    run = environment.commands[-2]
    assert uv_builds == ["x86_64"] and "/tmp/reef-uv.tar.gz" in environment.uploads
    assert tools.endswith("apt-get update && apt-get install -y git")
    clone = f"git clone --quiet --depth 1 --branch {hermes.install.ref} {hermes.install.repository}"
    assert clone in install and 'rm -rf "$HOME/.reef-agent/src/.git"' in install
    assert '"$HOME/.local/bin/uv" pip install --quiet --python "$HOME/.reef-agent/venv/bin/python" -e' in install
    assert run.startswith('agent="$HOME/.reef-agent/venv/bin/hermes"')
    assert agent.get_version_command() == '"$HOME/.reef-agent/venv/bin/hermes" --version'


@pytest.fixture
def agent_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str, str]]:
    """The agent bundles that an agent asked for, as ``(package, version, cpu, libc)``. Each one is a stand-in
    archive, not an npm install."""
    asked: list[tuple[str, str, str, str]] = []

    def bundle(package: str, version: str, cpu: str, libc: str, cache: Path) -> Path:
        asked.append((package, version, cpu, libc))
        (tmp_path / f"agent-{cpu}-{libc}.tar.gz").write_text(f"the {cpu} {libc} bundle")
        return tmp_path / f"agent-{cpu}-{libc}.tar.gz"

    monkeypatch.setattr("reef.harness.harbor.agent.agent_bundle", bundle)
    return asked


def test_an_npm_agent_is_a_bundle_from_the_host_with_an_uploaded_node_on_glibc_and_the_image_node_on_musl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_bundles: list[tuple[str, str, str, str]]
) -> None:
    pi = get_adapter("pi")
    builds: list[str] = []

    def archive(name: str, cache: Path) -> Path:
        builds.append(name)
        (tmp_path / f"{name}.tar.gz").write_text(f"the {name} build")
        return tmp_path / f"{name}.tar.gz"

    monkeypatch.setattr("reef.harness.harbor.agent.node_archive", archive)

    def installed(name: str, environment: RecordingEnvironment) -> tuple[str, str]:
        (tmp_path / name).mkdir()
        agent = AdapterAgent(tmp_path / name, **adapter_agent_kwargs(pi, pi.install, tmp_path / "files", 60.0))
        asyncio.run(agent.install(environment))
        asyncio.run(agent.run("Play.", environment, AgentContext()))
        (install,) = (command for command in environment.commands if "npm rebuild" in command)
        return install, environment.commands[-2]

    # Reef uploads the Node of the container's platform and the agent that npm installed on the host for it. The
    # container unpacks both outside the user's home and reaches no registry: no curl, no package manager, no
    # npm install.
    glibc = RecordingEnvironment(missing=("curl",), platform="x86_64\nglibc 2.27\n")
    install, run = installed("glibc", glibc)
    assert builds == ["x64-glibc-217"] and agent_bundles == [(pi.install.package, pi.install.version, "x64", "glibc")]
    assert glibc.uploads == {
        "/tmp/reef-node.tar.gz": "the x64-glibc-217 build",
        "/tmp/reef-agent.tar.gz": "the x64 glibc bundle",
    }
    assert "tar -xzf /tmp/reef-node.tar.gz -C /opt/reef/node --strip-components=1" in install
    assert "tar -xzf /tmp/reef-agent.tar.gz -C /opt/reef && " in install
    assert "npm rebuild --global --prefix=/opt/reef/npm --no-audit --no-fund && rm -f /tmp/reef-node" in install
    assert not any("curl" in command or "apt-get" in command or "npm install" in command for command in glibc.commands)
    assert f"{NPM_AGENT_PATH} && " in install and f"{NPM_AGENT_PATH} && " in run

    # The Node builds for musl need libraries that a bare musl image lacks: the image's packages give Node, and the
    # agent is still the bundle from the host, built for musl.
    musl = RecordingEnvironment(missing=("node", "npm"), manager="apk", musl=True, platform="aarch64\n")
    install, run = installed("musl", musl)
    assert any(command.endswith("apk add --no-cache nodejs npm") for command in musl.commands)
    assert builds == ["x64-glibc-217"] and agent_bundles[-1] == (
        pi.install.package,
        pi.install.version,
        "arm64",
        "musl",
    )
    assert musl.uploads == {"/tmp/reef-agent.tar.gz": "the arm64 musl bundle"}
    assert "reef-node.tar.gz -C" not in install and "npm rebuild --global --prefix=/opt/reef/npm" in install


def test_an_agent_bundle_is_one_npm_install_on_the_host_for_the_containers_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = tmp_path / "node"
    runs: list[list[str]] = []

    def npm(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        runs.append(command)
        if "broken@1" in command:
            raise subprocess.CalledProcessError(1, command, stderr="npm error 404 broken@1 is not in this registry")
        prefix = Path(next(argument for argument in command if argument.startswith("--prefix=")).split("=", 1)[1])
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "agent").write_text("#!/bin/sh\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("reef.harness.harbor.agent_bundle.host_node", lambda cache: node)
    monkeypatch.setattr("reef.harness.harbor.agent_bundle.subprocess.run", npm)

    bundle = agent_bundle("@scope/agent", "1.2.3", "arm64", "musl", tmp_path / "cache")
    assert bundle.name == "scope-agent@1.2.3-linux-arm64-musl.tar.gz"
    (command,) = runs
    assert command[:3] == [str(node / "bin" / "node"), str(node / "lib/node_modules/npm/bin/npm-cli.js"), "install"]
    # npm installs for the container, not for this host, and runs no script of the package here.
    assert {"--os=linux", "--cpu=arm64", "--libc=musl", "--ignore-scripts", "@scope/agent@1.2.3"} <= set(command)
    with tarfile.open(bundle, "r:gz") as tar:
        assert [(member.name, member.uid) for member in tar.getmembers()] == [
            ("npm", 0),
            ("npm/bin", 0),
            ("npm/bin/agent", 0),
        ]
    # The same four values again are the kept archive; another platform is another install.
    assert agent_bundle("@scope/agent", "1.2.3", "arm64", "musl", tmp_path / "cache") == bundle and len(runs) == 1
    assert agent_bundle("@scope/agent", "1.2.3", "x64", "glibc", tmp_path / "cache") != bundle and len(runs) == 2

    with pytest.raises(AgentBundleError, match="npm cannot install broken@1 for linux x64 glibc: npm error 404"):
        agent_bundle("broken", "1", "x64", "glibc", tmp_path / "cache")
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == [
        "scope-agent@1.2.3-linux-arm64-musl.tar.gz",
        "scope-agent@1.2.3-linux-x64-glibc.tar.gz",
    ]


def test_the_host_node_is_the_build_of_the_hosts_system_unpacked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "node.tar.xz"
    with tarfile.open(published, "w:xz") as tar:
        content = b"#!/bin/sh\n"
        info = tarfile.TarInfo("node-v0-darwin-arm64/bin/node")
        info.size, info.mode = len(content), 0o755
        tar.addfile(info, io.BytesIO(content))
    build = NodeBuild(published.as_uri(), hashlib.sha256(published.read_bytes()).hexdigest())
    monkeypatch.setitem(NODE_BUILDS, "darwin-arm64", build)
    monkeypatch.setattr("reef.harness.harbor.node_runtime.platform.system", lambda: "Darwin")
    monkeypatch.setattr("reef.harness.harbor.node_runtime.platform.machine", lambda: "arm64")

    directory = host_node(tmp_path / "cache")
    assert (directory / "bin" / "node").read_bytes() == b"#!/bin/sh\n"
    published.unlink()
    assert host_node(tmp_path / "cache") == directory

    monkeypatch.setattr("reef.harness.harbor.node_runtime.platform.system", lambda: "Windows")
    with pytest.raises(NodeRuntimeError, match="no Node build for a Windows host"):
        host_node(tmp_path / "cache")
    assert node_cpu("x86_64\n") == "x64" and node_cpu("aarch64") == "arm64"
    with pytest.raises(NodeRuntimeError, match="riscv64"):
        node_cpu("riscv64")


def run_bash(script: str, home: Path, path: Path) -> subprocess.CompletedProcess[str]:
    """Run ``script`` as Harbor runs a command: in bash with pipefail. ``path`` comes first on the PATH."""
    env = {"HOME": str(home), "PATH": f"{path}:/usr/bin:/bin"}
    return subprocess.run(["bash", "-c", f"set -o pipefail; {script}"], env=env, capture_output=True, text=True)


def test_the_musl_check_passes_for_the_ldd_of_musl_alone(tmp_path: Path) -> None:
    def ldd(name: str, text: str, exit_code: int) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "ldd").write_text(f"#!/bin/sh\necho '{text}' >&2\nexit {exit_code}\n")
        (directory / "ldd").chmod(0o755)
        return directory

    # musl's ldd prints its version on stderr and exits 1; glibc's exits 0.
    musl = ldd("musl", "musl libc (x86_64)", 1)
    glibc = ldd("glibc", "ldd (Ubuntu GLIBC 2.39-0ubuntu8) 2.39", 0)
    assert run_bash(MUSL_CHECK, tmp_path, musl).returncode == 0
    assert run_bash(MUSL_CHECK, tmp_path, glibc).returncode != 0


def test_the_node_build_follows_the_machine_and_the_glibc_of_the_container() -> None:
    assert node_build_name("x86_64\nglibc 2.39\n") == "x64"
    assert node_build_name("aarch64\nglibc 2.28\n") == "arm64"
    # The official x64 build needs glibc 2.28. ldd's first line gives the version when getconf is absent.
    assert node_build_name("x86_64\nglibc 2.27\n") == "x64-glibc-217"
    assert node_build_name("x86_64\nldd (GNU libc) 2.17\n") == "x64-glibc-217"
    for platform, reason in (
        ("x86_64\nglibc 2.12\n", "x64 with glibc 2.12"),
        ("aarch64\nglibc 2.27\n", "arm64 with glibc 2.27"),
        ("riscv64\nglibc 2.39\n", "reports"),
        ("x86_64\n", "reports"),
    ):
        with pytest.raises(NodeRuntimeError, match=reason):
            node_build_name(platform)


def test_a_node_archive_is_the_checked_download_as_gzip_and_is_downloaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "node.tar.xz"
    with tarfile.open(published, "w:xz") as tar:
        content = b"#!/bin/sh\n"
        info = tarfile.TarInfo("node-v0/bin/node")
        info.size, info.mode = len(content), 0o755
        tar.addfile(info, io.BytesIO(content))
    build = NodeBuild(published.as_uri(), hashlib.sha256(published.read_bytes()).hexdigest())
    monkeypatch.setitem(NODE_BUILDS, "x64", build)

    archive = node_archive("x64", tmp_path / "cache")
    with tarfile.open(archive, "r:gz") as tar:
        (member,) = tar.getmembers()
        assert (member.name, member.mode) == ("node-v0/bin/node", 0o755)
    published.unlink()
    assert node_archive("x64", tmp_path / "cache") == archive

    # A download with another checksum is refused, and the cache keeps nothing of it.
    with tarfile.open(published, "w:xz"):
        pass
    monkeypatch.setitem(NODE_BUILDS, "arm64", NodeBuild(published.as_uri(), build.sha256))
    with pytest.raises(NodeRuntimeError, match="does not have the SHA-256"):
        node_archive("arm64", tmp_path / "cache")
    assert [path.name for path in (tmp_path / "cache").iterdir()] == [archive.name]
    monkeypatch.setitem(NODE_BUILDS, "arm64", NodeBuild((tmp_path / "missing.tar.xz").as_uri(), build.sha256))
    with pytest.raises(NodeRuntimeError, match="cannot download"):
        node_archive("arm64", tmp_path / "cache")


def test_a_uv_archive_is_the_checked_download_of_the_pinned_version_and_is_downloaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert uv_build_name(UV_VERSION, "x86_64\n") == "x86_64" and uv_build_name(UV_VERSION, "arm64") == "aarch64"
    with pytest.raises(UvRuntimeError, match=r"not of uv 0\.1\.0"):
        uv_build_name("0.1.0", "x86_64")
    with pytest.raises(UvRuntimeError, match="reports 'riscv64'"):
        uv_build_name(UV_VERSION, "riscv64\n")

    published = tmp_path / "uv.tar.gz"
    published.write_bytes(b"the published archive")
    build = UvBuild(published.as_uri(), hashlib.sha256(published.read_bytes()).hexdigest())
    monkeypatch.setitem(UV_BUILDS, "x86_64", build)
    archive = uv_archive("x86_64", tmp_path / "cache")
    assert archive.read_bytes() == b"the published archive"
    published.unlink()
    assert uv_archive("x86_64", tmp_path / "cache") == archive

    # A download with another checksum is refused, and the cache keeps nothing of it.
    published.write_bytes(b"another archive")
    monkeypatch.setitem(UV_BUILDS, "aarch64", UvBuild(published.as_uri(), build.sha256))
    with pytest.raises(UvRuntimeError, match="does not have the SHA-256"):
        uv_archive("aarch64", tmp_path / "cache")
    assert [path.name for path in (tmp_path / "cache").iterdir()] == [archive.name]
    monkeypatch.setitem(UV_BUILDS, "aarch64", UvBuild((tmp_path / "missing.tar.gz").as_uri(), build.sha256))
    with pytest.raises(UvRuntimeError, match="cannot download"):
        uv_archive("aarch64", tmp_path / "cache")


def test_the_agent_writes_the_atif_of_its_sessions_for_the_verifier_and_drops_a_planted_one(tmp_path: Path) -> None:
    pi = get_adapter("pi")
    logs = tmp_path / "logs"
    shutil.copytree(Path(__file__).parent / "data" / "pi_session_real", logs / "sessions")
    (logs / "trajectory.cont-3.json").write_text('{"steps": []}')
    agent = AdapterAgent(logs, **adapter_agent_kwargs(pi, pi.install, tmp_path / "files", 60.0))
    agent.populate_context_post_run(AgentContext())
    assert sorted(path.name for path in logs.glob("trajectory*.json")) == ["trajectory.json"]
    document = json.loads((logs / "trajectory.json").read_text())
    assert document["steps"][-1]["message"] == "READY"
    assert "continued_trajectory_ref" not in document


def test_a_run_in_segments_writes_one_file_for_each_segment_with_its_continuation(tmp_path: Path) -> None:
    pi = get_adapter("pi")
    agent = AdapterAgent(tmp_path, **adapter_agent_kwargs(pi, pi.install, tmp_path / "files", 60.0))
    agent.reader = SimpleNamespace(to_atif=lambda path: ({"steps": [1]}, {"steps": [2]}, {"steps": [3]}))
    agent.populate_context_post_run(AgentContext())
    first, second, third = (
        json.loads((tmp_path / name).read_text())
        for name in ["trajectory.json", "trajectory.cont-1.json", "trajectory.cont-2.json"]
    )
    assert first == {"steps": [1], "continued_trajectory_ref": "trajectory.cont-1.json"}
    assert second == {"steps": [2], "continued_trajectory_ref": "trajectory.cont-2.json"}
    assert third == {"steps": [3]}


class BridgedContainer(RecordingEnvironment):
    """A task container whose relay holds one model call: the bridge client's exchanges take it once, and the
    container keeps each exchange's command and the answer that the next exchange gives back."""

    def __init__(self, call: Mapping[str, object]) -> None:
        super().__init__()
        self.calls = [json.dumps(call)]
        self.exchanges: list[str] = []
        self.answered = asyncio.Event()

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        await super().exec(command, user=user, env=env, cwd=cwd, timeout_sec=timeout_sec)
        if "client.py" not in command or command.endswith(" ready"):
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        self.exchanges.append(command)
        if command.endswith(".json"):
            self.answered.set()
        await asyncio.sleep(0.01)
        return SimpleNamespace(return_code=0, stdout=self.calls.pop() if self.calls else "", stderr="")


def test_a_bridged_agent_answers_the_relayed_calls_from_the_proxy_port_while_its_command_runs(tmp_path: Path) -> None:
    async def model(request: web.Request) -> web.Response:
        return web.json_response({"answer": (await request.json())["question"]}, headers={"X-Model": "m"})

    async def run() -> BridgedContainer:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", model)
        runner = web.AppRunner(app)
        await runner.setup()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        await web.SockSite(runner, listener).start()
        body = base64.b64encode(json.dumps({"question": "42?"}).encode()).decode()
        call = {"id": "c1", "method": "POST", "path": "/v1/chat/completions", "headers": {}, "body": body}
        container = BridgedContainer(call)
        python = get_adapter("python")
        kwargs = adapter_agent_kwargs(python, python.install, tmp_path / "files", 60.0)
        agent = AdapterAgent(tmp_path / "logs", bridge_port=port, **kwargs)
        async with agent.bridged(container):
            await asyncio.wait_for(container.answered.wait(), 10)
        await runner.cleanup()
        return container

    (tmp_path / "logs").mkdir()
    container = asyncio.run(run())
    relay = next(command for command in container.commands if "relay.py" in command)
    assert "--tunnel-port 47811 --tunnel-host 127.0.0.1" in relay
    assert container.uploads["/tmp/reef-relay/relay.py"].startswith('"""The relay a remote sandbox')
    piece = json.loads(container.uploads["/tmp/reef-relay/replies/c1.json"])
    assert (piece["status"], piece["end"], piece["sequence"]) == (200, True, 0)
    assert json.loads(base64.b64decode(piece["data"])) == {"answer": "42?"}
    assert piece["headers"]["X-Model"] == "m"
    assert any(command.endswith("c1 /tmp/reef-relay/replies/c1.json") for command in container.exchanges)


def test_a_tree_agent_bridges_the_port_of_its_loopback_binding(tmp_path: Path) -> None:
    python = get_adapter("python")
    (tmp_path / "tree" / "program").mkdir(parents=True)
    (tmp_path / "tree" / "program" / "main.py").write_text("print('hi')\n")
    spec = tree_agent(
        python, api="openai", tree_dir=tmp_path / "tree", files_dir=tmp_path / "files", timeout=60.0, bridge=True
    )
    kwargs = {**spec["kwargs"], "model": "m", "base_url": "http://127.0.0.1:8901", "api_key": "reef"}
    assert TreeAgent(tmp_path / "logs", **kwargs).bridge_port == 8901


def test_a_container_reaches_a_loopback_model_route_at_the_agent_host() -> None:
    route = ModelBinding(base_url="http://127.0.0.1:8901/reef/scenarios/s/evaluation", model="m")
    assert container_binding(route, "host.docker.internal").base_url == (
        "http://host.docker.internal:8901/reef/scenarios/s/evaluation"
    )
    public = ModelBinding(base_url="https://api.example.com", model="m")
    assert container_binding(public, "host.docker.internal") == public


def test_the_tools_that_a_task_gives_its_agent_are_found_in_its_config(tmp_path: Path) -> None:
    task = harbor_task(tmp_path)
    pi, python = get_adapter("pi"), get_adapter("python")
    assert task_agent_tools(python, task) == ()

    config = (task / "task.toml").read_text()
    server = '[[environment.mcp_servers]]\nname = "search"\ntransport = "sse"\nurl = "http://search:8000/sse"\n'
    (task / "task.toml").write_text(f'{config}\n[environment]\nskills_dir = "/skills"\n\n{server}')
    # pi names the directory where it reads a task's skills; the python adapter names none.
    assert task_agent_tools(pi, task) == ("mcp_servers",)
    assert task_agent_tools(python, task) == ("mcp_servers", "skills_dir")
    # opencode reads a task's MCP servers, but no adapter gives its agent a server's HTTP headers; Codex reads no
    # SSE server.
    assert task_agent_tools(get_adapter("opencode"), task) == ()
    assert task_agent_tools(get_adapter("codex"), task) == ("mcp_servers",)
    (task / "task.toml").write_text(f'{config}\n{server}headers = {{ Authorization = "Bearer ${{TOKEN}}" }}\n')
    assert task_agent_tools(get_adapter("opencode"), task) == ("mcp_servers headers",)


MCP_SERVERS = [
    MCPServerConfig(name="search", transport="streamable-http", url="http://search:8000/mcp"),
    MCPServerConfig(name="files", transport="stdio", command="files-server", args=["--root", "/app"]),
]


def test_the_agents_declare_to_harbor_the_task_settings_that_reef_gives_an_adapter() -> None:
    # Harbor ends a trial before the container starts when its task has skills or MCP servers and the agent does
    # not declare them. Reef refuses HTTP headers itself, so the agents do not declare them.
    assert AdapterAgent.capabilities.skills and not AdapterAgent.capabilities.mcp_servers
    assert TreeAgent.capabilities.skills and TreeAgent.capabilities.mcp_servers
    assert not TreeAgent.capabilities.mcp_http_headers


def test_each_adapter_writes_the_mcp_servers_of_a_task_as_its_agent_reads_them() -> None:
    def overlay(adapter: str) -> tuple[str, dict]:
        ((target, data),) = task_mcp_overlays(get_adapter(adapter), MCP_SERVERS)
        return target, data

    assert overlay("opencode") == (
        "agent",
        {
            "mcp": {
                "search": {"type": "remote", "url": "http://search:8000/mcp", "oauth": False},
                "files": {"type": "local", "command": ["files-server", "--root", "/app"]},
            }
        },
    )
    assert overlay("claude") == (
        "mcp",
        {
            "mcpServers": {
                "search": {"type": "http", "url": "http://search:8000/mcp"},
                "files": {"type": "stdio", "command": "files-server", "args": ["--root", "/app"]},
            }
        },
    )
    expected = {
        "search": {"url": "http://search:8000/mcp"},
        "files": {"command": "files-server", "args": ["--root", "/app"]},
    }
    assert overlay("codex") == ("primary", {"mcp_servers": expected})
    assert overlay("hermes") == ("primary", {"mcp_servers": expected})

    assert task_mcp_overlays(get_adapter("pi"), []) == ()
    with pytest.raises(ValueError, match="'pi' does not give its agent the MCP servers"):
        task_mcp_overlays(get_adapter("pi"), MCP_SERVERS)
    sse = [MCPServerConfig(name="events", transport="sse", url="http://events:8000/sse")]
    with pytest.raises(ValueError, match="'codex' cannot use the sse MCP server 'events'"):
        task_mcp_overlays(get_adapter("codex"), sse)


def test_a_tree_agent_renders_the_mcp_servers_of_its_task_and_a_tree_cannot_write_them(tmp_path: Path) -> None:
    (tmp_path / "tree").mkdir()
    binding = {"model": "m", "base_url": "http://127.0.0.1:8901", "api_key": "reef"}

    def rendered(adapter: str, api: str, **task: object) -> Path:
        files = tmp_path / f"{adapter}-{len(task)}"
        spec = tree_agent(get_adapter(adapter), api=api, tree_dir=tmp_path / "tree", files_dir=files, timeout=60.0)
        TreeAgent(tmp_path / "logs", **{**spec["kwargs"], **binding}, **task)
        return files

    codex = (rendered("codex", "responses", mcp_servers=MCP_SERVERS) / "codex" / "config.toml").read_text()
    assert '[mcp_servers.search]\nurl = "http://search:8000/mcp"' in codex
    state = rendered("claude", "anthropic", mcp_servers=MCP_SERVERS) / "claude" / ".claude.json"
    assert set(json.loads(state.read_text())["mcpServers"]) == {"search", "files"}
    # A run without MCP servers leaves Claude Code's state file to Claude Code.
    assert not (rendered("claude", "anthropic") / "claude" / ".claude.json").exists()

    # The checks of a config target hold on the tree: only Reef writes these keys, for a task.
    servers = json.dumps({"mcpServers": {"own": {"type": "http", "url": "http://own"}}})
    with pytest.raises(RenderError, match="Reef writes the MCP servers of a task"):
        render_tree(HarnessTree({"config/mcp.json": servers}), get_adapter("claude"))
    with pytest.raises(RenderError, match="a Codex plugin carries no config keys"):
        render_tree(HarnessTree({"config/primary.json": '{"mcp_servers": {}}'}), get_adapter("codex"))


def test_the_agent_copies_the_skills_of_the_task_into_its_skills_directory_before_the_tree(tmp_path: Path) -> None:
    pi = get_adapter("pi")
    assert pi.task_skills_path == "pi-agent/skills"
    kwargs = adapter_agent_kwargs(pi, pi.install, tmp_path / "files", 60.0)

    class OrderedEnvironment(RecordingEnvironment):
        async def upload_dir(self, source, target) -> None:
            self.events.append(f"upload {target}")

    def run_events(logs: Path, **task: str) -> list[str]:
        logs.mkdir()
        environment = OrderedEnvironment()
        asyncio.run(AdapterAgent(logs, **kwargs, **task).run("Play.", environment, AgentContext()))
        return environment.events

    copy = f"mkdir -p {CONTAINER_ROOT}/pi-agent/skills && cp -R /skills/. {CONTAINER_ROOT}/pi-agent/skills/"
    events = run_events(tmp_path / "with", skills_dir="/skills")
    copied = next(index for index, event in enumerate(events) if copy in event)
    assert copied < events.index(f"upload {CONTAINER_ROOT}")
    assert not any("cp -R" in event for event in run_events(tmp_path / "without"))
