"""A prompt adapter's agent plays a Harbor task directory in the task's container, scored by the task's verifier."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import socket
import subprocess
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
    AdapterAgent,
    TreeAgent,
    adapter_agent_kwargs,
    can_run_in_task_container,
    run_harbor_task,
    task_agent_tools,
    task_mcp_overlays,
    tree_agent,
)
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
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
    ``manager``, and is a musl image when ``musl``."""

    def __init__(self, *, missing: tuple[str, ...] = (), manager: str = "apt-get", musl: bool = False) -> None:
        self.missing = missing
        self.manager = manager
        self.musl = musl
        self.commands: list[str] = []
        self.uploads: dict[str, str] = {}
        self.events: list[str] = []
        self.stream_handle = SimpleNamespace(provider="modal")

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.commands.append(command)
        self.events.append(command)
        if command.startswith("for manager in "):
            return SimpleNamespace(return_code=0, stdout=self.manager, stderr="")
        absent = any(command.startswith(f"command -v {name} >/dev/null") for name in self.missing)
        failed = absent or (command == MUSL_CHECK and not self.musl)
        return SimpleNamespace(return_code=1 if failed else 0, stdout="", stderr="")

    async def upload_dir(self, source, target) -> None:
        pass

    async def upload_file(self, source, target) -> None:
        self.uploads[target] = Path(source).read_text()

    async def is_dir(self, path) -> bool:
        return False


def test_the_python_adapter_installs_uv_and_runs_its_program_with_the_instruction(tmp_path: Path) -> None:
    python = get_adapter("python")
    (tmp_path / "logs").mkdir()
    agent = AdapterAgent(tmp_path / "logs", **adapter_agent_kwargs(python, python.install, tmp_path / "files", 60.0))
    environment = RecordingEnvironment()
    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("Play the scene.", environment, AgentContext()))

    install, run = environment.commands[1], environment.commands[-2]
    assert f"curl -LsSf https://astral.sh/uv/{python.install.version}/install.sh" in install
    assert 'agent="$HOME/.local/bin/uv"' in run
    assert f"\"$agent\" run --script {CONTAINER_ROOT}/program/main.py 'Play the scene.'" in run
    assert f"REEF_PROGRAM_SESSION_DIR={CONTAINER_ROOT}/program/sessions" in run


def test_the_hermes_adapter_installs_its_pinned_checkout_editable_and_runs_its_binary(tmp_path: Path) -> None:
    hermes = get_adapter("hermes")
    (tmp_path / "logs").mkdir()
    agent = AdapterAgent(tmp_path / "logs", **adapter_agent_kwargs(hermes, hermes.install, tmp_path / "files", 60.0))
    environment = RecordingEnvironment(missing=("curl", "git"))
    asyncio.run(agent.install(environment))
    asyncio.run(agent.run("Play the scene.", environment, AgentContext()))

    # The image gets curl and git from its package manager.
    tools, install, run = environment.commands[3], environment.commands[4], environment.commands[-2]
    assert tools.endswith("apt-get update && apt-get install -y curl git")
    clone = f"git clone --quiet --depth 1 --branch {hermes.install.ref} {hermes.install.repository}"
    assert clone in install and 'rm -rf "$HOME/.reef-agent/src/.git"' in install
    assert '"$HOME/.local/bin/uv" pip install --quiet --python "$HOME/.reef-agent/venv/bin/python" -e' in install
    assert run.startswith('agent="$HOME/.reef-agent/venv/bin/hermes"')
    assert agent.get_version_command() == '"$HOME/.reef-agent/venv/bin/hermes" --version'


def test_an_npm_agent_uses_the_node_of_nvm_on_a_glibc_image_and_of_the_packages_on_a_musl_image(
    tmp_path: Path,
) -> None:
    pi = get_adapter("pi")
    package = f"{pi.install.package}@{pi.install.version}"

    def installed(name: str, environment: RecordingEnvironment) -> tuple[str, str]:
        (tmp_path / name).mkdir()
        agent = AdapterAgent(tmp_path / name, **adapter_agent_kwargs(pi, pi.install, tmp_path / "files", 60.0))
        asyncio.run(agent.install(environment))
        asyncio.run(agent.run("Play.", environment, AgentContext()))
        (install,) = (command for command in environment.commands if "npm install -g" in command)
        return install, environment.commands[-2]

    glibc = RecordingEnvironment()
    install, run = installed("glibc", glibc)
    assert "nvm install 22" in install and f"npm install -g {package} && " in install
    assert not any("nodejs" in command for command in glibc.commands)

    # The Node that nvm downloads does not run on musl: the image's packages give Node, and the agent goes under
    # the agent user's own home.
    musl = RecordingEnvironment(missing=("node", "npm"), manager="apk", musl=True)
    install, run = installed("musl", musl)
    assert any(command.endswith("apk add --no-cache nodejs npm") for command in musl.commands)
    assert "nvm install" not in install
    # npm alone only warns when the image's Node is older than the agent needs.
    assert f"npm install -g --engine-strict --prefix $HOME/.reef-npm {package} && " in install
    for command in (install, run):
        assert f"{NPM_AGENT_PATH} && " in command


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


def test_the_agent_path_follows_what_the_install_wrote(tmp_path: Path) -> None:
    def agent(directory: Path, text: str) -> None:
        directory.mkdir(parents=True)
        (directory / "agent").write_text(f"#!/bin/sh\necho {text}\n")
        (directory / "agent").chmod(0o755)

    # An image with an nvm of its own: nvm.sh puts its Node, and the agent that it holds, on the PATH.
    home = tmp_path / "home"
    agent(home / "nvm-node" / "bin", "from-nvm")
    (home / ".nvm").mkdir()
    (home / ".nvm" / "nvm.sh").write_text(f'export PATH="{home}/nvm-node/bin:$PATH"\n')
    found = f"{NPM_AGENT_PATH} && agent"
    assert run_bash(found, home, tmp_path).stdout == "from-nvm\n"

    # After a musl install, the prefix holds the agent, and it wins over the image's nvm.
    agent(home / ".reef-npm" / "bin", "from-prefix")
    assert run_bash(found, home, tmp_path).stdout == "from-prefix\n"


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
