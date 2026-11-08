"""A prompt adapter's agent plays a Harbor task directory in the task's container, scored by the task's verifier."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from harbor.models.agent.context import AgentContext

from reef.evaluation.harness.evaluator import container_binding
from reef.evaluation.harness.scoring import verifier_reward
from reef.harness.adapters import get_adapter
from reef.harness.harbor.agent import (
    CONTAINER_ROOT,
    AdapterAgent,
    TreeAgent,
    adapter_agent_kwargs,
    can_run_in_task_container,
    run_harbor_task,
    task_agent_tools,
    tree_agent,
)
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import TaskRunError
from reef.harness.tasks.trajectory import failed_trial_error

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


def test_only_an_adapter_with_an_npm_or_uv_install_and_a_text_prompt_plays_a_task_in_a_container(
    tmp_path: Path,
) -> None:
    assert all(
        can_run_in_task_container(get_adapter(name)) for name in ("claude", "pi", "opencode", "codex", "python")
    )
    # An adapter whose own command takes the task directory runs it itself; hermes installs from git.
    assert not can_run_in_task_container(replace(get_adapter("pi"), takes_task_directory=True))
    assert not can_run_in_task_container(get_adapter("hermes"))
    with pytest.raises(TaskRunError, match="'hermes' cannot play a Harbor task directory"):
        run_harbor_task(get_adapter("hermes"), {}, str(harbor_task(tmp_path)))
    with pytest.raises(TaskRunError, match="is not a Harbor task directory"):
        run_harbor_task(get_adapter("pi"), {}, str(tmp_path / "missing"))


class RecordingEnvironment:
    """A task container that keeps each command an agent runs in it and each file it uploads, and holds no session
    directory. ``events`` is both, in order."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: dict[str, str] = {}
        self.events: list[str] = []
        self.stream_handle = SimpleNamespace(provider="modal")

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.commands.append(command)
        self.events.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

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

    install, run = environment.commands[1], environment.commands[-1]
    assert f"curl -LsSf https://astral.sh/uv/{python.install.version}/install.sh" in install
    assert 'agent="$HOME/.local/bin/uv"' in run
    assert f"\"$agent\" run --script {CONTAINER_ROOT}/program/main.py 'Play the scene.'" in run
    assert f"REEF_PROGRAM_SESSION_DIR={CONTAINER_ROOT}/program/sessions" in run


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
    assert task_agent_tools(task) == ()

    config = (task / "task.toml").read_text()
    server = '[[environment.mcp_servers]]\nname = "search"\ntransport = "sse"\nurl = "http://search:8000/sse"\n'
    (task / "task.toml").write_text(f'{config}\n[environment]\nskills_dir = "/skills"\n\n{server}')
    assert task_agent_tools(task) == ("mcp_servers", "skills_dir")
