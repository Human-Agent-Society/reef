"""The native_harbor adapter: the native loop as a Harbor agent, its tools in the task container.

A fake Harbor environment runs each command with ``sh`` on this host, under a temporary tree whose ``workspace`` is
the task's workdir and whose ``reef`` is the support directory, with only a few commands on PATH and stderr folded
into stdout as Harbor's Docker exec does. A fake reef-eval stands in for the trial where the runner's row is checked.
No test here runs Docker or a real model."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from reef_service.test_harness_render import NATIVE_HOOK, NATIVE_TOOL, NODES, golden_tree
from reef_service.test_native_harness import _call, _FakeModel, _reply, _seed_nodes
from reef_service.test_native_team import TEAM_NODES, MemberModel, WritingModel, crew_graph, member_files

import reef.harness.runners.native as native
from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.adapters.native_harbor.quirks import unshipped_code
from reef.harness.episodes.e2b import E2BExecutor
from reef.harness.episodes.executor import EpisodeLaunchError, LocalExecutor, SandboxExecutor, SandboxUnavailable
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import EpisodeError, EpisodeResult, run_episode
from reef.harness.episodes.trajectory import reader_for
from reef.harness.runners.harbor_trial import (
    INFRASTRUCTURE_MARKERS_ENV,
    InfrastructureMarker,
    infrastructure_error,
    infrastructure_markers,
)
from reef.harness.runners.native import MAX_COMPLETION_TOKENS, run_loop
from reef.harness.runners.native.__main__ import main as native_main
from reef.harness.runners.native.control import EpisodeControl, RequestPolicy
from reef.harness.runners.native.seed import SEED_HOOKS, SEED_NODES, SEED_TOOLS
from reef.harness.runners.native.task import AGENT_IMPORT_PATH, ENVIRONMENT_ENV, TRIALS_DIR_ENV
from reef.harness.tree.render import render_composition
from reef.train.cordis_backend.backend import EpisodeEvaluationWorker, tree_files
from reef.train.cordis_backend.strategies import resolve_episode_scorer, verifier_reward

TASK = "tasks/sum-and-product"
#: What the fake environment's PATH holds besides python3: what a slim task image with git has.
COMMANDS = ("cat", "chmod", "env", "find", "git", "mkdir", "mktemp", "rm", "rmdir", "sh")
PID_TOOL = (
    "native_tool",
    {
        "name": "pid",
        "description": "The id of the process the call runs in.",
        "parameters": {"type": "object", "properties": {}},
        # Output on both streams: the child's reply must still come back alone.
        "code": "import os\nimport sys\nprint('loading')\n\n\ndef run(args, workdir):\n"
        "    print('running', file=sys.stderr)\n    return str(os.getpid())\n",
    },
)
BIG_TOOL = (
    "native_tool",
    {
        "name": "big",
        "description": "A result over the cap.",
        "parameters": {"type": "object", "properties": {}},
        "code": "def run(args, workdir):\n    return 'x' * 30000\n",
    },
)
RETRY_HOOK = (
    "native_hook",
    {
        "name": "retry",
        "event": "request_error",
        "code": "def listen(payload, next):\n    return {'kind': 'retry', 'delay_ms': 0}\n",
    },
)
USAGE = {"prompt_tokens": 200, "completion_tokens": 100}


def harbor_agent_class():
    pytest.importorskip("harbor", reason="harbor is not installed")
    from reef.harness.runners.native.harbor import NativeTeamAgent

    return NativeTeamAgent


class FakeEnvironment:
    """A Harbor environment on this host: ``exec`` runs ``sh -c`` in the workspace with PATH on ``bin`` alone and
    stderr folded into stdout, uploads copy, and every command and upload is recorded."""

    def __init__(self, tmp_path: Path, *, is_python_installed: bool = True) -> None:
        self.workspace_path = tmp_path / "workspace"
        self.support_path = tmp_path / "reef"
        self.verifier_path = tmp_path / "verifier"
        self.bin_path = tmp_path / "bin"
        self.workspace_path.mkdir()
        self.verifier_path.mkdir()
        self.bin_path.mkdir()
        for name in COMMANDS:
            (self.bin_path / name).symlink_to(shutil.which(name))
        if is_python_installed:
            (self.bin_path / "python3").symlink_to(Path(sys.executable).resolve())
        self.default_user = None
        self.commands: list[tuple[float, str]] = []
        self.uploads: list[tuple[str, int]] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append((time.monotonic(), command))
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd or self.workspace_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": str(self.bin_path), "HOME": str(self.workspace_path)},
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout_sec)
        return SimpleNamespace(stdout=stdout.decode(), stderr=None, return_code=process.returncode)

    async def upload_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        self.uploads.append((str(target_path), target.stat().st_size))

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    def child_runs(self) -> list[str]:
        return [command for _, command in self.commands if "sandboxed.py" in command]


def render_tree(tmp_path: Path, model: _FakeModel, nodes, *, max_output_tokens: int = 32000) -> Path:
    """The native_harbor tree over ``nodes`` and the model's binding, rendered under ``episode``; its native root."""
    descriptor = get_adapter("native_harbor")
    binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy", max_output_tokens=max_output_tokens)
    for relative, text in render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor).items():
        path = tmp_path / "episode" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path / "episode" / "native"


def make_agent(tmp_path: Path, root: Path, environment: FakeEnvironment, **options):
    return harbor_agent_class()(
        tmp_path / "logs",
        "fake",
        tree_path=str(root),
        session_path=str(root / "sessions"),
        max_completion_tokens=options.pop("max_completion_tokens", 32000),
        support_path=str(environment.support_path),
        verifier_path=str(environment.verifier_path),
        **options,
    )


async def play(agent, environment: FakeEnvironment, *, timeout_seconds: float | None = None):
    from harbor.models.agent.context import AgentContext

    context = AgentContext()
    await agent.setup(environment)
    await asyncio.wait_for(agent.run("put hello in notes.txt", environment, context), timeout_seconds)
    return context


def stop(model: _FakeModel) -> None:
    model.shutdown()
    model.server_close()


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def typed(found: list[dict], type_: str) -> list[dict]:
    return [event["data"] for event in found if event["type"] == type_]


def test_the_adapter_is_bundled_and_renders_what_native_renders() -> None:
    descriptor = get_adapter("native_harbor")
    assert "native_harbor" in available_adapters()
    assert (descriptor.binary, descriptor.argv) == ("reef-native", ("task", "--task", "{prompt}"))
    assert descriptor.is_prompt_task_directory and descriptor.self_isolating
    assert descriptor.env == {
        "REEF_NATIVE_DIR": "{root}/native",
        "REEF_NATIVE_SESSION_DIR": "{root}/native/sessions",
        TRIALS_DIR_ENV: "{root}/native_harbor/trials",
        "HOME": "{root}/workspace",
    }
    assert (descriptor.trajectory_format, descriptor.trajectory_path) == ("native-jsonl", "native/sessions")
    assert descriptor.writable_paths == ("native/sessions", "native_harbor/trials")
    assert descriptor.tree_path == "native/tree.json"
    nodes = [node for node in NODES if node[0] not in ("agent_command", "code_extension")]
    assert render_composition([*nodes, NATIVE_TOOL, NATIVE_HOOK], descriptor) == golden_tree("native")
    # The binding adds the reply budget of a model call, as a number.
    (node,) = ModelBinding(base_url="http://127.0.0.1:9", model="m", api_key="k").compose_nodes(descriptor)
    assert node == (
        "config",
        {
            "target": "models",
            "data": {
                "api": "openai",
                "base_url": "http://127.0.0.1:9",
                "api_key": "k",
                "model": "m",
                "max_output_tokens": 32000,
            },
        },
    )


def seed_files(nodes=()) -> dict[str, str]:
    """The seed tree plus ``nodes`` as an episode carries it: the rendered files and the entries list."""
    descriptor = get_adapter("native_harbor")
    entries = [*SEED_NODES, *({"id": config["name"], "name": kind, "config": config} for kind, config in nodes)]
    files = render_composition([(entry["name"], entry["config"]) for entry in entries], descriptor)
    return {**files, **tree_files(descriptor, entries)}


def test_outside_the_sandbox_only_the_seed_hooks_and_no_loop_run_in_the_process_that_writes_the_row(
    tmp_path: Path,
) -> None:
    validate = get_adapter("native_harbor").validate_execution
    assert unshipped_code(seed_files()) == []
    validate(seed_files(), LocalExecutor())
    guard = dict(SEED_HOOKS[0]["config"], code=SEED_HOOKS[0]["config"]["code"].replace("(3, 5, 8)", "(2,)"))
    spy = (
        "native_hook",
        {"name": "spy", "event": "pre_step", "code": "def listen(payload, next):\n    return next()\n"},
    )
    loop = ("native_loop", {"name": "main", "code": "def run_turn(ctx):\n    ctx.end()\n"})
    tuned = seed_files()
    tuned["native/tree.json"] = json.dumps(
        [{**entry, "config": guard} if entry["id"] == "loop_guard" else entry for entry in SEED_NODES]
    )
    assert unshipped_code(tuned) == ["native_hook loop_guard"]
    assert unshipped_code(seed_files([spy])) == ["native_hook spy"]
    assert unshipped_code(seed_files([loop])) == ["native_loop main"]
    # A loop the entries list carries is mounted by a tree boot whether or not a file was rendered for it.
    only_listed = seed_files()
    only_listed["native/tree.json"] = json.dumps(
        [*SEED_NODES, {"id": "turn", "name": "native_loop", "config": loop[1]}]
    )
    assert unshipped_code(only_listed) == ["native_loop main"]
    with pytest.raises(EpisodeLaunchError, match="native_hook spy is not code Reef ships, so it needs evolution"):
        validate(seed_files([spy]), LocalExecutor())
    # The episode lifecycle asks the validator before it writes the root.
    with pytest.raises(EpisodeError, match="imports hook and loop code into the process that writes the verifier row"):
        run_episode(get_adapter("native_harbor"), seed_files([loop]), TASK, binary=str(tmp_path / "missing"))


def test_in_the_sandbox_the_task_runs_on_e2b_and_tree_code_is_admitted() -> None:
    validate = get_adapter("native_harbor").validate_execution
    spy = (
        "native_hook",
        {"name": "spy", "event": "pre_step", "code": "def listen(payload, next):\n    return next()\n"},
    )
    with pytest.raises(
        EpisodeLaunchError, match=r"native_harbor Docker cannot run under evolution\.executor: sandbox"
    ):
        validate(seed_files(), SandboxExecutor(env={ENVIRONMENT_ENV: "docker"}))
    with pytest.raises(EpisodeLaunchError, match="requires egress_hosts and E2B_API_KEY"):
        validate(seed_files(), SandboxExecutor(egress_hosts=("api.e2b.app",), env={ENVIRONMENT_ENV: "e2b"}))
    jailed = SandboxExecutor(egress_hosts=("api.e2b.app",), env={ENVIRONMENT_ENV: "e2b", "E2B_API_KEY": "k"})
    validate(seed_files([spy]), jailed)
    # The E2B episode executor runs the runner remotely, where Harbor would need a container of its own.
    with pytest.raises(SandboxUnavailable, match="manages its own containers"):
        E2BExecutor(api_key="test-key", template="custom").for_adapter(get_adapter("native_harbor"))


class ToolsModel(_FakeModel):
    """Writes a note, reads it and asks for the process id in one step, then answers; ``usage`` when given."""

    def __init__(self, usage: dict | None = None) -> None:
        super().__init__()
        self.usage = usage

    def script(self, body: dict) -> dict:
        if any(message.get("role") == "tool" for message in body["messages"]):
            reply = _reply(content="done")
        else:
            calls = [
                _call("write_file", {"path": "notes.txt", "content": "hello"}, "c1"),
                _call("read_file", {"path": "notes.txt"}, "c2"),
                _call("pid", {}, "c3"),
            ]
            reply = _reply(tool_calls=calls)
        return reply if self.usage is None else {**reply, "usage": self.usage}


def test_every_tool_call_runs_once_in_the_task_environment_and_the_turn_runs_in_its_workdir(tmp_path: Path) -> None:
    model = ToolsModel()
    environment = FakeEnvironment(tmp_path)
    try:
        root = render_tree(tmp_path, model, [*_seed_nodes(SEED_TOOLS), PID_TOOL])
        asyncio.run(play(make_agent(tmp_path, root, environment), environment))
    finally:
        stop(model)
    found = events(root / "sessions" / "session.jsonl")
    header = found[0]["data"]
    assert header["enforcement"] == "task-environment"
    assert Path(header["cwd"]).resolve() == environment.workspace_path.resolve()
    results = typed(found, "tool/result")
    assert [result["content"] for result in results[:2]] == ["wrote 5 characters to notes.txt", "hello"]
    assert {result["enforcement"]["mode"] for result in results} == {"task-environment"}
    # One child per call, in the container's python3, which imported the module afresh: not this process. What the
    # module printed on either stream did not reach the reply.
    assert len(environment.child_runs()) == 3 and not any(result["is_error"] for result in results)
    assert results[2]["content"].isdigit() and results[2]["content"] != str(os.getpid())
    assert (environment.workspace_path / "notes.txt").read_text() == "hello"
    assert sorted(path.name for path in (environment.support_path / "tools").glob("*.py")) == [
        "execute.py",
        "pid.py",
        "read_file.py",
        "run_bash.py",
        "write_file.py",
    ]
    # Each request file went up for its call and was removed by it.
    assert list((environment.support_path / "requests").iterdir()) == []
    assert typed(found, "turn/end")[-1]["reason"] == {"kind": "completed"}


def test_the_bridge_returns_a_commands_stdout_alone_and_a_wrapper_failure_as_its_stderr(tmp_path: Path) -> None:
    harbor_agent_class()
    from reef.harness.runners.native.harbor import HarborTaskEnvironment

    environment = FakeEnvironment(tmp_path)

    async def run(command: str):
        bridge = HarborTaskEnvironment(environment, asyncio.get_running_loop())
        return await asyncio.to_thread(bridge.exec, command, cwd=None, timeout_seconds=30)

    # The fake folds stderr into stdout, as Harbor's Docker exec does; the bridge takes them apart again.
    done = asyncio.run(run("echo out; echo err >&2; exit 3"))
    assert (done.return_code, done.stdout, done.stderr) == (3, "out\n", "err\n")
    (environment.bin_path / "mktemp").unlink()
    failed = asyncio.run(run("echo out"))
    assert (failed.return_code, failed.stdout) == (125, "") and "mktemp" in failed.stderr


class BigModel(_FakeModel):
    def script(self, body: dict) -> dict:
        if any(message.get("role") == "tool" for message in body["messages"]):
            return _reply(content="read it")
        return _reply(tool_calls=[_call("big", {}, "c1")])


def test_a_clipped_result_is_saved_whole_in_the_container_and_gone_before_the_verifier(tmp_path: Path) -> None:
    model = BigModel()
    environment = FakeEnvironment(tmp_path)
    try:
        root = render_tree(tmp_path, model, [BIG_TOOL])
        asyncio.run(play(make_agent(tmp_path, root, environment), environment))
    finally:
        stop(model)
    (result,) = typed(events(root / "sessions" / "session.jsonl"), "tool/result")
    assert result["meta"]["truncated"] and result["meta"]["output_file"] == ".reef/tool-output/1-c1.txt"
    saved = str(environment.workspace_path / ".reef" / "tool-output" / "1-c1.txt")
    assert (saved, 30000) in environment.uploads
    assert not (environment.workspace_path / ".reef").exists()


def test_setup_needs_python3_in_the_task_image(tmp_path: Path) -> None:
    model = ToolsModel()
    environment = FakeEnvironment(tmp_path, is_python_installed=False)
    try:
        root = render_tree(tmp_path, model, _seed_nodes(SEED_TOOLS))
        with pytest.raises(RuntimeError, match="the task image has no python3, which native_harbor tools need"):
            asyncio.run(play(make_agent(tmp_path, root, environment), environment))
    finally:
        stop(model)
    assert model.requests == []


def test_the_harbor_agent_asks_for_the_bindings_output_limit_and_reports_the_tokens_to_harbor(tmp_path: Path) -> None:
    model = ToolsModel(USAGE)
    environment = FakeEnvironment(tmp_path)
    try:
        root = render_tree(tmp_path, model, [*_seed_nodes(SEED_TOOLS), PID_TOOL], max_output_tokens=777)
        context = asyncio.run(play(make_agent(tmp_path, root, environment, max_completion_tokens=777), environment))
        # The same tree under the native loop alone keeps the loop's own cap.
        (tmp_path / "alone").mkdir()
        assert run_loop("again", root, tmp_path / "alone-sessions", tmp_path / "alone") == 0
    finally:
        stop(model)
    assert [body["max_tokens"] for body in model.requests] == [777, 777, MAX_COMPLETION_TOKENS, MAX_COMPLETION_TOKENS]
    assert (context.n_input_tokens, context.n_output_tokens) == (400, 200)


class RateLimitedModel(_FakeModel):
    """Answers 429 to the first ``limited`` requests, then plays the note script."""

    def __init__(self, limited: int) -> None:
        super().__init__()
        self.limited = limited

    def status(self, body: dict) -> int:
        return 429 if len(self.requests) <= self.limited else 200


def test_a_transient_failure_is_retried_until_it_passes_under_the_harbor_agent_and_four_times_under_native(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(native, "TRANSIENT_RETRY_MAX_DELAY_SECONDS", 0)
    model = RateLimitedModel(5)
    environment = FakeEnvironment(tmp_path)
    try:
        # The tree's hook asks for retries; under the Harbor agent's policy a transient failure never reaches it.
        root = render_tree(tmp_path, model, [*_seed_nodes(SEED_TOOLS), RETRY_HOOK])
        asyncio.run(play(make_agent(tmp_path, root, environment), environment))
    finally:
        stop(model)
    found = events(root / "sessions" / "session.jsonl")
    failures = typed(found, "request/error")
    assert [failure["attempt"] for failure in failures] == [1, 2, 3, 4, 5]
    assert all(failure["error"]["status"] == 429 and failure["error"]["is_transient"] for failure in failures)
    assert typed(found, "hook/decision") == [] and typed(found, "turn/end")[-1]["reason"] == {"kind": "completed"}
    assert len(model.requests) == 8 and (environment.workspace_path / "notes.txt").read_text() == "hello"

    native_model = RateLimitedModel(5)
    try:
        native_root = render_tree(tmp_path / "native", native_model, [*_seed_nodes(SEED_TOOLS), RETRY_HOOK])
        (tmp_path / "native-work").mkdir()
        code = run_loop("put hello in notes.txt", native_root, tmp_path / "native-sessions", tmp_path / "native-work")
    finally:
        stop(native_model)
    assert code == 1 and len(native_model.requests) == native.MAX_REQUEST_ATTEMPTS


class ClientErrorModel(_FakeModel):
    def status(self, body: dict) -> int:
        return 400


class MalformedModel(_FakeModel):
    def script(self, body: dict) -> dict:
        return {"id": "fake", "object": "chat.completion"}


@pytest.mark.parametrize("model_class", [ClientErrorModel, MalformedModel])
def test_a_client_error_or_a_malformed_reply_is_not_retried_under_the_policy(tmp_path: Path, model_class) -> None:
    model = model_class()
    try:
        root = render_tree(tmp_path, model, _seed_nodes(SEED_TOOLS))
        (tmp_path / "work").mkdir()
        control = EpisodeControl(request_policy=RequestPolicy(is_retry_until_stopped=True))
        code = run_loop("put hello in notes.txt", root, tmp_path / "sessions", tmp_path / "work", control=control)
    finally:
        stop(model)
    (failure,) = typed(events(tmp_path / "sessions" / "session.jsonl"), "request/error")
    assert code == 1 and len(model.requests) == 1 and failure["error"]["is_transient"] is False


class SlowModel(_FakeModel):
    """Takes a second over every call, and every answer is a write."""

    def script(self, body: dict) -> dict:
        time.sleep(1.0)
        return _reply(tool_calls=[_call("write_file", {"path": "notes.txt", "content": "late"}, "c1")])


def test_a_cancel_stops_the_turn_before_its_next_call_and_before_its_next_tool(tmp_path: Path) -> None:
    model = SlowModel()
    environment = FakeEnvironment(tmp_path)
    try:
        root = render_tree(tmp_path, model, _seed_nodes(SEED_TOOLS))
        with pytest.raises(TimeoutError):
            asyncio.run(play(make_agent(tmp_path, root, environment), environment, timeout_seconds=0.5))
    finally:
        stop(model)
    found = events(root / "sessions" / "session.jsonl")
    # The call in flight at the cancel came back; its tool never ran, and no call went out after it.
    assert len(model.requests) == 1 and environment.child_runs() == []
    (result,) = typed(found, "tool/result")
    assert result["error"]["code"] == "STOPPED"
    assert typed(found, "turn/end")[-1]["reason"] == {"kind": "stopped", "reason": "cancelled"}
    assert not (environment.workspace_path / "notes.txt").exists()


def test_a_team_works_in_worktrees_in_the_container_and_only_merged_files_reach_the_workdir(tmp_path: Path) -> None:
    model = WritingModel({"peer.1": [("a.txt", "from one\n")], "peer.2": [("b.txt", "from two\n")]})
    environment = FakeEnvironment(tmp_path)
    crew = crew_graph(mode="team", agents=["peer", "peer"], workspace="own")
    try:
        root = render_tree(tmp_path, model, [*TEAM_NODES, ("native_graph", crew)])
        asyncio.run(play(make_agent(tmp_path, root, environment), environment))
    finally:
        stop(model)
    sessions = root / "sessions"
    team_path = environment.support_path / "team"
    headers = {instance: found[0]["data"] for instance, found in member_files(sessions).items()}
    # Each member worked in a worktree under the support directory, outside the task's workdir.
    assert {instance: header["workdir"] for instance, header in headers.items()} == {
        "peer.1": str(team_path / "s1-peer.1"),
        "peer.2": str(team_path / "s1-peer.2"),
    }
    merges = typed(events(sessions / "session.jsonl"), "team/merge")
    assert [(merge["agent"], merge["result"]) for merge in merges] == [("peer.1", "merged"), ("peer.2", "merged")]
    # The verifier finds the merged files in the workdir, no git state there, and no team state anywhere.
    assert (environment.workspace_path / "a.txt").read_text() == "from one\n"
    assert (environment.workspace_path / "b.txt").read_text() == "from two\n"
    assert sorted(path.name for path in environment.workspace_path.iterdir()) == ["a.txt", "b.txt"]
    assert not team_path.exists()
    # Every git command ran in the environment, through the commands Harbor's exec ran.
    assert any(" git " in command and "--git-dir=" in command for _, command in environment.commands)


SLOW_WRITE_TOOL = (
    "native_tool",
    {
        "name": "slow_write",
        "description": "Sleeps two seconds, then writes late.txt.",
        "parameters": {"type": "object", "properties": {}},
        "code": (
            "import time\nfrom pathlib import Path\n\n\ndef run(args, workdir):\n"
            "    time.sleep(2)\n    Path(workdir, 'late.txt').write_text('late')\n    return 'wrote late.txt'\n"
        ),
    },
)


class SlowMemberModel(MemberModel):
    """The root hands the task on; each member runs the slow write once, then answers."""

    def reply(self, instance: str, body: dict) -> dict:
        if instance == "root":
            return _reply(content="go")
        if not any(message.get("role") == "tool" for message in body["messages"]):
            return _reply(tool_calls=[_call("slow_write", {}, "s1")])
        return _reply(content="wrote late.txt")


def test_past_the_cancel_grace_no_command_of_the_episode_reaches_the_container(tmp_path: Path, monkeypatch) -> None:
    harbor_agent_class()
    import reef.harness.runners.native.harbor as harbor_module

    monkeypatch.setattr(harbor_module, "CANCEL_GRACE_SECONDS", 0.3)
    model = SlowMemberModel()
    environment = FakeEnvironment(tmp_path)
    crew = crew_graph(mode="team", agents=["peer"], workspace="own")

    async def cancelled_then_later() -> tuple[list[str], bool, list[str]]:
        from harbor.models.agent.context import AgentContext

        agent = make_agent(tmp_path, root, environment)
        await agent.setup(environment)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.run("put hello in notes.txt", environment, AgentContext()), 1.0)
        # Harbor would start the verifier here; the member's tool call was still running.
        at_verifier = sorted(path.name for path in environment.workspace_path.iterdir())
        is_team_gone = not (environment.support_path / "team").exists()
        await asyncio.sleep(3)
        return at_verifier, is_team_gone, sorted(path.name for path in environment.workspace_path.iterdir())

    try:
        root = render_tree(tmp_path, model, [*TEAM_NODES, SLOW_WRITE_TOOL, ("native_graph", crew)])
        at_verifier, is_team_gone, later = asyncio.run(cancelled_then_later())
    finally:
        stop(model)
    assert at_verifier == later == [] and is_team_gone
    (merge,) = typed(events(root / "sessions" / "session.jsonl"), "team/merge")
    assert merge["result"] == "failed" and "takes no more commands" in merge["error"]


PLANT_TOOL = (
    "native_tool",
    {
        "name": "plant",
        "description": "Writes text at an absolute path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
        "code": (
            "from pathlib import Path\n\n\ndef run(args, workdir):\n"
            "    Path(args['path']).write_text(args['text'])\n    return 'planted'\n"
        ),
    },
)


class PlantingModel(_FakeModel):
    """Writes a reward and a marker file where Harbor's verifier writes, then answers."""

    verifier_path: Path

    def script(self, body: dict) -> dict:
        if any(message.get("role") == "tool" for message in body["messages"]):
            return _reply(content="done")
        calls = [
            _call("plant", {"path": str(self.verifier_path / "reward.txt"), "text": "1"}, "c1"),
            _call("plant", {"path": str(self.verifier_path / ".m.json"), "text": "{}"}, "c2"),
        ]
        return _reply(tool_calls=calls)


def test_what_a_tool_wrote_where_the_verifier_writes_is_gone_before_the_verifier(tmp_path: Path) -> None:
    model = PlantingModel()
    environment = FakeEnvironment(tmp_path)
    model.verifier_path = environment.verifier_path
    try:
        root = render_tree(tmp_path, model, [PLANT_TOOL])
        asyncio.run(play(make_agent(tmp_path, root, environment), environment))
    finally:
        stop(model)
    results = typed(events(root / "sessions" / "session.jsonl"), "tool/result")
    assert [result["content"] for result in results] == ["planted", "planted"]
    assert list(environment.verifier_path.iterdir()) == []


class FakeLab:
    """reef-eval's Lab for the runner's row: records the agent spec and hands back ``row``, after ``write_trial``."""

    specs: list[dict] = []
    row = SimpleNamespace(rewards={"reward": 1.0}, tags={}, uri=None)
    trial_files: dict[str, str] = {}

    def __init__(self, trials_dir: Path) -> None:
        self.trials_dir = trials_dir

    async def run(self, task: str, agent: dict, **options: object) -> SimpleNamespace:
        FakeLab.specs.append(agent)
        trial = self.trials_dir / "trials" / "sum-and-product__a1"
        for relative, text in FakeLab.trial_files.items():
            (trial / relative).parent.mkdir(parents=True, exist_ok=True)
            (trial / relative).write_text(text)
        trial.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(**{**vars(FakeLab.row), "uri": trial.resolve().as_uri()})


@pytest.fixture
def runner_env(tmp_path: Path, monkeypatch) -> Path:
    """The environment the adapter's episode gives ``reef-native task`` over a rendered tree; the sessions path."""
    FakeLab.specs, FakeLab.row, FakeLab.trial_files = [], SimpleNamespace(rewards={"reward": 1.0}, tags={}), {}
    monkeypatch.setitem(sys.modules, "reef_eval", SimpleNamespace(Lab=FakeLab))
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="fake", api_key="dummy", max_output_tokens=900)
    descriptor = get_adapter("native_harbor")
    for relative, text in render_composition([*_seed_nodes(), *binding.compose_nodes(descriptor)], descriptor).items():
        path = tmp_path / "root" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for key, value in descriptor.env.items():
        monkeypatch.setenv(key, value.replace("{root}", str(tmp_path / "root")))
    monkeypatch.delenv(INFRASTRUCTURE_MARKERS_ENV, raising=False)
    monkeypatch.delenv("REEF_EPISODE_TOKENS", raising=False)
    return tmp_path / "root" / "native" / "sessions"


def score(sessions: Path, exit_code: int):
    """The episode as the backend scores it under ``verifier_reward``."""
    worker = EpisodeEvaluationWorker(
        descriptor=get_adapter("native_harbor"),
        scorer=resolve_episode_scorer(verifier_reward),
        binary=None,
        timeout=10,
        executor=LocalExecutor(),
        forbid_residue=False,
    )
    return worker._score_result(EpisodeResult(exit_code, "", "", reader_for("native-jsonl")(sessions), ()), TASK)


def test_the_runner_writes_one_flat_verifier_row_that_the_scorer_reads(runner_env: Path, monkeypatch) -> None:
    monkeypatch.setenv("REEF_EPISODE_TOKENS", "5000")
    assert native_main(["task", "--task", TASK]) == 0
    (row,) = events(runner_env / "verifier.jsonl")
    assert row == {
        "type": "verifier",
        "task": TASK,
        "rewards": {"reward": 1.0},
        "reward": 1.0,
        "failed": False,
        "error": "",
    }
    scored = score(runner_env, 0)
    assert scored.score == 1.0 and scored.failure is None
    (spec,) = FakeLab.specs
    assert spec == {
        "import_path": AGENT_IMPORT_PATH,
        "model_name": "fake",
        "kwargs": {
            "tree_path": str(runner_env.parent),
            "session_path": str(runner_env),
            "max_completion_tokens": 900,
            "token_limit": 5000,
        },
    }
    # No credential rides in the trial config Harbor writes.
    assert "dummy" not in json.dumps(spec)


def test_a_trial_without_rewards_is_a_failed_row_that_scores_none(runner_env: Path) -> None:
    FakeLab.row = SimpleNamespace(rewards=None, tags={"error": "docker compose build failed"})
    assert native_main(["task", "--task", TASK]) == 1
    (row,) = events(runner_env / "verifier.jsonl")
    assert (row["failed"], row["error"], row["reward"]) == (True, "docker compose build failed", None)
    scored = score(runner_env, 1)
    assert scored.score is None and scored.failure.stage == "trial"


def test_an_infrastructure_marker_fails_the_row_and_keeps_the_rewards(runner_env: Path, monkeypatch) -> None:
    FakeLab.trial_files = {"verifier/m.json": json.dumps({"errors": {"__infra__": "the evaluator crashed"}})}
    markers = [{"file_name": "m.json", "key": "errors", "values": ["__infra__", "__evaluator__"]}]
    monkeypatch.setenv(INFRASTRUCTURE_MARKERS_ENV, json.dumps(markers))
    assert native_main(["task", "--task", TASK]) == 1
    (row,) = events(runner_env / "verifier.jsonl")
    assert row["failed"] and row["rewards"] == {"reward": 1.0}
    assert row["error"] == "infrastructure failure: m.json errors names __infra__"
    assert score(runner_env, 1).score is None


def test_without_markers_the_reward_stands_and_a_malformed_list_is_a_failed_row(runner_env: Path, monkeypatch) -> None:
    FakeLab.trial_files = {"verifier/m.json": json.dumps({"errors": {"__infra__": "the evaluator crashed"}})}
    assert native_main(["task", "--task", TASK]) == 0
    monkeypatch.setenv(INFRASTRUCTURE_MARKERS_ENV, '{"file_name": "m.json"}')
    assert native_main(["task", "--task", TASK]) == 1
    first, second = events(runner_env / "verifier.jsonl")
    assert not first["failed"] and second["failed"] and INFRASTRUCTURE_MARKERS_ENV in second["error"]


def test_a_marker_reads_keys_items_or_a_string_and_a_torn_file_is_no_hit(tmp_path: Path) -> None:
    marker = InfrastructureMarker("m.json", "errors", ("__infra__",))
    for value, is_hit in ({"__infra__": 1}, True), (["x", "__infra__"], True), ("__infra__", True), ("x", False):
        (tmp_path / "m.json").write_text(json.dumps({"errors": value}))
        assert (infrastructure_error(tmp_path, [marker]) is not None) == is_hit, value
    (tmp_path / "m.json").write_text('{"errors": {"__in')
    assert infrastructure_error(tmp_path, [marker]) is None and infrastructure_error(None, [marker]) is None
    assert infrastructure_markers({}) == ()
    for bad in ("[1]", '[{"file_name": "a/m.json", "key": "k", "values": ["v"]}]', "not json"):
        with pytest.raises(Exception, match=INFRASTRUCTURE_MARKERS_ENV):
            infrastructure_markers({INFRASTRUCTURE_MARKERS_ENV: bad})


def test_a_runner_error_before_the_trial_still_writes_a_failed_row(runner_env: Path, monkeypatch) -> None:
    monkeypatch.delenv(TRIALS_DIR_ENV)
    assert native_main(["task", "--task", TASK]) == 1
    (row,) = events(runner_env / "verifier.jsonl")
    assert row["failed"] and TRIALS_DIR_ENV in row["error"] and FakeLab.specs == []
    assert score(runner_env, 1).score is None


def test_harbor_builds_the_agent_from_the_spec_the_runner_hands_it(runner_env: Path, tmp_path: Path) -> None:
    agent_class = harbor_agent_class()
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig

    assert native_main(["task", "--task", TASK]) == 0
    (spec,) = FakeLab.specs
    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"path": Path("recipes/basic/harbor")}).model_dump(),
            "trials_dir": tmp_path / "trials",
            "agent": spec,
        }
    )
    agent = AgentFactory.create_agent_from_config(AgentConfig.model_validate(spec), logs_dir=tmp_path / "logs")
    assert config.agent.import_path == AGENT_IMPORT_PATH and isinstance(agent, agent_class)
    assert (agent.tree_path, agent.session_path) == (runner_env.parent, runner_env)
    assert (agent.max_completion_tokens, agent.token_limit, str(agent.support_path)) == (900, None, "/reef")
    assert agent.to_agent_info().name == "reef-native"
