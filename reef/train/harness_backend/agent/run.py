"""Run the deployment's coding agent in one step's workspace.

``AgentExecution`` is the agent bound for one scenario: its adapter, the
executor that isolates it, its binary and its time limit.
``launch_agent`` runs one process of an adapter's harness over rendered
files in directories the caller owns, as ``run_episode`` does in a
throwaway root, and ``read_session`` reads back its session log. ``run_agent``
implements ``Services.run_agent``: it starts the gateway with the workspace's
tools (``check`` and ``evaluate``), renders the agent's tree after Reef's
protocol rules (``protocol.md``), runs the agent in the workspace, and keeps
its session log in ``output/``.

The agent's root is the directory that holds the step's workspace and the
history, so the agent reads both. Its working directory is the workspace;
``harness/``, ``state/``, ``scratch/`` and ``output/`` are writable, and
``material/``, the history and the agent's own rendered files are not.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from reef.core.errors import ReefError
from reef.evaluation.harness.history import EPISODES_DIR
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import headless_argv, headless_env
from reef.harness.episodes.trajectory import reader_for
from reef.harness.gateway import AgentGateway, AgentTool
from reef.harness.tree import HarnessTree, render_tree
from reef.sandbox.environment import download_directory, execute
from reef.sandbox.process import EpisodeExecutor, EpisodeLaunchError, EpisodeTimeout, ProcessOutcome
from reef.train.harness_backend.contract import AgentOutcome, Services
from reef.train.harness_backend.workspace import Workspace

#: Reef's rules for every agent run, before the agent's own: the workspace, the outputs and the tools.
PROTOCOL = Path(__file__).with_name("protocol.md").read_text(encoding="utf-8").strip()
#: In the step's ``scratch/``: the agent's home, which holds its rendered harness and its sessions.
AGENT_HOME = "agent"
#: In the step's ``output/``: the agent's session log, one JSON event per line.
SESSION_FILE = "agent-session.jsonl"
#: The tree a tool reads when its arguments name none.
DEFAULT_TREE = "harness"


@dataclass(frozen=True)
class AgentExecution:
    """Where and how the deployment's coding agent runs, bound for one scenario.

    Values: ``descriptor`` is the agent's adapter, which takes a text
    prompt. ``executor`` is bound to the adapter's program, and ``binary`` is
    the agent's binary there. ``timeout_s`` > 0 bounds one run.
    Purpose: ``AgentSettings.bind`` makes one per scenario; ``run_agent``
    runs the agent with it.
    """

    descriptor: AdapterDescriptor
    executor: EpisodeExecutor
    binary: str
    timeout_s: float

    def __post_init__(self) -> None:
        if self.descriptor.is_prompt_task_directory:
            raise ValueError(f"adapter {self.descriptor.name!r} plays task directories, so it cannot run a prompt")
        if self.timeout_s <= 0:
            raise ValueError("AgentExecution.timeout_s must be positive")


def agent_env(descriptor: AdapterDescriptor, binary: str, home: Path) -> dict[str, str]:
    """The environment of an agent process: the headless environment pointed at ``home``, and the binary's
    directory first on ``PATH``, so a nested call of the harness finds the same binary."""
    env = headless_env(descriptor, home)
    env["PATH"] = os.pathsep.join([str(Path(binary).absolute().parent), os.environ.get("PATH", "")])
    return env


def launch_agent(
    descriptor: AdapterDescriptor,
    executor: EpisodeExecutor,
    argv: Sequence[str],
    files: Mapping[str, str],
    env: Mapping[str, str],
    *,
    root: Path,
    home: Path,
    cwd: Path,
    timeout: float,
    writable: Sequence[Path] = (),
    readonly: Sequence[Path] = (),
) -> ProcessOutcome:
    """Run ``argv`` as ``descriptor``'s harness over ``files``, and wait for it.

    ``root`` holds everything the process may see, and ``cwd``, its working
    directory, is writable. ``files`` are written under ``home`` and stay
    read-only; the descriptor's state directories under ``home``
    (``writable_paths``) and ``writable`` stay writable, and ``readonly``
    names more paths the process only reads. Raises ``EpisodeTimeout`` and
    ``EpisodeLaunchError`` as the executor does.
    """
    for relative, text in descriptor.place_root(files, str(home)).items():
        target = home / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    cwd.mkdir(parents=True, exist_ok=True)
    state = tuple(home / PurePosixPath(relative) for relative in descriptor.writable_paths)
    for path in state:
        path.mkdir(parents=True, exist_ok=True)
    return execute(
        executor,
        list(argv),
        root=root,
        workspace=cwd,
        env=dict(env),
        timeout=timeout,
        writable_paths=(*state, *writable),
        readonly_paths=(*(home / PurePosixPath(relative) for relative in files), *readonly),
    )


def read_session(descriptor: AdapterDescriptor, home: Path) -> list[Mapping[str, Any]]:
    """The session log of the agent whose home is ``home``, read by the descriptor's trajectory reader."""
    return list(reader_for(descriptor.trajectory_format)(home / descriptor.trajectory_path))


def fill(text: str, workspace: Workspace) -> str:
    """``text`` with ``{step}`` as the step's number and ``{history}`` as the history's path from the workspace."""
    history = os.path.relpath(workspace.history_dir, workspace.root)
    return text.replace("{step}", str(workspace.step)).replace("{history}", history)


class AgentRun:
    """One agent run in a step's workspace: what its tools read, and the session that runs it.

    Values: ``workspace`` is the step's workspace and ``services`` the step's
    services. ``root`` is the run's root: the directory that holds the
    workspace and the history. ``session`` is the executor of the running
    agent, ``None`` before it starts. ``lock`` makes the run's tool calls
    take turns: a call reads the agent's files and may write the history.
    Purpose: ``run_agent`` makes one per run; its tools read trees through
    ``read_tree``.
    """

    def __init__(self, workspace: Workspace, services: Services) -> None:
        self.workspace = workspace
        self.services = services
        self.root = Path(os.path.commonpath([workspace.root.resolve(), workspace.history_dir.resolve()]))
        self.session: EpisodeExecutor | None = None
        self.lock = threading.Lock()

    def read_tree(self, arguments: Mapping[str, Any]) -> HarnessTree:
        """The tree in the workspace directory ``arguments["tree"]`` names (``harness`` when none), as the agent
        left it; ``ValueError`` for a name outside the workspace and for a tree that cannot be read."""
        name = arguments.get("tree", DEFAULT_TREE)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tree must name a directory of the step's workspace, such as harness")
        workspace = self.workspace.root.resolve()
        directory = (workspace / name).resolve()
        if directory != workspace and workspace not in directory.parents:
            raise ValueError(f"tree {name!r} is outside the step's workspace")
        if self.session is not None:
            # A remote agent edits a copy; a tool reads that copy as it stands now.
            download_directory(self.session, self.root, directory.relative_to(self.root).as_posix())
        return HarnessTree.read(directory)


class CheckTool(AgentTool):
    """``check``: validate a tree of the workspace for the harness's adapter."""

    name = "check"
    description = (
        "Validate a tree of the step's workspace for the harness's adapter, as the step validates its candidate. "
        "Answers passed (true or false) and problems (a list of messages)."
    )
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "tree": {"type": "string", "description": "A directory of the step's workspace.", "default": "harness"}
        },
        "additionalProperties": False,
    }

    def __init__(self, run: AgentRun) -> None:
        self.run = run

    def call(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = set(arguments) - {"tree"}
        if unknown:
            raise ValueError(f"check takes tree, not {', '.join(sorted(unknown))}")
        with self.run.lock:
            result = self.run.services.check(self.run.read_tree(arguments))
        verdict = "passed" if result.passed else f"refused: {result.message()}"
        self.run.services.calls.note("check", f"the agent's tree {verdict}", failed=not result.passed)
        return {"passed": result.passed, "problems": list(result.problems)}


class EvaluateTool(AgentTool):
    """``evaluate``: score a tree of the workspace on tasks, through the score cache and the step's budget."""

    name = "evaluate"
    description = (
        "Score a tree of the step's workspace on tasks, repeats episodes each. A scored episode in the history is "
        "reused; new episodes spend the step's episode budget. Answers tree (its checksum) and episodes: one "
        "record per episode with its task, repeat, score (null when it could not run), failure, reply, and "
        "directory (its files, from the workspace)."
    )
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "tree": {"type": "string", "description": "A directory of the step's workspace.", "default": "harness"},
            "tasks": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "repeats": {"type": "integer", "minimum": 1, "default": 1},
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }

    def __init__(self, run: AgentRun) -> None:
        self.run = run

    def call(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = set(arguments) - {"tree", "tasks", "repeats"}
        if unknown:
            raise ValueError(f"evaluate takes tree, tasks and repeats, not {', '.join(sorted(unknown))}")
        tasks = arguments.get("tasks")
        if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
            raise ValueError("tasks must be a non-empty list of tasks")
        repeats = arguments.get("repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be an integer of at least 1")
        with self.run.lock:
            tree = self.run.read_tree(arguments)
            records = self.run.services.evaluate(tree, tasks, repeats=repeats)
        history = self.run.workspace.history_dir / EPISODES_DIR
        episodes = [
            {
                "task": record.task,
                "repeat": record.repeat,
                "score": record.score,
                "failure": None if record.failure is None else record.failure.cause,
                "reply": record.reply,
                "directory": os.path.relpath(history / record.episode, self.run.workspace.root),
            }
            for record in records
        ]
        return {"tree": tree.checksum(), "episodes": episodes}


def run_agent(
    execution: AgentExecution, workspace: Workspace, services: Services, prompt: str, *, tree: HarnessTree, model: str
) -> AgentOutcome:
    """``Services.run_agent``: run the agent of ``execution`` once in ``workspace``, and keep its session log."""
    binding = services.models[model]
    descriptor = execution.descriptor
    run = AgentRun(workspace, services)
    gateway = AgentGateway(binding, services.calls, (CheckTool(run), EvaluateTool(run)))
    home = workspace.scratch / AGENT_HOME
    gateway.start()
    started = time.monotonic()
    try:
        # The agent holds the gateway's address and no credential; the gateway calls the binding with its key.
        reached = ModelBinding(
            base_url=gateway.base_url,
            model=binding.model,
            api=binding.api,
            max_output_tokens=binding.max_output_tokens,
            metadata=binding.metadata,
        )
        rules = "\n\n".join(text for text in (fill(PROTOCOL, workspace), tree.files.get("rules.md", "")) if text)
        try:
            files = render_tree(
                HarnessTree({**tree.files, "rules.md": rules}), descriptor, reached.config_overlays(descriptor)
            )
        except ReefError as exc:
            raise ValueError(f"the agent's tree does not render for adapter {descriptor.name!r}: {exc}") from exc
        env = {**agent_env(descriptor, execution.binary, home), "REEF_AGENT_URL": gateway.base_url}
        services.calls.note("agent", f"the coding agent started (at most {execution.timeout_s:g} s)")
        try:
            run.session = execution.executor.open_session(forward_ports=(gateway.port,), timeout_s=execution.timeout_s)
            try:
                process = launch_agent(
                    descriptor,
                    run.session,
                    headless_argv(descriptor, execution.binary, prompt),
                    files,
                    env,
                    root=run.root,
                    home=home,
                    cwd=workspace.root,
                    timeout=execution.timeout_s,
                    writable=(workspace.harness, workspace.state, workspace.scratch, workspace.output),
                    readonly=(workspace.material,),
                )
            finally:
                # The executor copies the agent's files back, also after a time out.
                run.session.close()
            outcome = AgentOutcome(process.exit_code, False, None, round(time.monotonic() - started, 1))
        except EpisodeTimeout:
            outcome = AgentOutcome(None, True, None, round(time.monotonic() - started, 1))
        except EpisodeLaunchError as exc:
            outcome = AgentOutcome(None, False, str(exc), round(time.monotonic() - started, 1))
    finally:
        gateway.stop()
    keep_session(descriptor, home, workspace.output / SESSION_FILE)
    services.calls.note("agent", f"the coding agent {outcome.describe()}", failed=not outcome.succeeded)
    return outcome


def keep_session(descriptor: AdapterDescriptor, home: Path, target: Path) -> None:
    """Write the session log under ``home`` to ``target``, one JSON event per line, when there is one."""
    try:
        events = read_session(descriptor, home)
    except (OSError, ValueError, ReefError):
        # A run that never started, or ended mid-write, may leave no log the reader takes.
        return
    if events:
        target.write_text("".join(json.dumps(event, default=str) + "\n" for event in events), encoding="utf-8")


__all__ = [
    "AGENT_HOME",
    "SESSION_FILE",
    "AgentExecution",
    "AgentRun",
    "CheckTool",
    "EvaluateTool",
    "agent_env",
    "fill",
    "keep_session",
    "launch_agent",
    "read_session",
    "run_agent",
]
