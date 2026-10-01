"""The native loop as a Harbor agent; ``reef-native task`` hands Harbor this class by its import path.

Harbor builds the task container and calls ``setup`` and ``run`` on the trial's event loop, then runs its verifier.
``run`` walks the tree on a worker thread (a team's members on threads of their own), and the model calls stay on
this host. Every tool call and every team git command goes to the container through ``HarborTaskEnvironment``,
which schedules Harbor's coroutines on the trial's event loop and waits for them. When Harbor cancels ``run`` at the
task's agent timeout, the episode's stop flag is set and ``run`` waits up to ``CANCEL_GRACE_SECONDS`` for the turn
to end, so the members stop and a stage merges. However ``run`` returns, the bridge then closes, so a turn still
running starts no command in the container, and the saved tool output, the team's git state and anything already
in Harbor's verifier directory are removed: the verifier sees the workdir as it stands and writes its output afresh.

Only Harbor imports this module, so the rest of the native runner never needs Harbor.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import shlex
import threading
import uuid
from collections.abc import Coroutine
from pathlib import Path, PurePosixPath
from typing import TypeVar

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from reef.core.version import __version__
from reef.harness.runners.native import run_loop
from reef.harness.runners.native.control import EpisodeControl, RequestPolicy, TeamBudget
from reef.harness.runners.native.enforce import CHILD
from reef.harness.runners.native.environment import (
    FILE_TIMEOUT_SECONDS,
    SUPPORT_PATH,
    EnvironmentCommandRunner,
    TaskEnvironment,
    TaskEnvironmentEnforcer,
    TaskEnvironmentError,
)
from reef.harness.runners.native.workspaces import CommandOutcome

#: How long a cancelled run waits for its turn to end before Harbor goes on to the verifier.
CANCEL_GRACE_SECONDS = 120.0
#: How long a worker thread waits past a command's own timeout before it gives the command up.
EXEC_GRACE_SECONDS = 30.0
#: How long one command of the agent's own (the setup, the workdir, the cleanup) may take.
AGENT_COMMAND_TIMEOUT_SECONDS = 120

ResultT = TypeVar("ResultT")


class HarborTaskEnvironment(TaskEnvironment):
    """A Harbor environment for worker threads: each call runs on the trial's event loop while the thread waits.

    Harbor's Docker exec returns a command's stderr inside its stdout, so every command's stderr goes to a file in
    the container and comes back after a marker line; the tool child's reply is then stdout alone. Once ``close``
    is called, every call fails at once."""

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop) -> None:
        self.environment = environment
        self.loop = loop
        self.close_lock = threading.Lock()
        self.is_closed = False

    def exec(self, command: str, *, cwd: str | None, timeout_seconds: float) -> CommandOutcome:
        marker = f"reef-stderr-{uuid.uuid4().hex}"
        wrapped = (
            f'err=$(mktemp) || exit 125; ( {command} ) 2>"$err"; code=$?; '
            f'printf "\\n%s\\n" {marker}; cat "$err"; rm -f "$err"; exit $code'
        )
        call = self.environment.exec(wrapped, cwd=cwd, timeout_sec=math.ceil(timeout_seconds))
        result = self.wait(call, timeout_seconds)
        output = result.stdout or ""
        stdout, found, stderr = output.rpartition(f"\n{marker}\n")
        if not found:  # the wrapper itself failed (no mktemp, say): what came back is its error
            stdout, stderr = "", output
        return CommandOutcome(result.return_code, stdout, stderr + (result.stderr or ""))

    def upload_file(self, source_path: Path, target_path: str) -> None:
        self.wait(self.environment.upload_file(source_path, target_path), FILE_TIMEOUT_SECONDS)

    def wait(self, call: Coroutine[object, object, ResultT], timeout_seconds: float) -> ResultT:
        """``call``'s result from the trial's event loop, or a TaskEnvironmentError naming what went wrong."""
        with self.close_lock:
            if self.is_closed:
                call.close()
                raise TaskEnvironmentError("the agent's run has ended; the Harbor environment takes no more commands")
            try:
                future = asyncio.run_coroutine_threadsafe(call, self.loop)
            except RuntimeError as exc:  # the trial's event loop has closed
                call.close()
                raise TaskEnvironmentError(f"the Harbor environment is gone: {exc}") from exc
        try:
            return future.result(timeout=timeout_seconds + EXEC_GRACE_SECONDS)
        except Exception as exc:
            # Harbor raises what its backend raises: a RuntimeError when a Docker exec runs out of time, an SDK error
            # from E2B. A wait past the grace is a TimeoutError here.
            future.cancel()
            raise TaskEnvironmentError(f"{type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        """Refuse every later call, so no command of the episode starts after this; one in flight runs to its end,
        as a command of any Harbor agent does at its timeout."""
        with self.close_lock:
            self.is_closed = True


class NativeTeamAgent(BaseAgent):
    """Reef's native loop, with any team its tree runs, as a Harbor agent.

    Harbor passes the trial config's agent ``kwargs``: where the rendered tree and its sessions are on this host, the
    reply budget of a model call from the binding, and the episode's token budget. The model and its key come from
    the tree's ``models.json``, so no credential goes into the trial config. ``verifier_path`` is where Harbor's
    verifier writes in the container."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        tree_path: str,
        session_path: str,
        max_completion_tokens: int,
        token_limit: int | None = None,
        support_path: str = str(SUPPORT_PATH),
        verifier_path: str = str(EnvironmentPaths.verifier_dir),
        **harbor_options: object,
    ) -> None:
        super().__init__(logs_dir, model_name, **harbor_options)
        self.tree_path = Path(tree_path)
        self.session_path = Path(session_path)
        self.max_completion_tokens = max_completion_tokens
        self.token_limit = token_limit
        self.support_path = PurePosixPath(support_path)
        self.verifier_path = PurePosixPath(verifier_path)

    @staticmethod
    def name() -> str:
        return "reef-native"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        """Check that the image has python3, then put the tool child and the tool modules under the support path."""
        found = await environment.exec("command -v python3", timeout_sec=AGENT_COMMAND_TIMEOUT_SECONDS)
        if found.return_code != 0:
            raise RuntimeError("the task image has no python3, which native_harbor tools need")
        support = shlex.quote(str(self.support_path))

        async def run_as_root(command: str) -> None:
            done = await environment.exec(command, user="root", timeout_sec=AGENT_COMMAND_TIMEOUT_SECONDS)
            if done.return_code != 0:
                raise RuntimeError(f"native_harbor setup failed: {command} exited {done.return_code}: {done.stdout}")

        await run_as_root(f"mkdir -p {support}/tools {support}/requests")
        await environment.upload_file(CHILD, str(self.support_path / "sandboxed.py"))
        tools_path = self.tree_path / "tools"
        if tools_path.is_dir():
            await environment.upload_dir(tools_path, str(self.support_path / "tools"))
        # The agent user writes requests and team git state here, whoever the uploads wrote the files as.
        await run_as_root(f"chmod -R a+rwX {support}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """The root turn on a worker thread in the task's workdir; the token counts land in ``context`` either way."""
        found = await environment.exec("pwd", timeout_sec=AGENT_COMMAND_TIMEOUT_SECONDS)
        workdir = PurePosixPath((found.stdout or "").strip())
        if found.return_code != 0 or not workdir.is_absolute():
            raise RuntimeError(f"the task environment names no working directory: {found.stdout!r}")
        bridge = HarborTaskEnvironment(environment, asyncio.get_running_loop())
        control = EpisodeControl(
            budget=TeamBudget(self.token_limit),
            command_runner=EnvironmentCommandRunner(bridge),
            team_path=self.support_path / "team",
            request_policy=RequestPolicy(is_retry_until_stopped=True),
            max_completion_tokens=self.max_completion_tokens,
        )
        enforcer = TaskEnvironmentEnforcer(bridge, self.support_path)
        # to_thread runs in a copy of this task's context and every member's thread in a copy of the root's, so the
        # environment overlay Harbor keeps in a context variable reaches every command.
        turn = asyncio.ensure_future(
            asyncio.to_thread(
                run_loop,
                instruction,
                self.tree_path,
                self.session_path,
                Path(str(workdir)),
                enforcer=enforcer,
                control=control,
            )
        )
        try:
            try:
                exit_code = await asyncio.shield(turn)
            except asyncio.CancelledError:
                control.stop.set("cancelled")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(turn), CANCEL_GRACE_SECONDS)
                raise
        finally:
            # A turn still running past the grace starts no command in the container.
            bridge.close()
            context.n_input_tokens = control.budget.input_tokens
            context.n_output_tokens = control.budget.output_tokens
            # The verifier sees the task's own files and writes its own output: the full results the loop saved in
            # the workdir, a team's git state left behind, and whatever a tool wrote where the verifier writes go.
            reef_path = shlex.quote(str(workdir / ".reef"))
            team_path = shlex.quote(str(self.support_path / "team"))
            verifier_path = shlex.quote(str(self.verifier_path))
            await environment.exec(
                f"rm -rf {reef_path}/tool-output {team_path}; rmdir {reef_path} 2>/dev/null; "
                f"rm -rf {verifier_path}/* {verifier_path}/.[!.]* {verifier_path}/..?* 2>/dev/null; true",
                user="root",
                timeout_sec=AGENT_COMMAND_TIMEOUT_SECONDS,
            )
        if exit_code != 0:
            raise NonZeroAgentExitCodeError(
                f"reef-native exited {exit_code}: its root turn ended in error ({self.session_path / 'session.jsonl'})"
            )
