"""Team stages of the native loop: the members of one stage run at once, each on its own thread, host and budget.

A member is one agent turn in its own session file, as a sequential subagent is, but it reads its own
``NativeHost`` (its own mount directory, so hook module state is per member) and runs on its agent's step budget,
not on the caller's remaining steps. The caller waits for every member and then reads one message that names each
member's outcome and text. The episode's token budget and stop flag are shared: once either is set, every member
ends its turn at its next step.

With ``workspace: own`` each member works in a git worktree of its own (``reef.harness.runners.native.workspaces``),
and its branch is merged into the caller's workdir when the stage ends; with ``shared`` every member works in the
caller's workdir and nothing is merged.
"""

from __future__ import annotations

import contextvars
import copy
import os
import shutil
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.harness.runners.native import LoadError, Session
from reef.harness.runners.native.graph import GraphError, Run, _Stop, _walk, narrow_allow
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.workspaces import MergeResult, TeamWorkspaceError

#: Characters of one member's final text that the caller's message carries.
TEAM_RESULT_CHARS = 4000
#: What each member's system prompt ends with, so a member knows who else is in its stage run.
TEAM_MEMBER_NOTE = (
    "You are {instance}, one {role} in a team of {roster}. {caller} started the team and reads your final answer."
)


@dataclass(frozen=True)
class MemberStart:
    """One member of a stage run: its instance name ``<role>.<k>``, the agent it runs, and what it is told."""

    instance: str
    role: str
    prompt: str
    #: Rules for this member alone, appended to its agent's prompt; empty when there are none.
    rules: str = ""


@dataclass(frozen=True)
class MemberResult:
    """How one member's turn ended: the outcome its graph named, its last text, and the steps it took."""

    instance: str
    role: str
    outcome: str
    text: str
    steps: int
    workdir: str


def team_outcome(outcomes: Sequence[str], *, is_budget_ended: bool) -> str:
    """The stage's outcome over its members' outcomes: ``budget``, then ``ask``, then ``gave_up``, then ``completed``.

    ``is_budget_ended`` is whether the episode's token budget was spent or its stop flag set."""
    if is_budget_ended or "budget" in outcomes:
        return "budget"
    for outcome in ("ask", "gave_up"):
        if outcome in outcomes:
            return outcome
    return "completed"


class TeamStageRun:
    """One parallel or team stage run: start the members together, wait for every one, report to the caller."""

    def __init__(
        self, caller: Run, stage_name: str, mode: str, workspace: str, members: Sequence[MemberStart]
    ) -> None:
        self.caller = caller
        self.stage_name = stage_name
        self.mode = mode
        self.workspace = workspace
        self.members = tuple(members)

    def run(self) -> tuple[str, dict[str, object]]:
        """The stage's outcome and its ``stage/exit`` detail; the caller's own step counter does not move."""
        caller, loop = self.caller, self.caller.loop
        stage_run = loop.next_team_stage_run()
        caller.session.write(
            "team/start",
            {
                "step": caller.step,
                "stage": self.stage_name,
                "mode": self.mode,
                "workspace": self.workspace,
                "stage_run": stage_run,
                "members": [{"agent": start.instance, "role": start.role} for start in self.members],
            },
        )
        results: list[MemberResult] = []
        merges: list[MergeResult] = []
        failures: list[str] = []
        try:
            workdirs = self.member_workdirs(stage_run)
        except TeamWorkspaceError as exc:
            failures.append(f"the team did not start: {exc}")
        else:
            # Turns are numbered in member order here, before any thread starts, so the file names do not depend on
            # which thread runs first.
            turns = [loop.open_turn(start.instance) for start in self.members]
            results = self.run_members(turns, workdirs)
            if self.workspace == "own":
                try:
                    merges = loop.workspaces.merge(stage_run, [result.instance for result in results])
                except TeamWorkspaceError as exc:
                    failures.append(f"the merge did not finish: {exc}")
        for merge in merges:
            caller.session.write(
                "team/merge",
                {
                    "step": caller.step,
                    "stage": self.stage_name,
                    "agent": merge.agent,
                    "branch": merge.branch,
                    "result": merge.result,
                    "files": list(merge.files),
                },
            )
        control = loop.control
        outcome = team_outcome(
            [*(result.outcome for result in results), *("gave_up" for _ in failures)],
            is_budget_ended=control.budget.is_spent or control.stop.is_set,
        )
        caller.session.write(
            "team/end",
            {
                "step": caller.step,
                "stage": self.stage_name,
                "outcome": outcome,
                "members": [
                    {"agent": result.instance, "role": result.role, "outcome": result.outcome, "steps": result.steps}
                    for result in results
                ],
            },
        )
        lines = []
        for result in results:
            text = result.text if len(result.text) <= TEAM_RESULT_CHARS else result.text[:TEAM_RESULT_CHARS] + " ..."
            ended = f"{result.instance} ({result.role}) ended with {result.outcome}"
            lines.append(f"{ended}: {text}" if text else ended)
        for merge in merges:
            if merge.result == "conflict":
                kept = loop.workspaces.member_path(stage_run, merge.agent)
                lines.append(
                    f"{merge.agent}: not merged, it conflicts in {', '.join(merge.files)}; its changes stay in {kept} "
                    f"on branch {merge.branch}"
                )
            else:
                lines.append(f"{merge.agent}: {'merged' if merge.result == 'merged' else 'changed no file'}")
        lines.extend(failures)
        caller.say(
            "\n\n".join(lines), {"kind": "team", "stage": self.stage_name, "mode": self.mode, "outcome": outcome}
        )
        detail: dict[str, object] = {
            "mode": self.mode,
            "agents": [result.instance for result in results],
            "outcomes": {result.instance: result.outcome for result in results},
            "steps": sum(result.steps for result in results),
        }
        if self.workspace == "own":
            detail["merges"] = {merge.agent: merge.result for merge in merges}
        return outcome, detail

    def member_workdirs(self, stage_run: int) -> list[Path]:
        """Each member's workdir: for ``own`` a new worktree of the caller's workdir as it stands, else that one."""
        workdir = self.caller.workdir
        if self.workspace != "own":
            return [workdir for _ in self.members]
        workspaces = self.caller.loop.workspaces
        if not workspaces.is_git_available():
            raise TeamWorkspaceError("git is not installed where the tools run")
        base = workspaces.start(stage_run)
        return [Path(str(workspaces.add_member(stage_run, start.instance, base))) for start in self.members]

    def run_members(self, turns: Sequence[tuple[Session, int]], workdirs: Sequence[Path]) -> list[MemberResult]:
        """Every member on its own thread, joined in member order; an error that is not a turn's end is raised here."""
        results: dict[str, MemberResult] = {}
        failures: list[BaseException] = []

        def member_body(start: MemberStart, session: Session, turn: int, workdir: Path) -> None:
            try:
                results[start.instance] = self.run_member(start, session, turn, workdir)
            except BaseException as exc:  # a bug, not a turn's end: raised in the caller once every member ended
                failures.append(exc)

        threads = [
            # A copy of this thread's context each: a context variable set by whoever runs the episode stays set.
            threading.Thread(
                target=contextvars.copy_context().run,
                args=(member_body, start, session, turn, workdir),
                name=f"reef-native-{start.instance}",
            )
            for start, (session, turn), workdir in zip(self.members, turns, workdirs, strict=True)
        ]
        for thread in threads:
            thread.start()
        # No timeout: the step budgets, the token budget and the stop flag end every member, and the executor's wall
        # clock bounds a tool that hangs.
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]
        return [results[start.instance] for start in self.members]

    def run_member(self, start: MemberStart, session: Session, turn: int, workdir: Path) -> MemberResult:
        """One member's turn on its own host; a tree that cannot load or a model error ends it ``gave_up``."""
        caller, loop = self.caller, self.caller.loop
        header = {
            **loop.header,
            "task": start.prompt,
            "agent": start.instance,
            "role": start.role,
            "turn": turn,
            "parent": caller.agent,
            "stage": self.stage_name,
            "mode": self.mode,
            "workspace": self.workspace,
            "workdir": str(workdir),
        }
        # Threads share the process id, so each member mounts under a directory named for it as well.
        mount_path = loop.session_dir / "mounts" / f"boot-{os.getpid()}-{start.instance}"
        shutil.rmtree(mount_path, ignore_errors=True)
        host: NativeHost | None = None
        try:
            try:
                host = NativeHost.from_root(loop.root, mount_path)
                agent = host.agents[start.role]
                # A copy: the budget below is this member's, and the host's graph outlives the turn.
                graph = copy.copy(host.graph(str(agent.get("graph", "seed"))))
            except (LoadError, GraphError, ValueError) as exc:
                session.write("session", {**header, "tools": [], "hooks": {}, "graph": None})
                session.write("turn/start", {"turn": turn, "parent": caller.agent})
                loop._abort(session, {"code": "LOAD_ERROR", "message": str(exc)[:600]}, turn=turn)
                return MemberResult(start.instance, start.role, "gave_up", "", 0, str(workdir))
            graph.max_steps = int(agent.get("max_steps") or graph.max_steps)
            roster = ", ".join(member.instance for member in self.members)
            note = TEAM_MEMBER_NOTE.format(
                instance=start.instance, role=start.role, roster=roster, caller=caller.agent
            )
            child = Run(
                loop,
                start.prompt,
                caller.binding,
                host,
                workdir,
                allow=narrow_allow(caller.allow, agent.get("tools")),
                parent=caller,
                agent=start.instance,
                turn=turn,
                session=session,
                skills=agent.get("skills"),
                agent_prompt="\n\n".join(part for part in (str(agent.get("prompt", "")), start.rules, note) if part),
                max_tool_calls=agent.get("max_tool_calls"),
            )
            tools = child.tools
            session.write(
                "session",
                {
                    **header,
                    "tools": sorted(tools),
                    "capabilities": {name: list(tools[name].capabilities) for name in sorted(tools)},
                    "hooks": {hook.name: event for event, listeners in host.hooks.items() for hook in listeners},
                    "graph": graph.source,
                    "max_steps": graph.max_steps,
                },
            )
            session.write("turn/start", {"turn": turn, "parent": caller.agent})
            try:
                outcome, text = _walk(child, graph)
            except _Stop:
                # A model error or a graph past its transition bound: the member's file holds the error end, and the
                # team goes on without it.
                outcome, text = "gave_up", ""
            return MemberResult(start.instance, start.role, outcome, text, child.step, str(workdir))
        finally:
            session.close()
            if host is not None:
                host.dispose()
            # A boot that failed may leave its mount directory behind; a member's never outlives its turn.
            shutil.rmtree(mount_path, ignore_errors=True)
