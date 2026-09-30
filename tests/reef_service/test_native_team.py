"""The native episode's controls and its team stages.

One token budget per episode, spent by every agent turn and set only by the caller; a stop flag every run reads
before its next step; and the members of a team stage, each on its own thread, host, session file and step budget.
The core tests run ``TeamStageRun`` on a root turn put together as ``run_loop`` puts it together; the mode tests run
a graph that names the stage."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from reef_service.test_harness_recipe import MODEL, batch, make_binary
from reef_service.test_native_harness import (
    CHECKER,
    _call,
    _delegating_graph,
    _FakeModel,
    _launcher,
    _reply,
    _seed_nodes,
    _SummarizingModel,
)

from reef.harness.adapters import get_adapter
from reef.harness.episodes.executor import EPISODE_TOKENS_ENV
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import EpisodeResult, run_episode
from reef.harness.episodes.trajectory import reader_for
from reef.harness.runners.native import TEAM_DIR, Session, ToolModule, ToolRunner, _Loop, run_loop
from reef.harness.runners.native.control import EpisodeControl, EpisodeStop, TeamBudget, episode_token_limit
from reef.harness.runners.native.enforce import InProcessEnforcer, ToolFailed
from reef.harness.runners.native.graph import Run, _tokens, run_graph
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.inbox import (
    TEAM_MAX_SENDS_PER_MEMBER,
    TEAM_MESSAGE_MAX_CHARS,
    Assignment,
    Inbox,
    TeamMember,
)
from reef.harness.runners.native.seed import SEED_TOOLS
from reef.harness.runners.native.team import (
    MemberStart,
    TeamAssignRunner,
    TeamStageRun,
    TeamWaitRunner,
    run_team_stage,
    team_outcome,
)
from reef.harness.runners.native.workspaces import CommandOutcome, HostCommandRunner, TeamWorkspaces
from reef.harness.tree.nodes import NODE_KINDS
from reef.harness.tree.render import RenderError, render_composition
from reef.train.cordis_backend import CordisBackend, Mutation
from reef.train.cordis_backend.backend import (
    EpisodeEvaluationWorker,
    _agent_work,
    _stage_path,
    sum_message_counts,
    team_message_counts,
    tree_files,
)
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

USAGE = {"prompt_tokens": 200, "completion_tokens": 100}
READ = _reply(tool_calls=[_call("read_file", {"path": "missing.txt"}, "c1")])
COMPACT_GRAPH = {
    "name": "main",
    "start": "think",
    "max_steps": 6,
    "stages": {
        "think": {"kind": "model"},
        "act": {"kind": "tools"},
        "squeeze": {"kind": "compact", "fire_ratio": 0.5, "keep_ratio": 0.2},
        "done": {"kind": "end", "reason": "completed"},
    },
    "edges": [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "done"},
        {"from": "act", "when": "done", "to": "squeeze"},
        {"from": "squeeze", "when": "done", "to": "think"},
    ],
}


class ReadingModel(_FakeModel):
    """Answers every call with a read of a missing file, so only a budget ends the turn; ``usage`` is reported when
    given, and the root of a delegating graph gets ``root_text`` instead of the read."""

    def __init__(self, usage: dict | None = None, root_text: str | None = None) -> None:
        super().__init__()
        self.usage = usage
        self.root_text = root_text

    def script(self, body: dict) -> dict:
        is_root = "You are the checker" not in body["messages"][0]["content"]
        reply = _reply(content=self.root_text) if is_root and self.root_text else READ
        return reply if self.usage is None else {**reply, "usage": self.usage}


class CountedSummarizingModel(_SummarizingModel):
    """The compaction script, every call reporting 10 input and 5 output tokens."""

    def script(self, body: dict) -> dict:
        return {**super().script(body), "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def stop(model: _FakeModel) -> None:
    model.shutdown()
    model.server_close()


def run_turn(tmp_path: Path, model: _FakeModel, nodes, **options) -> tuple[int, Path]:
    """One in process root turn over ``nodes`` and the model's binding; the exit status and the sessions directory."""
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
    for relative, text in render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor).items():
        path = tmp_path / "tree" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sessions, work = tmp_path / "sessions", tmp_path / "work"
    work.mkdir()
    return run_loop("find the notes", tmp_path / "tree" / "native", sessions, work, **options), sessions


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_spent_budget_ends_the_turn_at_the_next_step_before_its_call(tmp_path: Path) -> None:
    model = ReadingModel(USAGE)
    control = EpisodeControl(TeamBudget(1000))
    try:
        code, sessions = run_turn(tmp_path, model, _seed_nodes(SEED_TOOLS), control=control)
    finally:
        stop(model)
    # Three calls spend 900 of 1000, so a fourth goes out; the fifth step ends the turn before its call.
    assert code == 0 and len(model.requests) == 4
    reason = events(sessions / "session.jsonl")[-1]["data"]["reason"]
    assert reason == {"kind": "max-tokens", "tokens": 1000, "spent": 1200}
    assert (control.budget.input_tokens, control.budget.output_tokens, control.budget.is_spent) == (800, 400, True)


def test_an_agent_that_spends_the_budget_ends_on_it_and_its_stage_routes_on_budget(tmp_path: Path) -> None:
    model = ReadingModel(USAGE, root_text="the count is 9592")
    nodes = [*_seed_nodes(SEED_TOOLS), ("native_graph", _delegating_graph()), CHECKER]
    try:
        code, sessions = run_turn(tmp_path, model, nodes, control=EpisodeControl(TeamBudget(500)))
    finally:
        stop(model)
    # The root's call and the checker's first call spend 600; the checker's second step never asks.
    assert code == 0 and len(model.requests) == 2
    (checker_path,) = sorted((sessions / "agents").glob("*.jsonl"))
    assert events(checker_path)[-1]["data"]["reason"] == {"kind": "max-tokens", "tokens": 500, "spent": 600}
    root = events(sessions / "session.jsonl")
    exits = [e["data"] for e in root if e["type"] == "stage/exit" and e["data"]["stage"] == "delegate"]
    assert [(stage_exit["outcome"], stage_exit["to"]) for stage_exit in exits] == [("budget", "quit")]
    assert root[-1]["data"]["reason"] == {"kind": "gave_up"}


def test_a_reply_without_usage_is_charged_an_estimate_the_agent_counters_do_not_count(tmp_path: Path) -> None:
    model = ReadingModel()
    control = EpisodeControl(TeamBudget(300))
    try:
        code, sessions = run_turn(tmp_path, model, _seed_nodes(SEED_TOOLS), control=control)
    finally:
        stop(model)
    # Each call is charged what it sent and what came back, at four characters per token.
    estimate = sum(_tokens(body["messages"]) + _tokens([READ["choices"][0]["message"]]) for body in model.requests)
    assert code == 0 and control.budget.spent_tokens == estimate >= 300
    reason = events(sessions / "session.jsonl")[-1]["data"]["reason"]
    assert reason == {"kind": "max-tokens", "tokens": 300, "spent": estimate}
    assert len(model.requests) < 12  # the budget ended the turn, not the seed graph's step budget
    work = _agent_work(reader_for("native-jsonl")(sessions))
    assert (work["root"]["input_tokens"], work["root"]["output_tokens"]) == (0, 0)


def test_a_summary_call_is_charged_and_a_budget_without_a_limit_never_ends_a_turn(tmp_path: Path) -> None:
    model = CountedSummarizingModel()
    window = ("config", {"target": "models", "data": {"context_window": 120}})
    control = EpisodeControl(TeamBudget(None))
    try:
        code, sessions = run_turn(
            tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", COMPACT_GRAPH), window], control=control
        )
    finally:
        stop(model)
    assert code == 0 and events(sessions / "session.jsonl")[-1]["data"]["reason"] == {"kind": "completed"}
    assert any(body["messages"][0]["content"].startswith("Summarize the work so far") for body in model.requests)
    assert control.budget.spent_tokens == 15 * len(model.requests) and not control.budget.is_spent


def test_the_caller_chooses_the_enforcer_before_the_tree_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REEF_NATIVE_ENFORCE", "seccomp")  # a mode no enforcer serves: a load error when selected
    model = _FakeModel()
    try:
        code, sessions = run_turn(tmp_path, model, _seed_nodes(SEED_TOOLS), enforcer=InProcessEnforcer())
    finally:
        stop(model)
    root = events(sessions / "session.jsonl")
    assert code == 0 and root[0]["data"]["enforcement"] == "none"
    assert root[-1]["data"]["reason"] == {"kind": "completed"}


def test_the_budget_reaches_the_loop_only_through_the_episode_environment(tmp_path: Path) -> None:
    assert episode_token_limit({}) is None and episode_token_limit({EPISODE_TOKENS_ENV: "5000"}) == 5000
    for bad in ("0", "-1", "5k", "", "1.5"):
        with pytest.raises(ValueError, match=f"{EPISODE_TOKENS_ENV}=.* must be a positive integer"):
            episode_token_limit({EPISODE_TOKENS_ENV: bad})
    model = _FakeModel()
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
    # A tree can put the key in models.json; the loop never reads a budget there.
    nodes = [*_seed_nodes(SEED_TOOLS), ("config", {"target": "models", "data": {"episode_tokens": 1}})]
    files = render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor)
    task = "put hello in notes.txt and read it back"
    try:
        assert json.loads(files["native/models.json"])["episode_tokens"] == 1
        free = run_episode(descriptor, files, task, binary=_launcher(tmp_path))
        assert free.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
        limited = run_episode(descriptor, files, task, binary=_launcher(tmp_path), env={EPISODE_TOKENS_ENV: "1"})
        reason = limited.trajectory[-1]["data"]["reason"]
        assert (reason["kind"], reason["tokens"]) == ("max-tokens", 1) and limited.exit_code == 0
        refused = run_episode(descriptor, files, task, binary=_launcher(tmp_path), env={EPISODE_TOKENS_ENV: "many"})
        assert refused.exit_code == 2 and EPISODE_TOKENS_ENV in refused.stderr
    finally:
        stop(model)


# -- team stages: members on their own threads ----------------------------------------------------------------------

PEER = ("native_agent", {"name": "peer", "prompt": "You are a peer. Do your part, then answer in one line."})
CRITIC = ("native_agent", {"name": "critic", "prompt": "You are the critic. Check the work, then answer in one line."})
TEAM_NODES = [*_seed_nodes(SEED_TOOLS), PEER, CRITIC]


def member_of(body: dict) -> str:
    """The instance a request comes from, by the note its system prompt ends with; ``root`` for the caller."""
    found = re.search(r"You are (\S+), one ", body["messages"][0]["content"])
    return found.group(1) if found else "root"


class MemberModel(_FakeModel):
    """Answers each member by its instance name; with ``gather``, that many members (the root is none of them) meet
    at a barrier on their first call, so no reply goes out until all of them are in flight at once. At the barrier it
    sets ``stop`` when given and lists ``mounts_path`` when given."""

    def __init__(self, gather: int = 0, stop: EpisodeStop | None = None, mounts_path: Path | None = None) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.met: set[str] = set()
        self.is_barrier_broken = False
        self.mounts_seen: list[str] = []

        def at_barrier() -> None:
            if stop is not None:
                stop.set("cancelled")
            if mounts_path is not None:
                self.mounts_seen = sorted(path.name for path in mounts_path.iterdir())

        self.barrier = threading.Barrier(gather, action=at_barrier, timeout=10) if gather else None

    def script(self, body: dict) -> dict:
        instance = member_of(body)
        with self.lock:
            is_first = instance not in self.met
            self.met.add(instance)
        if is_first and self.barrier is not None and instance != "root":
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                self.is_barrier_broken = True
        return self.reply(instance, body)

    def reply(self, instance: str, body: dict) -> dict:
        return _reply(content=f"done by {instance}")


class ReadingMemberModel(MemberModel):
    def reply(self, instance: str, body: dict) -> dict:
        return READ


class FailingPeerModel(MemberModel):
    """peer.2's every call is refused with a 400; everyone else answers."""

    def status(self, body: dict) -> int:
        return 400 if member_of(body) == "peer.2" else 200


class TeamTurn:
    """A root turn put together as ``run_loop`` puts it together, so a team stage can run on it directly."""

    def __init__(self, tmp_path: Path, model: _FakeModel, nodes, *, control=None, is_tree: bool = False) -> None:
        descriptor = get_adapter("native")
        binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
        files = render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor)
        if is_tree:
            entries = [
                {"id": f"entry-{index}", "name": kind, "config": config} for index, (kind, config) in enumerate(nodes)
            ]
            files.update(tree_files(descriptor, entries))
        for relative, text in files.items():
            path = tmp_path / "tree" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.root, self.sessions, self.work = tmp_path / "tree" / "native", tmp_path / "sessions", tmp_path / "work"
        self.work.mkdir()
        self.control = control or EpisodeControl()
        session = Session(self.sessions / "session.jsonl")
        header = {"version": 1, "task": "split the work", "model": "fake", "cwd": str(self.work)}
        self.host = NativeHost.from_root(self.root, self.sessions / "mounts" / f"boot-{os.getpid()}")
        session.write("session", {**header, "agent": "root", "turn": 1, "agents": sorted(self.host.agents)})
        session.write("turn/start", {"turn": 1})
        main_path = PurePosixPath(self.work)
        self.workspaces = TeamWorkspaces(
            self.control.command_runner, main_path=main_path, team_path=main_path / TEAM_DIR
        )
        self.loop = _Loop(
            session,
            self.root,
            self.sessions,
            header,
            enforcer=InProcessEnforcer(),
            control=self.control,
            workspaces=self.workspaces,
        )
        self.run = Run(self.loop, "split the work", binding, self.host, self.work)

    def stage(self, members, *, mode: str = "team", workspace: str = "shared"):
        return TeamStageRun(self.run, "crew", mode, workspace, members).run()

    def close(self) -> list[dict]:
        """The episode's trajectory once the root's session, host and team workspaces are closed."""
        self.host.dispose()
        self.workspaces.close()
        self.loop.session.close()
        return list(reader_for("native-jsonl")(self.sessions))

    def finish(self) -> list[dict]:
        self.run.end_turn_quietly({"kind": "completed"})
        return self.close()


def peers(count: int, prompt: str = "count the primes below 100") -> list[MemberStart]:
    return [MemberStart(f"peer.{index}", "peer", prompt) for index in range(1, count + 1)]


def member_files(sessions: Path) -> dict[str, list[dict]]:
    """Each agent file's events, by the instance its header names."""
    files = {}
    for path in sorted((sessions / "agents").glob("*.jsonl")):
        found = events(path)
        files[found[0]["data"]["agent"]] = found
    return files


def test_members_run_at_once_each_in_its_own_session_file_and_the_caller_reads_every_result(tmp_path: Path) -> None:
    model = MemberModel(gather=2)
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        members = [
            MemberStart("peer.1", "peer", "count the primes"),
            MemberStart("critic.1", "critic", "check it", "Be strict."),
        ]
        outcome, detail = turn.stage(members)
        trajectory = turn.finish()
    finally:
        stop(model)
    # Both first calls were in flight together: the barrier released them only once both had arrived.
    assert not model.is_barrier_broken and outcome == "completed"
    assert detail == {
        "mode": "team",
        "agents": ["peer.1", "critic.1"],
        "outcomes": {"peer.1": "completed", "critic.1": "completed"},
        "steps": 2,
    }
    assert sorted(path.name for path in (turn.sessions / "agents").glob("*.jsonl")) == [
        "002-peer.1.jsonl",
        "003-critic.1.jsonl",
    ]
    files = member_files(turn.sessions)
    peer_header, critic_header = files["peer.1"][0]["data"], files["critic.1"][0]["data"]
    assert (peer_header["role"], peer_header["turn"], peer_header["parent"]) == ("peer", 2, "root")
    assert (critic_header["stage"], critic_header["mode"], critic_header["workspace"]) == ("crew", "team", "shared")
    assert critic_header["workdir"] == str(turn.work) and critic_header["task"] == "check it"
    critic_start = next(e for e in files["critic.1"] if e["type"] == "step/start")
    peer_end = files["peer.1"][-1]
    assert peer_end["type"] == "turn/end" and critic_start["time"] <= peer_end["time"]
    # Each member's system prompt ends with its agent's prompt, its own rules and who it is.
    critic_system = next(r for r in model.requests if member_of(r) == "critic.1")["messages"][0]["content"]
    assert critic_system.endswith(
        "Be strict.\n\nYou are critic.1, one critic in a team of peer.1, critic.1. "
        "root started the team and reads your final answer. team_send sends a message to a member, a role, all, "
        "or root; it arrives at the receiver's next step. team_wait waits for one."
    )
    peer_system = next(r for r in model.requests if member_of(r) == "peer.1")["messages"][0]["content"]
    assert "Be strict." not in peer_system
    root = events(turn.sessions / "session.jsonl")
    kinds = [e["type"] for e in root]
    assert kinds[2:] == ["team/start", "team/end", "user/message", "turn/end"]
    assert root[2]["data"] == {
        "step": 0,
        "stage": "crew",
        "mode": "team",
        "workspace": "shared",
        "stage_run": 1,
        "members": [{"agent": "peer.1", "role": "peer"}, {"agent": "critic.1", "role": "critic"}],
    }
    assert root[3]["data"]["members"] == [
        {"agent": "peer.1", "role": "peer", "outcome": "completed", "steps": 1},
        {"agent": "critic.1", "role": "critic", "outcome": "completed", "steps": 1},
    ]
    assert root[4]["data"] == {
        "step": 0,
        "source": {"kind": "team", "stage": "crew", "mode": "team", "outcome": "completed"},
        "content": (
            "peer.1 (peer) ended with completed: done by peer.1\n\n"
            "critic.1 (critic) ended with completed: done by critic.1"
        ),
    }
    # The caller's own step counter did not move, and the per agent counters count each member apart.
    assert turn.run.step == 0
    assert {agent: work["steps"] for agent, work in _agent_work(trajectory).items()} == {
        "peer.1": 1,
        "critic.1": 1,
        "root": 0,
    }


def test_each_member_boots_its_own_host_and_mount_directory_and_none_outlives_the_stage(tmp_path: Path) -> None:
    mounts = tmp_path / "sessions" / "mounts"
    model = MemberModel(gather=2, mounts_path=mounts)
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES, is_tree=True)
        outcome, _ = turn.stage(peers(2))
        after = sorted(path.name for path in mounts.iterdir())
        turn.finish()
    finally:
        stop(model)
    pid = os.getpid()
    assert outcome == "completed" and not model.is_barrier_broken
    assert model.mounts_seen == [f"boot-{pid}", f"boot-{pid}-peer.1", f"boot-{pid}-peer.2"]
    assert after == [f"boot-{pid}"]


def test_hook_module_state_is_per_member(tmp_path: Path) -> None:
    counter = (
        "native_hook",
        {
            "name": "count_steps",
            "event": "pre_step",
            "code": (
                "CALLS = [0]\n\n\ndef listen(payload, next):\n    CALLS[0] += 1\n    decision = next()\n"
                "    return {**decision, 'messages': [f'hook call {CALLS[0]}']}\n"
            ),
        },
    )
    model = MemberModel()
    try:
        turn = TeamTurn(tmp_path, model, [*TEAM_NODES, counter])
        turn.stage(peers(2))
        turn.finish()
    finally:
        stop(model)
    for instance, found in member_files(turn.sessions).items():
        said = [e["data"]["content"] for e in found if e["type"] == "user/message"]
        assert said == ["hook call 1"], instance


def test_a_member_runs_on_its_agents_step_budget_not_the_callers_remaining_steps(tmp_path: Path) -> None:
    model = ReadingMemberModel()
    nodes = [*_seed_nodes(SEED_TOOLS), ("native_agent", {**PEER[1], "max_steps": 3})]
    try:
        turn = TeamTurn(tmp_path, model, nodes)
        turn.run.max_steps, turn.run.step = 3, 1  # the caller has two steps left
        outcome, detail = turn.stage(peers(1))
        turn.finish()
    finally:
        stop(model)
    (found,) = member_files(turn.sessions).values()
    assert found[-1]["data"]["reason"] == {"kind": "max-steps", "steps": 3} and len(model.requests) == 3
    assert (outcome, detail["steps"], turn.run.step) == ("budget", 3, 1)


def test_a_member_model_error_ends_that_member_gave_up_and_not_the_episode(tmp_path: Path) -> None:
    model = FailingPeerModel()
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        outcome, detail = turn.stage(peers(2))
        trajectory = turn.finish()
    finally:
        stop(model)
    files = member_files(turn.sessions)
    assert files["peer.2"][-1]["data"]["reason"]["kind"] == "error"
    assert files["peer.1"][-1]["data"]["reason"] == {"kind": "completed"}
    assert outcome == "gave_up" and detail["outcomes"] == {"peer.1": "completed", "peer.2": "gave_up"}
    # The root still ended, so the member's error is not the episode's.
    path = _stage_path(trajectory)
    assert path["reason"] == "completed" and "error" not in path


def test_the_stop_flag_ends_every_member_at_its_next_step_and_no_call_goes_out_after_it(tmp_path: Path) -> None:
    control = EpisodeControl()
    model = ReadingMemberModel(gather=2, stop=control.stop)
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES, control=control)
        outcome, _ = turn.stage(peers(2))
        turn.finish()
    finally:
        stop(model)
    # The stop was set while both first calls were in flight: those two are the only calls.
    assert len(model.requests) == 2 and outcome == "budget"
    for found in member_files(turn.sessions).values():
        (result,) = [e["data"] for e in found if e["type"] == "tool/result"]
        assert result["error"]["code"] == "STOPPED" and result["is_error"]
        assert found[-1]["data"]["reason"] == {"kind": "stopped", "reason": "cancelled"}


class HaltRunner(ToolRunner):
    def __init__(self, stop_flag: EpisodeStop) -> None:
        self.stop_flag = stop_flag

    def __call__(self, args, workdir, /):
        self.stop_flag.set("halted")
        return "halted"


class HaltThenReadModel(_FakeModel):
    def script(self, body: dict) -> dict:
        return _reply(tool_calls=[_call("halt", {}, "c1"), _call("read_file", {"path": "x"}, "c2")])


def test_a_stop_inside_a_batch_keeps_the_rest_of_the_batch_from_running(tmp_path: Path) -> None:
    model = HaltThenReadModel()
    try:
        turn = TeamTurn(tmp_path, model, _seed_nodes(SEED_TOOLS))
        turn.host.add_tool(ToolModule("halt", "Stop the episode.", {}, HaltRunner(turn.control.stop)))
        code = run_graph(turn.run, turn.host.graph("main"))
        turn.close()
    finally:
        stop(model)
    root = events(turn.sessions / "session.jsonl")
    results = [e["data"] for e in root if e["type"] == "tool/result"]
    assert [(r["name"], r["is_error"]) for r in results] == [("halt", False), ("read_file", True)]
    assert results[1]["error"] == {"code": "STOPPED", "message": "the episode stopped before this call ran"}
    assert code == 0 and len(model.requests) == 1
    assert root[-1]["data"]["reason"] == {"kind": "stopped", "reason": "halted"}


def test_the_stage_outcome_puts_budget_before_ask_before_gave_up_before_completed() -> None:
    assert team_outcome(["budget", "completed"], is_budget_ended=False) == "budget"
    assert team_outcome(["completed", "completed"], is_budget_ended=True) == "budget"
    assert team_outcome(["ask", "gave_up"], is_budget_ended=False) == "ask"
    assert team_outcome(["gave_up", "completed"], is_budget_ended=False) == "gave_up"
    assert team_outcome(["completed", "completed"], is_budget_ended=False) == "completed"


# -- workspace: own, a git worktree per member ---------------------------------------------------------------------


class WritingModel(MemberModel):
    """Each member writes its planned files, one per step, then answers; the plan is keyed by instance."""

    def __init__(self, writes: dict[str, list[tuple[str, str]]]) -> None:
        super().__init__()
        self.writes = writes

    def reply(self, instance: str, body: dict) -> dict:
        done = sum(1 for message in body["messages"] if message.get("role") == "tool")
        planned = self.writes.get(instance, [])
        if done < len(planned):
            path, content = planned[done]
            return _reply(tool_calls=[_call("write_file", {"path": path, "content": content}, f"w{done}")])
        return _reply(content=f"{instance} wrote {len(planned)} files")


def git_status(turn: TeamTurn) -> str:
    return turn.workspaces.git("status", "--porcelain", cwd=turn.workspaces.main_path).stdout


def root_events(turn: TeamTurn, type_: str) -> list[dict]:
    return [e["data"] for e in events(turn.sessions / "session.jsonl") if e["type"] == type_]


def test_members_in_their_own_worktrees_merge_into_the_workdir_and_leave_no_git_there(tmp_path: Path) -> None:
    model = WritingModel(
        {
            "peer.1": [("a.txt", "from one\n"), (".reef/tool-output/x.txt", "scratch")],
            "peer.2": [("b.txt", "from two\n")],
        }
    )
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        outcome, detail = turn.stage(peers(2), workspace="own")
        team_path = turn.work / TEAM_DIR
        worktrees = sorted(path.name for path in team_path.iterdir())
        status, is_git_in_workdir = git_status(turn), (turn.work / ".git").exists()
        turn.finish()
    finally:
        stop(model)
    assert outcome == "completed" and detail["merges"] == {"peer.1": "merged", "peer.2": "merged"}
    assert (turn.work / "a.txt").read_text() == "from one\n" and (turn.work / "b.txt").read_text() == "from two\n"
    # Tool output under .reef/ never merges, and the merged worktrees are gone; only the git directory is left.
    assert not (turn.work / ".reef" / "tool-output").exists() and worktrees == ["git"] and status == ""
    assert [(m["agent"], m["branch"], m["result"], m["files"]) for m in root_events(turn, "team/merge")] == [
        ("peer.1", "reef/s1/peer.1", "merged", []),
        ("peer.2", "reef/s1/peer.2", "merged", []),
    ]
    headers = {instance: found[0]["data"] for instance, found in member_files(turn.sessions).items()}
    assert headers["peer.2"]["workdir"] == str(team_path / "s1-peer.2") and headers["peer.2"]["workspace"] == "own"
    (said,) = root_events(turn, "user/message")
    assert said["content"].endswith("peer.1: merged\n\npeer.2: merged")
    # The main worktree never gets a .git, and closing removes Reef's git directory with every worktree.
    assert not is_git_in_workdir and not (turn.work / ".git").exists() and not team_path.exists()


def test_a_conflict_is_named_and_leaves_the_workdir_clean_with_the_members_changes_kept(tmp_path: Path) -> None:
    model = WritingModel({"peer.1": [("same.txt", "one\n")], "peer.2": [("same.txt", "two\n")]})
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        (turn.work / "same.txt").write_text("base\n")
        outcome, detail = turn.stage(peers(2), workspace="own")
        kept = turn.work / TEAM_DIR / "s1-peer.2"
        kept_text, status = (kept / "same.txt").read_text(), git_status(turn)
        turn.finish()
    finally:
        stop(model)
    # A conflict is the caller's to resolve, not a failure of the team.
    assert outcome == "completed" and detail["merges"] == {"peer.1": "merged", "peer.2": "conflict"}
    assert (turn.work / "same.txt").read_text() == "one\n" and status == "" and kept_text == "two\n"
    assert root_events(turn, "team/merge")[1]["files"] == ["same.txt"]
    (said,) = root_events(turn, "user/message")
    assert said["content"].endswith(
        f"peer.2: not merged, it conflicts in same.txt; its changes stay in {kept} on branch reef/s1/peer.2"
    )
    assert not kept.exists()


def test_a_second_stage_run_branches_from_the_workdir_as_the_caller_left_it(tmp_path: Path) -> None:
    model = WritingModel({"peer.1": [("a.txt", "one\n")]})
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        turn.stage(peers(1), workspace="own")
        (turn.work / "a.txt").write_text("changed by the caller\n")
        outcome, detail = turn.stage(peers(1), workspace="own")
        turn.finish()
    finally:
        stop(model)
    # The second run starts from the caller's edit, so writing the first text again is a change, and it merges.
    assert (outcome, detail["merges"]) == ("completed", {"peer.1": "merged"})
    assert [m["branch"] for m in root_events(turn, "team/merge")] == ["reef/s1/peer.1", "reef/s2/peer.1"]
    assert [e["stage_run"] for e in root_events(turn, "team/start")] == [1, 2]
    headers = [events(path)[0]["data"] for path in sorted((turn.sessions / "agents").glob("*.jsonl"))]
    assert [Path(header["workdir"]).name for header in headers] == ["s1-peer.1", "s2-peer.1"]
    assert (turn.work / "a.txt").read_text() == "one\n"


def test_a_member_that_changes_nothing_merges_nothing_and_a_tasks_own_git_is_left_alone(tmp_path: Path) -> None:
    model = WritingModel({})
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        (turn.work / ".git").mkdir()
        (turn.work / ".git" / "HEAD").write_text("the task's own repository\n")
        outcome, detail = turn.stage(peers(1), workspace="own")
        turn.finish()
    finally:
        stop(model)
    assert (outcome, detail["merges"]) == ("completed", {"peer.1": "empty"})
    assert (turn.work / ".git" / "HEAD").read_text() == "the task's own repository\n"
    (said,) = root_events(turn, "user/message")
    assert said["content"].endswith("peer.1: changed no file")


def test_shared_members_write_into_the_callers_workdir_and_no_git_runs(tmp_path: Path) -> None:
    model = WritingModel({"peer.1": [("a.txt", "one\n")], "peer.2": [("b.txt", "two\n")]})
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        outcome, detail = turn.stage(peers(2), workspace="shared")
        turn.finish()
    finally:
        stop(model)
    assert outcome == "completed" and "merges" not in detail and not root_events(turn, "team/merge")
    assert sorted(path.name for path in turn.work.iterdir()) == ["a.txt", "b.txt"]


class NoGitRunner(HostCommandRunner):
    def run(self, argv, *, cwd, timeout_seconds):
        if "--version" in argv:
            return CommandOutcome(127, "", "git: command not found")
        return super().run(argv, cwd=cwd, timeout_seconds=timeout_seconds)


def test_an_own_stage_without_git_ends_gave_up_before_any_member_starts(tmp_path: Path) -> None:
    model = WritingModel({"peer.1": [("a.txt", "one\n")]})
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES, control=EpisodeControl(command_runner=NoGitRunner()))
        outcome, detail = turn.stage(peers(2), workspace="own")
        turn.finish()
    finally:
        stop(model)
    assert (outcome, detail["agents"], detail["merges"]) == ("gave_up", [], {}) and not model.requests
    assert not (turn.sessions / "agents").exists()
    (said,) = root_events(turn, "user/message")
    assert said["content"] == "the team did not start: git is not installed where the tools run"


def test_the_host_command_runner_reports_a_missing_command_and_a_timeout_as_outcomes(tmp_path: Path) -> None:
    runner = HostCommandRunner()
    assert runner.run(["reef-no-such-command"], cwd=str(tmp_path), timeout_seconds=5).return_code == 127
    slow = runner.run(["sleep", "5"], cwd=str(tmp_path), timeout_seconds=0.2)
    assert slow.return_code == 124 and "did not finish in 0.2 s" in slow.stderr
    done = runner.run(["sh", "-c", "echo hi; exit 3"], cwd=str(tmp_path), timeout_seconds=5)
    assert done == CommandOutcome(3, "hi\n", "")


# -- team_send and team_wait -------------------------------------------------------------------------------------


def message_events(found: list[dict]) -> list[dict]:
    return [e for e in found if e["type"] == "user/message" and e["data"]["source"]["kind"] == "message"]


class ChattingModel(MemberModel):
    """Each member runs its plan of (tool, arguments) calls, one per step, then answers; a listener waits after its
    plan until a message has reached it, and answers then."""

    def __init__(self, plans: dict[str, list[tuple[str, dict]]], listeners: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.plans = plans
        self.listeners = listeners

    def reply(self, instance: str, body: dict) -> dict:
        heard = [m["content"] for m in body["messages"] if m["role"] == "user" and m["content"].startswith("Message")]
        done = sum(1 for message in body["messages"] if message.get("role") == "tool")
        plan = self.plans.get(instance, [])
        if done < len(plan):
            name, arguments = plan[done]
            return _reply(tool_calls=[_call(name, arguments, f"t{done}")])
        if instance in self.listeners and not heard:
            return _reply(tool_calls=[_call("team_wait", {"seconds": 30}, f"t{done}")])
        return _reply(content=f"{instance} heard: {heard[-1]}" if heard else f"{instance} is done")


def test_one_message_each_way_is_a_send_in_one_file_and_a_message_at_the_next_step_in_the_other(
    tmp_path: Path,
) -> None:
    plans = {
        "peer.1": [("team_send", {"to": "peer.2", "text": "hello from one"})],
        "peer.2": [("team_send", {"to": "peer.1", "text": "hello from two"})],
    }
    model = ChattingModel(plans, listeners=("peer.1", "peer.2"))
    listed = ("native_agent", {**PEER[1], "tools": ["read_file"]})
    try:
        turn = TeamTurn(tmp_path, model, [*_seed_nodes(SEED_TOOLS), listed])
        outcome, _ = turn.stage(peers(2))
        trajectory = turn.finish()
    finally:
        stop(model)
    assert outcome == "completed"
    files = member_files(turn.sessions)
    for sender, receiver in (("peer.1", "peer.2"), ("peer.2", "peer.1")):
        # The agent's tools list does not hide the team tools.
        assert files[sender][0]["data"]["tools"] == ["read_file", "team_send", "team_wait"]
        (sent,) = [e["data"] for e in files[sender] if e["type"] == "team/send"]
        assert (sent["to"], sent["delivered"], sent["undelivered"]) == (receiver, [receiver], [])
        (heard,) = message_events(files[receiver])
        assert heard["data"]["source"] == {"kind": "message", "from": sender, "message_id": f"{sender}-1"}
        assert heard["data"]["content"] == f"Message from {sender}: {sent['text']}"
        # Delivered at the top of a step: the next event is that step's start.
        following = files[receiver][files[receiver].index(heard) + 1]
        assert following["type"] == "step/start" and following["data"]["step"] == heard["data"]["step"]
    assert team_message_counts(trajectory) == {
        "peer.1": {"sent": 1, "received": 1, "undelivered": 0},
        "peer.2": {"sent": 1, "received": 1, "undelivered": 0},
    }
    # The message events move no per agent counter.
    plain = [e for e in trajectory if not e["type"].startswith("team/") and e not in message_events(trajectory)]
    assert _agent_work(plain) == _agent_work(trajectory)


def test_a_broadcast_reaches_every_other_open_member_and_not_the_sender(tmp_path: Path) -> None:
    plans = {"peer.1": [("team_send", {"to": "all", "text": "split: I take the parser"})]}
    model = ChattingModel(plans, listeners=("peer.2", "peer.3"))
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        outcome, _ = turn.stage(peers(3))
        turn.finish()
    finally:
        stop(model)
    files = member_files(turn.sessions)
    (sent,) = [e["data"] for e in files["peer.1"] if e["type"] == "team/send"]
    assert sent["delivered"] == ["peer.2", "peer.3"] and outcome == "completed"
    assert not message_events(files["peer.1"])
    assert [len(message_events(files[instance])) for instance in ("peer.2", "peer.3")] == [1, 1]


def test_a_message_to_a_member_that_has_ended_is_reported_undelivered(tmp_path: Path) -> None:
    # peer.1 answers at once; peer.2 waits until it is alone, then writes to peer.1.
    plans = {"peer.2": [("team_wait", {"seconds": 30}), ("team_send", {"to": "peer.1", "text": "are you there?"})]}
    model = ChattingModel(plans)
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        turn.stage(peers(2))
        trajectory = turn.finish()
    finally:
        stop(model)
    files = member_files(turn.sessions)
    results = [e["data"]["content"] for e in files["peer.2"] if e["type"] == "tool/result"]
    assert results == ["every other member has ended; no message will come", "peer.1 has ended; not delivered"]
    (sent,) = [e["data"] for e in files["peer.2"] if e["type"] == "team/send"]
    assert (sent["delivered"], sent["undelivered"]) == ([], ["peer.1"])
    assert team_message_counts(trajectory) == {"peer.2": {"sent": 1, "received": 0, "undelivered": 1}}


class LateNoteModel(MemberModel):
    """peer.1 writes to the caller and then to peer.2 while peer.2's only call is held open, so peer.2 ends with
    the note unread."""

    def __init__(self) -> None:
        super().__init__()
        self.is_sent = threading.Event()

    def reply(self, instance: str, body: dict) -> dict:
        done = sum(1 for message in body["messages"] if message.get("role") == "tool")
        if instance == "peer.2":
            self.is_sent.wait(10)
            return _reply(content="peer.2 is done")
        if done == 0:
            return _reply(tool_calls=[_call("team_send", {"to": "root", "text": "the parser is done"}, "t0")])
        if done == 1:
            return _reply(tool_calls=[_call("team_send", {"to": "peer", "text": "late note"}, "t1")])
        self.is_sent.set()
        return _reply(content="peer.1 is done")


def test_a_message_to_the_caller_joins_the_stage_result_and_an_unread_one_is_named(tmp_path: Path) -> None:
    model = LateNoteModel()
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        turn.stage(peers(2))
        turn.finish()
    finally:
        stop(model)
    (said,) = root_events(turn, "user/message")
    assert said["content"].endswith(
        "peer.2 (peer) ended with completed: peer.2 is done\n\nMessage from peer.1: the parser is done"
    )
    files = member_files(turn.sessions)
    sent = [e["data"] for e in files["peer.1"] if e["type"] == "team/send"]
    assert [(e["to"], e["delivered"]) for e in sent] == [("root", ["root"]), ("peer", ["peer.2"])]
    assert files["peer.2"][-1]["type"] == "team/unread"
    assert files["peer.2"][-1]["data"] == {"messages": [{"message_id": "peer.1-2", "from": "peer.1"}]}


def test_message_text_is_redacted_in_both_files_and_an_overlong_message_is_a_tool_error(tmp_path: Path) -> None:
    secret = "sk-" + "a" * 24
    model = ChattingModel(
        {
            "peer.1": [
                ("team_send", {"to": "peer.2", "text": "x" * (TEAM_MESSAGE_MAX_CHARS + 1)}),
                ("team_send", {"to": "peer.2", "text": f"the key is {secret}"}),
            ]
        },
        listeners=("peer.2",),
    )
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        turn.stage(peers(2))
        turn.finish()
    finally:
        stop(model)
    files = member_files(turn.sessions)
    first, second = [e["data"] for e in files["peer.1"] if e["type"] == "tool/result"]
    assert (
        first["error"]["code"] == "TOOL_FAILED" and f"at most {TEAM_MESSAGE_MAX_CHARS} characters" in first["content"]
    )
    assert second["content"] == "sent to peer.2"
    (sent,) = [e["data"] for e in files["peer.1"] if e["type"] == "team/send"]
    (heard,) = message_events(files["peer.2"])
    assert sent["text"] == "the key is [redacted credential]"
    assert heard["data"]["content"] == "Message from peer.1: the key is [redacted credential]"
    assert secret not in (turn.sessions / "agents").joinpath("003-peer.2.jsonl").read_text()


def test_the_inbox_caps_sends_and_resolves_members_roles_all_and_the_caller() -> None:
    inbox = Inbox({"peer.1": "peer", "peer.2": "peer", "critic.1": "critic"}, "lead", EpisodeStop())
    assert inbox.recipients("peer.1", "peer") == ["peer.2"]
    assert inbox.recipients("peer.1", "all") == ["peer.2", "critic.1"]
    assert inbox.recipients("peer.1", "lead") == ["lead"] and inbox.recipients("critic.1", "peer.2") == ["peer.2"]
    with pytest.raises(ValueError, match=re.escape("no member, role or caller is named 'nobody'; members: peer.1,")):
        inbox.recipients("peer.1", "nobody")
    with pytest.raises(ValueError, match="cannot send a message to itself"):
        inbox.recipients("peer.1", "peer.1")
    for _ in range(TEAM_MAX_SENDS_PER_MEMBER):
        inbox.send("peer.1", "peer.2", "hi")
    with pytest.raises(ValueError, match=f"at most {TEAM_MAX_SENDS_PER_MEMBER} messages"):
        inbox.send("peer.1", "peer.2", "one too many")
    assert len(inbox.take("peer.2")) == TEAM_MAX_SENDS_PER_MEMBER and inbox.take("peer.2") == []


def test_team_wait_returns_on_a_message_the_stop_flag_or_when_it_is_alone_and_never_past_its_cap() -> None:
    def timed(inbox: Inbox, act) -> tuple[int, float]:
        threading.Timer(0.2, act).start()
        started = time.monotonic()
        return inbox.wait("peer.1", 30), time.monotonic() - started

    members = {"peer.1": "peer", "peer.2": "peer"}
    inbox = Inbox(members, "root", EpisodeStop())
    waiting, seconds = timed(inbox, lambda: inbox.send("peer.2", "peer.1", "here"))
    assert waiting == 1 and seconds < 5
    stop_flag = EpisodeStop()
    inbox = Inbox(members, "root", stop_flag)
    waiting, seconds = timed(inbox, lambda: stop_flag.set("cancelled"))
    assert waiting == 0 and seconds < 5  # the flag is read at least once a second
    inbox = Inbox(members, "root", EpisodeStop())
    waiting, seconds = timed(inbox, lambda: inbox.close("peer.2"))
    assert waiting == 0 and seconds < 5
    assert Inbox(members, "root", EpisodeStop()).wait("peer.1", 0.1) == 0

    class RecordingInbox(Inbox):
        def wait(self, member: str, timeout_seconds: float) -> int:
            self.asked = timeout_seconds
            return 0

    recording = RecordingInbox(members, "root", EpisodeStop())
    runner = TeamWaitRunner(TeamMember("peer.1", "peer", recording))
    assert runner({"seconds": 10**6}, "") == "no message after 300 s" and recording.asked == 300
    runner({"seconds": 0}, "")
    assert recording.asked == 1


def test_team_tool_names_are_reserved_for_built_in_tools() -> None:
    for name in ("team_assign", "team_send", "team_wait"):
        config = {"name": name, "description": "x", "code": "def run(args, workdir):\n    return ''\n"}
        with pytest.raises(ValueError, match="reserved for built-in tools"):
            NODE_KINDS["native_tool"](None, config)


def test_message_counts_are_per_agent_and_add_up_over_episodes() -> None:
    trajectory = [
        {"type": "session", "data": {"agent": "peer.1"}},
        {"type": "team/send", "data": {"delivered": [], "undelivered": ["peer.2"]}},
        {"type": "session", "data": {"agent": "peer.2"}},
        {"type": "user/message", "data": {"source": {"kind": "message", "from": "peer.1"}}},
        {"type": "user/message", "data": {"source": {"kind": "team"}}},
        {"type": "session", "data": {"agent": "root"}},
    ]
    counts = team_message_counts(trajectory)
    assert counts == {
        "peer.1": {"sent": 1, "received": 0, "undelivered": 1},
        "peer.2": {"sent": 0, "received": 1, "undelivered": 0},
    }
    assert sum_message_counts([counts, counts, {}])["peer.1"] == {"sent": 2, "received": 0, "undelivered": 2}
    worker = EpisodeEvaluationWorker(
        descriptor=get_adapter("native"),
        scorer=resolve_episode_scorer(lambda task, result: 1.0),
        binary=None,
        timeout=10,
        executor=__import__("reef.harness.episodes.executor", fromlist=["LocalExecutor"]).LocalExecutor(),
        forbid_residue=False,
    )
    scored = worker._score_result(EpisodeResult(0, "", "", tuple(trajectory), ()), "task")
    assert scored.messages == counts


def test_a_side_whose_episodes_sent_messages_reports_them_per_agent(tmp_path: Path, monkeypatch) -> None:
    import reef.train.cordis_backend.backend as reef_cordis_backend

    trajectory = (
        {"type": "session", "data": {"agent": "peer.1"}},
        {"type": "team/send", "data": {"delivered": ["peer.2"], "undelivered": []}},
    )
    monkeypatch.setattr(
        reef_cordis_backend, "run_episode", lambda *args, **kwargs: EpisodeResult(0, "", "", trajectory, ())
    )
    b = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda n, s, m: Mutation("create", "r1", {"name": "rules", "config": {"text": "x"}})),
        score_episode=resolve_episode_scorer(lambda task, result: 1.0),
        tasks=("task one",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
    )
    prepared = b.prepare_step(batch(), b.initial_state(), 0)
    metrics = b.evaluate(prepared.candidate, sides=("candidate",)).metrics
    assert metrics["candidate_messages"] == {"peer.1": {"sent": 1, "received": 0, "undelivered": 0}}


# -- mode parallel: a lead assigns workers with team_assign --------------------------------------------------------

WORKER = ("native_agent", {"name": "worker", "prompt": "You are a worker. Do the task you are given in one line."})


def crew_graph(name: str = "main", **crew) -> dict:
    """The delegating graph with its subagent stage ``delegate`` made of ``crew``: think, act while the model calls
    tools, the stage on its answer, then one more model step."""
    graph = _delegating_graph()
    return {**graph, "name": name, "stages": {**graph["stages"], "delegate": {"kind": "subagent", **crew}}}


class LeadModel(MemberModel):
    """A run that holds ``team_assign`` makes ``assignments`` in its first step, answers after the tool results, and
    then answers with the message it read; any other root answers with what it read; a worker, with its task."""

    def __init__(self, assignments: list[dict], gather: int = 0) -> None:
        super().__init__(gather)
        self.assignments = assignments

    def reply(self, instance: str, body: dict) -> dict:
        messages = body["messages"]
        if instance != "root":
            return _reply(content=f"{instance} did: {messages[1]['content']}")
        if "team_assign" not in [tool["function"]["name"] for tool in body.get("tools") or ()]:
            return _reply(content=f"root read: {messages[-1]['content']}")
        if not any(message["role"] == "tool" for message in messages):
            calls = [_call("team_assign", arguments, f"a{index}") for index, arguments in enumerate(self.assignments)]
            return _reply(tool_calls=calls)
        if messages[-1]["role"] == "user":
            return _reply(content=f"lead read: {messages[-1]['content']}")
        return _reply(content="assigned")


def root_typed(sessions: Path, type_: str) -> list[dict]:
    return [e["data"] for e in events(sessions / "session.jsonl") if e["type"] == type_]


def test_a_lead_assigns_two_workers_that_run_at_once_and_its_next_step_reads_both_results(tmp_path: Path) -> None:
    assignments = [
        {"agent": "worker", "task": "count the primes below 50", "rules": "Answer with the count alone."},
        {"agent": "worker", "task": "count the primes below 100"},
    ]
    model = LeadModel(assignments, gather=2)
    crew = crew_graph(mode="parallel", agents=["worker"], workspace="shared")
    try:
        code, sessions = run_turn(tmp_path, model, [*_seed_nodes(SEED_TOOLS), WORKER, ("native_graph", crew)])
    finally:
        stop(model)
    # Both workers' first calls were in flight together.
    assert code == 0 and not model.is_barrier_broken
    seed_tools = [tool["config"]["name"] for tool in SEED_TOOLS]
    assert events(sessions / "session.jsonl")[0]["data"]["tools"] == sorted([*seed_tools, "team_assign"])
    (declaration,) = [t for t in model.requests[0]["tools"] if t["function"]["name"] == "team_assign"]
    assert declaration["function"]["parameters"]["properties"]["agent"]["enum"] == ["worker"]
    assert [r["content"] for r in root_typed(sessions, "tool/result")] == ["assigned worker.1", "assigned worker.2"]
    files = member_files(sessions)
    assert {instance: (found[0]["data"]["role"], found[0]["data"]["task"]) for instance, found in files.items()} == {
        "worker.1": ("worker", "count the primes below 50"),
        "worker.2": ("worker", "count the primes below 100"),
    }
    # A worker talks to its team but assigns no one; the rules reach the worker they were given to alone.
    assert files["worker.1"][0]["data"]["tools"] == sorted([*seed_tools, "team_send", "team_wait"])
    systems = {member_of(r): r["messages"][0]["content"] for r in model.requests if member_of(r) != "root"}
    assert "Answer with the count alone." in systems["worker.1"] and "Answer with" not in systems["worker.2"]
    kinds = [e["type"] for e in events(sessions / "session.jsonl")]
    assert kinds.index("team/start") < kinds.index("team/end") < kinds.index("user/message")
    (said,) = root_typed(sessions, "user/message")
    assert said["source"] == {"kind": "team", "stage": "delegate", "mode": "parallel", "outcome": "completed"}
    assert said["content"] == (
        "worker.1 (worker) ended with completed: worker.1 did: count the primes below 50\n\n"
        "worker.2 (worker) ended with completed: worker.2 did: count the primes below 100"
    )
    # The lead's next step reads the stage's one message, and its answer ends the turn.
    assert model.requests[-1]["messages"][-1] == {"role": "user", "content": said["content"]}
    exits = [e for e in root_typed(sessions, "stage/exit") if e["stage"] == "delegate"]
    assert exits == [
        {
            "step": 2,
            "stage": "delegate",
            "outcome": "completed",
            "to": "answer",
            "mode": "parallel",
            "agents": ["worker.1", "worker.2"],
            "outcomes": {"worker.1": "completed", "worker.2": "completed"},
            "steps": 2,
        }
    ]
    assert root_typed(sessions, "turn/end")[-1]["reason"] == {"kind": "completed"}


def test_an_agent_no_parallel_stage_runs_is_a_tool_error_and_a_stage_with_no_work_completes(tmp_path: Path) -> None:
    model = LeadModel([{"agent": "critic", "task": "check it"}])
    crew = crew_graph(mode="parallel", agents=["worker"])
    try:
        code, sessions = run_turn(tmp_path, model, [*_seed_nodes(SEED_TOOLS), WORKER, CRITIC, ("native_graph", crew)])
    finally:
        stop(model)
    (result,) = root_typed(sessions, "tool/result")
    assert result["error"]["code"] == "TOOL_FAILED"
    assert result["content"] == "Error: no parallel stage runs 'critic'; agent must be one of worker"
    (said,) = root_typed(sessions, "user/message")
    assert (said["content"], said["source"]["outcome"]) == ("no work was assigned to worker", "completed")
    (stage_exit,) = [e for e in root_typed(sessions, "stage/exit") if e["stage"] == "delegate"]
    assert (stage_exit["outcome"], stage_exit["to"], stage_exit["agents"]) == ("completed", "answer", [])
    assert code == 0 and not (sessions / "agents").exists() and not root_typed(sessions, "team/start")


def test_team_assign_holds_eight_workers_and_a_stage_takes_only_the_agents_it_runs(tmp_path: Path) -> None:
    runner = TeamAssignRunner(SimpleNamespace(assignments=[]), ["worker"])
    for index in range(1, 9):
        assert runner({"agent": "worker", "task": f"part {index}"}, "") == f"assigned worker.{index}"
    with pytest.raises(ToolFailed, match="8 workers already wait for a parallel stage"):
        runner({"agent": "worker", "task": "part 9"}, "")
    model = MemberModel()
    try:
        turn = TeamTurn(tmp_path, model, TEAM_NODES)
        turn.run.assignments = [
            Assignment("peer", "a", ""),
            Assignment("critic", "b", ""),
            Assignment("peer", "c", ""),
        ]
        stage = {"kind": "subagent", "mode": "parallel", "agents": ["peer"], "workspace": "shared"}
        outcome, detail = run_team_stage(turn.run, stage, "crew")
        turn.finish()
    finally:
        stop(model)
    assert (outcome, detail["agents"]) == ("completed", ["peer.1", "peer.2"])
    assert turn.run.assignments == [Assignment("critic", "b", "")]


def test_an_agent_whose_graph_holds_a_parallel_stage_gets_team_assign_and_leads_its_own_workers(
    tmp_path: Path,
) -> None:
    model = LeadModel([{"agent": "worker", "task": "count the primes below 50"}])
    lead = ("native_agent", {"name": "lead", "prompt": "You are the lead. Split the work.", "graph": "leading"})
    graph = _delegating_graph()
    main = {**graph, "stages": {**graph["stages"], "delegate": {"kind": "subagent", "agent": "lead"}}}
    leading = crew_graph("leading", mode="parallel", agents=["worker"])
    nodes = [*_seed_nodes(SEED_TOOLS), WORKER, lead, ("native_graph", main), ("native_graph", leading)]
    try:
        code, sessions = run_turn(tmp_path, model, nodes)
    finally:
        stop(model)
    headers = {found[0]["data"]["agent"]: found[0]["data"] for found in member_files(sessions).values()}
    assert "team_assign" in headers["lead"]["tools"]
    assert "team_assign" not in events(sessions / "session.jsonl")[0]["data"]["tools"]
    assert (headers["worker.1"]["parent"], headers["worker.1"]["task"]) == ("lead", "count the primes below 50")
    # The workspace defaults to own: the worker had a worktree, and it changed no file.
    assert headers["worker.1"]["workspace"] == "own"
    (stage_exit,) = [
        e["data"] for e in member_files(sessions)["lead"] if e["type"] == "stage/exit" and "mode" in e["data"]
    ]
    assert stage_exit["merges"] == {"worker.1": "empty"}
    handed = root_typed(sessions, "user/message")[0]
    assert handed["source"]["agent"] == "lead"
    assert handed["content"] == (
        "lead read: worker.1 (worker) ended with completed: worker.1 did: count the primes below 50\n\n"
        "worker.1: changed no file"
    )
    assert code == 0 and root_typed(sessions, "turn/end")[-1]["reason"] == {"kind": "completed"}


def test_render_refuses_unknown_members_a_cycle_through_them_nested_team_stages_and_a_member_with_then() -> None:
    descriptor = get_adapter("native")
    seed = _seed_nodes(SEED_TOOLS)
    crew = ("native_graph", crew_graph(mode="parallel", agents=["worker"]))
    files = render_composition([*seed, WORKER, crew], descriptor)
    assert json.loads(files["native/graphs/main.json"])["stages"]["delegate"]["agents"] == ["worker"]
    with pytest.raises(RenderError, match="stage 'delegate' names agents the tree lacks: critic, worker"):
        render_composition(
            [*seed, ("native_graph", crew_graph(mode="parallel", agents=["worker", "critic"]))], descriptor
        )
    looped = ("native_agent", {**WORKER[1], "graph": "side"})
    side = ("native_graph", crew_graph("side", mode="parallel", agents=["worker"]))
    with pytest.raises(RenderError, match="'worker' is called in a cycle"):
        render_composition([*seed, looped, crew, side], descriptor)
    below = ("native_graph", crew_graph("side", mode="parallel", agents=["critic"]))
    nested = "native_agent 'worker' runs inside a team stage, and its graph 'side' holds team stage 'delegate'"
    with pytest.raises(RenderError, match=f"{nested}; team stages do not nest"):
        render_composition([*seed, looped, CRITIC, crew, below], descriptor)
    # A team stage anywhere below a member is refused too: here the member calls a lead that runs one.
    calls_lead = ("native_graph", crew_graph("side", agent="lead"))
    lead = ("native_agent", {"name": "lead", "prompt": "You lead.", "graph": "leading"})
    leading = ("native_graph", crew_graph("leading", mode="parallel", agents=["critic"]))
    with pytest.raises(RenderError, match="native_agent 'lead' runs inside a team stage"):
        render_composition([*seed, looped, CRITIC, lead, crew, calls_lead, leading], descriptor)
    with pytest.raises(RenderError, match="native_agent 'worker' is a team member and cannot carry then"):
        render_composition([*seed, ("native_agent", {**WORKER[1], "then": ["critic"]}), CRITIC, crew], descriptor)
    with pytest.raises(RenderError, match="does not render native_graph"):
        render_composition([crew], get_adapter("pi"))
