"""The native episode's controls and its team stages.

One token budget per episode, spent by every agent turn and set only by the caller; a stop flag every run reads
before its next step; and the members of a team stage, each on its own thread, host, session file and step budget.
Until a graph can name a team stage, the stage runs through ``TeamStageRun`` on a root turn put together as
``run_loop`` puts it together."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import pytest
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
from reef.harness.episodes.run import run_episode
from reef.harness.episodes.trajectory import reader_for
from reef.harness.runners.native import Session, ToolModule, ToolRunner, _Loop, run_loop
from reef.harness.runners.native.control import EpisodeControl, EpisodeStop, TeamBudget, episode_token_limit
from reef.harness.runners.native.enforce import InProcessEnforcer
from reef.harness.runners.native.graph import Run, _tokens, run_graph
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.seed import SEED_TOOLS
from reef.harness.runners.native.team import MemberStart, TeamStageRun, team_outcome
from reef.harness.tree.render import render_composition
from reef.train.cordis_backend.backend import _agent_work, _stage_path, tree_files

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
    """Answers each member by its instance name; with ``gather``, that many members meet at a barrier on their first
    call, so no reply goes out until all of them are in flight at once. At the barrier it sets ``stop`` when given
    and lists ``mounts_path`` when given."""

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
        if is_first and self.barrier is not None:
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
        self.loop = _Loop(
            session, self.root, self.sessions, header, enforcer=InProcessEnforcer(), control=self.control
        )
        self.run = Run(self.loop, "split the work", binding, self.host, self.work)

    def stage(self, members, *, mode: str = "team", workspace: str = "shared"):
        return TeamStageRun(self.run, "crew", mode, workspace, members).run()

    def close(self) -> list[dict]:
        """The episode's trajectory once the root's session and host are closed."""
        self.host.dispose()
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
        "root started the team and reads your final answer."
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
