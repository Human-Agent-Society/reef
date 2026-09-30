"""The native episode's controls: one token budget per episode, spent by every agent turn, set only by the caller."""

from __future__ import annotations

import json
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
from reef.harness.runners.native import run_loop
from reef.harness.runners.native.control import EpisodeControl, TeamBudget, episode_token_limit
from reef.harness.runners.native.enforce import InProcessEnforcer
from reef.harness.runners.native.graph import _tokens
from reef.harness.runners.native.seed import SEED_TOOLS
from reef.harness.tree.render import render_composition
from reef.train.cordis_backend.backend import _agent_work

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
