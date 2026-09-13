"""Host-side contracts of the CEO-Bench example (recipes/sao/examples/ceobench)."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_example_entrypoints import EXAMPLE_DIRS, _load_harness

EXAMPLE_DIR = EXAMPLE_DIRS["ceobench"]
SCORE_PATH = EXAMPLE_DIR / "harbor" / "tests" / "score.py"


def _load_score_module():
    namespace = runpy.run_path(str(SCORE_PATH), run_name="score")
    return SimpleNamespace(**namespace)


class _Environment:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.commands: list[tuple[str, dict]] = []
        self.downloads: list[tuple[str, Path]] = []

    async def exec(self, command, env=None):
        self.commands.append((command, dict(env or {})))
        return SimpleNamespace(return_code=self.return_code, stdout="", stderr="")

    async def download_dir(self, source_dir, target_dir):
        self.downloads.append((source_dir, Path(target_dir)))


class _Sidecar:
    server_address = ("0.0.0.0", 29123)

    def __init__(self) -> None:
        self.stopped = False

    def shutdown(self) -> None:
        self.stopped = True


class _Capture:
    def __init__(self, turns: list[dict]) -> None:
        self._turns = turns

    def snapshot(self):
        return list(self._turns)


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def report(self, scenario, payload, *, recipe=None):
        self.calls.append((scenario, payload))
        return {"accepted": True}


def _dashboard(week: int, day: int, cash: int) -> str:
    return f"=== Week {week} Dashboard (Day {day}) ===\n\nCash: ${cash:,}\nIndividual Subscribers: 0\n"


def _turn(receipt: str, dashboard: str | None, prompt_tokens: int, completion_tokens: int) -> dict:
    messages = [{"role": "system", "content": "You are the CEO."}]
    if dashboard is not None:
        messages.append({"role": "user", "content": dashboard})
    return {
        "status": 200,
        "receipt": receipt,
        "request": {"messages": messages},
        "response": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    }


def _agent(agent_module, monkeypatch, turns: list[dict], *, service_url="http://10.0.0.7:28900"):
    import threading

    agent = object.__new__(agent_module.HarborAgent)
    agent.model_name = "reef"
    agent.logs_dir = Path("/tmp/trial/agent")
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    agent._service_url = service_url
    agent._scenario = "ceobench-host-test"
    agent._seed = 7
    agent._days = 14
    agent._client = _Client()
    agent._capture = _Capture([])
    agent._ledger = agent_module.WeekLedger()
    agent._ledger_lock = threading.Lock()
    agent._max_tokens = 0
    sidecar = _Sidecar()

    def start_sidecar():
        agent._capture = _Capture(turns)
        return sidecar

    monkeypatch.setattr(agent, "_start_sidecar", start_sidecar)
    monkeypatch.setattr(agent_module, "WEEK_POLL_S", 0.01)
    return agent, sidecar


@pytest.mark.unit
def test_runner_command_points_the_benchmark_agent_at_the_sidecar(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    command = agent_module.runner_command("http://10.0.0.7:29123/v1", "reef", 42, 500)

    assert command.startswith(f"mkdir -p {agent_module.RUNS_DIR} && cd {agent_module.CEOBENCH_DIR} && uv run")
    assert "saas_bench.agents.bash_agent.run_test" in command
    for flag in (
        "--provider openai",
        "--base-url http://10.0.0.7:29123/v1",
        "--model reef",
        "--seed 42",
        "--days 500",
    ):
        assert flag in command
    assert command.endswith(f"--workspace {agent_module.RUNS_DIR} > {agent_module.RUNS_DIR}/runner.log 2>&1")


@pytest.mark.unit
def test_forwarded_environment_keeps_simulator_settings_out_of_the_repository(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    forwarded = agent_module.forwarded_environment(
        {
            "SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER": "openai",
            "OPENAI_BASE_URL": "http://sim:30100/v1",
            "ANTHROPIC_API_KEY": "secret",
            "AWS_REGION": "us-east-2",
            "HOME": "/home/x",
            "REEF_TOKEN": "reef-local",
        }
    )

    assert forwarded == {
        "SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER": "openai",
        "OPENAI_BASE_URL": "http://sim:30100/v1",
        "ANTHROPIC_API_KEY": "secret",
        "AWS_REGION": "us-east-2",
        "SAAS_BENCH_OPENAI_CHAT_COMPLETIONS": "1",
    }


@pytest.mark.unit
def test_turn_week_parses_the_latest_dashboard(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    fresh = _turn("r", _dashboard(0, 0, 1_000_000), 1, 1)
    assert agent_module.turn_week(fresh) == (0, 0, 1_000_000.0)

    # A transcript that still holds an earlier week's dashboard resolves to the latest one.
    carried = _turn("r", _dashboard(0, 0, 1_000_000), 1, 1)
    carried["request"]["messages"].append({"role": "tool", "content": "ok\n" + _dashboard(1, 7, 982_311)})
    assert agent_module.turn_week(carried) == (1, 7, 982_311.0)

    broke = _turn("r", "=== Week 9 Dashboard (Day 63) ===\n\nCash: -$1,234\n", 1, 1)
    assert agent_module.turn_week(broke) == (9, 63, -1234.0)

    assert agent_module.turn_week({"request": {"messages": [{"role": "user", "content": "hello"}]}}) is None


@pytest.mark.unit
def test_week_ledger_groups_turns_and_closes_weeks_with_the_next_dashboard(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    ledger = agent_module.WeekLedger()
    ledger.observe(
        [
            _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
            {"status": 500, "receipt": None, "response": {"error": {"message": "engine restarting"}}},
            _turn("r-2", _dashboard(0, 0, 1_000_000), 20, 5),
            _turn("r-3", _dashboard(1, 7, 982_311), 30, 7),
            _turn("r-4", None, 40, 9),  # no dashboard of its own: still week 1
        ]
    )

    assert [(turn["receipt"], turn["tokens"], turn["week"]) for turn in ledger.turns] == [
        ("r-1", 13, 0),
        ("r-2", 25, 0),
        ("r-3", 37, 1),
        ("r-4", 49, 1),
    ]
    assert ledger.weeks[0]["turns"] == [("r-1", 13), ("r-2", 25)]
    assert ledger.weeks[1] == {"day": 7, "cash_start": 982_311.0, "turns": [("r-3", 37), ("r-4", 49)]}
    # Week 0 closed when week 1's dashboard appeared; week 1 waits for the final cash.
    assert ledger.finished_weeks() == [(0, 982_311.0)]
    assert ledger.finished_weeks(final_cash=793_047.0) == [(0, 982_311.0), (1, 793_047.0)]
    ledger.posted.add(0)
    assert ledger.finished_weeks() == []
    assert ledger.summary(final_cash=793_047.0) == [
        {"week": 0, "day": 0, "cash_start": 1_000_000.0, "cash_end": 982_311.0, "turns": 2, "reported": True},
        {"week": 1, "day": 7, "cash_start": 982_311.0, "cash_end": 793_047.0, "turns": 2, "reported": False},
    ]


@pytest.mark.unit
def test_harness_reports_a_week_as_soon_as_the_next_one_starts(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    monkeypatch.setenv("SAAS_BENCH_ENTERPRISE_LLM_PROVIDER", "openai")
    turns = [
        _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
        {"status": 500, "receipt": None, "response": {"error": {"message": "engine restarting"}}},
        _turn("r-2", _dashboard(0, 0, 1_000_000), 20, 5),
        _turn("r-3", _dashboard(1, 7, 982_311), 30, 7),
    ]
    agent, sidecar = _agent(agent_module, monkeypatch, turns)
    environment = _Environment()
    context = SimpleNamespace(metadata={"prior": True}, n_input_tokens=0, n_output_tokens=0)

    asyncio.run(agent.run("play", environment, context))

    (command, env), *rest = environment.commands
    assert not rest
    assert "--base-url http://10.0.0.7:29123/v1" in command and "--seed 7 --days 14" in command
    assert env["SAAS_BENCH_OPENAI_CHAT_COMPLETIONS"] == "1"
    assert env["SAAS_BENCH_ENTERPRISE_LLM_PROVIDER"] == "openai"
    assert sidecar.stopped
    assert environment.downloads == [(agent_module.RUNS_DIR, Path("/tmp/trial/agent/ceobench"))]

    # Week 0 was reported (its two turns) once week 1's dashboard appeared; week 1 waits.
    payloads = [payload for _, payload in agent._client.calls]
    assert [payload["references"] for payload in payloads] == [["r-1"], ["r-2"]]
    assert {payload["score"] for payload in payloads} == {(982_311 - 1_000_000) / 1_000_000}
    assert payloads[1]["metadata"]["ceobench"] == {
        "week": 0,
        "day": 0,
        "cash_start": 1_000_000.0,
        "cash_end": 982_311.0,
        "turn": 1,
        "turns": 2,
    }
    assert len({payload["agent_record_id"] for payload in payloads}) == 2

    assert context.metadata["reef"] == {
        "agent_record_ids": ["r-1", "r-2", "r-3"],
        "agent_record_tokens": [13, 25, 37],
        "agent_record_weeks": [0, 0, 1],
    }
    assert context.metadata["ceobench"] == {
        "seed": 7,
        "days": 14,
        "turns": 4,
        "exit_code": 0,
        "weeks": [
            {"week": 0, "day": 0, "cash_start": 1_000_000.0, "cash_end": 982_311.0, "turns": 2, "reported": True},
            {"week": 1, "day": 7, "cash_start": 982_311.0, "cash_end": None, "turns": 1, "reported": False},
        ],
    }
    assert context.metadata["prior"] is True
    assert (context.n_input_tokens, context.n_output_tokens) == (60, 15)


@pytest.mark.unit
def test_last_week_closes_with_the_verifier_final_cash(monkeypatch, tmp_path) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    turns = [
        _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
        _turn("r-2", _dashboard(1, 7, 982_311), 30, 7),
    ]
    agent, _ = _agent(agent_module, monkeypatch, turns)
    agent.logs_dir = tmp_path / "agent"
    agent._report_watch_from = 0.0
    asyncio.run(agent.run("play", _Environment(), SimpleNamespace(metadata=None, n_input_tokens=0, n_output_tokens=0)))
    assert [payload["references"] for _, payload in agent._client.calls] == [["r-1"]]

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-9",
                "verifier_result": {"rewards": {"reward": 0.793, "final_cash": 793_047.0, "survival_days": 14}},
            }
        ),
        encoding="utf-8",
    )
    agent._report_trial_result()

    scenario, payload = agent._client.calls[-1]
    assert scenario == "ceobench-host-test"
    assert payload["references"] == ["r-2"]
    assert payload["score"] == (793_047 - 982_311) / 1_000_000
    assert payload["metadata"]["ceobench"]["week"] == 1 and payload["metadata"]["ceobench"]["cash_end"] == 793_047.0
    assert "week 1" in payload["feedback"]
    # A second look changes nothing: the week is already posted.
    agent._report_trial_result()
    assert len(agent._client.calls) == 2


@pytest.mark.unit
def test_harness_fails_the_trial_when_the_runner_exits_nonzero(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    agent, sidecar = _agent(agent_module, monkeypatch, [{"status": 200, "receipt": "r-1", "response": {}}])
    environment = _Environment(return_code=3)
    context = SimpleNamespace(metadata=None, n_input_tokens=0, n_output_tokens=0)

    with pytest.raises(RuntimeError, match="exited 3"):
        asyncio.run(agent.run("play", environment, context))

    # The receipts and the run directory are kept even for a failed episode.
    assert sidecar.stopped
    assert context.metadata["reef"] == {
        "agent_record_ids": ["r-1"],
        "agent_record_tokens": [0],
        "agent_record_weeks": [None],
    }
    assert context.metadata["ceobench"]["weeks"] == []
    assert environment.downloads


@pytest.mark.unit
def test_turns_longer_than_the_training_window_are_not_reported(monkeypatch) -> None:
    _, report_module = _load_harness(monkeypatch, "ceobench")
    client = _Client()

    posted = report_module.post_week_reports(
        client,
        "ceobench-host-test",
        week=3,
        day=21,
        cash_start=900_000.0,
        cash_end=910_000.0,
        turns=[("r-1", 9000), ("r-2", 50000), ("r-3", 48000)],
        max_tokens=49152,
    )

    # The 50k-token turn was served and recorded but cannot be trained on.
    assert posted == [{"accepted": True}] * 2
    payloads = [payload for _, payload in client.calls]
    assert [payload["references"] for payload in payloads] == [["r-1"], ["r-3"]]
    assert [payload["metadata"]["ceobench"]["turn"] for payload in payloads] == [0, 2]
    assert {payload["score"] for payload in payloads} == {0.01}
    assert "week 3" in payloads[0]["feedback"] and "900000 -> 910000" in payloads[0]["feedback"]
    # No limit reports every turn.
    client.calls.clear()
    report_module.post_week_reports(
        client,
        "ceobench-host-test",
        week=3,
        day=21,
        cash_start=900_000.0,
        cash_end=910_000.0,
        turns=[("r-1", 1), ("r-2", 2), ("r-3", 3)],
    )
    assert len(client.calls) == 3


@pytest.mark.unit
def test_scorer_locates_the_single_run_and_prefers_the_checkpointed_database(tmp_path) -> None:
    score = _load_score_module()
    runs = tmp_path / "runs"
    run_dir = runs / "run_abc123"
    live = run_dir / "agent_workspace" / "sessions" / "s1"
    live.mkdir(parents=True)
    (live / "world.nmdb").write_bytes(b"live")

    assert score.find_run_dir(runs) == run_dir
    assert score.find_world_db(run_dir) == live / "world.nmdb"
    (run_dir / "world.nmdb").write_bytes(b"checkpointed")
    assert score.find_world_db(run_dir) == run_dir / "world.nmdb"

    (runs / "run_def456").mkdir()
    with pytest.raises(RuntimeError, match="expected one run"):
        score.find_run_dir(runs)
    with pytest.raises(FileNotFoundError):
        score.find_run_dir(tmp_path / "empty")


@pytest.mark.unit
def test_entrypoint_runs_one_episode_and_drains_training(monkeypatch, capsys) -> None:
    calls = []

    class Lab:
        def __init__(self, path):
            self.path = Path(path)

        async def run(self, task, agent, tags=None):
            calls.append((self.path, Path(task), agent, tags))
            return SimpleNamespace(rewards={"reward": 0.9, "final_cash": 900000.0}, tags={}, uri="file:///trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    for key, value in {
        "REEF_SERVICE_URL": "http://127.0.0.1:1/",  # nothing listens: the drain skips
        "REEF_SCENARIO": "ceobench-host-test",
        "REEF_TOKEN": "reef-local",
        "CEOBENCH_SEED": "43",
        "CEOBENCH_DAYS": "14",
    }.items():
        monkeypatch.setenv(key, value)

    runpy.run_path(str(EXAMPLE_DIR / "run.py"))

    # One episode: the policy adapts inside the episode it is scored on.
    (lab, task, agent, tags), *rest = calls
    assert not rest
    assert str(task.relative_to(EXAMPLE_DIR)) == "harbor"
    assert agent == {"name": "harness:HarborAgent", "model_name": "reef", "kwargs": {"seed": 43, "days": 14}}
    assert tags == {"seed": 43, "days": 14}
    assert lab == EXAMPLE_DIR / "work" / "lab"
    out = capsys.readouterr().out
    assert "reward" in out and "not reachable" in out


@pytest.mark.unit
def test_entrypoint_fails_when_harbor_reports_an_error(monkeypatch) -> None:
    class Lab:
        def __init__(self, _path):
            pass

        async def run(self, _task, _agent, tags=None):
            return SimpleNamespace(rewards={}, tags={"error": "environment failed"}, uri="file:///failed-trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    monkeypatch.setenv("REEF_SERVICE_URL", "http://127.0.0.1:1/")

    with pytest.raises(RuntimeError, match="Harbor trial failed: environment failed"):
        runpy.run_path(str(EXAMPLE_DIR / "run.py"))


@pytest.mark.unit
def test_task_pins_the_upstream_commit_and_ships_no_credentials() -> None:
    dockerfile = (EXAMPLE_DIR / "harbor" / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG CEOBENCH_COMMIT=d2b7b32e5301a571b77f5f68bd1032adbcd5b464" in dockerfile
    patch = (EXAMPLE_DIR / "harbor" / "environment" / "reef.patch").read_text(encoding="utf-8")
    assert "SAAS_BENCH_OPENAI_CHAT_COMPLETIONS" in patch and "SAAS_BENCH_" in patch
    for path in EXAMPLE_DIR.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".toml", ".md", ".patch"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "sk-ant-" not in text and "AKIA" not in text, path
    serve = (EXAMPLE_DIR / "serve.yaml").read_text(encoding="utf-8")
    assert "batch-size: 1" in serve and "recipes.sao.recipe:SAORecipe" in serve
    assert json.loads(json.dumps({"ok": True}))  # keeps json imported for the reward fixture above
