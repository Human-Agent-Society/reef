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


def _agent(agent_module, monkeypatch, turns: list[dict], *, service_url="http://10.0.0.7:28900"):
    agent = object.__new__(agent_module.HarborAgent)
    agent.model_name = "reef"
    agent.logs_dir = Path("/tmp/trial/agent")
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    agent._service_url = service_url
    agent._seed = 7
    agent._days = 14
    agent._capture = _Capture([])
    sidecar = _Sidecar()

    def start_sidecar():
        agent._capture = _Capture(turns)
        return sidecar

    monkeypatch.setattr(agent, "_start_sidecar", start_sidecar)
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
def test_harness_runs_one_episode_and_keeps_every_receipt_in_call_order(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    monkeypatch.setenv("SAAS_BENCH_ENTERPRISE_LLM_PROVIDER", "openai")
    turns = [
        {"status": 200, "receipt": "r-1", "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 3}}},
        {"status": 500, "receipt": None, "response": {"error": {"message": "engine restarting"}}},
        {"status": 200, "receipt": "r-2", "response": {"usage": {"prompt_tokens": 20, "completion_tokens": 5}}},
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
    assert context.metadata["reef"] == {"agent_record_ids": ["r-1", "r-2"], "agent_record_tokens": [13, 25]}
    assert context.metadata["ceobench"] == {"seed": 7, "days": 14, "turns": 3, "exit_code": 0}
    assert context.metadata["prior"] is True
    assert (context.n_input_tokens, context.n_output_tokens) == (30, 8)


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
    assert context.metadata["reef"] == {"agent_record_ids": ["r-1"], "agent_record_tokens": [0]}
    assert environment.downloads


@pytest.mark.unit
def test_verifier_reward_is_reported_once_per_turn(monkeypatch) -> None:
    _, report_module = _load_harness(monkeypatch, "ceobench")

    class Client:
        def __init__(self):
            self.calls = []

        def report(self, scenario, payload, *, recipe=None):
            self.calls.append((scenario, payload, recipe))
            return {"accepted": True}

    result = {
        "id": "trial-7",
        "task_name": "ceobench",
        "agent_result": {"metadata": {"reef": {"agent_record_ids": ["r-1", "r-2", "r-3"]}}},
        "verifier_result": {"rewards": {"reward": 0.875, "final_cash": 875000.0, "survival_days": 14, "bankrupt": 0}},
    }
    client = Client()

    posted = report_module.post_reports(result, client=client, scenario="ceobench-host-test")

    assert posted == [{"accepted": True}] * 3
    assert [scenario for scenario, _, _ in client.calls] == ["ceobench-host-test"] * 3
    assert [recipe for _, _, recipe in client.calls] == [None] * 3
    payloads = [payload for _, payload, _ in client.calls]
    assert [payload["references"] for payload in payloads] == [["r-1"], ["r-2"], ["r-3"]]
    assert {payload["score"] for payload in payloads} == {0.875}
    assert len({payload["agent_record_id"] for payload in payloads}) == 3
    assert payloads[1]["metadata"]["ceobench"] == {
        "turn": 1,
        "turns": 3,
        "reward": 0.875,
        "final_cash": 875000.0,
        "survival_days": 14,
        "bankrupt": 0,
    }
    assert "875000.0" in payloads[0]["feedback"] and "14 days" in payloads[0]["feedback"]


@pytest.mark.unit
def test_turns_longer_than_the_training_window_are_not_reported(monkeypatch) -> None:
    _, report_module = _load_harness(monkeypatch, "ceobench")

    class Client:
        def __init__(self):
            self.calls = []

        def report(self, scenario, payload, *, recipe=None):
            self.calls.append(payload)
            return {"accepted": True}

    result = {
        "id": "trial-8",
        "task_name": "ceobench",
        "agent_result": {
            "metadata": {
                "reef": {"agent_record_ids": ["r-1", "r-2", "r-3"], "agent_record_tokens": [9000, 50000, 48000]}
            }
        },
        "verifier_result": {"rewards": {"reward": 1.01, "final_cash": 1010000.0, "survival_days": 14, "bankrupt": 0}},
    }
    client = Client()

    posted = report_module.post_reports(result, client=client, scenario="ceobench-host-test", max_tokens=49152)

    # The 50k-token turn was served and recorded but cannot be trained on.
    assert len(posted) == 2
    assert [payload["references"] for payload in client.calls] == [["r-1"], ["r-3"]]
    assert [payload["metadata"]["ceobench"]["turn"] for payload in client.calls] == [0, 2]
    # No limit reports every turn.
    client.calls.clear()
    report_module.post_reports(result, client=client, scenario="ceobench-host-test")
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
def test_entrypoint_runs_one_episode_per_seed_and_drains_training(monkeypatch, capsys) -> None:
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
        "CEOBENCH_SEEDS": "42,43",
        "CEOBENCH_DAYS": "14",
    }.items():
        monkeypatch.setenv(key, value)

    runpy.run_path(str(EXAMPLE_DIR / "run.py"))

    assert [str(task.relative_to(EXAMPLE_DIR)) for _, task, _, _ in calls] == ["harbor", "harbor"]
    assert [agent for _, _, agent, _ in calls] == [
        {"name": "harness:HarborAgent", "model_name": "reef", "kwargs": {"seed": 42, "days": 14}},
        {"name": "harness:HarborAgent", "model_name": "reef", "kwargs": {"seed": 43, "days": 14}},
    ]
    assert [tags for _, _, _, tags in calls] == [
        {"position": 0, "seed": 42, "days": 14},
        {"position": 1, "seed": 43, "days": 14},
    ]
    assert all(lab == EXAMPLE_DIR / "work" / "lab" for lab, _, _, _ in calls)
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
