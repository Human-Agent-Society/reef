"""Execution phase, evidence, and runtime compatibility are independent of reward."""

import asyncio
import copy
import json
import os
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipes.meta_harness.examples.terminal_bench.history import HistoryBinding
from recipes.meta_harness.examples.terminal_bench.runtime import prepare_harbor, tmux_source
from reef.harness.terminus.runner import pinned_task, trial_record
from reef.harness.terminus.tree import TerminusTreeError
from reef.harness.terminus.trial import trial_outcome


def raw_trial(*, exception=None, reward=0, executed=True, cost=0.02):
    return {
        "task_name": "a",
        "agent_execution": {"started_at": "start"} if executed else None,
        "agent_result": {"cost_usd": cost} if executed else None,
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": {"exception_type": exception} if exception else None,
        "started_at": "2026-09-04T00:00:00",
        "config": {"task": {"path": "a"}},
    }


@pytest.mark.parametrize("reward", [0, 1])
def test_agent_timeout_with_verifier_reward_is_a_measurement(reward):
    result = trial_outcome(raw_trial(exception="AgentTimeoutError", reward=reward))
    assert result["valid"]
    assert result["phase"] == "agent_timeout_verified"
    assert result["reward"] == reward


def test_setup_timeout_is_unmeasured_and_known_not_to_have_run_model():
    result = trial_outcome(raw_trial(exception="TimeoutException", reward=None, executed=False))
    assert not result["valid"]
    assert result["phase"] == "setup_failure"
    assert result["cost_usd"] == 0


@pytest.mark.parametrize("cost", [0, None])
def test_cost_is_not_an_execution_flag(cost):
    result = trial_outcome(raw_trial(cost=cost))
    assert result["valid"]
    assert result["cost_usd"] == cost


@pytest.mark.parametrize("reward", [None, True, float("nan"), float("inf")])
def test_missing_or_nonfinite_rewards_cannot_enter_frontier(reward):
    assert not trial_outcome(raw_trial(reward=reward))["valid"]


def test_transport_failure_with_reward_is_flagged():
    assert not trial_outcome(raw_trial(exception="TimeoutException", reward=1))["valid"]


def test_trial_record_preserves_failure_phase_usage_and_all_steps(tmp_path):
    path = tmp_path / "trial"
    path.mkdir()
    raw = raw_trial(cost=None)
    (path / "result.json").write_text(json.dumps(raw))
    (path / "trajectory.json").write_text(json.dumps({"steps": [{"message": "verifier evidence"}]}))
    result = trial_record("terminal-bench/a", {"reward": 0}, tmp_path)
    assert result["outcome"]["valid"]
    assert result["observed_cost_usd"] is None
    assert result["steps"] == [{"message": "verifier evidence"}]
    assert result["attempts"][0]["task_config"] == {"path": "a"}


def test_missing_result_never_becomes_a_free_valid_trial(tmp_path):
    result = trial_record("terminal-bench/a", {}, tmp_path)
    assert not result["outcome"]["valid"]
    assert result["observed_cost_usd"] is None


def test_task_pin_is_explicit_and_path_safe():
    commit = "a" * 40
    assert pinned_task("terminal-bench/bn-fit-modify", commit)["git_commit_id"] == commit
    assert pinned_task("terminal-bench/install-windows-3.11", commit)["path"] == "install-windows-3.11"
    for task in ["../../a", "terminal-bench/../a", "/tmp/a", "terminal-bench/a/b"]:
        with pytest.raises(TerminusTreeError):
            pinned_task(task, commit)


def test_history_tools_read_frozen_full_evidence_and_validate_pages():
    records = {"seed": {"trajectory": [{"message": str(i)} for i in range(12)]}}
    history = HistoryBinding(SimpleNamespace(api="openai"), records)
    records["seed"]["trajectory"].clear()
    assert history.read("seed", 8)["steps"] == [{"message": str(i)} for i in range(8, 12)]
    for kwargs in [
        {"record_id": "../other"},
        {"record_id": "seed", "start": -1},
        {"record_id": "seed", "limit": 100},
        {"record_id": "seed", "start": True},
    ]:
        with pytest.raises(ValueError):
            history.read(**kwargs)


def test_proposer_receives_tool_results_before_returning_composition():
    bodies = []

    def complete(body):
        bodies.append(copy.deepcopy(body))
        if len(bodies) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {
                                        "name": "read_evaluation",
                                        "arguments": json.dumps({"record_id": "failed-candidate"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": '{"entries": []}'}}]}

    history = HistoryBinding(
        SimpleNamespace(api="openai", complete=complete),
        {"failed-candidate": {"reward": 0, "trajectory": [{"message": "command failed: wrong path"}]}},
    )
    assert history.chat([{"role": "user", "content": "propose"}]) == '{"entries": []}'
    assert "wrong path" in bodies[1]["messages"][-1]["content"]
    assert bodies[1]["reasoning_effort"] == "xhigh"


def test_e2b_install_timeout_is_caught_without_mutating_hardlinked_runtime(tmp_path):
    pytest.importorskip("harbor")
    pytest.importorskip("e2b", reason="install the Terminal-Bench example dependencies")
    from e2b.exceptions import TimeoutException

    original = tmux_source()
    installed = tmp_path / "tmux.py"
    neighbor = tmp_path / "other-runtime.py"
    installed.write_bytes(original.read_bytes())
    os.link(installed, neighbor)
    before = neighbor.read_bytes()
    prepare_harbor(installed)
    prepare_harbor(installed)  # idempotent
    assert neighbor.read_bytes() == before
    text = installed.read_text()
    start = text.index("    async def _attempt_tmux_installation(")
    end = text.index("    async def _install_recording_tools(", start)
    scope = {"asyncio": asyncio}
    exec(textwrap.dedent(text[start:end]), scope)

    async def unavailable():
        raise TimeoutException("command timed out")

    fake = SimpleNamespace(_install_recording_tools=unavailable, _TOOL_INSTALL_BUDGET_SEC=1, _logger=Mock())
    asyncio.run(scope["_attempt_tmux_installation"](fake))
    fake._logger.warning.assert_called_once()


@pytest.mark.parametrize("residue, valid", [((), True), (("unexpected.txt",), False)])
def test_campaign_episode_uses_configured_executor_timeout_and_residue_policy(monkeypatch, residue, valid):
    from recipes.meta_harness.examples.terminal_bench.campaign import TerminalBenchBackend, score_episode
    from reef.harness.episode import EpisodeResult

    calls = []
    executor = object()
    descriptor = object()

    def run(descriptor_arg, files, task, **kwargs):
        calls.append((descriptor_arg, files, task, kwargs))
        return EpisodeResult(
            0,
            "",
            "",
            (
                {
                    "type": "verifier",
                    "reward": 0,
                    "outcome": {"valid": True, "phase": "verified"},
                    "observed_cost_usd": 0.4,
                },
            ),
            residue,
        )

    monkeypatch.setattr("recipes.meta_harness.examples.terminal_bench.campaign.run_episode", run)
    backend = SimpleNamespace(
        _descriptor=descriptor,
        _binary="configured-binary",
        _episode_timeout_s=28800,
        _executor=executor,
        _forbid_residue=True,
        _score_episode=score_episode,
    )
    identity = {"id": "trial", "task": "terminal-bench/a", "repeat": 0, "attempt": 1}
    result = TerminalBenchBackend._episode(backend, {"file": "contents"}, identity)
    assert result["valid"] is valid
    assert result["cost_usd"] == 0.4
    assert calls == [
        (
            descriptor,
            {"file": "contents"},
            "terminal-bench/a",
            {"binary": "configured-binary", "timeout": 28800, "executor": executor},
        )
    ]


def test_campaign_rejects_nonfinite_custom_scoring(monkeypatch):
    from recipes.meta_harness.examples.terminal_bench.campaign import TerminalBenchBackend
    from reef.harness.episode import EpisodeResult

    monkeypatch.setattr(
        "recipes.meta_harness.examples.terminal_bench.campaign.run_episode",
        lambda *args, **kwargs: EpisodeResult(
            0, "", "", ({"type": "verifier", "outcome": {"valid": True}, "observed_cost_usd": 0.1},), ()
        ),
    )
    backend = SimpleNamespace(
        _descriptor=None,
        _binary="configured",
        _episode_timeout_s=28800,
        _executor=None,
        _forbid_residue=True,
        _score_episode=lambda *args: float("nan"),
    )
    with pytest.raises(ValueError, match="non-finite"):
        TerminalBenchBackend._episode(backend, {}, {"task": "terminal-bench/a"})


def test_campaign_keeps_runner_failure_diagnostics_and_unknown_cost(monkeypatch):
    from recipes.meta_harness.examples.terminal_bench.campaign import TerminalBenchBackend
    from reef.harness.episode import EpisodeResult

    monkeypatch.setattr(
        "recipes.meta_harness.examples.terminal_bench.campaign.run_episode",
        lambda *args, **kwargs: EpisodeResult(1, "", "Traceback: launch failed", (), ()),
    )
    backend = SimpleNamespace(
        _descriptor=None, _binary=None, _episode_timeout_s=28800, _executor=None, _forbid_residue=True
    )
    record = TerminalBenchBackend._episode(backend, {}, {"task": "terminal-bench/a"})
    assert record["phase"] == "missing_diagnostics"
    assert record["runner_stderr"] == "Traceback: launch failed"
    assert record["cost_usd"] is None
    assert not record["valid"]
