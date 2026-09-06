import copy
from types import SimpleNamespace

import pytest
from tests.reef_service.test_meta_harness_driver import campaign, finish, open_campaign  # noqa: F401

from recipes.meta_harness.examples.terminal_bench import agent_failure_policy as policy
from recipes.meta_harness.examples.terminal_bench.campaign import TerminalBenchBackend

ORIGINAL_EPISODE = TerminalBenchBackend._episode


@pytest.fixture
def failed():
    command = "python check.py >/tmp/check.out 2>&1; status=$?; tail -15 /tmp/check.out; exit $status\n"
    step = {
        "source": "agent",
        "step_id": 9,
        "tool_calls": [{"function_name": "bash_command", "arguments": {"keystrokes": command}}],
        "observation": {"results": [{"content": "/app $ " + command}]},
    }
    return {
        "valid": False,
        "reward": None,
        "phase": "execution_failure",
        "exception_type": "RuntimeError",
        "error": "tmux send-keys failed: no server running on /tmp/tmux-0/default",
        "agent_execution_started": True,
        "timing": {"verifier": None},
        "cost_usd": 0.1,
        "trajectory": [step],
        "residue": [],
    }


def test_raw_failed_trial_stays_invalid_and_unverified_while_benchmark_score_is_zero(failed):
    original = copy.deepcopy(failed)
    scored = policy.annotate(failed, policy.PROTOCOL)
    assert scored["valid"] is False and scored["reward"] is None and scored["cost_usd"] == 0.1
    assert policy.score(scored, policy.PROTOCOL) == 0 and policy.eligible(scored, policy.PROTOCOL)
    assert not policy.eligible(scored) and policy.score(scored) is None
    assert failed == original and policy.annotate(failed) == original


def pipeline_failure(failed):
    row = copy.deepcopy(failed)
    row["trajectory"][0]["tool_calls"][0]["arguments"][
        "keystrokes"
    ] = "cd /app/ocaml; make -C testsuite one DIR=tests/basic 2>&1 | tee /tmp/basic.log; exit ${PIPESTATUS[0]}\n"
    row["trajectory"][0]["observation"]["results"][0][
        "content"
    ] = "36 tests failed\nexit\nasciinema: recording finished\n/app $\n"
    return row


def test_pipeline_exit_requires_explicit_v3_and_preserves_raw_failure(failed):
    row = pipeline_failure(failed)
    original = copy.deepcopy(row)
    assert not policy.eligible(policy.annotate(row, policy.ERREXIT_PROTOCOL), policy.ERREXIT_PROTOCOL)
    scored = policy.annotate(row, policy.PIPESTATUS_PROTOCOL)
    assert policy.score(scored, policy.PIPESTATUS_PROTOCOL) == 0
    assert row == original and scored["reward"] is None and scored["valid"] is False
    assert (
        policy.score(policy.annotate(errexit_failure(failed), policy.ERREXIT_PROTOCOL), policy.PIPESTATUS_PROTOCOL)
        == 0
    )


@pytest.mark.parametrize(
    "command",
    [
        "echo 'x | tee /tmp/out; exit ${PIPESTATUS[0]}'\n",
        "(x | tee /tmp/out; exit ${PIPESTATUS[0]})\n",
        "x | tee /tmp/out; exit ${PIPESTATUS[0]} &\n",
        "if false; then x | tee /tmp/out; exit ${PIPESTATUS[0]}; fi\n",
        "x | tee /tmp/out; echo ${PIPESTATUS[0]}\n",
        "x | tee /tmp/out; exit ${PIPESTATUS[1]}\n",
        "x | exit ${PIPESTATUS[0]}\n",
        "x | tee /tmp/out\nexit ${PIPESTATUS[0]}\n",
    ],
)
def test_pipeline_classifier_rejects_quoted_conditional_background_or_subshell_exit(command):
    assert not policy._pipeline_exit(command)


@pytest.mark.parametrize("change", ["no_exit", "no_recording_end", "other_step", "unknown_cost", "verifier_ran"])
def test_pipeline_failure_needs_the_same_step_exit_and_recording_end(failed, change):
    row = pipeline_failure(failed)
    content = row["trajectory"][0]["observation"]["results"][0]
    if change == "no_exit":
        content["content"] = "asciinema: recording finished\n"
    elif change == "no_recording_end":
        content["content"] = "exit\n/app $\n"
    elif change == "other_step":
        row["trajectory"].append(
            {"source": "agent", "observation": copy.deepcopy(row["trajectory"][0]["observation"])}
        )
        content["content"] = "unrelated output"
    elif change == "unknown_cost":
        row["cost_usd"] = None
    else:
        row["timing"]["verifier"] = {"started_at": "now"}
    assert not policy.eligible(policy.annotate(row, policy.PIPESTATUS_PROTOCOL), policy.PIPESTATUS_PROTOCOL)


@pytest.mark.parametrize(
    "command",
    [
        "exit\n",
        "false; exit 1\n",
        "exit $?\n",
        'exit "$status"\n',
        "exit ${status}\n",
        "python check.py >/tmp/out 2>&1; status=$?; tail /tmp/out; exit $status\n",
    ],
)
def test_direct_shell_exit_recognizes_exit_status_and_redirection(command):
    assert policy._direct_exit(command)


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'exit 0'\n",
        "(exit 0)\n",
        "echo 'exit 0'\n",
        "echo '; exit 0'\n",
        "exit 0 | cat\n",
        "exit 0 &\n",
        "echo x\nexit 0\n",
        "cat <<EOF\nexit 0\nEOF\n",
        "logout\n",
    ],
)
def test_subshell_quoted_or_ambiguous_exit_is_not_an_interactive_exit(command):
    assert not policy._direct_exit(command)


@pytest.mark.parametrize(
    "change",
    [
        "setup",
        "unknown_cost",
        "nan_cost",
        "boolean_cost",
        "transport",
        "residue",
        "verifier_ran",
        "no_observed_prompt",
        "no_agent_command",
        "reward",
        "other_error",
    ],
)
def test_infrastructure_or_unevidenced_failures_remain_unscored(failed, change):
    if change == "setup":
        failed["agent_execution_started"] = False
    elif change in ("unknown_cost", "nan_cost", "boolean_cost"):
        failed["cost_usd"] = {"unknown_cost": None, "nan_cost": float("nan"), "boolean_cost": True}[change]
    elif change == "transport":
        failed["phase"] = "episode_failure"
    elif change == "residue":
        failed["residue"] = ["leftover"]
    elif change == "verifier_ran":
        failed["timing"]["verifier"] = {"started_at": "now"}
    elif change == "no_observed_prompt":
        failed["trajectory"][0]["observation"] = {"results": []}
    elif change == "no_agent_command":
        failed["trajectory"][0]["source"] = "system"
    elif change == "reward":
        failed["reward"] = 1
    else:
        failed["error"] = "remote process killed"
    projected = policy.annotate(failed, policy.PROTOCOL)
    assert policy.score(projected, policy.PROTOCOL) is None and not policy.eligible(projected, policy.PROTOCOL)
    assert "benchmark_failure" not in projected


def test_score_rechecks_retained_evidence(failed):
    row = policy.annotate(failed, policy.PROTOCOL)
    row["trajectory"][0]["tool_calls"][0]["arguments"]["keystrokes"] = "echo changed\n"
    with pytest.raises(ValueError, match="terminal evidence"):
        policy.score(row, policy.PROTOCOL)


def test_real_episode_retains_raw_failure_and_adds_separate_benchmark_score(failed, monkeypatch):
    outcome = {k: v for k, v in failed.items() if k not in ("trajectory", "residue")}
    result = SimpleNamespace(
        trajectory=[{"type": "verifier", "outcome": outcome, "observed_cost_usd": 0.1}, *failed["trajectory"]],
        exit_code=1,
        residue=[],
        stdout="",
        stderr="",
    )
    monkeypatch.setattr("recipes.meta_harness.examples.terminal_bench.campaign.run_episode", lambda *a, **kw: result)
    backend = SimpleNamespace(
        plan={"agent_failure_policy": policy.PROTOCOL},
        _descriptor=None,
        _binary=None,
        _episode_timeout_s=28800,
        _executor=None,
        _forbid_residue=True,
    )
    row = ORIGINAL_EPISODE(backend, {}, {"id": "a", "task": "terminal-bench/a", "attempt": 1})
    assert row["valid"] is False and row["reward"] is None and row["exit_code"] == 1
    assert policy.score(row, policy.PROTOCOL) == 0


def errexit_failure(failed):
    row = copy.deepcopy(failed)
    command = 'set -e\nactual=bad\ntest "$actual" = good\n'
    step = row["trajectory"][0]
    step["tool_calls"][0]["arguments"]["keystrokes"] = command
    step["observation"]["results"][0][
        "content"
    ] = '/app $ set -e\n/app $ actual=bad\n/app $ test "$actual" = good\nasciinema: recording finished\n'
    return row


def test_errexit_requires_explicit_v2_and_keeps_original_raw_evidence(failed):
    row = errexit_failure(failed)
    original = copy.deepcopy(row)
    assert not policy.eligible(policy.annotate(row, policy.PROTOCOL), policy.PROTOCOL)
    result = policy.annotate(row, policy.ERREXIT_PROTOCOL)
    assert policy.score(result, policy.ERREXIT_PROTOCOL) == 0
    assert not policy.eligible(result, policy.PROTOCOL)
    assert result["reward"] is None and not result["valid"] and result["cost_usd"] == 0.1
    assert row == original
    direct_v1 = policy.annotate(failed, policy.PROTOCOL)
    assert policy.score(direct_v1, policy.ERREXIT_PROTOCOL) == 0


@pytest.mark.parametrize(
    "change",
    ["subshell", "quoted", "disabled", "no_prompt", "no_recording_end", "no_tmux_error", "verifier", "unknown_cost"],
)
def test_errexit_policy_requires_observed_agent_shell_termination(failed, change):
    row = errexit_failure(failed)
    call = row["trajectory"][0]["tool_calls"][0]["arguments"]
    observation = row["trajectory"][0]["observation"]["results"][0]
    if change == "subshell":
        call["keystrokes"] = "bash -c 'set -e; false'\n"
    elif change == "quoted":
        call["keystrokes"] = "echo 'set -e'\n"
    elif change == "disabled":
        call["keystrokes"] = "set -e\nset +e\nfalse\n"
    elif change == "no_prompt":
        observation["content"] = observation["content"].replace("/app $ set -e", "echo set -e")
    elif change == "no_recording_end":
        observation["content"] = "/app $ set -e\n/app $"
    elif change == "no_tmux_error":
        row["error"] = "stream failed"
    elif change == "verifier":
        row["timing"]["verifier"] = {"started_at": "now"}
    else:
        row["cost_usd"] = None
    assert not policy.eligible(policy.annotate(row, policy.ERREXIT_PROTOCOL), policy.ERREXIT_PROTOCOL)
