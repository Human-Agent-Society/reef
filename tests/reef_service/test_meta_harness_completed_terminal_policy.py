"""Scoring a completed terminal loss does not establish its cause."""

import copy
import json

import pytest
from tests.reef_service.test_meta_harness_health import raw_trial

from recipes.meta_harness.examples.terminal_bench import agent_failure_policy as policy


def completed_loss():
    return {
        "valid": False,
        "reward": None,
        "phase": "execution_failure",
        "exception_type": "RuntimeError",
        "error": "tmux send-keys failed: no server running on /tmp/tmux-0/default",
        "agent_execution_started": True,
        "cost_usd": 0.1,
        "residue": [],
        "started_at": "2026-09-05T00:00:00Z",
        "finished_at": "2026-09-05T00:02:01Z",
        "timing": {
            "verifier": None,
            "agent_execution": {"started_at": "2026-09-05T00:00:01Z", "finished_at": "2026-09-05T00:02:00Z"},
        },
        "trajectory": [
            {
                "source": "agent",
                "step_id": 209,
                "tool_calls": [],
                "observation": {"results": [{"content": "New Terminal Output:\n/app $"}]},
            }
        ],
    }


def test_completed_terminal_loss_is_zero_without_inventing_causation():
    row = completed_loss()
    original = copy.deepcopy(row)
    for old in policy.SUPPORTED[:-1]:
        assert policy.score(policy.annotate(row, old), old) is None
    scored = policy.annotate(row, policy.COMPLETED_PROTOCOL)
    assert policy.score(scored, policy.COMPLETED_PROTOCOL) == 0
    assert scored["benchmark_failure"]["attribution"] == "unresolved"
    assert scored["benchmark_failure"]["causal_evidence"] is None
    assert scored["valid"] is False and scored["reward"] is None
    assert {k: scored[k] for k in original} == original == row


@pytest.mark.parametrize(
    "change",
    [
        "no_start",
        "no_finish",
        "no_agent_finish",
        "bad_time",
        "naive_time",
        "reverse_time",
        "agent_outside_trial",
        "no_trace",
        "unknown_bill",
        "nan_bill",
        "bool_bill",
        "transport",
        "transport_diagnostics",
        "cleanup_diagnostics",
        "residue",
        "not_started",
        "verifier",
        "reward",
        "other_error",
        "other_exception",
        "bad_terminal_hash",
        "bad_terminal_protocol",
    ],
)
def test_incomplete_unaccounted_or_different_failures_remain_unscored(change):
    row = completed_loss()
    if change == "no_start":
        row.pop("started_at")
    elif change == "no_finish":
        row.pop("finished_at")
    elif change == "no_agent_finish":
        row["timing"]["agent_execution"].pop("finished_at")
    elif change == "bad_time":
        row["finished_at"] = "not a timestamp"
    elif change == "naive_time":
        row["finished_at"] = "2026-09-05T00:02:01"
    elif change == "reverse_time":
        row["finished_at"] = "2026-09-04T00:00:00Z"
    elif change == "agent_outside_trial":
        row["timing"]["agent_execution"]["started_at"] = "2026-09-04T00:00:00Z"
    elif change == "no_trace":
        row["trajectory"] = []
    elif change == "unknown_bill":
        row["cost_usd"] = None
    elif change == "nan_bill":
        row["cost_usd"] = float("nan")
    elif change == "bool_bill":
        row["cost_usd"] = True
    elif change == "transport":
        row["phase"] = "missing_runner_summary"
    elif change == "transport_diagnostics":
        row["transport_diagnostics"] = {"phase": "download_failed"}
    elif change == "cleanup_diagnostics":
        row["e2b_diagnostics"] = {"cleanup_confirmed": False}
    elif change == "residue":
        row["residue"] = ["unconfirmed sandbox"]
    elif change == "not_started":
        row["agent_execution_started"] = False
    elif change == "verifier":
        row["timing"]["verifier"] = {"started_at": "now"}
    elif change == "reward":
        row["reward"] = 1
    elif change == "other_error":
        row["error"] = "model API rate limited"
    elif change == "other_exception":
        row["exception_type"] = "TimeoutException"
    else:
        from tests.reef_service.test_meta_harness_terminal_evidence import pane_record

        row["terminal_evidence"] = pane_record()
        row["terminal_evidence"]["text_sha256" if change == "bad_terminal_hash" else "protocol"] = "invalid"
    assert policy.score(policy.annotate(row, policy.COMPLETED_PROTOCOL), policy.COMPLETED_PROTOCOL) is None


@pytest.mark.parametrize("change", ["finished_at", "trajectory", "terminal_evidence", "cost_usd"])
def test_completed_proof_rechecks_raw_times_trace_optional_pane_and_bill(change):
    row = policy.annotate(completed_loss(), policy.COMPLETED_PROTOCOL)
    if change == "finished_at":
        row[change] = "2026-09-05T00:03:00Z"
    elif change == "trajectory":
        row[change][0]["observation"]["results"][0]["content"] = "changed"
    elif change == "cost_usd":
        row[change] = 0.2
    else:
        from tests.reef_service.test_meta_harness_terminal_evidence import pane_record

        row[change] = pane_record()
    with pytest.raises(ValueError, match="retained terminal evidence"):
        policy.score(row, policy.COMPLETED_PROTOCOL)


def test_upstream_reader_preserves_raw_failure_and_reports_unknown_attribution(tmp_path):
    path = tmp_path / "a__r0/result.json"
    path.parent.joinpath("agent").mkdir(parents=True)
    row = completed_loss()
    raw = raw_trial(exception="RuntimeError", reward=None, cost=0.1)
    raw.update(
        started_at=row["started_at"], finished_at=row["finished_at"], agent_execution=row["timing"]["agent_execution"]
    )
    raw["exception_info"]["exception_message"] = row["error"]
    path.write_text(json.dumps(raw))
    original = path.read_bytes()
    path.parent.joinpath("agent/trajectory.json").write_text(json.dumps({"steps": row["trajectory"]}))
    health = policy.job_health(tmp_path, ["a"], policy.COMPLETED_PROTOCOL)
    assert health["valid"] and health["verified"] == 0
    assert health["agent_failures_scored_zero"] == 0
    assert health["unattributed_terminal_failures_scored_zero"] == 1
    assert path.read_bytes() == original
