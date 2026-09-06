import copy
import dataclasses
import hashlib
import json

import pytest

from recipes.meta_harness.examples.terminal_bench import agent_failure_policy as policy
from recipes.meta_harness.examples.terminal_bench import terminal_evidence as terminal
from recipes.meta_harness.examples.terminal_bench.campaign import CAMPAIGN_STATE_KEY, TerminalBenchBackend

from .test_meta_harness_agent_failure_policy import failed  # noqa: F401
from .test_meta_harness_driver import campaign, finish, open_campaign  # noqa: F401
from .test_meta_harness_health import raw_trial

COMMAND = (
    "set -e; cd /app; test -s /app/model.bin; ./checker test /app/model.bin /app/data.txt 1; "
    'size=$(stat -c%s /app/model.bin); echo "bytes=$size megabytes=$(awk "BEGIN{print $size/1024/1024}")"; '
    'test "$size" -lt 150'
)
PANE = "\x1b[?2004h/app $ " + COMMAND + "\n\x1b[?2004l\nN\t10\nbytes=380 megabytes=0.00036\n"


def pane_record(text=PANE):
    return {
        "protocol": terminal.PROTOCOL,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "truncated": False,
    }


@pytest.fixture
def pane_failure(failed):  # noqa: F811
    row = copy.deepcopy(failed)
    # The failing batch is absent from ATIF: Harbor writes it only after all
    # its commands complete. The pane contains the actual submitted command.
    row["trajectory"] = [{"source": "agent", "step_id": 28, "tool_calls": [], "observation": {"results": []}}]
    row["terminal_evidence"] = pane_record()
    return row


def test_v4_requires_explicit_policy_and_revalidates_terminal_evidence(pane_failure):
    original = copy.deepcopy(pane_failure)
    for previous in policy.SUPPORTED[: policy.SUPPORTED.index(policy.PANE_PROTOCOL)]:
        assert not policy.eligible(policy.annotate(pane_failure, previous), previous)
    scored = policy.annotate(pane_failure, policy.PANE_PROTOCOL)
    assert policy.score(scored, policy.PANE_PROTOCOL) == 0
    assert scored["valid"] is False and scored["reward"] is None and scored["cost_usd"] == 0.1
    assert scored["benchmark_failure"]["observed_integer"] == 380
    assert scored["benchmark_failure"]["required_less_than"] == 150
    assert pane_failure == original
    scored["terminal_evidence"] = pane_record(PANE.replace("bytes=380", "bytes=80"))
    with pytest.raises(ValueError, match="terminal evidence"):
        policy.score(scored, policy.PANE_PROTOCOL)


@pytest.mark.parametrize(
    "text",
    [
        PANE.replace("set -e;", "set +e;"),
        PANE.replace("set -e; ", "set -e; set +e; "),
        PANE.replace("set -e; ", "set -e; trap ':' ERR; "),
        PANE.replace("set -e; ", "set -e; eval 'set +e'; "),
        PANE.replace("set -e; ", "set -e; source /app/setup; "),
        PANE.replace("set -e; ", "if false; then set -e; "),
        PANE.replace("set -e;", "(set -e;"),
        PANE.replace('test "$size" -lt 150', 'test "$size" -lt 150 || true'),
        PANE.replace('test "$size" -lt 150', 'test "$size" -lt 150 &'),
        PANE.replace('test "$size" -lt 150', 'test "$different" -lt 150'),
        PANE.replace('echo "bytes=$size', 'echo "bytes=$different'),
        PANE.replace("bytes=380", "bytes=80"),
        PANE.replace("bytes=380", "other=380"),
        PANE + "/app $ \n",
        PANE + "Killed\n",
        PANE + "asciinema: recording finished\n",
        PANE + "\x1b[?2004h/app $ echo later\n\x1b[?2004l\nlater\n",
        PANE.replace("\x1b[?2004l", ""),
        PANE.replace("\x1b[?2004h", ""),
    ],
)
def test_ambiguous_or_nonfatal_terminal_tails_are_not_scored(text):
    assert policy._pane_size_check(pane_record(text)) is None


def test_terminal_wrap_and_equal_boundary_remain_a_failed_less_than_check():
    text = PANE.replace("/app/model.bin);", "/app/mo\ndel.bin);").replace("bytes=380", "bytes=150")
    assert policy._pane_size_check(pane_record(text))["observed_integer"] == 150


@pytest.mark.parametrize("change", ["unknown_cost", "transport", "verifier_ran", "residue", "other_error"])
def test_pane_does_not_admit_an_infrastructure_or_unpriced_failure(pane_failure, change):
    if change == "unknown_cost":
        pane_failure["cost_usd"] = None
    elif change == "transport":
        pane_failure["phase"] = "episode_failure"
    elif change == "verifier_ran":
        pane_failure["timing"]["verifier"] = {"started_at": "now"}
    elif change == "residue":
        pane_failure["residue"] = ["unremoved"]
    else:
        pane_failure["error"] = "remote process killed"
    assert policy.evidence(pane_failure, policy.PANE_PROTOCOL) is None


def staged_episode(tmp_path, *, valid=False):
    sessions = tmp_path / "terminus/sessions"
    trial = tmp_path / "terminus/trials/trials/task"
    sessions.mkdir(parents=True)
    (trial / "agent").mkdir(parents=True)
    summary = sessions / "trial.json"
    summary.write_text(json.dumps({"outcome": {"valid": valid, "reward": None, "cost_usd": 0.1}}))
    raw = trial / "result.json"
    raw.write_text(json.dumps(raw_trial(cost=0.1, reward=None, exception="RuntimeError")))
    pane = trial / "agent/terminus_2.pane"
    pane.write_text(PANE)
    return sessions, summary, raw, pane


def test_shared_executor_retains_redacted_pane_without_changing_raw_artifacts(tmp_path):
    sessions, summary, raw, pane = staged_episode(tmp_path)
    pane.write_text("provider=private-provider-key\n" + PANE)
    before = {p: p.read_bytes() for p in (raw, pane)}
    terminal.preserve_terminal_evidence(tmp_path, [sessions], {"OPENAI_API_KEY": "private-provider-key"})
    result = json.loads(summary.read_text())
    evidence = result["outcome"].pop("terminal_evidence")
    assert result["outcome"] == {"valid": False, "reward": None, "cost_usd": 0.1}
    assert "private-provider-key" not in evidence["text"] and "[REDACTED]" in evidence["text"]
    assert evidence["text_sha256"] == hashlib.sha256(evidence["text"].encode()).hexdigest()
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("case", ["not_writable", "valid", "missing_pane", "multiple_results"])
def test_shared_preservation_does_not_rewrite_inapplicable_summaries(tmp_path, case):
    sessions, summary, raw, pane = staged_episode(tmp_path, valid=case == "valid")
    if case == "missing_pane":
        pane.unlink()
    elif case == "multiple_results":
        extra = raw.parent.parent / "other"
        extra.mkdir()
        (extra / "result.json").write_text(raw.read_text())
    before = summary.read_bytes()
    terminal.preserve_terminal_evidence(tmp_path, [] if case == "not_writable" else [sessions], {})
    assert summary.read_bytes() == before


def test_terminal_evidence_is_bounded_and_rejects_symlinks(tmp_path):
    path = tmp_path / "pane"
    path.write_bytes(b"x" * (terminal.MAX_BYTES + 200))
    result = terminal.read_terminal_evidence(path)
    assert result["truncated"] and len(result["text"].encode()) == terminal.MAX_BYTES
    link = tmp_path / "linked"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular collected file"):
        terminal.read_terminal_evidence(link)


def test_raw_upstream_trial_and_pane_are_preserved_when_v4_projects_zero(tmp_path, pane_failure):
    (tmp_path / "agent").mkdir()
    raw = raw_trial(cost=0.1, reward=None, exception="RuntimeError")
    raw["exception_info"]["exception_message"] = pane_failure["error"]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(raw))
    (tmp_path / "agent/trajectory.json").write_text(json.dumps({"steps": pane_failure["trajectory"]}))
    pane = tmp_path / "agent/terminus_2.pane"
    pane.write_text(PANE)
    before = [p.read_bytes() for p in (path, pane)]
    row = policy.read_trial(path, policy.PANE_PROTOCOL)
    assert policy.score(row, policy.PANE_PROTOCOL) == 0
    assert [p.read_bytes() for p in (path, pane)] == before


def test_shared_e2b_download_passes_pane_through_the_normal_reef_reader(tmp_path, monkeypatch, pane_failure):
    import io
    import tarfile

    pytest.importorskip("e2b", reason="install the Terminal-Bench example dependencies")
    from e2b import Sandbox

    from recipes.meta_harness.examples.terminal_bench.e2b_executor import E2BEpisodeExecutor
    from reef.harness.trajectory import read_terminus_atif

    from .test_meta_harness_e2b import SandboxDouble

    sandbox = SandboxDouble()
    raw = json.dumps(raw_trial(cost=0.1, reward=None, exception="RuntimeError")).encode()
    outcome = {k: v for k, v in pane_failure.items() if k not in ("trajectory", "terminal_evidence")}
    content = {
        "terminus/sessions/trial.json": json.dumps(
            {"outcome": outcome, "observed_cost_usd": 0.1, "steps": []}
        ).encode(),
        "terminus/trials/a/result.json": raw,
        "terminus/trials/a/agent/terminus_2.pane": PANE.encode(),
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in content.items():
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
    sandbox.evidence = stream.getvalue()
    monkeypatch.setattr(Sandbox, "create", lambda *args, **kwargs: sandbox)
    workspace, sessions, trials = (tmp_path / name for name in ("workspace", "terminus/sessions", "terminus/trials"))
    for path in (workspace, sessions, trials):
        path.mkdir(parents=True)
    executor = E2BEpisodeExecutor("snapshot:v1", hashlib.sha256(b"manifest").hexdigest())
    executor.launch(
        ["reef-terminus-e2b", "--task", "terminal-bench/a"],
        root=tmp_path,
        workspace=workspace,
        env={"E2B_API_KEY": "SECRET"},
        timeout=600,
        writable_paths=[sessions, trials],
    )
    verifier = read_terminus_atif(sessions)[0]
    row = verifier["outcome"]
    assert row["terminal_evidence"] == pane_record()
    assert policy.score(policy.annotate(row, policy.PANE_PROTOCOL), policy.PANE_PROTOCOL) == 0
    assert (trials / "a/result.json").read_bytes() == raw
    assert sandbox.killed and sum("isolated_runner" in command for command, _ in sandbox.calls) == 1


def test_reef_commits_terminal_evidence_and_zero_once_without_retry(campaign, pane_failure, monkeypatch):  # noqa: F811
    recipe, factory, calls, prompts = campaign
    tasks = ("terminal-bench/a", "terminal-bench/b", "terminal-bench/c")
    recipe = dataclasses.replace(
        recipe,
        tasks=tasks,
        episode_workers=1,
        plan={
            **recipe.plan,
            "tasks": list(tasks),
            "concurrency": 1,
            "max_attempts": 1,
            "agent_failure_policy": policy.PANE_PROTOCOL,
        },
    )
    original = TerminalBenchBackend._episode

    def episode(self, files, identity):
        row = original(self, files, identity)
        if "verify" in files.get("terminus/AGENTS.md", "") and identity["task"].endswith("/a"):
            row = policy.annotate({**row, **pane_failure, "cost_usd": 0.02}, policy.PANE_PROTOCOL)
        return row

    monkeypatch.setattr(TerminalBenchBackend, "_episode", episode)
    dispatcher, scenario = open_campaign(recipe, factory)
    try:
        final = finish(scenario)
        assert final["status"] == "complete" and len(calls) == 6 and len(prompts) == 1
        row = next(r for r in final["records"].values() if r.get("benchmark_failure"))
        assert row["terminal_evidence"] == pane_record() and row["attempt"] == 1
        assert not row["valid"] and row["reward"] is None
        assert final["rounds"][-1]["mean"] == pytest.approx(2 / 3)
        assert scenario.trainer.state[CAMPAIGN_STATE_KEY] == final
    finally:
        dispatcher.close()
