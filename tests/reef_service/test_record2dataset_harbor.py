"""A designer reply as a Harbor task, the authoring gate, and the oracle check through the harbor command line."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

from reef.core.tasks import read_harbor_task, write_harbor_task
from reef.record2dataset import GeneratedHarborTask, HarborReply, OracleUnavailable, harbor_task, oracle_check
from reef.record2dataset.harbor import (
    HarborRuns,
    content_hash,
    dockerfile_logical_lines,
    dockerfile_parse_errors,
    missing_copy_sources,
    reply_errors,
    run_harbor_agent,
)

REPLY = HarborReply(
    instruction=(
        "A service on this machine writes the port it listens on under /var/run. Find that file and write the "
        "port number, and nothing else, to /workspace/port.txt.\n"
    ),
    environment={
        "Dockerfile": "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y tmux && echo 8471 > /var/run/app.port\nWORKDIR /workspace\n"
    },
    tests={
        "test.sh": (
            "#!/bin/sh\nmkdir -p /logs/verifier\n"
            'test "$(cat /workspace/port.txt 2>/dev/null)" = 8471 && echo 1 > /logs/verifier/reward.txt '
            "|| echo 0 > /logs/verifier/reward.txt\n"
        )
    },
    solution={"solve.sh": "#!/bin/sh\ncat /var/run/app.port > /workspace/port.txt\n"},
    hint="Look under /var/run for what the service left behind.",
)


def generated(**overrides: object) -> GeneratedHarborTask:
    fields: dict[str, object] = {
        "reply": REPLY,
        "skill": "inspection",
        "generation": 3,
        "index": 1,
        "source_record_id": "rec-designer-3",
        "step": 9,
        "difficulty": "easy",
    }
    fields.update(overrides)
    return GeneratedHarborTask(**fields)  # type: ignore[arg-type]


FAKE_HARBOR = '''#!{python}
"""A stand in for the harbor command line: writes the reward its agent name earns under the jobs directory."""
import json, sys
from pathlib import Path

arguments = sys.argv[1:]
agent = arguments[arguments.index("-a") + 1]
jobs = Path(arguments[arguments.index("-o") + 1])
script = json.loads(Path(__file__).with_suffix(".json").read_text())
exit_code = script.get("exits", {{}}).get(agent)
if exit_code is not None:
    sys.stderr.write("docker daemon is not running")
    sys.exit(exit_code)
if script.get("is_ignoring_term"):
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if script.get("ready_path"):
    Path(script["ready_path"]).touch()
if script.get("sleep_s"):
    import time
    time.sleep(script["sleep_s"])
trial = jobs / "job" / "trial"
trial.mkdir(parents=True)
reward = script["rewards"].get(agent)
exception = script.get("exceptions", {{}}).get(agent)
failure = script.get("agent_failures", {{}}).get(agent)
if failure:
    (trial / "agent").mkdir()
    (trial / "agent" / "exit-code.txt").write_text(str(failure["exit_code"]) + "\\n")
    (trial / "agent" / "oracle.txt").write_text(failure["output"])
(jobs / "job" / "result.json").write_text(json.dumps({{"stats": {{"n_trials": 1}}}}))
result = {{
    "task_name": "t",
    "verifier_result": {{"rewards": {{"reward": reward}}}} if reward is not None else None,
    "exception_info": {{"exception_type": "AgentTimeoutError", "exception_message": exception}} if exception else None,
}}
(trial / "result.json").write_text(json.dumps(result))
'''


def fake_harbor(tmp_path: Path, rewards: dict[str, float | None], **script: object) -> str:
    path = tmp_path / "harbor"
    path.write_text(FAKE_HARBOR.format(python=sys.executable))
    path.with_suffix(".json").write_text(json.dumps({"rewards": rewards, **script}))
    path.chmod(0o755)
    return str(path)


# ----------------------------------------------------------------------------------------------- the task


def test_the_task_carries_the_files_the_hint_beside_the_solution_and_the_metadata(tmp_path: Path) -> None:
    task = harbor_task(generated())
    assert task.name == "harbor-00003-001-inspection"
    root = write_harbor_task(task, tmp_path)
    assert (root / "instruction.md").read_text() == REPLY.instruction
    assert (root / "environment" / "Dockerfile").read_text() == REPLY.environment["Dockerfile"]
    assert (root / "tests" / "test.sh").read_text() == REPLY.tests["test.sh"]
    assert (root / "solution" / "solve.sh").read_text() == REPLY.solution["solve.sh"]
    assert (root / "solution" / "hint.txt").read_text() == REPLY.hint + "\n"
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["metadata"] == {
        "skill": "inspection",
        "generation": 3,
        "step": 9,
        "index": 1,
        "difficulty": "easy",
        "reef": {"digest": task.digest, "source_agent_record_ids": ["rec-designer-3"]},
    }
    assert document["environment"]["network_mode"] == "no-network" and document["agent"]["timeout_sec"] == 900
    assert document["verifier"] == {"timeout_sec": 300, "user": "root"}, "the verifier reads root only state"
    assert read_harbor_task(root) == task


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reply": "files"}, "HarborReply"),
        ({"skill": "Inspection"}, "skill"),
        ({"generation": -1}, "generation must be"),
        ({"source_record_id": ""}, "source_record_id"),
        ({"difficulty": ""}, "difficulty"),
    ],
)
def test_a_bad_harbor_task_is_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        harbor_task(generated(**overrides))


def test_a_reply_without_a_hint_writes_no_hint_file() -> None:
    task = harbor_task(generated(reply=reply_with(hint="")))
    assert "hint.txt" not in task.solution and task.solution["solve.sh"] == REPLY.solution["solve.sh"]


def test_a_bad_agent_timeout_is_refused() -> None:
    with pytest.raises(ValueError, match="agent_timeout_s"):
        harbor_task(generated(), agent_timeout_s=0)


# ----------------------------------------------------------------------------------------------- the structural gate


def reply_with(**changes: object) -> HarborReply:
    fields = {
        "instruction": REPLY.instruction,
        "environment": dict(REPLY.environment),
        "tests": dict(REPLY.tests),
        "solution": dict(REPLY.solution),
        "hint": REPLY.hint,
    }
    fields.update(changes)
    return HarborReply(**fields)  # type: ignore[arg-type]


def test_a_substantive_reply_has_no_errors() -> None:
    assert reply_errors(REPLY) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"instruction": "Do it.\n"}, "fewer than 80 characters"),
        ({"environment": {"Dockerfile": "FROM python:3.12-slim\n"}}, "needs FROM and one of RUN, COPY, ADD or ENV"),
        (
            {"environment": {"Dockerfile": "FROM python:3.12-slim\nRUN echo 8471 > /var/run/app.port\n"}},
            "does not install tmux",
        ),
        ({"environment": {"Dockerfile": "RUN echo 1\n"}}, "needs FROM"),
        (
            {
                "environment": {
                    "Dockerfile": "FROM python:3.12-slim\nRUN cat > /app/c.toml << 'EOF'\n[database]\nhost=db\nEOF\n"
                }
            },
            "heredoc body",
        ),
        ({"environment": {"Dockerfile": "FROM python:3.12-slim\n[database]\n"}}, "Dockerfile: [database]"),
        ({"solution": {"solve.sh": "#!/bin/sh\n# nothing yet\n"}}, "solve.sh has no command"),
        ({"tests": {"test.sh": "#!/bin/sh\necho 1\n"}}, "no file under tests/ names /logs/verifier/reward.txt"),
        (
            {"solution": {"solve.sh": "#!/bin/sh\n# Use this file to solve the task\nls\n"}},
            "untouched scaffold line in solution/solve.sh",
        ),
        (
            {
                "environment": {
                    "Dockerfile": "FROM python:3.12-slim\n# Install or copy over any environment dependencies here\nRUN true\n"
                }
            },
            "untouched scaffold line in Dockerfile",
        ),
        (
            {
                "environment": {
                    "Dockerfile": "FROM python:3.12-slim\nRUN apt-get install -y tmux\nCOPY app.conf /etc/app.conf\n"
                }
            },
            "COPY 'app.conf' names no file under environment/",
        ),
        (
            {
                "environment": {
                    "Dockerfile": "FROM python:3.12-slim\nRUN apt-get install -y tmux\nCOPY environment/app.conf /etc/\n",
                    "app.conf": "port=1\n",
                }
            },
            "COPY 'environment/app.conf' names no file under environment/",
        ),
        (
            {
                "environment": {
                    "Dockerfile": "FROM python:3.12-slim\nRUN apt-get install -y tmux\nADD --chown=1:1 logs/ /var/log/app\n",
                    "app.conf": "port=1\n",
                }
            },
            "COPY 'logs/' names no file",
        ),
    ],
)
def test_a_reply_that_is_not_a_substantive_task_is_refused(changes: dict[str, object], message: str) -> None:
    errors = reply_errors(reply_with(**changes))
    assert any(message in error for error in errors), errors
    with pytest.raises(ValueError, match="not a substantive task"):
        harbor_task(generated(reply=reply_with(**changes)))


def test_copy_sources_the_environment_holds_pass_the_gate() -> None:
    dockerfile = (
        "FROM python:3.12-slim\nRUN apt-get install -y tmux\n"
        "COPY app.conf /etc/app.conf\nCOPY ./logs/ /var/log/app/\nCOPY --chown=1:1 *.conf scripts /opt/\n"
        'COPY ["app.conf", "/etc/copy.conf"]\nCOPY . /src\nCOPY --from=builder /built /opt/built\n'
        "ADD https://example.invalid/x.tar /opt/x\n"
    )
    environment = {
        "Dockerfile": dockerfile,
        "app.conf": "port=1\n",
        "logs/app.log": "ok\n",
        "scripts/run.sh": "true\n",
    }
    assert reply_errors(reply_with(environment=environment)) == []
    assert missing_copy_sources(dockerfile, environment) == []
    assert missing_copy_sources("COPY a.txt b.txt /dst\nCOPY --chmod=644 c/ /c\n", {"b.txt": ""}) == ["a.txt", "c/"]


def test_the_classic_parser_view_joins_continuations_and_drops_comments() -> None:
    text = "FROM python:3.12-slim\n# a note\nCOPY a \\\n  # between\n  /a\nRUN true \\\n\n"
    assert dockerfile_logical_lines(text) == ["FROM python:3.12-slim", "COPY a /a", "RUN true"]


def test_the_scaffold_sentences_are_looked_for_in_their_own_files_only() -> None:
    quoting = reply_with(
        instruction=REPLY.instruction + "The stub says 'Use this file to solve the task'; replace that stub.\n"
    )
    assert reply_errors(quoting) == []


def test_a_verifier_that_writes_reward_json_is_accepted() -> None:
    json_verifier = reply_with(
        tests={"test.sh": "#!/bin/sh\nmkdir -p /logs/verifier\necho '{\"reward\": 1}' > /logs/verifier/reward.json\n"}
    )
    assert reply_errors(json_verifier) == []


def test_the_classic_parser_lint_joins_continuations_and_skips_comments() -> None:
    text = (
        "FROM python:3.12-slim\n# a note\nRUN apt-get update \\\n  # between\n && apt-get install -y sudo \\\n"
        " && rm -rf /var/lib/apt/lists/*\nCOPY a b\n"
    )
    assert dockerfile_parse_errors(text) == []
    assert dockerfile_parse_errors("FROM x\necho hi\n") == ["echo hi"]


def test_the_content_hash_ignores_the_hint_and_the_metadata_but_not_the_files() -> None:
    first = harbor_task(generated())
    same_files = harbor_task(generated(index=2, reply=reply_with(hint="Another hint.")))
    other_files = harbor_task(generated(reply=reply_with(instruction=REPLY.instruction + "Hurry.\n")))
    assert content_hash(first) == content_hash(same_files) != content_hash(other_files)
    assert len(content_hash(first)) == 16


def test_a_category_from_the_vocabulary_lands_in_the_metadata_with_tags() -> None:
    task = harbor_task(generated(category="system_administration"))
    assert task.metadata["category"] == "system_administration"
    assert task.metadata["tags"] == ["system_administration", "inspection"]
    with pytest.raises(ValueError, match="category must be one of"):
        generated(category="cooking")


# ----------------------------------------------------------------------------------------------- the oracle check


def test_partial_credit_for_doing_nothing_is_accepted_as_long_as_it_is_below_one(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    result = oracle_check(root, harbor=fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.3}))
    assert result.is_solvable and result.nop_reward == 0.3


def test_a_task_the_oracle_solves_and_the_nop_agent_does_not_is_solvable(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    result = oracle_check(root, harbor=fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0}))
    assert result.is_solvable and (result.oracle_reward, result.nop_reward) == (1.0, 0.0)
    assert sorted(p.name for p in (tmp_path / "tasks" / ".harbor-jobs" / root.name).iterdir()) == ["nop", "oracle"]


@pytest.mark.parametrize(
    ("rewards", "reason"),
    [
        ({"oracle": 0.0, "nop": 0.0}, "the reference solution scored 0.0, not 1"),
        ({"oracle": None, "nop": 0.0}, "the reference solution scored None, not 1"),
        ({"oracle": 1.0, "nop": 1.0}, "doing nothing scored 1.0, not below 1"),
        ({"oracle": 1.0, "nop": None}, "doing nothing scored None, not below 1"),
    ],
)
def test_a_task_that_is_unsolvable_or_free_is_refused_with_the_reason(tmp_path: Path, rewards, reason: str) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    result = oracle_check(root, harbor=fake_harbor(tmp_path, rewards))
    assert not result.is_solvable and result.reason == reason


def test_a_trial_that_ended_in_an_exception_says_so_in_the_reason(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    harbor = fake_harbor(tmp_path, {"oracle": None, "nop": 0.0}, exceptions={"oracle": "agent timed out after 900 s"})
    result = oracle_check(root, harbor=harbor)
    assert not result.is_solvable
    assert (
        result.reason == "the reference solution scored None, not 1; the trial ended with agent timed out after 900 s"
    )


def test_a_failed_oracle_run_ends_the_check_before_the_nop_agent_runs(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    result = oracle_check(root, harbor=fake_harbor(tmp_path, {"oracle": 0.0, "nop": 0.0}))
    assert not result.is_solvable and result.oracle_reward == 0.0 and result.nop_reward is None
    jobs = root.parent / ".harbor-jobs" / root.name
    assert (jobs / "oracle").is_dir() and not (jobs / "nop").exists(), "the nop run would only cost a build"


def test_a_build_failure_reaches_the_reason_as_its_last_lines(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    log = "Docker compose command failed\n" + "\n".join(f"#{n} [1/9] RUN apt-get update" for n in range(40))
    log += "\nE: Unable to locate package tmuxx\nERROR: failed to solve: process did not complete successfully"
    harbor = fake_harbor(tmp_path, {"oracle": None, "nop": 0.0}, exceptions={"oracle": log})
    result = oracle_check(root, harbor=harbor)
    assert not result.is_solvable and result.reason.endswith("process did not complete successfully")
    assert "Unable to locate package tmuxx" in result.reason and len(result.reason) < 400


def test_a_reference_solution_that_fails_names_its_exit_code_and_last_line(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    harbor = fake_harbor(
        tmp_path,
        {"oracle": 0.0, "nop": 0.0},
        agent_failures={
            "oracle": {
                "exit_code": 2,
                "output": "starting\n/solution/solve.sh: 2: cannot create /workspace/port.txt: Directory nonexistent\n",
            }
        },
    )
    result = oracle_check(root, harbor=harbor)
    assert result.reason == (
        "the reference solution scored 0.0, not 1; solve.sh exited 2: "
        "/solution/solve.sh: 2: cannot create /workspace/port.txt: Directory nonexistent"
    )


def test_a_harbor_run_that_hangs_is_stopped_at_the_timeout(tmp_path: Path) -> None:
    import time

    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    started = time.monotonic()
    harbor = fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0}, sleep_s=30)
    with pytest.raises(OracleUnavailable, match="harbor run -a oracle did not finish within 1 s"):
        oracle_check(root, harbor=harbor, timeout_s=1.0)
    assert time.monotonic() - started < 10.0


def harbor_run_on_a_thread(tmp_path: Path, **script: object) -> tuple[threading.Thread, HarborRuns, dict[str, str]]:
    """A sleeping fake harbor under ``run_harbor_agent`` on its own thread, the way a job runs it, once it is up."""
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    ready_path = tmp_path / "ready"
    harbor = fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0}, sleep_s=60, ready_path=str(ready_path), **script)
    runs = HarborRuns()
    outcome: dict[str, str] = {}

    def run() -> None:
        try:
            run_harbor_agent(root, "oracle", root.parent / "jobs", harbor=harbor, timeout_s=60.0, runs=runs)
        except RuntimeError as exc:
            outcome["error"] = str(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 10.0
    while (not runs.pids() or not ready_path.exists()) and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread, runs, outcome


def test_terminate_all_ends_the_harbor_run_in_flight_and_refuses_a_new_one(tmp_path: Path) -> None:
    thread, runs, outcome = harbor_run_on_a_thread(tmp_path)
    (pid,) = runs.pids()
    started = time.monotonic()
    assert runs.terminate_all(grace_s=5.0) == (pid,)
    thread.join(timeout=5.0)
    assert not thread.is_alive() and time.monotonic() - started < 5.0
    assert outcome["error"] == "harbor run -a oracle was stopped with the generator"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert runs.pids() == ()
    # A check the stopping generator refuses to start could not run: an error, never a refused task.
    with pytest.raises(OracleUnavailable, match="the generator is stopping; no new harbor run"):
        oracle_check(tmp_path / "tasks" / "harbor-00003-001-inspection", harbor=str(tmp_path / "harbor"), runs=runs)


def test_a_harbor_run_that_ignores_the_term_is_killed_after_the_grace(tmp_path: Path) -> None:
    thread, runs, outcome = harbor_run_on_a_thread(tmp_path, is_ignoring_term=True)
    started = time.monotonic()
    runs.terminate_all(grace_s=0.5)
    thread.join(timeout=5.0)
    assert not thread.is_alive() and 0.4 <= time.monotonic() - started < 5.0
    assert outcome["error"] == "harbor run -a oracle was stopped with the generator"


def test_a_missing_harbor_command_line_raises_rather_than_refusing_the_task(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    with pytest.raises(OracleUnavailable, match="the harbor command line is not installed") as raised:
        oracle_check(root)
    assert isinstance(raised.value, RuntimeError), "a caller that catches RuntimeError still sees it"


def test_a_harbor_run_that_exits_nonzero_raises_with_its_output(tmp_path: Path) -> None:
    path = tmp_path / "harbor"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('docker is not running')\nsys.exit(2)\n")
    path.chmod(0o755)
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    with pytest.raises(OracleUnavailable, match="harbor run -a oracle exited 2: docker is not running") as raised:
        oracle_check(root, harbor=str(path))
    assert isinstance(raised.value.__cause__, RuntimeError), "the harbor run's own error is the cause"


def test_a_nop_run_that_exits_nonzero_raises_after_a_good_oracle_run(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    harbor = fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0}, exits={"nop": 3})
    with pytest.raises(OracleUnavailable, match="harbor run -a nop exited 3: docker daemon is not running"):
        oracle_check(root, harbor=harbor)
    assert (root.parent / ".harbor-jobs" / root.name / "oracle").is_dir(), "the oracle run happened first"
    stderr_path = root.parent / ".harbor-jobs" / root.name / "nop" / "harbor-stderr.txt"
    assert stderr_path.read_text() == "docker daemon is not running", "the whole stderr stays beside the run"


def test_a_reference_solution_that_scores_zero_is_a_refusal_not_an_error(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    result = oracle_check(root, harbor=fake_harbor(tmp_path, {"oracle": 0.0, "nop": 0.0}))
    assert not result.is_solvable and result.reason == "the reference solution scored 0.0, not 1"
    for_free = oracle_check(root, harbor=fake_harbor(tmp_path, {"oracle": 1.0, "nop": 1.0}))
    assert not for_free.is_solvable and for_free.reason == "doing nothing scored 1.0, not below 1"


def test_a_second_check_starts_from_a_clean_jobs_directory(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    harbor = fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0})
    assert oracle_check(root, harbor=harbor).is_solvable
    assert oracle_check(root, harbor=harbor).is_solvable
