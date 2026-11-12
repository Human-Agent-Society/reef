"""Claude Code as AutoEnvScaling's Designer: its run, its check command and gateway, its task's sources, and the token.

No test runs Claude Code or a sandbox: a scripted E2B session stands in for both, and runs the agent's
``task_check`` commands against the real loopback gateway.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from tests.recipes.autoenvscaling.support import (
    NAMES,
    REQUEST,
    SESSION_ID,
    TOKEN,
    ClaudeExecutor,
    NoChecks,
    ScriptedClaudeSession,
    StandInDesigner,
    autoenvscaling,
    checked,
    generating_config,
    report,
    result_json,
    run_tasks,
    serving_reef,
    source_of,
    sources,
    stand_in_e2b_package,
    state,
    state_document,
)
from tests.support.task_service_stubs import task_for

from recipes.beta.autoenvscaling import task_check
from recipes.beta.autoenvscaling.check_gateway import CheckGateway
from recipes.beta.autoenvscaling.claude_designer import CLAUDE_CHECK_TEXT, DESIGNER_TOOLS, ClaudeCodeDesigner
from recipes.beta.autoenvscaling.designer import (
    AGENT_PROMPT,
    REBUILD_RULES_TEXT,
    VARIANT_INPUTS_TEXT,
    VARIANT_PROMPT,
    AgentTools,
    CandidateChecks,
    DesignedTask,
    DesignerOutageError,
    ModelRefusedError,
    TaskVariant,
    VariantDesigner,
)
from recipes.beta.autoenvscaling.designer_credentials import CredentialChecks, CredentialHeldError, rename_holding
from recipes.beta.autoenvscaling.generator import AutoEnvScalingGenerator
from reef.core.plugins import PluginConfigError
from reef.core.tasks import write_harbor_task
from reef.core.tasks.split import TaskSplit, extend_split_by_source
from reef.execution.sandbox.e2b import E2BSession
from reef.execution.sandbox.process import ProcessOutcome
from reef.recipe.generated_tasks.generator import RolledOutTask
from reef.recipe.generated_tasks.processor import split_source_ids

# Claude Code's result text when the model provider refused the run's last model call.
SAFEGUARDS_TEXT = (
    "API Error: Opus 4.8's safeguards flagged this message. Our intentionally broad safeguards allow us to deliver "
    "more capabilities faster, but can sometimes flag legitimate cybersecurity work. Apply to the Cyber Verification "
    "Program to reduce these interruptions. Learn more: "
    "https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude\n\n"
    "Details: `[cyber]`\n\nRequest ID: req_011Cgr2aqWqyhnxgsFFCWiR8"
)


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_e2b_package(monkeypatch)


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_check, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(task_check, "WAIT_SECONDS", 0.3)


@pytest.fixture
def reef_url() -> Iterator[str]:
    """A stand-in Reef that serves every source session."""
    server, _ = serving_reef("unused")
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def cached_repositories(cache_dir: Path) -> Path:
    """The snapshots of the stand-in sources' repositories, kept as the snapshot cache keeps them: no test clones."""
    for entry in sources(2):
        repository = entry["repository"]
        name = f"{hashlib.sha256(str(repository['url']).encode()).hexdigest()[:8]}-{repository['commit']}"
        (cache_dir / name).mkdir(parents=True, exist_ok=True)
        (cache_dir / name / "README.md").write_text("the repository at the base commit\n")
    return cache_dir


def claude_code(tmp_path: Path, reef_url: str, session: E2BSession, *, max_turns: int = 40) -> ClaudeCodeDesigner:
    return ClaudeCodeDesigner(
        reef_url=reef_url,
        token="reef-token",
        claude_token=TOKEN,
        model="opus",
        executor=ClaudeExecutor(session=session),
        cache_dir=cached_repositories(tmp_path / "cache"),
        timeout_seconds=60.0,
        max_turns=max_turns,
    )


def designed_in(
    tmp_path: Path, reef_url: str, session: E2BSession, checks: CandidateChecks | None = None
) -> DesignedTask:
    """Claude Code's run on the first stand-in session, in ``tmp_path / "run"``."""
    return claude_code(tmp_path, reef_url, session).design(
        source_of(sources(1)[0]),
        tmp_path / "run",
        checks if checks is not None else NoChecks(1),
        tags={"role": "designer", "generation": "0", "index": "0"},
        stop=threading.Event(),
    )


def files_holding(directory: Path, text: str) -> list[str]:
    return sorted(str(path) for path in directory.rglob("*") if path.is_file() and text.encode() in path.read_bytes())


def holds_part_of_token(text: str) -> bool:
    """Whether ``text`` holds any 10 characters of the token in a row."""
    return any(TOKEN[start : start + 10] in text for start in range(len(TOKEN) - 9))


class CountingChecks(CandidateChecks):
    """Checks that pass at once, or wait for ``release`` when it is given; ``count`` is how many ran."""

    def __init__(self, *, release: threading.Event | None = None) -> None:
        super().__init__(6)
        self.release = release
        self.count = 0

    def check(self, task_dir: Path) -> dict[str, object]:
        self.count += 1
        if self.release is not None:
            self.release.wait(10)
        return {"check": self.count, "ran": True, "passed": True, "oracle_reward": 1.0, "nop_reward": 0.0}


@pytest.mark.unit
def test_runs_pinned_claude_headless_with_token_in_environment(tmp_path: Path, reef_url: str) -> None:
    session = ScriptedClaudeSession()
    designed = designed_in(tmp_path, reef_url, session)
    root = tmp_path / "run"
    argv, env = session.argv, session.launch_env
    assert argv[:3] == ["claude", "-p", AGENT_PROMPT]
    for option, value in (
        ("--output-format", "json"),
        ("--model", "opus"),
        ("--max-turns", "40"),
        ("--tools", DESIGNER_TOOLS),
        ("--allowedTools", DESIGNER_TOOLS),
    ):
        assert argv[argv.index(option) + 1] == value
    assert "--disable-slash-commands" in argv and "--strict-mcp-config" in argv
    # The Designer passes the token in env; the executor owns the sandbox wrapper command.
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert not any(TOKEN in value for value in argv)
    assert not any(name.startswith("ANTHROPIC_") for name in env)
    assert env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1" and env["DISABLE_ERROR_REPORTING"] == "1"
    assert env["CLAUDE_CONFIG_DIR"] == f"{root}/claude" and env["HOME"] == f"{root}/home"
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == env["BASH_MAX_TIMEOUT_MS"] == "600000"
    assert env["REEF_DESIGNER_CHECK"] == f"{root}/tools/task_check.py" and env["REEF_DESIGNER_URL"].startswith(
        "http://"
    )
    assert session.writable == [root / "claude", root / "home"]
    assert argv[argv.index("--plugin-dir") + 1] == f"{root}/plugin"
    plugin = root / "plugin"
    readonly = [
        root / "claude" / "settings.json",
        plugin / "rules.md",
        plugin / ".claude-plugin" / "plugin.json",
        plugin / "output-styles" / "reef.md",
        root / "tools" / "task_check.py",
    ]
    assert sorted(session.readonly) == sorted(readonly)
    rules = (plugin / "rules.md").read_text()
    assert REBUILD_RULES_TEXT in rules and CLAUDE_CHECK_TEXT.strip() in rules
    # No model binding: the settings point Claude Code at no other endpoint.
    assert "env" not in json.loads((root / "claude" / "settings.json").read_text())
    transcript = f"claude/projects/-home-user-reef-run-workspace/{SESSION_ID}.jsonl"
    assert designed.record_ids == () and designed.transcript == transcript and (root / transcript).is_file()
    assert "port.txt" in designed.reply
    # Of Claude Code's own files only the transcripts, the rules and the settings stay.
    assert sorted(path.name for path in (root / "claude").iterdir()) == ["projects", "settings.json"]
    assert not (root / "home").exists() and not (root / "inputs" / "repo").exists()
    assert session.is_closed.is_set()


@pytest.mark.unit
def test_task_check_script_waits_and_rechecks(tmp_path: Path) -> None:
    release = threading.Event()
    held = CountingChecks(release=release)
    tools = AgentTools(tmp_path, CredentialChecks(held, token=TOKEN))
    gateway = CheckGateway(tools)
    gateway.start()
    deadline = str(int((time.time() + 3600) * 1000))
    environ = {"REEF_DESIGNER_URL": gateway.base_url, "REEF_DESIGNER_DEADLINE_MS": deadline}
    try:
        first = checked([], {**environ, "REEF_DESIGNER_FINISH_SECONDS": "60"})
        assert "state: running\ncheck_id: 1\n" in first and "--wait 1` to wait" in first
        assert "Remaining time:" in first and "Finish now" not in first
        # Asked again while the check runs, the command waits for that one and starts no other.
        assert "check_id: 1\n" in checked([], {**environ, "REEF_DESIGNER_FINISH_SECONDS": "60"})
        release.set()
        done = checked(["--wait", "1"], {**environ, "REEF_DESIGNER_FINISH_SECONDS": "60"})
        assert '"state": "done"' in done and '"passed": true' in done and held.count == 1
        assert "check_id must name a check this run started" in checked(
            ["--wait", "7"], {**environ, "REEF_DESIGNER_FINISH_SECONDS": "60"}
        )
        # A check that ends while the command waits gives its result in the same call.
        quick = checked([], {**environ, "REEF_DESIGNER_FINISH_SECONDS": "7200"})
        assert '"check": 2' in quick and '"passed": true' in quick and "Finish now" in quick
    finally:
        gateway.stop()


@pytest.mark.unit
def test_task_check_script_stops_waiting_at_the_run_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_check, "WAIT_SECONDS", 30.0)
    release = threading.Event()
    gateway = CheckGateway(AgentTools(tmp_path, CredentialChecks(CountingChecks(release=release), token=TOKEN)))
    gateway.start()
    past = str(int((time.time() - 1) * 1000))
    environ = {
        "REEF_DESIGNER_URL": gateway.base_url,
        "REEF_DESIGNER_DEADLINE_MS": past,
        "REEF_DESIGNER_FINISH_SECONDS": "60",
    }
    try:
        started = time.monotonic()
        output = checked([], environ)
        assert time.monotonic() - started < 10, "the command stops at the deadline, not after its 30 s wait"
        assert "state: running\ncheck_id: 1\n" in output and "The run's time is up" in output
        assert "--wait" not in output and "Remaining time: 0 seconds." in output and "Finish now" in output
    finally:
        release.set()
        gateway.stop()


def test_credential_checks_need_the_token() -> None:
    with pytest.raises(ValueError, match="need the run's token"):
        CredentialChecks(CountingChecks(), token="")


def test_credential_checks_preserve_delegate_configuration_and_budget(tmp_path: Path) -> None:
    delegate = CountingChecks()
    delegate.is_reviewing = True
    checks = CredentialChecks(delegate, token=TOKEN)
    assert checks.max_checks == delegate.max_checks and checks.is_reviewing
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    instruction = task_dir / "instruction.md"
    instruction.write_text(TOKEN)
    assert checks.check(task_dir)["ran"] is False
    assert delegate.count == 0
    instruction.write_text("A clean task")
    assert checks.check(task_dir) == {"ran": True, "passed": True, "check": 1, "oracle_reward": 1.0, "nop_reward": 0.0}
    assert delegate.count == 1


@pytest.mark.parametrize(
    ("route", "body", "expected_status"),
    [
        ("/check", b"not json", 400),
        ("/check", b"[]", 400),
        ("/check", b'{"check_id": 999}', 200),
        ("/check", b'{"check_id": "not-a-check-id"}', 200),
        ("/trial", b'{"task": "check 1"}', 404),
        ("/wrong-token/check", b"{}", 404),
    ],
)
def test_check_gateway_validates_requests(tmp_path: Path, route: str, body: bytes, expected_status: int) -> None:
    checks = CountingChecks()
    gateway = CheckGateway(AgentTools(tmp_path, checks))
    gateway.start()
    try:
        if route.startswith("/wrong-token"):
            url = f"http://127.0.0.1:{gateway.port}/t{route}"
        else:
            url = f"{gateway.base_url}{route}"
        request = urllib.request.Request(url, data=body, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
                result = json.load(response)
        except urllib.error.HTTPError as error:
            status = error.code
            result = json.load(error)
        assert status == expected_status
        assert "error" in result
        assert checks.count == 0
    finally:
        gateway.stop()


def test_check_gateway_relays_no_model_call(tmp_path: Path) -> None:
    gateway = CheckGateway(AgentTools(tmp_path, CredentialChecks(CountingChecks(), token=TOKEN)))
    gateway.start()

    def status(route: str, *, method: str = "POST") -> int:
        data = json.dumps({"model": "claude-opus", "messages": []}).encode() if method == "POST" else None
        request = urllib.request.Request(f"{gateway.base_url}{route}", data=data, method=method)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    try:
        for route in ("/v1/messages", "/v1/chat/completions", "/v1/responses", "/v1/images", "/v1/embeddings"):
            assert status(route) == 404, route
        assert status("/models?modality=text", method="GET") == 404
        assert status("/check") == 200
    finally:
        gateway.stop()


def test_claude_task_names_session_and_transcript_not_a_record(tmp_path: Path, reef_url: str) -> None:
    session = ScriptedClaudeSession(checks=([], []))
    generator = autoenvscaling(tmp_path, claude_code(tmp_path, reef_url, session), count=1, max_checks=1)
    results = run_tasks(generator, (0.0, 1.0))
    asyncio.run(generator.record_results(results, generation=0))
    [result] = results
    passed, spent = session.check_outputs
    assert '"passed": true' in passed and '"checks_left": 0' in passed
    assert "all 1 checks are used" in spent, "max_checks holds on the host"
    assert isinstance(result, RolledOutTask)
    task = result.task
    transcript = f"{SESSION_ID}.jsonl"
    assert task.source_agent_record_ids == ("session-1",) and "designer_record_id" not in task.metadata
    assert task.metadata["designer_transcript"] == transcript and task.metadata["designer_harness"] == "claude-code"
    # The run's directory is gone; the state keeps the transcript as saved files.
    assert (state(tmp_path).files(task.metadata["designer_transcript_files"]) / transcript).is_file()
    [proposal] = report(tmp_path)["proposals"]
    assert proposal["designer_record_id"] == "" and proposal["task_name"] == task.name


def test_claude_variant_shares_parent_split(tmp_path: Path, reef_url: str) -> None:
    # A parent pi wrote: its sources are its last recorded call and the session.
    parent = task_for("harbor-00000-000-autoenvscaling", "designer-1", 8100)
    parent = dataclasses.replace(
        parent,
        source_agent_record_ids=("designer-1", "session-1"),
        metadata={**parent.metadata, "designer_record_id": "designer-1"},
    )
    parent_dir = write_harbor_task(parent, tmp_path / "parents")
    generator = autoenvscaling(tmp_path, claude_code(tmp_path, reef_url, ScriptedClaudeSession()), count=1)
    variant = TaskVariant(parent=parent.name, parent_dir=parent_dir, direction="harder", pass_rate=1.0, rollouts=4)
    task = asyncio.run(generator.rebuilt(generator.sources[0], 1, variant=variant))
    assert task.metadata["parent_task"] == parent.name and task.metadata["variant"] == "harder"
    assert split_source_ids(task) == split_source_ids(parent) == ("session-1",)
    for split in ("train", "eval"):
        previous = TaskSplit(
            train=(parent.name,) if split == "train" else (),
            eval=(parent.name,) if split == "eval" else (),
            seed=0,
            eval_fraction=0.5,
        )
        assigned = extend_split_by_source(
            previous=previous,
            previous_sources={parent.name: split_source_ids(parent)},
            sources={task.name: split_source_ids(task)},
        )
        assert task.name in (assigned.train if split == "train" else assigned.eval)


@pytest.mark.unit
def test_claude_vary_gives_claude_code_the_parent_and_the_variant_rules(tmp_path: Path, reef_url: str) -> None:
    parent_dir = write_harbor_task(
        task_for("harbor-00000-000-autoenvscaling", "designer-1", 8100), tmp_path / "parents"
    )
    variant = TaskVariant(parent="t-0", parent_dir=parent_dir, direction="simpler", pass_rate=0.125, rollouts=8)
    session = ScriptedClaudeSession()
    designer = claude_code(tmp_path, reef_url, session)
    assert isinstance(designer, VariantDesigner)
    designed = designer.vary(
        source_of(sources(1)[0]), tmp_path / "run", NoChecks(1), tags={}, stop=threading.Event(), variant=variant
    )
    root = tmp_path / "run"
    assert session.argv[:3] == ["claude", "-p", VARIANT_PROMPT.format(direction="simpler")]
    rules = (root / "plugin" / "rules.md").read_text()
    assert VARIANT_INPUTS_TEXT.strip() in rules and CLAUDE_CHECK_TEXT.strip() in rules
    assert (root / "inputs" / "parent" / "instruction.md").is_file()
    assert "(t-0) is 0.12 over 8 rollouts" in (root / "inputs" / "variant.md").read_text()
    assert designed.record_ids == () and "port.txt" in designed.reply


@pytest.mark.unit
def test_adaptive_config_builds_claude_code_as_a_variant_designer(tmp_path: Path) -> None:
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, designer_harness="claude-code", adaptive=True, count=2),
        scenario="one",
        environ={"E2B_API_KEY": "e2b-test", "CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
        state=state(tmp_path),
    )
    assert generator.is_adaptive and isinstance(generator.designer, ClaudeCodeDesigner)
    assert generator.variant_designer is generator.designer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stdout", "exit_code", "error"),
    [
        (result_json("error_max_turns", True, errors=["Reached maximum number of turns (40)"]), 1, ""),
        (result_json(), 0, ""),
        (result_json(is_error=True, result="Credit balance is too low"), 1, r"error \(success\): Credit balance"),
        (result_json(api_error_status=429, result="Rate limited"), 0, "Rate limited"),
        (result_json(result="API Error: 401 OAuth token revoked"), 0, "API Error: 401"),
        (result_json("error_during_execution", True, errors=["aborted"]), 1, r"error_during_execution\): aborted"),
        ("", 1, "exited 1 without a result: boom"),
    ],
)
def test_max_turns_keeps_workspace_and_model_error_is_outage(
    tmp_path: Path, reef_url: str, stdout: str, exit_code: int, error: str
) -> None:
    session = ScriptedClaudeSession(stdout=stdout or "not json", exit_code=exit_code, stderr="boom")
    if error:
        with pytest.raises(DesignerOutageError, match=error):
            designed_in(tmp_path, reef_url, session)
    else:
        assert "port.txt" in designed_in(tmp_path, reef_url, session).reply
    assert session.is_closed.is_set()


@pytest.mark.unit
@pytest.mark.parametrize(
    "stdout",
    [
        result_json(is_error=True, result=SAFEGUARDS_TEXT),
        result_json(is_error=True, stop_reason="refusal", result="API Error: Claude can't help with this."),
    ],
    ids=("safeguards-text", "refusal-stop-reason"),
)
def test_a_run_that_the_model_provider_refused_is_a_refusal_and_not_an_outage(
    tmp_path: Path, reef_url: str, stdout: str
) -> None:
    session = ScriptedClaudeSession(stdout=stdout, exit_code=1)
    with pytest.raises(ModelRefusedError, match="the model provider refused Claude Code's run: API Error: ") as raised:
        designed_in(tmp_path, reef_url, session)
    assert not isinstance(raised.value, DesignerOutageError)
    assert session.is_closed.is_set()


@pytest.mark.unit
@pytest.mark.parametrize(
    "fields",
    [
        {"result": "Claude AI usage limit reached|1762700000"},
        {"api_error_status": 429, "result": "API Error: 429 rate_limit_error"},
        {"result": "Invalid API key. Please run /login"},
        {"api_error_status": 401, "result": "API Error: 401 OAuth token has expired"},
        # Only Claude Code's error text starts with the refusal; the same words later in a text are no refusal.
        {"result": f"Claude AI usage limit reached. {SAFEGUARDS_TEXT}"},
    ],
    ids=("usage-limit", "rate-limit", "login", "expired-token", "refusal-words-later"),
)
def test_a_spent_plan_or_a_lost_sign_in_stays_an_outage(
    tmp_path: Path, reef_url: str, fields: dict[str, object]
) -> None:
    session = ScriptedClaudeSession(stdout=result_json("success", True, stop_reason="stop_sequence", **fields))
    with pytest.raises(DesignerOutageError, match=r"Claude Code's run ended with an error \(success\)"):
        designed_in(tmp_path, reef_url, session)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stdout", "exit_code", "failure"),
    [
        (result_json(is_error=True, result=SAFEGUARDS_TEXT), 0, "the model provider refused Claude Code's run"),
        (
            result_json(api_error_status=429, result="API Error: 429 rate_limit_error"),
            0,
            "Claude Code's run ended with an error (success): API Error: 429 rate_limit_error",
        ),
        ("not json", 1, "Claude Code exited 1 without a result: not json"),
    ],
    ids=("refusal", "rate-limit", "no-result"),
)
def test_a_refused_or_failed_run_whose_task_holds_the_token_ends_as_a_held_token(
    tmp_path: Path, reef_url: str, stdout: str, exit_code: int, failure: str
) -> None:
    session = ScriptedClaudeSession(stdout=stdout, exit_code=exit_code, task_leak=TOKEN)
    with pytest.raises(CredentialHeldError, match="held the run's Claude Code token") as raised:
        designed_in(tmp_path, reef_url, session)
    # The failure, which can meet the next run too, stays in the message and as the cause.
    assert str(raised.value.__cause__).startswith(failure)
    assert str(raised.value).endswith(f"; the run also ended with: {raised.value.__cause__}")
    assert not files_holding(tmp_path / "run", TOKEN)


class VariantFailingSession(ScriptedClaudeSession):
    """A sandbox whose Claude Code prints ``variant_stdout`` for a variant's run, and writes a session's task."""

    def __init__(self, variant_stdout: str) -> None:
        super().__init__()
        self.variant_stdout = variant_stdout

    def launch(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        timeout: float,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> ProcessOutcome:
        outcome = super().launch(
            argv,
            root=root,
            workspace=workspace,
            env=env,
            timeout=timeout,
            writable_paths=writable_paths,
            readonly_paths=readonly_paths,
        )
        return outcome if argv[2] == AGENT_PROMPT else ProcessOutcome(1, self.variant_stdout, "")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("variant_stdout", "is_refused"),
    [
        (result_json(is_error=True, stop_reason="refusal", result=SAFEGUARDS_TEXT), True),
        (result_json(is_error=True, stop_reason="stop_sequence", result="Invalid API key. Please run /login"), False),
    ],
    ids=("refusal", "login"),
)
def test_a_refused_variant_takes_its_parents_slot_and_a_lost_sign_in_leaves_it(
    tmp_path: Path, reef_url: str, variant_stdout: str, is_refused: bool
) -> None:
    first = autoenvscaling(tmp_path, StandInDesigner(), count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(first.record_results(run_tasks(first, (1.0, 1.0), (0.5, 0.5)), generation=0))
    designer = claude_code(tmp_path, reef_url, VariantFailingSession(variant_stdout))
    generator = autoenvscaling(tmp_path, designer, count=2, is_adaptive=True, variant_share=0.5)
    if is_refused:
        [task] = asyncio.run(generator.generate(REQUEST))
        assert "parent_task" not in task.metadata, "the generation goes on to the session's slot"
        [variant] = state_document(tmp_path, f"open/{REQUEST.generation:05d}")["refused"]
    else:
        with pytest.raises(DesignerOutageError, match="Please run /login"):
            asyncio.run(generator.generate(REQUEST))
        # An attempt that made no task has no results to wait for: its report is saved before it raises.
        [variant] = report(tmp_path, generator.generation)["proposals"]
    assert (variant["parent_task"], variant["stage"]) == (NAMES[0], "authoring" if is_refused else "outage")
    assert ("safeguards flagged this message" in variant["refusal"]) is is_refused
    [parent] = state_document(tmp_path, "adaptive")
    assert parent["varied"] is is_refused
    stand_in = StandInDesigner()
    again = autoenvscaling(tmp_path, stand_in, count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(again.generate(REQUEST))
    varied = [call["variant"].parent for call in stand_in.calls if call["variant"] is not None]
    assert varied == ([] if is_refused else [NAMES[0]]), "the next generation varies only a parent an outage left"


@pytest.mark.unit
def test_timeout_keeps_the_task_and_names_the_newest_transcript(tmp_path: Path, reef_url: str) -> None:
    older = tmp_path / "run" / "claude" / "projects" / "-elsewhere" / "older.jsonl"
    older.parent.mkdir(parents=True)
    older.write_text("{}\n")
    os.utime(older, (time.time() - 3600, time.time() - 3600))
    session = ScriptedClaudeSession(is_timeout=True)
    designed = designed_in(tmp_path, reef_url, session)
    assert "port.txt" in designed.reply and designed.record_ids == ()
    assert designed.transcript == f"claude/projects/-home-user-reef-run-workspace/{SESSION_ID}.jsonl"
    assert session.is_closed.is_set()


@pytest.mark.unit
def test_token_never_reaches_errors_logs_run_dir_or_task(
    tmp_path: Path, reef_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    failing = ScriptedClaudeSession(stdout=result_json(result=f"API Error: 401 {TOKEN} was refused"), leak=TOKEN)
    with pytest.raises(DesignerOutageError) as raised:
        designed_in(tmp_path / "failing", reef_url, failing)
    assert TOKEN not in str(raised.value) and "[redacted] was refused" in str(raised.value)
    run_dir = tmp_path / "failing" / "run"
    assert not files_holding(run_dir, TOKEN)
    assert files_holding(run_dir, "[redacted]") == [str(next(run_dir.glob(f"claude/projects/*/{SESSION_ID}.jsonl")))]

    holding = ScriptedClaudeSession(task_leak=TOKEN)
    with pytest.raises(DesignerOutageError, match="held the run's Claude Code token"):
        designed_in(tmp_path / "holding", reef_url, holding)
    assert not files_holding(tmp_path / "holding" / "run", TOKEN)
    assert "[redacted]" in (tmp_path / "holding" / "run" / "workspace" / "task" / "instruction.md").read_text()
    assert not any(TOKEN in record.getMessage() for record in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [(result_json(is_error=True, result="x" * 480 + TOKEN), ""), ("not json", TOKEN + "y" * 480)],
    ids=("result-tail", "stderr-head"),
)
def test_cut_error_text_keeps_no_part_of_the_token(tmp_path: Path, reef_url: str, stdout: str, stderr: str) -> None:
    session = ScriptedClaudeSession(stdout=stdout, exit_code=1, stderr=stderr)
    with pytest.raises(DesignerOutageError) as raised:
        designed_in(tmp_path, reef_url, session)
    assert "[redacted]" in str(raised.value) and not holds_part_of_token(str(raised.value))


@pytest.mark.unit
def test_result_fields_are_redacted_in_the_log_and_the_error(
    tmp_path: Path, reef_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    session = ScriptedClaudeSession(stdout=result_json(f"odd-{TOKEN}", num_turns=f"many-{TOKEN}"))
    with pytest.raises(DesignerOutageError, match=r"error \(odd-\[redacted\]\)") as raised:
        designed_in(tmp_path, reef_url, session)
    assert TOKEN not in str(raised.value)
    [ended] = [record.getMessage() for record in caplog.records if "ended with" in record.getMessage()]
    assert "many-[redacted] turns" in ended and TOKEN not in ended


@pytest.mark.unit
def test_launch_error_and_close_warning_are_redacted(
    tmp_path: Path, reef_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    session = ScriptedClaudeSession(launch_error=f"no process: env {TOKEN}", close_error=f"still up: {TOKEN}")
    with pytest.raises(DesignerOutageError, match=r"could not run: no process: env \[redacted\]") as raised:
        designed_in(tmp_path, reef_url, session)
    assert TOKEN not in str(raised.value)
    [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert "did not stop" in warning and "still up: [redacted]" in warning and TOKEN not in warning


@pytest.mark.unit
def test_a_name_holding_the_token_is_renamed(tmp_path: Path, reef_url: str) -> None:
    designed = designed_in(tmp_path, reef_url, ScriptedClaudeSession(session_name=TOKEN, is_timeout=True))
    run_dir = tmp_path / "run"
    assert designed.transcript == "claude/projects/-home-user-reef-run-workspace/[redacted].jsonl"
    assert (run_dir / designed.transcript).is_file()
    assert not [path for path in run_dir.rglob("*") if TOKEN in str(path)]


def test_a_name_that_cannot_take_the_redacted_name_is_deleted(tmp_path: Path) -> None:
    (tmp_path / f"a{TOKEN}").mkdir()
    (tmp_path / f"a{TOKEN}" / "x").write_text("x")
    (tmp_path / "a[redacted]").mkdir()
    (tmp_path / "a[redacted]" / "y").write_text("y")
    (tmp_path / f"b{TOKEN}").write_text("b")
    rename_holding(tmp_path, TOKEN)
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        "a[redacted]",
        "a[redacted]/y",
        "b[redacted]",
    ]


@pytest.mark.unit
def test_check_and_admission_refuse_a_task_link_to_the_token(tmp_path: Path, reef_url: str) -> None:
    checks = CountingChecks()
    session = ScriptedClaudeSession(checks=([],), link_leak=TOKEN)
    with pytest.raises(CredentialHeldError):
        designed_in(tmp_path, reef_url, session, checks)
    [output] = session.check_outputs
    assert "holds this run's credential" in output and checks.count == 0
    workspace = tmp_path / "run" / "workspace"
    assert not files_holding(tmp_path / "run", TOKEN) and (workspace / "notes.txt").read_text() == "[redacted]"


def test_task_holding_token_ends_generation_with_stage_credential(tmp_path: Path, reef_url: str) -> None:
    generator = autoenvscaling(tmp_path, claude_code(tmp_path, reef_url, ScriptedClaudeSession(task_leak=TOKEN)))
    assert run_tasks(generator) == ()
    asyncio.run(generator.record_results((), generation=0))
    document = report(tmp_path)
    assert [entry["stage"] for entry in document["proposals"]] == ["credential"]
    assert "held the run's Claude Code token" in document["error"]
    assert not files_holding(tmp_path / "state", TOKEN)


@pytest.mark.unit
def test_check_refuses_task_holding_token(tmp_path: Path, reef_url: str) -> None:
    checks = CountingChecks()
    session = ScriptedClaudeSession(checks=([],), task_leak=TOKEN)
    with pytest.raises(DesignerOutageError, match="not admitted"):
        designed_in(tmp_path, reef_url, session, checks)
    [output] = session.check_outputs
    assert "holds this run's credential" in output and TOKEN not in output
    assert checks.count == 0, "the generator service never saw the task"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fields", "environ", "message"),
    [
        ({"designer_harness": "cursor"}, {}, "designer_harness must be one of"),
        ({"claude_model": "opus"}, {}, "claude_model and claude_token_env need designer_harness: claude-code"),
        ({"designer_harness": "claude-code", "designer": "prompt"}, {}, "needs designer: agent"),
        ({"designer_harness": "claude-code"}, {}, "needs a Claude Code OAuth token in CLAUDE_CODE_OAUTH_TOKEN"),
        (
            {"designer_harness": "claude-code"},
            {"CLAUDE_CODE_OAUTH_TOKEN": " "},
            "OAuth token in CLAUDE_CODE_OAUTH_TOKEN",
        ),
        (
            {"designer_harness": "claude-code", "claude_token_env": "OWNER_TOKEN"},
            {"OWNER_TOKEN": "sk-ant-api03-leak-me"},
            "OWNER_TOKEN must hold a Claude Code OAuth token",
        ),
        ({"designer_harness": "claude-code", "claude_model": ""}, {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}, "claude_model"),
        *(
            (
                {"designer_harness": "claude-code", "agent_sandbox_options": {"env_from": [name]}},
                {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, name: "http://elsewhere"},
                f"must not forward {name}",
            )
            for name in (
                "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
                "HTTPS_PROXY",
                "MY_BASE_URL",
            )
        ),
        (
            {"designer_harness": "claude-code", "agent_sandbox_options": {"env_from": ["CLAUDE_CODE_OAUTH_TOKEN"]}},
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
            "must not forward CLAUDE_CODE_OAUTH_TOKEN",
        ),
        (
            {
                "designer_harness": "claude-code",
                "claude_token_env": "OWNER_TOKEN",
                "agent_sandbox_options": {"env_from": ["OWNER_TOKEN"]},
            },
            {"OWNER_TOKEN": TOKEN},
            "must not forward OWNER_TOKEN",
        ),
        (
            {"designer_harness": "claude-code", "agent_sandbox_options": {"env_from": ["SAVED"]}},
            {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "SAVED": f"key={TOKEN}"},
            "must not forward SAVED",
        ),
    ],
)
def test_claude_config_refusals(
    tmp_path: Path, fields: dict[str, object], environ: dict[str, str], message: str
) -> None:
    with pytest.raises(PluginConfigError, match=message) as raised:
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, **fields),
            scenario="one",
            environ={"E2B_API_KEY": "e2b-test", **environ},
            state=state(tmp_path),
        )
    assert TOKEN not in str(raised.value) and "leak-me" not in str(raised.value)


@pytest.mark.unit
def test_env_from_refuses_every_variable_the_run_sets(tmp_path: Path, reef_url: str) -> None:
    session = ScriptedClaudeSession()
    designed_in(tmp_path / "probe", reef_url, session)
    names = sorted(session.launch_env)
    assert {"HOME", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "DISABLE_ERROR_REPORTING"} <= set(names)
    environ = {"E2B_API_KEY": "e2b-test", "CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "OWNER_SETTING": "1"}
    for name in names:
        fields = {"designer_harness": "claude-code", "agent_sandbox_options": {"env_from": [name]}}
        with pytest.raises(PluginConfigError, match=f"must not forward {name}"):
            AutoEnvScalingGenerator.from_config(
                generating_config(tmp_path, **fields),
                scenario="one",
                environ={**environ, name: TOKEN if name == "CLAUDE_CODE_OAUTH_TOKEN" else "0"},
                state=state(tmp_path),
            )
    # A variable of the deployment's own still reaches the sandbox.
    fields = {"designer_harness": "claude-code", "agent_sandbox_options": {"env_from": ["OWNER_SETTING"]}}
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, **fields), scenario="one", environ=environ, state=state(tmp_path)
    )
    assert isinstance(generator.designer, ClaudeCodeDesigner)
    assert dict(generator.designer.executor.env) == {"OWNER_SETTING": "1"}


@pytest.mark.unit
def test_claude_config_builds_claude_code_on_the_agent_settings(tmp_path: Path) -> None:
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(
            tmp_path, designer_harness="claude-code", claude_token_env="OWNER_TOKEN", max_model_calls=80
        ),
        scenario="one",
        environ={"E2B_API_KEY": "e2b-test", "OWNER_TOKEN": f"{TOKEN}\n", "REEF_TOKEN": "reef-token"},
        state=state(tmp_path),
    )
    designer = generator.designer
    assert isinstance(designer, ClaudeCodeDesigner)
    assert (designer.claude_token, designer.model, designer.max_turns) == (TOKEN, "opus", 80)
    assert (
        designer.reef.token == "reef-token" and designer.cache_dir == Path(generator.temporary.name) / "repositories"
    )


@pytest.mark.unit
def test_stop_signal_closes_claude_sandbox(tmp_path: Path, reef_url: str) -> None:
    stop = threading.Event()
    session = ScriptedClaudeSession(stop=stop)
    with pytest.raises(DesignerOutageError, match="stopped"):
        claude_code(tmp_path, reef_url, session).design(
            source_of(sources(1)[0]), tmp_path / "run", NoChecks(1), tags={}, stop=stop
        )
    assert session.is_closed.is_set()
