"""AutoEnvScaling's authors: the agent's tools, gateway and run outcomes, the prompt Designer's one call, and the sources."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    USER_REQUEST,
    NoChecks,
    ScriptedSession,
    agent_files,
    fenced,
    run_agent,
    serving_reef,
    source_of,
    sources,
)
from tests.support.task_service_stubs import DOCUMENT

from recipes.beta.autoenvscaling import (
    AuthorError,
    CandidateChecks,
    PromptAuthor,
    Repository,
    SessionSource,
    SourceError,
    load_sources,
    render_session,
)
from recipes.beta.autoenvscaling.author import (
    BEHAVIOUR_RULES_TEXT,
    REBUILD_RULES_TEXT,
    AgentTools,
    ModelCalls,
    RecordingGateway,
    call_receipt,
    repository_snapshot,
)
from reef.harness.episodes.model_binding import ModelBinding
from reef.record2task.generator.designer import HARBOR_OUTPUT_TEXT, SYSTEM_PROMPT


class HeldChecks(CandidateChecks):
    """Checks that wait for the test to release them."""

    def __init__(self) -> None:
        super().__init__(6)
        self.release = threading.Event()
        self.count = 0

    def check(self, task_dir: Path) -> dict[str, object]:
        self.count += 1
        self.release.wait(10)
        return {"check": self.count, "ran": True, "passed": True}


def test_asking_for_a_check_while_one_runs_waits_for_that_one(tmp_path: Path) -> None:
    checks = HeldChecks()
    tools = AgentTools(tmp_path, checks)
    assert tools.check() == tools.check() == {"state": "running", "check_id": 1}
    assert tools.trial("check 1") == {"state": "running", "check_id": 1}
    checks.release.set()
    while (result := tools.trial("check 1"))["state"] == "running":
        time.sleep(0.01)
    assert result == {"state": "done", "check_id": 1, "check": 1, "ran": True, "passed": True}
    assert checks.count == 1, "the second ask started no check of its own"


def test_the_gateway_records_each_model_call_on_reef_and_keeps_its_receipt(tmp_path: Path) -> None:
    reef, served = serving_reef("ok")
    calls = ModelCalls(budget=1)
    upstream = ModelBinding(base_url=f"http://127.0.0.1:{reef.server_address[1]}", model="designer", api_key="token")
    headers = {"x-reef-scenario": "autoenvscaling", "x-reef-tag-role": "designer"}
    gateway = RecordingGateway(upstream, calls, AgentTools(tmp_path, NoChecks(1)), headers=headers)
    gateway.start()

    def post() -> int:
        request = urllib.request.Request(
            f"{gateway.base_url}/v1/chat/completions",
            data=json.dumps({"model": "any", "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    try:
        assert post() == 200
        assert post() == 429, "the run's model-call budget is spent"
    finally:
        gateway.stop()
        reef.shutdown()
        reef.server_close()
    assert calls.receipts == ["rec-1"]
    [call] = served
    assert call["model"] == "designer" and call["headers"]["authorization"] == "Bearer token"
    assert call["headers"]["x-reef-scenario"] == "autoenvscaling" and call["headers"]["x-reef-tag-role"] == "designer"
    streamed = b'event: response.completed\ndata: {"type":"response.completed","reef":{"agent_record_id":"rec-9"}}\n\n'
    assert call_receipt({}, streamed) == "rec-9" and call_receipt({}, b"{}") is None


def test_the_session_renderer_cuts_tool_output_first_and_never_the_users_words() -> None:
    request = "Please add a --dry-run flag to the deploy command. " * 10
    payload = {
        "messages": [
            {"role": "user", "content": request},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"p": "x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "y" * 50_000}]},
        ],
        "response": {"role": "assistant", "content": [{"type": "text", "text": "Done. " * 100}]},
    }
    whole = render_session(payload)
    assert "y" * 12_000 in whole and "y" * 12_001 not in whole and "[... 38000 characters cut]" in whole
    assert "[... cut to fit the session budget]" not in whole
    # Below what the user's words alone take: every other line is cut and the loop still ends.
    text = render_session(payload, budget_chars=100)
    assert f"[user] {request.strip()}" in text
    assert text.count("[... cut to fit the session budget]") == 2 and "y" * 201 not in text
    assert "[tool call Read]" in text and "[assistant] Done." in text


def test_a_sources_file_lists_sessions_in_order_and_refuses_a_commit_that_is_no_hash(tmp_path: Path) -> None:
    (tmp_path / "sources.json").write_text(json.dumps(sources(2)))
    assert [source.record_id for source in load_sources(tmp_path / "sources.json")] == ["session-1", "session-2"]
    (tmp_path / "bad.json").write_text(
        json.dumps([{**sources(1)[0], "repository": {"url": "u", "commit": "main", "path": "/app"}}])
    )
    with pytest.raises(ValueError, match="commit"):
        load_sources(tmp_path / "bad.json")
    (tmp_path / "empty.json").write_text("[]")
    with pytest.raises(ValueError, match="non-empty JSON list"):
        load_sources(tmp_path / "empty.json")


@pytest.mark.parametrize(
    ("stop_reason", "is_spent", "is_outage"),
    [("error", False, True), ("aborted", False, True), ("error", True, False), ("length", False, False)],
)
def test_only_a_model_failure_with_the_budget_unspent_is_an_outage(
    tmp_path: Path, stop_reason: str, is_spent: bool, is_outage: bool
) -> None:
    calls = ModelCalls(budget=0 if is_spent else 5)
    if is_spent:
        with pytest.raises(RuntimeError, match="spent"):
            calls.spend()
    message = {"role": "assistant", "stopReason": stop_reason, "errorMessage": "usage limit reached"}
    session = ScriptedSession([{"type": "message", "message": message}])
    if is_outage:
        with pytest.raises(AuthorError, match=f"ended with {stop_reason}: usage limit reached"):
            run_agent(tmp_path, session, calls, threading.Event())
    else:
        run_agent(tmp_path, session, calls, threading.Event())
    assert session.is_closed.is_set()


def test_a_stop_signal_closes_the_sandbox_and_ends_the_run(tmp_path: Path) -> None:
    stop = threading.Event()
    session = ScriptedSession([], stop=stop)
    with pytest.raises(AuthorError, match="stopped"):
        run_agent(tmp_path, session, ModelCalls(5), stop)
    assert session.is_closed.is_set()


def test_a_repository_snapshot_holds_the_base_commit_alone(tmp_path: Path) -> None:
    def git(*arguments: str) -> str:
        command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments]
        return subprocess.run(command, cwd=tmp_path / "origin", check=True, capture_output=True, text=True).stdout

    (tmp_path / "origin").mkdir()
    git("init", "-q")
    for name in ("base.txt", "later.txt"):
        (tmp_path / "origin" / name).write_text(name)
        git("add", name)
        git("commit", "-q", "-m", name)
    base, later = git("rev-parse", "HEAD~1", "HEAD").split()
    source = SessionSource("r", "sessions", Repository(url=f"file://{tmp_path / 'origin'}", commit=base, path="/app"))
    snapshot = repository_snapshot(source, tmp_path / "cache")
    assert sorted(path.name for path in snapshot.iterdir() if path.name != ".git") == ["base.txt"]
    log = subprocess.run(["git", "rev-list", "--all"], cwd=snapshot, check=True, capture_output=True, text=True).stdout
    assert len(log.split()) == 1 and later not in log
    assert repository_snapshot(source, tmp_path / "cache") == snapshot


@pytest.mark.parametrize(
    ("place", "commit", "error", "message"),
    [
        ("nowhere", "3f2c1ab", SourceError, "does not appear to be a git repository"),
        ("origin", "0123456789abcdef0123456789abcdef01234567", SourceError, "has no commit"),
        ("network", "3f2c1ab", AuthorError, "Failed to connect"),
    ],
)
def test_a_missing_repository_or_commit_is_a_bad_source_and_a_network_failure_an_outage(
    tmp_path: Path, place: str, commit: str, error: type[Exception], message: str
) -> None:
    (tmp_path / "origin").mkdir()
    for command in (
        ["init", "-q"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base"],
    ):
        subprocess.run(["git", *command], cwd=tmp_path / "origin", check=True, capture_output=True)
    url = {
        "nowhere": f"file://{tmp_path / 'nowhere'}",
        "origin": f"file://{tmp_path / 'origin'}",
        # Nothing listens on port 9: the connection is refused, as on a network failure.
        "network": "https://127.0.0.1:9/org/repo.git",
    }[place]
    source = SessionSource("r", "sessions", Repository(url=url, commit=commit, path="/app"))
    with pytest.raises(error, match=message):
        repository_snapshot(source, tmp_path / "cache")


@pytest.mark.parametrize("api", ["openai", "responses"])
def test_the_prompt_designer_asks_once_with_the_session_cut_to_its_budget(tmp_path: Path, api: str) -> None:
    reply = fenced(DOCUMENT)
    reef, calls = serving_reef(reply)
    try:
        author = PromptAuthor(
            reef_url=f"http://127.0.0.1:{reef.server_address[1]}",
            token="token",
            scenario="autoenvscaling",
            model="designer",
            api=api,
            session_chars=2000,
        )
        authored = author.author(
            source_of(sources(1)[0]),
            tmp_path / "run",
            NoChecks(1),
            tags={"role": "designer", "generation": "0", "index": "0"},
            stop=threading.Event(),
        )
    finally:
        reef.shutdown()
        reef.server_close()
    [call] = calls
    assert call["path"] == ("/v1/responses" if api == "responses" else "/v1/chat/completions")
    assert call["model"] == "designer" and call["headers"]["authorization"] == "Bearer token"
    assert call["headers"]["x-reef-scenario"] == "autoenvscaling" and call["headers"]["x-reef-tag-role"] == "designer"
    if api == "responses":
        system, prompt = call["instructions"], call["input"][0]["content"][0]["text"]
    else:
        system, prompt = call["messages"][0]["content"], call["messages"][1]["content"]
    assert system == SYSTEM_PROMPT
    assert REBUILD_RULES_TEXT in prompt and HARBOR_OUTPUT_TEXT in prompt
    assert "REPOSITORY: https://example.com/org/repo-1.git\nBASE COMMIT: 3f2c1ab" in prompt
    assert "[BEGIN recorded session" in prompt and f"[user] {USER_REQUEST.strip()}" in prompt
    # The session is cut to its budget: the tool result keeps its first characters only.
    assert "[... cut to fit the session budget]" in prompt and "y" * 1000 not in prompt
    assert authored.record_ids == ("rec-1",) and authored.reply == reply
    assert (tmp_path / "run" / "reply.md").read_text() == reply


def test_the_behaviour_rule_reaches_both_authors_and_is_absent_when_off(tmp_path: Path) -> None:
    rule = "TESTS CHECK BEHAVIOUR"
    assert rule in BEHAVIOUR_RULES_TEXT
    assert BEHAVIOUR_RULES_TEXT in agent_files(tmp_path / "on", has_behaviour_rule=True, checks=NoChecks(1))
    assert rule not in agent_files(tmp_path / "off", has_behaviour_rule=False, checks=NoChecks(1))

    reef, calls = serving_reef(fenced(DOCUMENT))
    try:
        for has_behaviour_rule in (True, False):
            PromptAuthor(
                reef_url=f"http://127.0.0.1:{reef.server_address[1]}",
                token="token",
                scenario="autoenvscaling",
                model="designer",
                api="openai",
                session_chars=2000,
                has_behaviour_rule=has_behaviour_rule,
            ).author(
                source_of(sources(1)[0]),
                tmp_path / f"prompt-{has_behaviour_rule}",
                NoChecks(1),
                tags={"role": "designer"},
                stop=threading.Event(),
            )
    finally:
        reef.shutdown()
        reef.server_close()
    with_rule, without_rule = (call["messages"][1]["content"] for call in calls)
    assert BEHAVIOUR_RULES_TEXT in with_rule and rule not in without_rule
