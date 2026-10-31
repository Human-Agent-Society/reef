"""Reefine's agent: the tree read back, the gateway, the trials, and whole runs with a stand-in pi."""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from tests.recipes.reefine.reefine_support import TREE, Served, designed, reefine_step, step_workspace
from tests.support.harness_steps import step_services

from recipes.beta.reefine import ReefineMethod
from recipes.beta.reefine import agent as reefine_agent
from recipes.beta.reefine.agent import AgentRun, CheckTool, ReefineAgent, TrialTool, agent_rules, answer_with_agent
from recipes.beta.reefine.agent_gateway import ReefineGateway
from recipes.beta.reefine.multimodal import PRESETS, MultimodalProvider
from recipes.beta.reefine.trial import trial_script
from reef.core.training_request import TrainingRequest
from reef.execution.sandbox.e2b import E2BSession
from reef.execution.sandbox.process import LocalExecutor, ProcessOutcome, ProgramTimeout, SandboxExecutor
from reef.execution.sandbox.remote import pack
from reef.harness.adapters import get_adapter
from reef.harness.gateway import reply_tool_calls, tool_summary
from reef.harness.tasks.model_binding import ModelBinding, ModelBindings
from reef.harness.tree import TreePatch
from reef.train.harness_backend import Services, StepContext, Workspace
from reef.train.harness_backend.agent.run import AgentExecution
from reef.train.harness_backend.calls import StepCalls
from reef.train.types import JobData

REVIEW = {"result": "complete", "covered": ["reads answers aloud"], "uncovered": [], "delivers": True}
SPEAK = "export default function (pi) { if (process.env.PI_OFFLINE) return; }"
ASK = TrainingRequest(text="Read my answers aloud with a natural voice", session="s1", release_id="base", id="r1")


class Upstream:
    """A served model and a multimodal provider in one server, remembering what reached it."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.gets: list[tuple[str, str | None]] = []
        #: The served model's next whole answer, in place of the review.
        self.answer: dict | None = None
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                upstream.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                if self.path == "/api/v1/audio/speech":
                    if body.get("model") != "real/tts":
                        self.reply(400, b'{"error":{"message":"Model not/real does not exist"}}', "application/json")
                    else:
                        self.reply(200, b"ID3audio", "audio/mpeg")
                    return
                if body.get("stream"):
                    events = [
                        {"choices": [{"delta": {"content": "hel"}}]},
                        {"choices": [{"delta": {"content": "lo"}}]},
                    ]
                    text = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
                    self.reply(200, text.encode(), "text/event-stream")
                    return
                reply = upstream.answer or {
                    "choices": [{"message": {"role": "assistant", "content": json.dumps(REVIEW)}}]
                }
                reply["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
                self.reply(200, json.dumps(reply).encode(), "application/json")

            def do_GET(self) -> None:
                upstream.gets.append((self.path, self.headers.get("Authorization")))
                self.reply(200, json.dumps({"data": [{"id": "real/tts"}]}).encode(), "application/json")

            def reply(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    server = Upstream()
    yield server
    server.close()


def provider_of(upstream: Upstream) -> MultimodalProvider:
    return MultimodalProvider(PRESETS["openrouter"], f"{upstream.url}/api", "sk-or-test")


def served_of(upstream: Upstream) -> ModelBindings:
    return ModelBindings(served=ModelBinding(base_url=upstream.url, model="served-model", api_key="served-key"))


def post(url: str, body: dict) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class NoRun:
    """An agent run whose check and trial answer what they were asked, for the gateway's routes."""

    def check(self) -> dict:
        return {"admitted": True}

    def trial(self, task: str, script: dict[str, object] | None = None) -> dict:
        return {"task": task} if script is None else {"script": script}


def no_tools() -> tuple[CheckTool, TrialTool]:
    run = NoRun()
    return CheckTool(run), TrialTool(run)


FAKE_PI = textwrap.dedent(
    """\
    #!{python}
    import json, os, pathlib, sys, time, urllib.request

    def post(url, body):
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={{"content-type": "application/json"}}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
        try:
            with opener.open(request, timeout=60) as response:
                return response.status, response.read().decode(errors="replace")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    sessions = pathlib.Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
    sessions.mkdir(parents=True, exist_ok=True)
    assert "PI_OFFLINE" not in os.environ, "an agent or a trial runs online"
    assert pathlib.Path(sys.argv[sys.argv.index("-e") + 1], "{tools}").exists() == (
        "REEF_AGENT_URL" in os.environ)
    # The run environment is minimal, so the test sets the mode in a file beside the binary.
    mode_file = pathlib.Path(sys.argv[0]).with_name("mode")
    mode = mode_file.read_text() if mode_file.exists() else ""
    if "REEF_AGENT_URL" in os.environ:
        base = os.environ["REEF_AGENT_URL"]
        if mode == "sleep":
            time.sleep(30)
        if mode != "idle":
            # The agent's current directory is the tree; its design and requires go beside it.
            pathlib.Path("../output/design.md").write_text("Speak each answer with a TTS model.")
            pathlib.Path("extensions").mkdir(exist_ok=True)
            pathlib.Path("extensions/speak.ts").write_text({speak!r})
            pathlib.Path("notes.txt").write_text("scratch")
            pathlib.Path("../output/requires.json").write_text(json.dumps(
                [{{"name": "afplay", "kind": "service", "check": "command -v afplay", "prompt": "Install a player"}}]))
            print(post(base + "/check", {{}}))
            print(post(base + "/trial", {{"task": "Say hello out loud"}}))
        text = "done"
    else:
        # The candidate harness: its extension calls the TTS route at REEF_SERVICE_URL, as in a user's session.
        status, body = post(os.environ["REEF_SERVICE_URL"] + "/v1/audio/speech",
                            {{"model": "not/real", "input": sys.argv[-1]}})
        text = "spoke with status %d" % status
    event = {{"type": "message", "message": {{"role": "assistant", "content": [{{"type": "text", "text": text}}]}}}}
    if mode == "stream-error":
        event["message"].update(stopReason="error", errorMessage="Stream ended without finish_reason")
    if mode == "budget":
        event["message"].update(stopReason="length")
    (sessions / "s.jsonl").write_text(json.dumps(event) + "\\n")
    """
)


def fake_pi(tmp_path: Path, mode: str = "") -> Path:
    """The stand-in pi in ``tmp_path``: the agent with ``REEF_AGENT_URL``, the candidate harness without it."""
    binary = tmp_path / "bin" / "pi"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text(FAKE_PI.format(python=sys.executable, tools=reefine_agent.AGENT_TOOLS_PATH, speak=SPEAK))
    binary.chmod(0o755)
    binary.with_name("mode").write_text(mode)
    return binary


def agent_of(binary: Path, provider: MultimodalProvider | None = None, *, timeout_s: float = 60.0) -> ReefineAgent:
    return ReefineAgent(AgentExecution(get_adapter("pi"), LocalExecutor(), str(binary), timeout_s), 30.0, provider)


def agent_step(tmp_path: Path, models: ModelBindings) -> tuple[Workspace, Services]:
    """A request step's workspace over ``TREE`` and its services, the served model ``models``."""
    workspace = step_workspace(tmp_path, request=ASK)
    return workspace, step_services(workspace, models)


@pytest.mark.unit
def test_the_agent_changes_the_steps_tree_tries_it_and_the_step_publishes_it_reviewed(tmp_path, upstream) -> None:
    method = ReefineMethod(agent_of(fake_pi(tmp_path), provider_of(upstream)))
    result, activity = reefine_step(tmp_path, method, JobData("demo:1", (), request=ASK), served_of(upstream))
    metrics = result.metrics
    assert metrics["selected"] is True and metrics["patch"] == {"extensions/speak.ts": SPEAK}
    notes = metrics["method_notes"]
    assert notes["design"] == "Speak each answer with a TTS model." and notes["review"]["result"] == "complete"
    assert notes["agent"]["trials"] == 1 and notes["agent"]["exit_code"] == 0
    # A file the agent left off the layout is not the candidate's: the notes say so and the record keeps it.
    assert notes["unread"] == ["harness/notes.txt is not a path of the harness tree, so it is not read"]
    assert metrics["method_summary"].startswith("Design:\nSpeak each answer with a TTS model.\n\nReview: complete")
    assert [item["name"] for item in metrics["training_request"]["requires"]] == ["afplay"]
    step_dir = Path(metrics["step_record"])
    assert not (step_dir / "harness" / "notes.txt").exists()
    assert (step_dir / "output" / "agent-harness" / "notes.txt").read_text() == "scratch"
    assert (step_dir / "output" / "agent-session.jsonl").read_text().count('"done"') == 1
    checks = [json.loads(line) for line in (step_dir / "output" / "agent-checks.jsonl").read_text().splitlines()]
    assert [check["kind"] for check in checks] == ["check", "trial"] and checks[0]["valid"] is True
    # The trial's speech call reached the provider, and its refusal is in the step's call record for the agent.
    speech = [r for r in upstream.requests if r["path"] == "/api/v1/audio/speech"]
    assert speech and speech[0]["body"]["model"] == "not/real"
    calls = json.loads((step_dir / "calls.json").read_text())
    assert any(entry.get("path") == "/v1/audio/speech" and entry.get("status") == 400 for entry in calls)
    # The live activity tells the run as it went: the agent's start, its check and trial, the refused speech call,
    # its end, and the review's model call.
    kinds = [line["kind"] for line in activity]
    assert kinds[0] == "agent" and "the coding agent started" in activity[0]["text"]
    assert any(line["kind"] == "check" and line["text"].startswith("validation passed") for line in activity)
    assert any(line["text"].startswith("trial 1 running the changed harness: Say hello out loud") for line in activity)
    refused = [line for line in activity if line["kind"] == "provider"]
    assert refused and refused[0]["failed"] and refused[0]["text"].startswith("/v1/audio/speech not/real → 400")
    assert any(line["kind"] == "agent" and "exited 0" in line["text"] for line in activity)
    assert kinds[-2:] == ["model", "model"] and activity[-1]["text"].startswith("served-model answered in")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "failure"),
    [
        ("idle", "the agent changed no file"),
        ("stream-error", "the agent's model response failed: Stream ended without finish_reason"),
        ("budget", "the agent's model response failed: the model's reply hit its token budget"),
    ],
)
def test_an_agent_that_changes_nothing_or_fails_its_model_call_hands_back_no_change(
    tmp_path, upstream, mode, failure
) -> None:
    """A run that writes nothing, or that pi ends on a failed model response (exiting zero all the same), is no
    answer. A reasoning model that spends the reply budget on its reasoning ends the turn with nothing: the step
    names the budget, so the person raises it instead of guessing."""
    workspace, services = agent_step(tmp_path, served_of(upstream))
    answered = answer_with_agent(
        workspace, services, TREE, ASK, agent_of(fake_pi(tmp_path, mode), provider_of(upstream))
    )
    assert answered.patch.changes == {} and answered.notes["failure"].startswith(failure)
    assert answered.notes["agent"]["exit_code"] == 0


@pytest.mark.unit
def test_off_pi_the_agent_stays_idle_and_the_served_model_answers(tmp_path) -> None:
    """The agent's rules and trials are pi's: on another adapter a method that holds an agent answers in text."""
    workspace = step_workspace(tmp_path, adapter="dsh")
    served = Served(designed({"rules.md": "Run the tests first.\n"}), json.dumps(REVIEW))
    method = ReefineMethod(agent_of(tmp_path / "no-such-pi"))
    method.step(StepContext(workspace, step_services(workspace, ModelBindings(served=served))))
    assert workspace.tree() == TREE.apply(TreePatch({"rules.md": "Run the tests first.\n"}))
    assert workspace.read_output().notes["review"] == REVIEW and "This harness is" in served.prompts[1]


@pytest.mark.unit
def test_a_run_past_its_limit_keeps_its_tree_and_progress_without_publishing(tmp_path, monkeypatch) -> None:
    def launch(*args, root, workspace, **kwargs):
        reference = (root / "scratch" / "reference" / "pi-extension-api.md").read_text()
        assert "pi.getActiveTools()" in reference and "**replaces**" in reference
        (workspace / "rules.md").write_text("Unfinished candidate")
        (root / "output" / "progress.md").write_text("API confirmed; tool isolation remains unchecked.")
        raise ProgramTimeout("deadline")

    monkeypatch.setattr(reefine_agent, "launch_pi", launch)
    workspace, services = agent_step(tmp_path, ModelBindings(served=ModelBinding(base_url="http://unused", model="m")))
    answered = answer_with_agent(workspace, services, TREE, ASK, agent_of(tmp_path / "pi"))
    assert answered.patch.changes == {} and "past its" in answered.notes["failure"]
    assert (workspace.output / "agent-harness" / "rules.md").read_text() == "Unfinished candidate"
    assert "remains unchecked" in (workspace.output / "progress.md").read_text()


@pytest.mark.unit
def test_the_tree_reads_back_only_the_changes_a_request_may_make(tmp_path) -> None:
    workspace, services = agent_step(tmp_path, ModelBindings(served=ModelBinding(base_url="http://m", model="m")))
    run = AgentRun(workspace, services, TREE, agent_of(tmp_path / "pi"))
    assert run.read() == (TREE, [])
    harness = workspace.harness
    (harness / "skills" / "answer-style" / "SKILL.md").write_text("---\nname: answer-style\ndescription: x\n---\nnew")
    (harness / "rules.md").unlink()
    for relative, text in {
        "extensions/speak.ts": "export default function (pi) {}",
        "commands/tone.md": "Say it again.",
        "config/primary.json": "{}",  # a config file, which a request on pi may not change
        "notes.txt": "scratch",
    }.items():
        (harness / relative).parent.mkdir(parents=True, exist_ok=True)
        (harness / relative).write_text(text)
    candidate, unread = run.read()
    assert TREE.diff(candidate).changes == {
        "commands/tone.md": "Say it again.",
        "extensions/speak.ts": "export default function (pi) {}",
        "rules.md": None,
        "skills/answer-style/SKILL.md": "---\nname: answer-style\ndescription: x\n---\nnew",
    }
    assert unread == [
        "harness/config/primary.json is a file a request may not change, so it is not read",
        "harness/notes.txt is not a path of the harness tree, so it is not read",
    ]
    # A link could point outside the tree: a file a request may write that is a link fails the check.
    (harness / "commands" / "linked.md").symlink_to(harness / "notes.txt")
    checked = run.check()
    assert checked["valid"] is False and checked["problems"] == ["harness/commands/linked.md is a symbolic link"]


@pytest.mark.unit
def test_a_binary_scratch_file_and_the_links_of_an_install_leave_the_change_readable(tmp_path) -> None:
    """One stray file does not cost the request: a binary off the layout and the node_modules/ of an npm install,
    its .bin links included, are left out and named, the directory once. A file a request may write must be text."""
    workspace, services = agent_step(tmp_path, ModelBindings(served=ModelBinding(base_url="http://m", model="m")))
    run = AgentRun(workspace, services, TREE, agent_of(tmp_path / "pi"))
    harness = workspace.harness
    (harness / "rules.md").write_text("Reply in French.\n")
    (harness / "out.mp3").write_bytes(b"\xff\xfb\x90\x00")
    (harness / "node_modules" / "say").mkdir(parents=True)
    (harness / "node_modules" / "say" / "cli.js").write_text("#!/usr/bin/env node\n")
    (harness / "node_modules" / ".bin").mkdir()
    (harness / "node_modules" / ".bin" / "say").symlink_to("../say/cli.js")
    checked = run.check()
    assert checked["valid"] is True and checked["files"] == {"rules.md": "written"}
    assert checked["unread"] == [
        "harness/node_modules/ holds no file of the harness tree, so nothing in it is read",
        "harness/out.mp3 is not a path of the harness tree, so it is not read",
    ]
    (harness / "extensions").mkdir()
    (harness / "extensions" / "speak.ts").write_bytes(b"\xff\xfe")
    assert run.check()["problems"][0].startswith("harness/extensions/speak.ts is not UTF-8 text")


@pytest.mark.unit
def test_concurrent_tool_calls_pull_the_sandbox_tree_one_at_a_time(tmp_path) -> None:
    """pi runs the tool calls of one turn at once, and each pull replaces the tree's directory: they take turns, and
    every one answers from a whole copy."""
    workspace, services = agent_step(tmp_path, ModelBindings(served=ModelBinding(base_url="http://m", model="m")))
    (workspace.harness / "skills" / "answer-style" / "SKILL.md").write_text("---\nname: x\n---\nnew")
    archive = pack(workspace.harness)

    class Sandbox:
        """What a pull asks of the E2B sandbox, counting how many pulls are inside it at once."""

        def __init__(self) -> None:
            self.commands = self
            self.files = self
            self.inside = 0
            self.most_at_once = 0
            self.counter = threading.Lock()

        def run(self, command: str, **options: object) -> None:
            return None

        def read(self, path: str, format: str = "text") -> bytes:
            with self.counter:
                self.inside += 1
                self.most_at_once = max(self.most_at_once, self.inside)
            time.sleep(0.05)  # long enough for unserialized pulls to overlap
            with self.counter:
                self.inside -= 1
            return archive

    sandbox = Sandbox()
    run = AgentRun(workspace, services, TREE, agent_of(tmp_path / "pi"))
    run.session = E2BSession(sandbox)
    results: list[dict] = []
    errors: list[BaseException] = []

    def check() -> None:
        try:
            results.append(run.check())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=check) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert all(r["valid"] and r["files"] == {"skills/answer-style/SKILL.md": "written"} for r in results)
    assert len(results) == 6 and sandbox.most_at_once == 1


def gateway_run(tmp_path: Path, *, timeout_s: float = 60.0) -> tuple[AgentRun, ReefineGateway]:
    """An agent run on a fresh step whose gateway is up; the caller stops the gateway."""
    workspace, services = agent_step(tmp_path, ModelBindings(served=ModelBinding(base_url="http://unused", model="m")))
    run = AgentRun(workspace, services, TREE, agent_of(tmp_path / "pi", timeout_s=timeout_s))
    gateway = ReefineGateway(run.served, services.calls, (CheckTool(run), TrialTool(run)), None)
    run.gateway = gateway
    gateway.start()
    return run, gateway


@pytest.mark.unit
def test_a_trial_reserves_finish_time_and_records_the_exact_candidate(tmp_path, monkeypatch) -> None:
    timeouts = []

    def launch(*args, timeout, **kwargs):
        timeouts.append(timeout)
        raise ProgramTimeout("trial deadline")

    monkeypatch.setattr(reefine_agent, "launch_pi", launch)
    run, gateway = gateway_run(tmp_path, timeout_s=100)
    try:
        (run.workspace.harness / "rules.md").write_text("A changed rule.\n")
        run.deadline = time.monotonic() + 15
        first = run.trial("hello")
        assert first["ran"] and "timed_out" in first and 0 < timeouts[0] <= 5
        assert first["candidate"] == run.check()["candidate"]
        (run.workspace.harness / "rules.md").write_text("Another changed rule.\n")
        assert first["candidate"] != run.check()["candidate"]
        run.deadline = time.monotonic() + 9
        refused = run.trial("another trial")
        assert not refused["ran"] and len(timeouts) == 1
        run.write_checks()
        records = [json.loads(line) for line in (run.workspace.output / "agent-checks.jsonl").read_text().splitlines()]
        assert records[0]["candidate"] == first["candidate"] and records[-1]["ran"] is False
    finally:
        gateway.stop()


@pytest.mark.unit
def test_the_agent_can_try_the_head_tree_before_it_changes_a_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reefine_agent, "launch_pi", lambda *args, **kwargs: (ProcessOutcome(0, "", ""), []))
    run, gateway = gateway_run(tmp_path)
    try:
        assert run.check()["valid"]
        assert run.trial("hello")["ran"]
    finally:
        gateway.stop()


@pytest.mark.unit
def test_a_scripted_driver_failure_is_returned_to_the_agent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        reefine_agent, "launch_pi", lambda *args, **kwargs: (ProcessOutcome(1, "", "node could not load pi"), [])
    )
    run, gateway = gateway_run(tmp_path)
    try:
        (run.workspace.harness / "rules.md").write_text("A changed rule.\n")
        result = run.trial("", trial_script({"steps": [{"prompt": "hello", "expect": {"model_called": True}}]}))
        assert not result["passed"] and result["stderr_tail"] == "node could not load pi"
    finally:
        gateway.stop()


@pytest.mark.unit
def test_the_gateway_serves_the_served_model_and_the_provider_and_nothing_else(upstream) -> None:
    record: list[dict] = []
    calls = StepCalls(3, record)
    served = ModelBinding(base_url=upstream.url, model="served-model", api_key="served-key")
    gateway = ReefineGateway(served, calls, no_tools(), provider_of(upstream))
    gateway.start()
    try:
        base = gateway.base_url
        status, body = post(
            f"{base}/v1/chat/completions", {"model": "expensive/other", "stream": True, "messages": []}
        )
        assert status == 200 and b"data: [DONE]" in body
        chat = upstream.requests[-1]
        assert (chat["body"]["model"], chat["auth"]) == ("served-model", "Bearer served-key")
        assert record[-1]["reply"] == "hello" and record[-1]["source"] == "agent"

        status, body = post(f"{base}/v1/audio/speech", {"model": "real/tts", "input": "hi"})
        assert (status, body) == (200, b"ID3audio")
        assert upstream.requests[-1]["auth"] == "Bearer sk-or-test"
        assert (record[-1]["path"], record[-1]["status"], record[-1]["source"]) == ("/v1/audio/speech", 200, "agent")
        assert gateway.provider_calls_since(0) == [{"path": "/v1/audio/speech", "model": "real/tts", "status": 200}]

        models = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"{base}/models?modality=speech")
        assert json.loads(models.read()) == {"data": [{"id": "real/tts"}]}
        assert upstream.gets[-1] == ("/api/v1/models?output_modalities=speech", "Bearer sk-or-test")

        assert post(f"{base}/trial", {"task": "say hi"}) == (200, b'{"task": "say hi"}')
        assert post(f"{base}/reef/scenarios/s/activate", {})[0] == 404
        wrong = base.replace(base.rsplit("/", 1)[1], "not-the-token")
        assert post(f"{wrong}/v1/chat/completions", {"messages": []})[0] == 404
        # The budget of three is spent: the chat, the speech call, and this one is refused.
        post(f"{base}/v1/chat/completions", {"messages": []})
        status, body = post(f"{base}/v1/chat/completions", {"messages": []})
        assert status == 429 and b"budget" in body
    finally:
        gateway.stop()


@pytest.mark.unit
def test_without_a_provider_a_provider_call_is_501(upstream) -> None:
    gateway = ReefineGateway(ModelBinding(base_url=upstream.url, model="m"), StepCalls(0, []), no_tools(), None)
    gateway.start()
    try:
        assert post(f"{gateway.base_url}/v1/images", {"model": "x", "prompt": "reef"})[0] == 501
    finally:
        gateway.stop()


@pytest.mark.unit
def test_the_gateway_notes_each_tool_the_agent_calls_and_what_it_says(upstream) -> None:
    calls = StepCalls(0, [])
    gateway = ReefineGateway(ModelBinding(base_url=upstream.url, model="m"), calls, no_tools(), None)
    gateway.start()
    try:
        upstream.answer = {
            "choices": [{"message": {"tool_calls": [{"function": {"name": "edit", "arguments": '{"path": "x.md"}'}}]}}]
        }
        post(f"{gateway.base_url}/v1/chat/completions", {"messages": []})
        upstream.answer = {"choices": [{"message": {"content": "Done: the extension speaks.\nMore detail."}}]}
        post(f"{gateway.base_url}/v1/chat/completions", {"messages": []})
    finally:
        gateway.stop()
    assert [(line["kind"], line["text"]) for line in calls.activity()] == [
        ("agent", "edit x.md"),
        ("agent", "Done: the extension speaks."),
    ]


@pytest.mark.unit
def test_an_isolated_sandbox_runs_the_jail_under_pasta_with_every_forward_named(tmp_path) -> None:
    executor = SandboxExecutor(network="isolated", forward_ports=(4123,))
    argv = executor.bwrap_argv(["pi"], root=tmp_path, workspace=tmp_path / "workspace", env={})
    separator = argv.index("--")
    assert argv[:separator] == [
        "pasta",
        "--config-net",
        "--no-map-gw",
        "--quiet",
        "-t",
        "none",
        "-u",
        "none",
        "-T",
        "4123",
        "-U",
        "none",
    ]
    assert argv[separator + 1] == "bwrap" and "--unshare-net" not in argv
    assert "--unshare-net" in SandboxExecutor().bwrap_argv(["pi"], root=tmp_path, workspace=tmp_path, env={})
    # A run's session adds its ports to the isolated network's; the default network already reaches the host.
    session = executor.open_session(forward_ports=(4124,), timeout_s=60)
    assert isinstance(session, SandboxExecutor) and session.forward_ports == (4123, 4124)
    assert executor.open_session(forward_ports=(4123,), timeout_s=60) is executor
    assert SandboxExecutor().open_session(forward_ports=(4124,), timeout_s=60) == SandboxExecutor()


@pytest.mark.unit
def test_the_agent_is_told_what_its_deployments_provider_serves() -> None:
    assert "no multimodal provider" in agent_rules(None)
    compatible = MultimodalProvider(PRESETS["openai-compatible"], "https://gateway.example", "k")
    rules = agent_rules(compatible)
    assert "openai-compatible (https://gateway.example)" in rules and "`/v1/decisions`" not in rules
    assert "$REEF_AGENT_URL/models?modality=speech" in rules and "<!-- provider -->" not in rules
    assert "(or `image`, `embeddings`)." in rules
    openrouter = MultimodalProvider(PRESETS["openrouter"], "https://openrouter.ai/api", "k")
    assert "(or `image`, `embeddings`, `decisions`)." in agent_rules(openrouter)


@pytest.mark.unit
def test_a_replys_tool_calls_read_the_same_in_every_dialect() -> None:
    chat_stream = "\n".join(
        "data: " + json.dumps(event)
        for event in (
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "write", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "a.ts",'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "content": "x"}'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "bash", "arguments": "{}"}}]}}]},
        )
    )
    assert reply_tool_calls(chat_stream + "\ndata: [DONE]") == [
        ("write", '{"path": "a.ts", "content": "x"}'),
        ("bash", "{}"),
    ]
    whole = {"choices": [{"message": {"tool_calls": [{"function": {"name": "read", "arguments": '{"path": "b"}'}}]}}]}
    assert reply_tool_calls(json.dumps(whole)) == [("read", '{"path": "b"}')]
    anthropic = "\n".join(
        "data: " + json.dumps(event)
        for event in (
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "bash"}},
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"comm'},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'and": "ls"}'},
            },
        )
    )
    assert reply_tool_calls(anthropic) == [("bash", '{"command": "ls"}')]
    done = {
        "type": "response.output_item.done",
        "item": {"type": "function_call", "call_id": "c1", "name": "harness_trial", "arguments": '{"task": "speak"}'},
    }
    assert reply_tool_calls("data: " + json.dumps(done)) == [("harness_trial", '{"task": "speak"}')]
    assert reply_tool_calls('{"choices": [{"message": {"content": "just text"}}]}') == []

    assert tool_summary("write", '{"content": "long", "path": "harness/extensions/speak.ts"}') == (
        "write harness/extensions/speak.ts"
    )
    assert tool_summary("bash", '{"command": "curl -s\\n  https://x"}') == "bash curl -s https://x"
    assert tool_summary("harness_check", "{}") == "harness_check"
    assert tool_summary("odd", "not json") == "odd not json"


@pytest.mark.unit
@pytest.mark.parametrize(
    "script",
    [
        {},
        {"steps": []},
        {"steps": [{"prompt": "hello"}]},
        {"steps": [{"new_session": True, "prompt": "hello"}]},
        {"steps": [{"prompt": "hello", "expect": {"unknown": True}}]},
        {"steps": [{"prompt": "hello", "expect": {"model_called": "yes"}}]},
        {"steps": [{"prompt": "hello", "expect": {"tools": "read"}}]},
        {"steps": [{"prompt": "hello", "expect": {"executed_tools": ["read"]}}]},
        {"fixture_tools": ["read"], "steps": [{"prompt": "hello", "expect": {"model_called": True}}]},
        {"fixture_tools": ["x", "x"], "steps": [{"prompt": "hello", "expect": {"model_called": True}}]},
        {
            "steps": [
                {"prompt": "hello", "tool_call": {"name": "x", "arguments": []}, "expect": {"model_called": True}}
            ]
        },
    ],
)
def test_script_rejects_ambiguous_or_unobservable_checks(script) -> None:
    with pytest.raises(ValueError):
        trial_script(script)


@pytest.mark.unit
def test_gateway_validates_scripted_trials_and_preserves_online_tasks(upstream) -> None:
    gateway = ReefineGateway(ModelBinding(base_url=upstream.url, model="m"), StepCalls(0, []), no_tools(), None)
    gateway.start()
    try:
        script = {"steps": [{"prompt": "hello", "expect": {"model_called": True}}]}
        status, body = post(f"{gateway.base_url}/trial", {"script": script})
        assert status == 200 and json.loads(body)["script"] == {**script, "fixture_tools": []}
        assert post(f"{gateway.base_url}/trial", {"script": script, "task": "hello"})[0] == 400
        assert post(f"{gateway.base_url}/trial", {"script": {"steps": []}})[0] == 400
        assert post(f"{gateway.base_url}/trial", {"task": "hello"}) == (200, b'{"task": "hello"}')
    finally:
        gateway.stop()
