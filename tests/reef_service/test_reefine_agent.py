"""Reefine's agent proposer: the workspace round trip, the gateway, and a whole run with a stand-in pi."""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.e2b import E2BSession, pack
from reef.harness.episodes.executor import EpisodeTimeout, LocalExecutor, ProcessOutcome
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.tree.mutations import Mutation
from reef.recipe.reefine import agent as reefine_agent
from reef.recipe.reefine.agent import AgentProposer, AgentRun, workspace_mutations, write_workspace
from reef.recipe.reefine.agent_gateway import AgentGateway, WorkspaceTools, reply_tool_calls, tool_summary
from reef.recipe.reefine.multimodal import PRESETS, MultimodalProvider
from reef.recipe.reefine.trial import trial_script
from reef.train.cordis_backend.backend import _budgeted_bindings, _StepCalls
from reef.train.cordis_backend.strategies import AgentHost, StepProposal

REVIEW = {"result": "complete", "covered": ["reads answers aloud"], "uncovered": [], "delivers": True}

ENTRIES = [
    {"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": "---\nname: x\n---\nold"}},
    {"id": "tone", "name": "rules", "config": {"text": "Be brief."}},
    {"id": "reef-requests", "name": "code_extension", "config": {"name": "reef-requests", "code": "// reef"}},
    {"id": "reef-pi-extension-api", "name": "skill", "config": {"name": "reef-pi-extension-api", "text": "api"}},
]
NODES = tuple((entry["name"], entry["config"]) for entry in ENTRIES)


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


class NoTools(WorkspaceTools):
    def check(self) -> dict:
        return {"admitted": True}

    def trial(self, task: str, script: dict[str, object] | None = None) -> dict:
        return {"task": task} if script is None else {"script": script}


@pytest.mark.unit
def test_the_workspace_round_trip_gives_only_the_changes_and_never_touches_reefs_own(tmp_path) -> None:
    write_workspace(tmp_path, ENTRIES)
    harness = tmp_path / "harness"
    assert (harness / "skills" / "answer-style.md").read_text() == "---\nname: x\n---\nold"
    assert (tmp_path / "reserved" / "reef-pi-extension-api.md").read_text() == "api"
    assert workspace_mutations(tmp_path, ENTRIES, NODES) == ([], [])

    (harness / "skills" / "answer-style.md").write_text("---\nname: x\n---\nnew")
    (harness / "extensions" / "speak.ts").write_text("export default function (pi) {}")
    (harness / "rules" / "tone.md").unlink()
    (harness / "commands" / "tone.md").write_text("Say it again.")  # the same id, another kind
    (tmp_path / "reserved" / "reef-requests.ts").write_text("// edited")  # ignored
    (harness / "notes.txt").write_text("scratch")
    mutations, problems = workspace_mutations(tmp_path, ENTRIES, NODES)
    assert [(m.op, m.id, (m.options or {}).get("name")) for m in mutations] == [
        ("update", "answer-style", "skill"),
        ("remove", "tone", None),
        ("create", "tone", "agent_command"),
        ("create", "speak", "code_extension"),
    ]
    assert mutations[0].options["config"] == {"name": "answer-style", "text": "---\nname: x\n---\nnew"}
    assert problems == ["harness/notes.txt is not read"]


@pytest.mark.unit
def test_the_gateway_serves_the_served_model_and_the_provider_and_nothing_else(upstream) -> None:
    record: list[dict] = []
    calls = _StepCalls(3, record)
    served = ModelBinding(base_url=upstream.url, model="served-model", api_key="served-key")
    gateway = AgentGateway(served, calls, NoTools(), provider_of(upstream))
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
        assert post(f"{base}/reef/scenarios/s/promote", {})[0] == 404
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
    gateway = AgentGateway(ModelBinding(base_url=upstream.url, model="m"), _StepCalls(0, []), NoTools(), None)
    gateway.start()
    try:
        assert post(f"{gateway.base_url}/v1/images", {"model": "x", "prompt": "reef"})[0] == 501
    finally:
        gateway.stop()


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
    assert pathlib.Path(os.environ["PI_CODING_AGENT_DIR"], "extensions", "{tools}.ts").exists() == (
        "REEF_PROPOSER_URL" in os.environ)
    # The episode environment is minimal, so the test sets the mode in a file beside the binary.
    mode_file = pathlib.Path(sys.argv[0]).with_name("mode")
    mode = mode_file.read_text() if mode_file.exists() else ""
    if "REEF_PROPOSER_URL" in os.environ:
        base = os.environ["REEF_PROPOSER_URL"]
        if mode == "sleep":
            time.sleep(30)
        if mode != "idle":
            pathlib.Path("design.md").write_text("Speak each answer with a TTS model.")
            pathlib.Path("harness/extensions/speak.ts").write_text(
                "export default function (pi) {{ if (process.env.PI_OFFLINE) return; }}")
            pathlib.Path("harness/requires.json").write_text(json.dumps(
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


def agent_host(tmp_path: Path, record: list[dict], *, timeout_s: float = 60.0) -> AgentHost:
    binary = tmp_path / "bin" / "pi"
    binary.parent.mkdir()
    binary.write_text(FAKE_PI.format(python=sys.executable, tools=reefine_agent.TOOLS_ENTRY_ID))
    binary.chmod(0o755)
    step_dir = tmp_path / "step"
    step_dir.mkdir()
    return AgentHost(
        descriptor=get_adapter("pi"),
        binary=str(binary),
        executor=LocalExecutor(),
        step_dir=step_dir,
        calls=_StepCalls(0, record),
        timeout_s=timeout_s,
        trial_timeout_s=30.0,
    )


def served_models(upstream: Upstream, record: list[dict], host: AgentHost) -> ModelBindings:
    models = ModelBindings(served=ModelBinding(base_url=upstream.url, model="served-model", api_key="served-key"))
    return _budgeted_bindings(models, host.calls)  # the step hands the proposer these


@pytest.mark.unit
def test_the_agent_writes_the_change_tries_it_and_hands_it_back_reviewed(tmp_path, upstream) -> None:
    record: list[dict] = []
    host = agent_host(tmp_path, record)
    request = {"id": "r1", "text": "Read my answers aloud with a natural voice", "requires": []}
    proposal = AgentProposer(provider_of(upstream))(
        NODES, (), served_models(upstream, record, host), requests=[request], entries=ENTRIES, agent_host=host
    )
    assert isinstance(proposal, StepProposal)
    assert [(m.op, m.id) for m in proposal.mutations] == [("create", "speak")]
    notes = proposal.notes
    assert notes["design"] == "Speak each answer with a TTS model."
    assert notes["review"]["result"] == "complete"
    assert notes["agent"]["trials"] == 1 and notes["agent"]["exit_code"] == 0
    assert request["requires"] == [
        {"name": "afplay", "kind": "service", "check": "command -v afplay", "prompt": "Install a player"}
    ]
    # The trial's speech call reached the provider and its refusal is on record for the agent to read.
    speech = [r for r in upstream.requests if r["path"] == "/api/v1/audio/speech"]
    assert speech and speech[0]["body"]["model"] == "not/real"
    assert any(entry.get("path") == "/v1/audio/speech" and entry.get("status") == 400 for entry in record)
    assert (host.step_dir / "agent-session.jsonl").read_text().count('"done"') == 1
    # The live activity tells the run as it went: the agent's start, its check and trial, the refused speech call,
    # its end, and the review's model call.
    activity = host.calls.activity()
    kinds = [line["kind"] for line in activity]
    assert kinds[0] == "proposer" and "the coding agent started" in activity[0]["text"]
    assert any(line["kind"] == "check" and line["text"].startswith("admission passed") for line in activity)
    assert any(line["text"].startswith("trial 1 running the changed harness: Say hello out loud") for line in activity)
    speech = [line for line in activity if line["kind"] == "provider"]
    assert speech and speech[0]["failed"] and speech[0]["text"].startswith("/v1/audio/speech not/real → 400")
    assert any(
        line["kind"] == "trial" and line["text"].startswith("trial 1 exited") and line["failed"] for line in activity
    )
    assert any(line["kind"] == "proposer" and "exited 0" in line["text"] for line in activity)
    assert kinds[-2:] == ["model", "model"] and activity[-1]["text"].startswith("served-model answered in")


@pytest.mark.unit
def test_concurrent_tool_calls_pull_the_sandbox_workspace_one_at_a_time(tmp_path) -> None:
    """pi runs the tool calls of one turn at once, and each pull replaces the workspace directory: they take turns,
    and every one answers from a whole copy."""
    workspace = tmp_path / "workspace"
    write_workspace(workspace, ENTRIES)
    (workspace / "harness" / "skills" / "answer-style.md").write_text("---\nname: x\n---\nnew")
    archive = pack(workspace)

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
    run = AgentRun(agent_host(tmp_path, []), ENTRIES, NODES, workspace, ModelBinding(base_url="http://m", model="m"))
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
    assert all(
        r["admitted"] and r["mutations"] == [{"op": "update", "id": "answer-style", "kind": "skill"}] for r in results
    )
    assert len(results) == 6 and sandbox.most_at_once == 1


@pytest.mark.unit
def test_an_agent_that_changes_nothing_or_runs_out_of_time_hands_back_no_mutation(tmp_path, upstream) -> None:
    record: list[dict] = []
    host = agent_host(tmp_path, record)
    models = served_models(upstream, record, host)
    mode = Path(host.binary).with_name("mode")
    mode.write_text("idle")
    propose = AgentProposer(provider_of(upstream))
    idle = propose(NODES, (), models, requests=[{"text": "x"}], entries=ENTRIES, agent_host=host)
    assert idle.mutations == () and idle.notes["failure"] == "the agent changed no entry"

    mode.write_text("sleep")
    slow_host = replace(host, timeout_s=1.0, step_dir=None)
    slow = propose(NODES, (), models, requests=[{"text": "x"}], entries=ENTRIES, agent_host=slow_host)
    assert slow.mutations == () and "past its" in slow.notes["failure"]


@pytest.mark.unit
def test_an_agent_with_a_failed_model_response_does_not_publish_its_partial_changes(tmp_path, upstream) -> None:
    record: list[dict] = []
    host = agent_host(tmp_path, record)
    Path(host.binary).with_name("mode").write_text("stream-error")
    proposal = AgentProposer(provider_of(upstream))(
        NODES, (), served_models(upstream, record, host), requests=[{"text": "x"}], entries=ENTRIES, agent_host=host
    )
    assert proposal.mutations == ()
    assert proposal.notes["agent"]["exit_code"] == 0
    assert proposal.notes["failure"] == "the agent's model response failed: Stream ended without finish_reason"


@pytest.mark.unit
def test_a_reply_cut_at_its_token_budget_says_so_rather_than_that_nothing_changed(tmp_path, upstream) -> None:
    """A reasoning model spends the reply budget on its reasoning and answers with nothing, which pi reads as the
    end of the turn and exits zero. The step names the budget, so the person raises it instead of guessing."""
    record: list[dict] = []
    host = agent_host(tmp_path, record)
    Path(host.binary).with_name("mode").write_text("budget")
    proposal = AgentProposer(provider_of(upstream))(
        NODES, (), served_models(upstream, record, host), requests=[{"text": "x"}], entries=ENTRIES, agent_host=host
    )
    assert proposal.mutations == () and proposal.notes["agent"]["exit_code"] == 0
    assert "hit its token budget" in proposal.notes["failure"] and "maxTokens" in proposal.notes["failure"]


@pytest.mark.unit
def test_without_a_request_or_an_agent_the_text_proposer_answers(monkeypatch) -> None:
    seen = []

    def text_proposer(nodes, samples, models, *, requests=(), entries=(), adapter=None):
        seen.append((tuple(requests), adapter))
        return Mutation("remove", "tone")

    monkeypatch.setattr(reefine_agent.evolution, "propose", text_proposer)
    models = ModelBindings(served=ModelBinding(base_url="http://127.0.0.1:9", model="m"))
    propose = AgentProposer()
    assert propose(NODES, (), models, entries=ENTRIES).id == "tone"
    assert propose(NODES, (), models, requests=[{"text": "x"}], entries=ENTRIES, agent_host=None).id == "tone"
    # A request for another adapter's harness goes to the text proposer with the adapter named, whatever the host.
    assert propose(NODES, (), models, requests=[{"text": "y"}], entries=ENTRIES, adapter="claude").id == "tone"
    assert seen == [((), None), (({"text": "x"},), None), (({"text": "y"},), "claude")]


@pytest.mark.unit
def test_an_isolated_sandbox_runs_the_jail_under_pasta_with_every_forward_named(tmp_path) -> None:
    from reef.harness.episodes.executor import SandboxExecutor

    executor = SandboxExecutor(network="isolated", forward_ports=(4123,))
    argv = executor._bwrap_argv(["pi"], root=tmp_path, workspace=tmp_path / "workspace", env={})
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
    assert "--unshare-net" in SandboxExecutor()._bwrap_argv(["pi"], root=tmp_path, workspace=tmp_path, env={})


@pytest.mark.unit
def test_proposer_agent_settings_choose_the_isolation(monkeypatch) -> None:
    from reef.harness.episodes.e2b import E2BExecutor
    from reef.harness.episodes.executor import SandboxExecutor, SandboxUnavailable
    from reef.recipe.cordis import proposer_agent_settings
    from reef.recipe.errors import RecipeConfigError

    assert proposer_agent_settings(None, {}) == (None, 1800.0, 300.0)
    executor, timeout_s, trial_s = proposer_agent_settings({"sandbox": "none", "timeout_s": 60}, {})
    assert isinstance(executor, LocalExecutor) and (timeout_s, trial_s) == (60.0, 300.0)
    assert isinstance(proposer_agent_settings({}, {"REEF_PROPOSER_SANDBOX": "none"})[0], LocalExecutor)
    with pytest.raises(RecipeConfigError, match="'bwrap', 'e2b' or 'none'"):
        proposer_agent_settings({"sandbox": "docker"}, {})
    # E2B: the key from the section, else the environment; none at all is refused at startup.
    with pytest.raises(RecipeConfigError, match=r"sandbox is e2b, but .*E2B_API_KEY"):
        proposer_agent_settings({"sandbox": "e2b"}, {})
    monkeypatch.setattr(E2BExecutor, "preflight", lambda self: None)  # the e2b package is an extra
    remote, timeout_s, _ = proposer_agent_settings({"sandbox": "e2b", "timeout_s": 900}, {"E2B_API_KEY": "e2b-env"})
    assert isinstance(remote, E2BExecutor) and remote.api_key == "e2b-env" and remote.timeout_s == 900.0
    assert remote.template == "" and "e2b-env" not in repr(remote)
    configured = proposer_agent_settings(
        {"sandbox": "e2b", "e2b_api_key": "e2b-cfg", "e2b_template": "mine"}, {"E2B_API_KEY": "e2b-env"}
    )[0]
    assert isinstance(configured, E2BExecutor) and (configured.api_key, configured.template) == ("e2b-cfg", "mine")
    monkeypatch.undo()
    with pytest.raises(RecipeConfigError, match="positive number"):
        proposer_agent_settings({"timeout_s": 0}, {})

    def unavailable(self) -> None:
        raise SandboxUnavailable("no bwrap here")

    monkeypatch.setattr(SandboxExecutor, "preflight", unavailable)
    # Unset on a host that cannot isolate: the agent is off and the text proposer answers.
    assert proposer_agent_settings({}, {})[0] is None
    with pytest.raises(RecipeConfigError, match="no bwrap here"):
        proposer_agent_settings({"sandbox": "bwrap"}, {})


@pytest.mark.unit
def test_the_agent_is_told_what_its_deployments_provider_serves() -> None:
    from reef.recipe.reefine.agent import agent_rules

    assert "no multimodal provider" in agent_rules(None)
    compatible = MultimodalProvider(PRESETS["openai-compatible"], "https://gateway.example", "k")
    rules = agent_rules(compatible)
    assert "openai-compatible (https://gateway.example)" in rules and "`/v1/decisions`" not in rules
    assert "$REEF_PROPOSER_URL/models?modality=speech" in rules and "<!-- provider -->" not in rules
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
def test_the_gateway_notes_each_tool_the_agent_calls_and_what_it_says(upstream) -> None:
    calls = _StepCalls(0, [])
    gateway = AgentGateway(ModelBinding(base_url=upstream.url, model="m"), calls, NoTools(), None)
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
    gateway = AgentGateway(ModelBinding(base_url=upstream.url, model="m"), _StepCalls(0, []), NoTools(), None)
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


@pytest.mark.unit
def test_trial_reserves_finish_time_and_records_the_exact_candidate(tmp_path, monkeypatch) -> None:
    host = agent_host(tmp_path, [], timeout_s=100)
    workspace = tmp_path / "workspace"
    write_workspace(workspace, ENTRIES)
    run = AgentRun(host, ENTRIES, NODES, workspace, ModelBinding(base_url="http://unused", model="m"))
    gateway = AgentGateway(run.served, host.calls, run, None)
    run.gateway = gateway
    gateway.start()
    timeouts = []

    def launch(*args, timeout, **kwargs):
        timeouts.append(timeout)
        raise EpisodeTimeout("trial deadline")

    monkeypatch.setattr(reefine_agent, "launch_pi", launch)
    try:
        run.deadline = time.monotonic() + 15
        first = run.trial("hello")
        assert first["ran"] and "timed_out" in first and 0 < timeouts[0] <= 5
        assert first["candidate"] == run.check()["candidate"]
        (workspace / "harness/rules/tone.md").write_text("A changed rule.")
        assert first["candidate"] != run.check()["candidate"]
        run.deadline = time.monotonic() + 9
        refused = run.trial("another trial")
        assert not refused["ran"] and len(timeouts) == 1
        records = [json.loads(line) for line in (host.step_dir / "agent-checks.jsonl").read_text().splitlines()]
        assert records[0]["candidate"] == first["candidate"] and records[-1]["ran"] is False
    finally:
        gateway.stop()


@pytest.mark.unit
def test_timeout_keeps_candidate_and_progress_without_publishing(tmp_path, upstream, monkeypatch) -> None:
    host = agent_host(tmp_path, [])

    def launch(*args, root, **kwargs):
        reference = (root / "workspace/reserved/reef-pi-extension-api.md").read_text()
        assert "pi.getActiveTools()" in reference and "**replaces**" in reference
        (root / "workspace/harness/rules/tone.md").write_text("Unfinished candidate")
        (root / "workspace/progress.md").write_text("API confirmed; tool isolation remains unchecked.")
        raise EpisodeTimeout("deadline")

    monkeypatch.setattr(reefine_agent, "launch_pi", launch)
    proposal = AgentProposer()(
        NODES, (), served_models(upstream, [], host), requests=[{"text": "x"}], entries=ENTRIES, agent_host=host
    )
    assert proposal.mutations == () and "past its" in proposal.notes["failure"]
    saved = host.step_dir / "agent-workspace"
    assert (saved / "harness/rules/tone.md").read_text() == "Unfinished candidate"
    assert "remains unchecked" in (saved / "progress.md").read_text()


@pytest.mark.unit
def test_script_driver_failure_is_returned_to_the_agent(tmp_path, monkeypatch) -> None:
    host = agent_host(tmp_path, [])
    workspace = tmp_path / "workspace"
    write_workspace(workspace, ENTRIES)
    run = AgentRun(host, ENTRIES, NODES, workspace, ModelBinding(base_url="http://unused", model="m"))
    gateway = AgentGateway(run.served, host.calls, run, None)
    run.gateway = gateway
    gateway.start()

    def launch(*args, **kwargs):
        return ProcessOutcome(1, "", "node could not load pi"), []

    monkeypatch.setattr(reefine_agent, "launch_pi", launch)
    try:
        result = run.trial("", trial_script({"steps": [{"prompt": "hello", "expect": {"model_called": True}}]}))
        assert not result["passed"] and result["stderr_tail"] == "node could not load pi"
    finally:
        gateway.stop()
