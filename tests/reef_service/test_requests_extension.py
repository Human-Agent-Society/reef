"""The harness requests extension: reef-pi's commands and tools, run under node with stubs.

The asset registers nothing under ``PI_OFFLINE``. With a UI the ask command
clarifies the request in the background: it calls the session's model with the
two tools, asks what is unclear through ``reef_ask_user``, files through
``reef_file_request`` and keeps the clarification as one collapsed chat entry,
out of the session's context; with ``--direct``
or headless it posts the request with the session id and the release file's
release, leaves inference receipts available for feedback, and reports
durable acceptance with a link to the request's page. A watch then reports
the step's result in the session as a custom message the chat keeps, and
the filed requests are stored beside the release file until reported, so a
session start reports what settled while pi was away. The result opens no
dialog; /versions <step> install starts the install through the
``reef-pi`` wrapper (its update, then the setup loop that asks once for what
the release needs). The versions command lists the chain, prints a step's
page link and promotes a pending release after a confirmation, then offers its install;
session start says the commands exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from reef.core.training_request import CLIENT_COMMANDS, TrainingRequest

ASSET = Path(__file__).parents[2] / "reef" / "harness" / "adapters" / "pi" / "requests.ts"
SKILL = Path(__file__).parents[2] / "reef" / "harness" / "adapters" / "pi" / "pi_extension_api.md"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

#: The page key the service hands out with a filed request and the catalog, in place of the token.
PAGE_KEY = "k3y-from-the-service"
ACCEPTED = {
    "agent_record_id": "q-1",
    "scenario": "code-repair",
    "request_type": "train",
    "page_path": f"/reef/harness/requests/q-1/page?scenario=code-repair&key={PAGE_KEY}",
}


def _listed(catalog: dict) -> dict:
    """``catalog`` as the service answers it: each row with its step page's path, the page key in its query."""
    rows = [
        {**row, "page_path": f"/reef/harness/releases/{step}/page?scenario=code-repair&key={PAGE_KEY}"}
        for step, row in enumerate(catalog["releases"])
    ]
    return {**catalog, "releases": rows}


# One runner for every case: it loads the asset with a stub pi, a stub ctx and a stub fetch, runs the command,
# tool or event TEST_STEP names (TEST_REPEAT times), waits TEST_WAIT_MS for the watch, and prints what the
# extension registered and every call it made. TEST_ARGS_2 is what a second TEST_REPEAT run passes instead of
# TEST_ARGS. TEST_SELECT and TEST_INPUT script the dialogs, one answer per
# call; TEST_CONFIRM is 1 (yes to every confirm), 0 or unset (no), or a JSON list consumed in order, the last one
# repeating; TEST_EXEC maps a joined argv of a pi.exec call, or a prefix of it, to the {stdout, stderr, code} the
# stub answers (no key: exit 0, nothing printed); TEST_SHUTDOWN_AFTER_MS fires session_shutdown mid run;
# TEST_CLOCK_SKEW_AFTER_MS moves the clock forward by TEST_CLOCK_SKEW_MS, past the watch's 30 minute cap by
# default. An answer that is a list is consumed in order, the last one repeating, for a route or a call whose
# answer changes over the polls. TEST_HANG lists the routes ("METHOD path") whose fetch never answers and ends
# only with the caller's abort. TEST_REPLIES scripts the model's replies to the background clarification, one
# per call (a {throw} entry rejects the call, a {hang} entry never answers); TEST_BRANCH is the session branch it reads as background, and
# TEST_NO_MODEL=1 leaves the session without a model.
RUNNER = """
import requests from "./requests.mjs";

const tools = {};
const commands = {};
const shortcuts = {};
const handlers = {};
const events = [];
const entryRenderers = [];
const execAnswers = JSON.parse(process.env.TEST_EXEC || "{}");
const execAnswerFor = (args) => {
  const joined = args.join(" ");
  const keys = Object.keys(execAnswers).filter((key) => joined === key || joined.startsWith(`${key} `));
  const key = keys.sort((a, b) => b.length - a.length)[0];
  const entry = key === undefined ? {} : execAnswers[key];
  const answer = !Array.isArray(entry) ? entry : entry.length > 1 ? entry.shift() : entry[0];
  return { stdout: "", stderr: "", code: 0, killed: false, ...answer };
};
const pi = {
  registerTool(definition) { tools[definition.name] = definition; },
  registerCommand(name, definition) { commands[name] = definition; },
  registerShortcut(key, definition) { shortcuts[key] = definition; },
  on(name, handler) { handlers[name] = handler; },
  sendUserMessage(text, options) { events.push({ kind: "user_message", text, options: options ?? null }); },
  sendMessage(message, options) { events.push({ kind: "message", message, options: options ?? null }); },
  appendEntry(customType, data) { events.push({ kind: "entry", customType, data }); },
  registerEntryRenderer(customType) { entryRenderers.push(customType); },
  exec: async (command, args) => { events.push({ kind: "exec", command, args }); return execAnswerFor(args); },
};
const selections = JSON.parse(process.env.TEST_SELECT || "[]");
const inputs = JSON.parse(process.env.TEST_INPUT || "[]");
const confirmRaw = JSON.parse(process.env.TEST_CONFIRM || "0");
const confirms = Array.isArray(confirmRaw) ? confirmRaw : [confirmRaw === 1];
const confirmAnswer = () => (confirms.length > 1 ? confirms.shift() : confirms[0]) === true;
const replies = JSON.parse(process.env.TEST_REPLIES || "[]");
const ctx = {
  hasUI: process.env.TEST_HEADLESS !== "1",
  model: process.env.TEST_NO_MODEL === "1" ? undefined : { provider: "reef", id: "served" },
  modelRegistry: {
    complete: async (model, context, options) => {
      events.push({ kind: "model_call", model, context: JSON.parse(JSON.stringify(context)), signal: options.signal instanceof AbortSignal });
      const reply = replies.shift();
      if (reply && reply.hang) return new Promise(() => {});
      if (!reply || reply.throw) throw new Error(reply ? reply.throw : "no scripted reply");
      return { role: "assistant", stopReason: "stop", ...reply };
    },
  },
  isIdle: () => process.env.TEST_BUSY !== "1",
  ui: {
    confirm: async (title, message) => { events.push({ kind: "confirm", title, message }); return confirmAnswer(); },
    notify: (message, type) => events.push({ kind: "notify", message, type }),
    select: async (title, options) => { events.push({ kind: "select", title, options }); return selections.shift() ?? undefined; },
    input: async (title, placeholder) => { events.push({ kind: "input", title, placeholder }); return inputs.shift() ?? undefined; },
    setStatus: (key, text) => events.push({ kind: "status", key, text: text ?? null }),
    setWidget: (key, content) => events.push({ kind: "widget", key, content: content ?? null }),
  },
  sessionManager: { getSessionId: () => "sess-1234", getBranch: () => JSON.parse(process.env.TEST_BRANCH || "[]") },
};
const answers = JSON.parse(process.env.TEST_ANSWERS || "{}");
const answerFor = (key) => {
  const entry = answers[key];
  if (!Array.isArray(entry)) return entry;
  return entry.length > 1 ? entry.shift() : entry[0];
};
const hanging = JSON.parse(process.env.TEST_HANG || "[]");
globalThis.fetch = async (url, init = {}) => {
  const method = init.method || "GET";
  const route = `${method} ${new URL(url).pathname}`;
  events.push({ kind: "fetch", method, url, headers: init.headers ?? {}, body: init.body ? JSON.parse(init.body) : null, signal: init.signal instanceof AbortSignal });
  if (hanging.includes(route)) {
    // A hung connection: nothing answers, and only the caller's abort ends the wait, as with node's fetch.
    return new Promise((_, reject) => init.signal?.addEventListener("abort", () => reject(new Error("This operation was aborted"))));
  }
  const answer = answerFor(route);
  if (!answer) throw new Error(`connection refused: ${url}`);
  return { ok: answer.status < 400, status: answer.status, json: async () => answer.body, text: async () => JSON.stringify(answer.body) };
};
const realNow = Date.now;
let skew = 0;
Date.now = () => realNow() + skew;
if (process.env.TEST_CLOCK_SKEW_AFTER_MS) {
  const skewMs = Number(process.env.TEST_CLOCK_SKEW_MS || 31 * 60 * 1000);
  setTimeout(() => { skew = skewMs; }, Number(process.env.TEST_CLOCK_SKEW_AFTER_MS));
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
requests(pi);
const out = { tools: Object.keys(tools), commands: Object.keys(commands), shortcuts: Object.keys(shortcuts), handlers: Object.keys(handlers).sort(), entryRenderers, events, result: null, error: null, fetchesAtShutdown: null };
const params = JSON.parse(process.env.TEST_PARAMS || "{}");
const run = async (index) => {
  const step = process.env.TEST_STEP;
  if (step === "command") {
    const later = index > 0 && process.env.TEST_ARGS_2 !== undefined;
    return commands["reefine"].handler(later ? process.env.TEST_ARGS_2 : process.env.TEST_ARGS || "", ctx);
  }
  if (step === "versions") return commands["versions"].handler(process.env.TEST_ARGS || "", ctx);
  if (step === "ask_user") return tools.reef_ask_user.execute("call-1", params, undefined, undefined, ctx);
  if (step === "file_request") return tools.reef_file_request.execute("call-1", params, undefined, undefined, ctx);
  if (step === "session_start") return handlers.session_start({ type: "session_start", reason: "startup" }, ctx);
  return null;
};
const shutdown = () => handlers.session_shutdown({ type: "session_shutdown" }, ctx);
try {
  const repeats = Number(process.env.TEST_REPEAT || 1);
  for (let index = 0; index < repeats; index++) out.result = await run(index);
} catch (error) {
  out.error = error.message;
}
if (process.env.TEST_SHUTDOWN_AFTER_MS) {
  setTimeout(async () => {
    out.fetchesAtShutdown = events.filter((event) => event.kind === "fetch").length;
    await shutdown();
  }, Number(process.env.TEST_SHUTDOWN_AFTER_MS));
}
// The look-in key, fired the way pi fires it: once per listed delay, so a case can open and close the spinner.
for (const at of JSON.parse(process.env.TEST_SHORTCUT_AT_MS || "[]")) {
  setTimeout(async () => {
    events.push({ kind: "shortcut", key: Object.keys(shortcuts)[0] });
    await shortcuts[Object.keys(shortcuts)[0]].handler(ctx);
  }, Number(at));
}
await sleep(Number(process.env.TEST_WAIT_MS || 0));
console.log(JSON.stringify(out));
// A live watch would keep node running: the shutdown handler clears it, as pi's does.
if (handlers.session_shutdown) await shutdown();
""".strip()


def _install_root(tmp_path: Path, *, with_release_file: bool = True) -> Path:
    """A pulled pi tree: the release file at the root and the models.json that points at the proxy in pi-agent."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir(parents=True)
    if with_release_file:
        (tmp_path / ".reef-harness-release").write_text(json.dumps({"release_id": "v1"}), encoding="utf-8")
    models = {"providers": {"reef": {"api": "openai-completions", "baseUrl": "http://127.0.0.1:4567/v1"}}}
    (agent_dir / "models.json").write_text(json.dumps(models), encoding="utf-8")
    return agent_dir


#: The runner's knobs; each case sets the ones it needs and the rest stay unset.
KNOBS = (
    "TEST_STEP",
    "TEST_ARGS",
    "TEST_ARGS_2",
    "TEST_PARAMS",
    "TEST_ANSWERS",
    "TEST_REPEAT",
    "TEST_CONFIRM",
    "TEST_HEADLESS",
    "TEST_BUSY",
    "TEST_SELECT",
    "TEST_INPUT",
    "TEST_WAIT_MS",
    "TEST_SHUTDOWN_AFTER_MS",
    "TEST_CLOCK_SKEW_AFTER_MS",
    "TEST_CLOCK_SKEW_MS",
    "TEST_HANG",
    "TEST_EXEC",
    "TEST_SHORTCUT_AT_MS",
    "TEST_REPLIES",
    "TEST_BRANCH",
    "TEST_NO_MODEL",
)
#: The environment the extension reads beyond the three it needs; each case sets what it needs and the rest stays unset.
SETTINGS = ("PI_OFFLINE", "REEF_TOKEN", "REEF_HARNESS_WATCH_MS", "REEF_HARNESS_FETCH_MS", "REEF_HARNESS_WRAPPER")


def _run(tmp_path: Path, agent_dir: Path, **env: str) -> dict[str, Any]:
    (tmp_path / "requests.mjs").write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    runner = tmp_path / "runner.mjs"
    runner.write_text(RUNNER, encoding="utf-8")
    full_env = {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "REEF_SERVICE_URL": "http://reef:8900",
        "REEF_SCENARIO": "code-repair",
        "REEF_HARNESS_DEST": str(tmp_path),
        **env,
    }
    for name in (*KNOBS, *SETTINGS):
        if name not in env:
            full_env.pop(name, None)
    completed = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True, env=full_env)
    return json.loads(completed.stdout)


def _fetches(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in out["events"] if event["kind"] == "fetch"]


def _notices(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in out["events"] if event["kind"] == "notify"]


def _dialogs(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in out["events"] if event["kind"] == "confirm"]


def _ask(
    tmp_path: Path,
    agent_dir: Path,
    answers: dict[str, Any],
    text: str = "text me when you are blocked",
    *,
    direct: bool = True,
    **env: str,
) -> dict[str, Any]:
    args = f"--direct {text}" if direct else text
    return _run(tmp_path, agent_dir, TEST_STEP="command", TEST_ARGS=args, TEST_ANSWERS=json.dumps(answers), **env)


def _tool(tmp_path: Path, agent_dir: Path, name: str, params: dict[str, Any], **env: str) -> dict[str, Any]:
    return _run(tmp_path, agent_dir, TEST_STEP=name, TEST_PARAMS=json.dumps(params), **env)


def _of_kind(out: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in out["events"] if event["kind"] == kind]


# What a filing says: the accepted notice ends with the request's page link, which carries the scenario and the
# page key the service handed out as query parameters, since a browser sends no header.
REQUEST_PAGE = "http://reef:8900/reef/harness/requests/q-1/page?scenario=code-repair"
#: What the links carry in place of the token: the page key the service handed out.
KEY = f"&key={PAGE_KEY}"
ACCEPTED_NOTICE = (
    f"Training request q-1 accepted; the step usually takes a few minutes. Watch it here: {REQUEST_PAGE}{KEY}"
)
REQUESTS_FILE = ".reef-harness-requests.json"


def _stored(tmp_path: Path) -> list[dict[str, Any]]:
    """The filed requests the extension keeps beside the release file, as written."""
    return json.loads((tmp_path / REQUESTS_FILE).read_text(encoding="utf-8"))


def test_the_extension_parses_as_plain_javascript(tmp_path: Path) -> None:
    """The asset stays free of annotations by design (see its header comment), so
    a plain node parse is the check; TS syntax would fail here first."""
    module = tmp_path / "requests.mjs"
    module.write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["node", "--check", str(module)], check=True, capture_output=True)


def test_the_assets_are_ascii_and_the_skill_body_is_a_short_pi_skill() -> None:
    for asset in (ASSET, SKILL):
        asset.read_text(encoding="utf-8").encode("ascii")
    lines = SKILL.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 200
    # pi drops a skill without a description, so the body is a SKILL.md with its frontmatter.
    assert lines[0] == "---"
    assert lines[1] == "name: reef-pi-extension-api"
    assert lines[2].startswith("description: ")


def test_the_extension_imports_node_only_and_the_ask_path_confirms_nothing() -> None:
    """The tools declare plain JSON schema parameters, so the one file needs no typebox import; pi-tui, which draws
    the clarification's entry, is the one module loaded lazily through pi's loader."""
    text = ASSET.read_text(encoding="utf-8")
    imports = re.findall(r'^import .* from "([^"]+)";$', text, flags=re.MULTILINE)
    assert imports and all(module.startswith("node:") for module in imports)
    assert re.findall(r'import\("([^"]+)"\)', text) == ["@earendil-works/pi-tui"]
    assert text.count("pi.registerTool(") == 2
    # Asking confirms nothing: from the first tool through the ask command no confirm runs. The confirms guard the
    # next steps (the install, the promote, a setup check) in the helpers before the tools and in /versions.
    helpers, _, rest = text.partition("pi.registerTool(")
    asking, found, versions = rest.partition('pi.registerCommand("versions"')
    assert helpers and found and "ui.confirm" not in asking
    assert "ui.confirm" in helpers and "ui.confirm" in versions


def test_offline_registers_nothing(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), PI_OFFLINE="1")
    assert out["tools"] == [] and out["commands"] == [] and out["handlers"] == [] and out["events"] == []


def test_a_missing_service_url_registers_nothing(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), REEF_SERVICE_URL="")
    assert out["tools"] == [] and out["commands"] == []


def test_a_missing_scenario_registers_nothing(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), REEF_SCENARIO="")
    assert out["tools"] == [] and out["commands"] == []


def test_registers_the_two_tools_the_commands_and_the_session_events(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path))
    assert out["tools"] == ["reef_ask_user", "reef_file_request"]
    assert out["commands"] == ["reefine", "versions"]
    assert out["handlers"] == ["session_shutdown", "session_start"]
    assert out["entryRenderers"] == ["reef-harness-clarify"]
    assert out["events"] == []


def test_the_command_prints_usage_with_no_argument(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    for args in ("   ", "--direct", "--direct   "):
        out = _run(tmp_path, agent_dir, TEST_STEP="command", TEST_ARGS=args)
        assert out["events"] == [
            {"kind": "notify", "message": "Usage: /reefine <what the harness should do>", "type": "warning"}
        ]


def test_the_command_submits_native_training_without_touching_receipts(tmp_path: Path) -> None:
    answers = {"POST /reef/train": {"status": 200, "body": ACCEPTED}}
    out = _ask(tmp_path, _install_root(tmp_path), answers, text="  text me when you are blocked ", REEF_TOKEN="tok")
    assert out["error"] is None
    (request,) = _fetches(out)
    assert request["url"] == "http://reef:8900/reef/train"
    assert request["method"] == "POST"
    assert request["headers"] == {
        "x-reef-scenario": "code-repair",
        "authorization": "Bearer tok",
        "content-type": "application/json",
    }
    # The request reports this machine, so the proposer builds for it and not for the sandbox it tries changes in.
    client = request["body"].pop("client")
    assert request["body"] == {"text": "text me when you are blocked", "session": "sess-1234", "release_id": "v1"}
    assert client["platform"] == sys.platform and client["arch"] and client["release"]
    assert list(client["commands"]) == list(CLIENT_COMMANDS)
    assert client["commands"]["node"] is True  # the test runs the extension under node, which is on the PATH
    assert TrainingRequest.from_dict({**request["body"], "client": client}).client == client
    assert _notices(out) == [{"kind": "notify", "message": ACCEPTED_NOTICE, "type": "info"}]
    # The watch starts once the request is filed: the footer names the record until the step settles.
    assert _of_kind(out, "status") == [{"kind": "status", "key": "reef", "text": "reef: request q-1 queued"}]
    assert _of_kind(out, "user_message") == []


def test_the_command_needs_no_capture_proxy_or_models_file(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    (agent_dir / "models.json").unlink()
    out = _ask(tmp_path, agent_dir, {"POST /reef/train": {"status": 200, "body": ACCEPTED}})
    assert out["error"] is None
    assert len(_fetches(out)) == 1
    assert _notices(out)[0]["message"] == ACCEPTED_NOTICE


def test_the_command_surfaces_auto_mode_refusal(tmp_path: Path) -> None:
    answers = {
        "POST /reef/train": {"status": 400, "body": {"error": "training requests require training_mode='manual'"}}
    }
    out = _ask(tmp_path, _install_root(tmp_path), answers)
    assert out["error"] is None
    assert len(_fetches(out)) == 1
    (notice,) = _notices(out)
    assert "training_mode='manual'" in notice["message"]
    assert notice["type"] == "error"


def test_the_command_reports_a_rejected_body_as_a_notice(tmp_path: Path) -> None:
    answers = {"POST /reef/train": {"status": 400, "body": {"error": "release_id must be a string"}}}
    out = _ask(tmp_path, _install_root(tmp_path), answers)
    assert out["error"] is None
    assert len(_fetches(out)) == 1
    (notice,) = _notices(out)
    assert notice["message"].startswith("reef refused the request (HTTP 400): ")
    assert "release_id must be a string" in notice["message"]
    assert notice["type"] == "error"


def test_the_command_reports_an_unreachable_reef_as_a_notice(tmp_path: Path) -> None:
    out = _ask(tmp_path, _install_root(tmp_path), {})
    assert out["error"] is None
    assert len(_fetches(out)) == 1
    (notice,) = _notices(out)
    assert notice["message"].startswith("reef unreachable at http://reef:8900: ")
    assert notice["type"] == "error"


def test_the_command_without_the_release_file_sends_nothing(tmp_path: Path) -> None:
    out = _ask(
        tmp_path,
        _install_root(tmp_path, with_release_file=False),
        {"POST /reef/train": {"status": 200, "body": ACCEPTED}},
    )
    (notice,) = out["events"]
    assert notice["kind"] == "notify" and notice["type"] == "error"
    assert ".reef-harness-release" in notice["message"] and "nothing was sent" in notice["message"]


def test_the_command_falls_back_to_the_release_file_beside_the_agent_dir(tmp_path: Path) -> None:
    answers = {"POST /reef/train": {"status": 200, "body": ACCEPTED}}
    out = _ask(tmp_path, _install_root(tmp_path), answers, REEF_HARNESS_DEST="")
    assert _fetches(out)[0]["body"]["release_id"] == "v1"


# The catalog /versions reads, oldest first: the creation row, a published win with a request, a rejected
# candidate whose row carries the head's id, a pending extension win, and a step the method skipped.
LONG_TEXT = "text me when you are blocked, and say what you tried before you stopped"
RELEASES = {
    "scenario": "code-repair",
    "releases": [
        {"release_id": "rel-0000-creation", "parent_release_id": None, "operation": "creation", "pending": False},
        {
            "release_id": "rel-1111-selected",
            "parent_release_id": "rel-0000-creation",
            "operation": "training",
            "pending": False,
            "current": True,
            "metrics": {
                "steps": 1,
                "selected": True,
                "training_request": {"id": "q-1", "text": LONG_TEXT},
                "proposer_input_tokens": 1200,
                "proposer_output_tokens": 80,
                "candidate_agents": {"root": {"turns": 1, "steps": 2, "input_tokens": 500, "output_tokens": 40}},
                "current_agents": {"root": {"turns": 1, "steps": 2, "input_tokens": 450, "output_tokens": 35}},
            },
        },
        {
            "release_id": "rel-1111-selected",
            "parent_release_id": "rel-0000-creation",
            "operation": "training",
            "pending": False,
            "current": False,
            "metrics": {"steps": 2, "selected": False, "training_request": {"id": "q-2", "text": "answer with care"}},
        },
        {
            "release_id": "rel-3333-pending",
            "parent_release_id": "rel-1111-selected",
            "operation": "training",
            "pending": True,
            "current": False,
            "metrics": {"steps": 3, "selected": True, "training_request": {"id": "q-3", "text": "log when blocked"}},
        },
        {
            "release_id": "rel-1111-selected",
            "parent_release_id": "rel-0000-creation",
            "operation": "training",
            "pending": False,
            "current": False,
            "metrics": {"steps": 4, "skipped": "no proposal"},
        },
    ],
}
RELEASES = _listed(RELEASES)
CATALOG = {"GET /reef/harness/releases": {"status": 200, "body": RELEASES}}
PROMOTED = {"POST /reef/scenarios/code-repair/promote": {"status": 200, "body": {"release_id": "rel-4444-promote"}}}


NO_WRAPPER = "reef: no reef-pi wrapper found; install it with reef-pi update, then reef-pi setup"
INSTALL_LATER = "reef: install it later with reef-pi update, then reef-pi setup"


def _versions(tmp_path: Path, agent_dir: Path, answers: dict[str, Any], args: str = "", **env: str) -> dict[str, Any]:
    return _run(tmp_path, agent_dir, TEST_STEP="versions", TEST_ARGS=args, TEST_ANSWERS=json.dumps(answers), **env)


def test_versions_lists_aligned_columns_with_requests_below_each_row(tmp_path: Path) -> None:
    out = _versions(tmp_path, _install_root(tmp_path), CATALOG, REEF_TOKEN="tok")
    assert out["error"] is None
    (catalog,) = _fetches(out)
    assert catalog["method"] == "GET" and catalog["url"] == "http://reef:8900/reef/harness/releases"
    assert catalog["headers"] == {"x-reef-scenario": "code-repair", "authorization": "Bearer tok"}
    (notice,) = _notices(out)
    assert notice["type"] == "info"
    assert notice["message"].splitlines() == [
        "Harness versions (5 entries, oldest first)",
        "",
        "Version  Release   Result    Status",
        "-----------------------------------",
        "v0       rel-0000  creation  -",
        "v1       rel-1111  selected  current",
        f'  "{LONG_TEXT[:57]}..."',
        "v2       rel-1111  rejected  -",
        '  "answer with care"',
        "v3       rel-3333  pending   -",
        '  "log when blocked"',
        "v4       rel-1111  skipped   -",
        "",
        "installed: running in this tree; current: served by Reef",
        "Details: /versions <version>",
        "Install: /versions <version> install",
    ]


@pytest.mark.parametrize("headless", ["0", "1"])
def test_versions_keeps_multiline_requests_out_of_columns(tmp_path: Path, headless: str) -> None:
    rows = [{"release_id": f"release-{step}", "operation": "creation"} for step in range(11)]
    rows[9]["metrics"] = {"training_request": {"text": "  支持复制图片\n\n Clarification...\t完成  "}}
    catalog = {"GET /reef/harness/releases": {"status": 200, "body": _listed({"releases": rows})}}
    out = _versions(tmp_path, _install_root(tmp_path), catalog, TEST_HEADLESS=headless)
    (notice,) = _notices(out)
    assert notice["message"].splitlines()[-7:-4] == [
        "v9       release-  creation  -",
        '  "支持复制图片 Clarification... 完成"',
        "v10      release-  creation  current",
    ]
    assert [event["kind"] for event in out["events"]] == ["fetch", "notify"]


def test_versions_with_a_step_offers_its_page_and_opens_it(tmp_path: Path) -> None:
    """The page holds the design, the review and the numbers, so the command offers it rather than reprinting it.
    The confirmation names what the step is; taking it opens the page in the browser, declining prints the URL."""
    agent_dir = _install_root(tmp_path)
    opened = _versions(tmp_path, agent_dir, CATALOG, args="3", TEST_CONFIRM="1", REEF_TOKEN="tok")
    assert opened["error"] is None
    assert [event["kind"] for event in opened["events"]] == ["fetch", "confirm", "exec"]
    prompt = opened["events"][1]
    assert prompt["title"] == "Open harness v3?"
    assert prompt["message"] == "rel-3333-pending (pending)"
    # The page a browser opens carries the scenario and the page key as query parameters, and is one argument.
    page = f"http://reef:8900/reef/harness/releases/3/page?scenario=code-repair{KEY}"
    assert opened["events"][2]["args"] == [page]
    # Declining opens nothing and leaves the URL to open by hand.
    declined = _versions(tmp_path, agent_dir, CATALOG, args="3", REEF_TOKEN="tok")
    assert [event["kind"] for event in declined["events"]] == ["fetch", "confirm", "notify"]
    assert declined["events"][2] == {"kind": "notify", "message": f"page: {page}", "type": "info"}
    # A launcher that is not there leaves the URL too, never an error.
    page1 = f"http://reef:8900/reef/harness/releases/1/page?scenario=code-repair{KEY}"
    missing = _versions(
        tmp_path, agent_dir, CATALOG, args="1", TEST_CONFIRM="1", TEST_EXEC=json.dumps({page1: {"code": 1}})
    )
    assert missing["error"] is None
    assert _notices(missing)[0]["message"] == f"reef: open it yourself: {page1}"
    # Headless has no dialog to open: the summary and the URL are printed instead.
    headless = _versions(tmp_path, agent_dir, CATALOG, args="1", TEST_HEADLESS="1")
    assert [event["kind"] for event in headless["events"]] == ["fetch", "notify"]
    assert headless["events"][1]["message"].splitlines() == [
        "Harness v1: rel-1111-selected (selected, current)",
        f"page: {page1}",
    ]


def test_versions_install_serves_a_pending_release_before_installing_it(tmp_path: Path) -> None:
    """A release held back from the served head installs like any other: the confirmation is the person saying it
    may run, so the install moves the head to it first and then installs what the promote minted."""
    agent_dir = _install_root(tmp_path)
    confirmed = _versions(
        tmp_path, agent_dir, {**CATALOG, **PROMOTED}, args="3 install", TEST_CONFIRM="1", REEF_TOKEN="tok"
    )
    assert confirmed["error"] is None
    assert [event["kind"] for event in confirmed["events"]] == ["fetch", "confirm", "fetch", "notify"]
    prompt = confirmed["events"][1]
    assert prompt["title"] == "Install release rel-3333 now?"
    # The confirmation says what makes this release different and where to read it before answering.
    assert "runs in pi with your privileges" in prompt["message"]
    assert f"http://reef:8900/reef/harness/releases/3/page?scenario=code-repair{KEY}" in prompt["message"]
    assert "token=" not in prompt["message"]
    promote = confirmed["events"][2]
    assert promote["method"] == "POST" and promote["url"] == "http://reef:8900/reef/scenarios/code-repair/promote"
    assert promote["headers"] == {
        "x-reef-scenario": "code-repair",
        "authorization": "Bearer tok",
        "content-type": "application/json",
    }
    assert promote["body"] == {"release_id": "rel-3333-pending"}
    # Without a wrapper on disk the install names the commands; the head it installs is the promote's own.
    assert confirmed["events"][3] == {"kind": "notify", "message": NO_WRAPPER, "type": "warning"}

    # Declining installs nothing and promotes nothing: the head does not move behind a refusal.
    declined = _versions(tmp_path, agent_dir, {**CATALOG, **PROMOTED}, args="3 install")
    assert [event["kind"] for event in declined["events"]] == ["fetch", "confirm", "notify"]
    assert declined["events"][2] == {"kind": "notify", "message": INSTALL_LATER, "type": "info"}

    # A refused promote installs nothing and says why.
    refused = _versions(
        tmp_path,
        agent_dir,
        {**CATALOG, "POST /reef/scenarios/code-repair/promote": {"status": 409, "body": {"error": "no"}}},
        args="3 install",
        TEST_CONFIRM="1",
    )
    assert [event["kind"] for event in refused["events"]] == ["fetch", "confirm", "fetch", "notify"]
    assert refused["events"][3]["type"] == "error"
    assert refused["events"][3]["message"].startswith("reef refused the promote (HTTP 409)")

    # A release already at the head needs no promote: the install runs straight through.
    published = _versions(tmp_path, agent_dir, {**CATALOG, **PROMOTED}, args="1 install", TEST_CONFIRM="1")
    assert [event["kind"] for event in published["events"]] == ["fetch", "confirm", "notify"]
    assert published["events"][2] == {"kind": "notify", "message": NO_WRAPPER, "type": "warning"}


def test_versions_reports_an_unreachable_reef_as_one_notice(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    for args in ("", "3", "3 install"):
        out = _versions(tmp_path, agent_dir, {}, args=args)
        assert out["error"] is None
        assert len(_fetches(out)) == 1
        (notice,) = _notices(out)
        assert notice["type"] == "error"
        assert notice["message"].startswith("reef unreachable at http://reef:8900: ")
        assert [event["kind"] for event in out["events"]] == ["fetch", "notify"]


def test_versions_refuses_a_missing_step_and_bad_arguments_with_a_notice(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    missing = _versions(tmp_path, agent_dir, CATALOG, args="9")
    assert _notices(missing) == [{"kind": "notify", "message": "no v9: the catalog holds v0 to v4", "type": "warning"}]
    for args in ("three", "-1", "3 publish", "3 promote", "install", "3 install now"):
        out = _versions(tmp_path, agent_dir, CATALOG, args=args)
        assert _fetches(out) == []
        assert _notices(out) == [
            {"kind": "notify", "message": "Usage: /versions [version] [install]", "type": "warning"}
        ]
    refused = _versions(
        tmp_path,
        agent_dir,
        {"GET /reef/harness/releases": {"status": 404, "body": {"error": "unknown scenario"}}},
    )
    (notice,) = _notices(refused)
    assert notice["type"] == "error" and notice["message"].startswith("reef refused the catalog read (HTTP 404)")


# The catalog as the service reports it: its current flag sits on the newest row, here a pending win.
NEWEST_PENDING = {
    "scenario": "code-repair",
    "releases": [
        {"release_id": "rel-0000-creation", "parent_release_id": None, "operation": "creation", "pending": False},
        {
            "release_id": "rel-1111-selected",
            "parent_release_id": "rel-0000-creation",
            "operation": "training",
            "pending": False,
            "current": False,
            "metrics": {"steps": 1, "selected": True, "training_request": {"id": "q-1", "text": "say when blocked"}},
        },
        {
            "release_id": "rel-1111-selected",
            "parent_release_id": "rel-0000-creation",
            "operation": "training",
            "pending": False,
            "current": False,
            "metrics": {"steps": 2, "selected": False},
        },
        {
            "release_id": "rel-1111-selected",
            "parent_release_id": "rel-0000-creation",
            "operation": "training",
            "pending": False,
            "current": False,
            "metrics": {"steps": 3, "selected": False},
        },
        {
            "release_id": "rel-4444-pending",
            "parent_release_id": "rel-1111-selected",
            "operation": "training",
            "pending": True,
            "current": True,
            "metrics": {"steps": 4, "selected": True},
        },
    ],
}
NEWEST_PENDING = _listed(NEWEST_PENDING)
# The same catalog after a person promoted the pending win: the promote row names it as its target.
PROMOTE_ROW = {
    "release_id": "rel-5555-promote",
    "parent_release_id": "rel-1111-selected",
    "operation": "promote",
    "pending": False,
    "current": True,
    "rollback_target_release_id": "rel-4444-pending",
}
AFTER_PROMOTE = {
    "scenario": "code-repair",
    "releases": [*NEWEST_PENDING["releases"][:4], {**NEWEST_PENDING["releases"][4], "current": False}, PROMOTE_ROW],
}
AFTER_PROMOTE = _listed(AFTER_PROMOTE)


def test_versions_marks_the_served_head_current_and_never_the_pending_row(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    catalog = {"GET /reef/harness/releases": {"status": 200, "body": NEWEST_PENDING}}
    listed = _versions(tmp_path, agent_dir, catalog)
    (notice,) = _notices(listed)
    assert notice["message"].splitlines()[4:-4] == [
        "v0       rel-0000  creation  -",
        "v1       rel-1111  selected  current",
        '  "say when blocked"',
        "v2       rel-1111  rejected  -",
        "v3       rel-1111  rejected  -",
        "v4       rel-4444  pending   -",
    ]
    head = _versions(tmp_path, agent_dir, catalog, args="1")
    assert _dialogs(head)[0]["message"] == "rel-1111-selected (selected, current)"
    pending = _versions(tmp_path, agent_dir, catalog, args="4")
    assert _dialogs(pending)[0]["message"] == "rel-4444-pending (pending)"


def test_versions_marks_the_installed_release_apart_from_the_head(tmp_path: Path) -> None:
    # The tree runs rel-1111; its row is marked installed (and current, since it is the head). A row that is
    # neither the installed release nor the newest stays unmarked, and the fallback release file.
    installed = _install_root(tmp_path, with_release_file=False)
    (tmp_path / ".reef-harness-release").write_text(json.dumps({"release_id": "rel-1111-selected"}), encoding="utf-8")
    listed = _versions(tmp_path, installed, CATALOG)
    (notice,) = _notices(listed)
    assert notice["message"].splitlines()[4:-4] == [
        "v0       rel-0000  creation  -",
        "v1       rel-1111  selected  installed, current",
        f'  "{LONG_TEXT[:57]}..."',
        "v2       rel-1111  rejected  -",
        '  "answer with care"',
        "v3       rel-3333  pending   -",
        '  "log when blocked"',
        "v4       rel-1111  skipped   -",
    ]
    shown = _versions(tmp_path, installed, CATALOG, args="1")
    assert _dialogs(shown)[0]["message"] == "rel-1111-selected (selected, installed (this tree), current)"


def test_versions_distinguishes_an_installed_older_release_from_a_newer_head(tmp_path: Path) -> None:
    # The tree runs rel-0000 (creation); a promoted rel-5555 is the head. The listing marks rel-0000 installed and
    # rel-5555 current, the two apart.
    installed = _install_root(tmp_path, with_release_file=False)
    (tmp_path / ".reef-harness-release").write_text(json.dumps({"release_id": "rel-0000-creation"}), encoding="utf-8")
    listed = _versions(tmp_path, installed, {"GET /reef/harness/releases": {"status": 200, "body": AFTER_PROMOTE}})
    (notice,) = _notices(listed)
    assert notice["message"].splitlines()[4:-4] == [
        "v0       rel-0000  creation        installed",
        "v1       rel-1111  selected        -",
        '  "say when blocked"',
        "v2       rel-1111  rejected        -",
        "v3       rel-1111  rejected        -",
        "v4       rel-4444  promoted at v5  -",
        "v5       rel-5555  promote         current",
    ]
    shown = _versions(
        tmp_path, installed, {"GET /reef/harness/releases": {"status": 200, "body": AFTER_PROMOTE}}, args="5"
    )
    assert _dialogs(shown)[0]["message"] == "rel-5555-promote (promote, current)"


def test_versions_reads_a_promoted_row_as_promoted_and_installs_it_without_promoting_again(tmp_path: Path) -> None:
    """A promote is a row of its own naming the release it promoted, so the row it names reads "promoted at
    vN". It is already served, so installing it needs no second promote."""
    agent_dir = _install_root(tmp_path)
    catalog = {"GET /reef/harness/releases": {"status": 200, "body": AFTER_PROMOTE}}
    listed = _versions(tmp_path, agent_dir, catalog)
    (notice,) = _notices(listed)
    assert notice["message"].splitlines()[-6:-4] == [
        "v4       rel-4444  promoted at v5  -",
        "v5       rel-5555  promote         current",
    ]
    shown = _versions(tmp_path, agent_dir, catalog, args="4")
    assert _dialogs(shown)[0]["message"] == "rel-4444-pending (promoted at v5)"
    installed = _versions(tmp_path, agent_dir, {**catalog, **PROMOTED}, args="4 install", TEST_CONFIRM="1")
    assert [event["kind"] for event in installed["events"]] == ["fetch", "confirm", "notify"]
    assert installed["events"][2] == {"kind": "notify", "message": NO_WRAPPER, "type": "warning"}


def test_versions_takes_only_a_run_of_digits_as_the_step(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    for args in ("1e0", "0x1", "1.0", "+1", "vv1", "V1", "1 install extra"):
        out = _versions(tmp_path, agent_dir, CATALOG, args=args)
        assert _fetches(out) == []
        assert _notices(out) == [
            {"kind": "notify", "message": "Usage: /versions [version] [install]", "type": "warning"}
        ]
    for args in ("01", "v1"):
        named = _versions(tmp_path, agent_dir, CATALOG, args=args)
        assert _dialogs(named)[0]["message"] == "rel-1111-selected (selected, current)"


# -- the clarify path, the two tools and the watch ---------------------------------------------------------------

CLARIFY_HEAD = (
    "The user asked for this harness change:\n\n```\ntext me when you are blocked\n```\n\nBefore filing it with "
)
QUESTIONS = {
    "questions": [
        {"question": "Which channel?", "options": ["Email", "SMS"], "recommended": "SMS"},
        {"question": "When?", "options": ["Always", "Nights", "Weekends"], "recommended": "not an option"},
    ]
}
OTHER = "Other (type an answer)"
CANCEL = "Cancel this request"
FILED = (
    "filed request q-1; reef is running the step, which usually takes a few minutes, and will report here "
    f"when it settles. Watch it here: {REQUEST_PAGE}{KEY}"
)


def _call(name: str, arguments: dict[str, Any], call_id: str = "c-1") -> dict[str, Any]:
    """A scripted model reply that calls one tool."""
    return {
        "content": [{"type": "toolCall", "id": call_id, "name": name, "arguments": arguments}],
        "stopReason": "toolUse",
    }


BRANCH = [
    {"type": "message", "message": {"role": "user", "content": "we keep missing the blocked builds"}},
    {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "I can watch them."}]}},
    {"type": "message", "message": {"role": "toolResult", "content": [{"type": "text", "text": "tool output"}]}},
    {"type": "custom", "customType": "other"},
]


def _clarify(tmp_path: Path, replies: list[dict[str, Any]], **env: str) -> dict[str, Any]:
    return _ask(
        tmp_path,
        _install_root(tmp_path),
        {"POST /reef/train": {"status": 200, "body": ACCEPTED}},
        direct=False,
        TEST_REPLIES=json.dumps(replies),
        TEST_BRANCH=json.dumps(BRANCH),
        TEST_WAIT_MS="300",
        **env,
    )


def test_the_command_with_a_ui_clarifies_in_the_background_and_keeps_one_entry(tmp_path: Path) -> None:
    thinking = {"type": "thinking", "thinking": "the channel is open"}
    filing = {
        "request": "text me when you are blocked",
        "clarifications": [{"question": "Which channel?", "answer": "SMS"}],
    }
    replies = [
        {"content": [thinking, *_call("reef_ask_user", QUESTIONS)["content"]], "stopReason": "toolUse"},
        _call("reef_file_request", filing, "c-2"),
    ]
    out = _clarify(tmp_path, replies, TEST_SELECT=json.dumps(["SMS (recommended)", "Nights"]), REEF_TOKEN="tok")
    assert out["error"] is None
    # The recommended option leads its question, marked; a recommendation that names no option changes nothing.
    assert [event["options"] for event in _of_kind(out, "select")] == [
        ["SMS (recommended)", "Email", OTHER, CANCEL],
        ["Always", "Nights", "Weekends", OTHER, CANCEL],
    ]
    # Nothing reaches the session: no message in it, and the model is called directly with the two tools.
    assert _of_kind(out, "user_message") == [] and _of_kind(out, "message") == []
    first, second = _of_kind(out, "model_call")
    assert first["signal"] is True and first["model"] == {"provider": "reef", "id": "served"}
    assert [tool["name"] for tool in first["context"]["tools"]] == ["reef_ask_user", "reef_file_request"]
    assert "Use only the reef_ask_user and reef_file_request tools" in first["context"]["systemPrompt"]
    # The model hears what reef-pi is, so it never takes the name for an unrelated web app.
    assert "a coding agent that runs in a terminal (not a web or browser app)" in first["context"]["systemPrompt"]
    (prompt,) = first["context"]["messages"]
    # The recent conversation rides as background, user and assistant text only, before the request.
    assert prompt["content"].startswith(
        "The recent conversation in the session, as background for what the request refers to:\n\n```\n"
        "user: we keep missing the blocked builds\n\nassistant: I can watch them.\n```\n\n" + CLARIFY_HEAD
    )
    assert "tool output" not in prompt["content"]
    # The model asks only what a default cannot settle, and may recommend an option.
    assert "a clear request needs no question, so file it at once" in prompt["content"]
    assert "- as few as the request needs, often none; one decision per question" in prompt["content"]
    # A capability the machine and a paid model both serve is asked about, not settled by a default.
    assert "which one runs is a decision, not a setup detail" in prompt["content"]
    assert "name it as recommended: it is shown first, marked" in prompt["content"]
    assert prompt["content"].endswith("Do not write the change yourself: reef's service writes it.")
    # The second call carries the answers back as the first call's tool result.
    tool_result = second["context"]["messages"][-1]
    assert tool_result["role"] == "toolResult" and tool_result["toolCallId"] == "c-1"
    assert json.loads(tool_result["content"][0]["text"])[1] == {"question": "When?", "answer": "Nights"}
    (request,) = _fetches(out)
    assert request["body"]["text"] == "text me when you are blocked\n\nClarifications:\n- Q: Which channel?\n  A: SMS"
    # The filing ends it at once: no further model call, its widget is cleared, and the step's watch starts.
    widgets = [event for event in _of_kind(out, "widget") if event["key"] == "reef-harness-clarify"]
    assert widgets[0]["content"][0].endswith("thinking it through - ctrl+q or /reefine to look in")
    assert widgets[-1]["content"] is None
    assert _of_kind(out, "status")[0] == {"kind": "status", "key": "reef", "text": "reef: request q-1 queued"}
    (entry,) = _of_kind(out, "entry")
    assert entry["customType"] == "reef-harness-clarify"
    data = entry["data"]
    assert data["outcome"] == "filed" and data["request"] == "text me when you are blocked"
    assert data["summary"].startswith(f"filed request q-1; watch it at {REQUEST_PAGE}{KEY} (")
    kinds = [item["kind"] for item in data["transcript"]]
    assert kinds == ["thinking", "reef_ask_user", "result", "reef_file_request", "result"]
    assert data["transcript"][-1]["text"] == FILED


def test_the_command_without_an_argument_says_what_is_running(tmp_path: Path) -> None:
    """The way in that needs no key the terminal may swallow and no click it may not offer: the command itself
    names the phase and the clarification's steps so far; with nothing running it prints the usage."""
    out = _clarify(tmp_path, [{"hang": True}], TEST_REPEAT="2", TEST_ARGS_2="")
    (notice,) = _notices(out)
    assert notice["type"] == "info"
    assert notice["message"].startswith("reef: clarifying 'text me when you are blocked' - thinking it through")


@pytest.mark.parametrize(
    ("replies", "selects", "outcome", "notice"),
    [
        ([_call("reef_ask_user", QUESTIONS)], [CANCEL], "cancelled", None),
        (
            [{"content": [{"type": "text", "text": "What should I file?"}]}],
            [],
            "unfiled",
            (
                "warning",
                "reef: the clarification ended without filing: What should I file?; ask again, or file it as is "
                "with /reefine --direct",
            ),
        ),
        (
            [{"throw": "connection reset"}],
            [],
            "failed",
            ("error", "reef: the clarification failed (connection reset); file it as is with /reefine --direct"),
        ),
    ],
    ids=["cancelled", "no-tool-call", "model-error"],
)
def test_a_clarification_that_does_not_file_says_why_and_sends_nothing(
    tmp_path: Path, replies: list[dict[str, Any]], selects: list[str], outcome: str, notice: tuple[str, str] | None
) -> None:
    """The entry says what happened; a cancel the person chose needs no notice beside it."""
    out = _clarify(tmp_path, replies, TEST_SELECT=json.dumps(selects))
    assert _fetches(out) == [] and _of_kind(out, "status") == []
    (entry,) = _of_kind(out, "entry")
    assert entry["data"]["outcome"] == outcome
    assert _notices(out) == ([] if notice is None else [{"kind": "notify", "message": notice[1], "type": notice[0]}])


def test_a_bad_tool_call_goes_back_to_the_model_as_an_error(tmp_path: Path) -> None:
    replies = [_call("reef_file_request", {"request": ""}), _call("reef_file_request", {"request": "text me"}, "c-2")]
    out = _clarify(tmp_path, replies)
    retry = _of_kind(out, "model_call")[1]["context"]["messages"][-1]
    assert retry["isError"] is True and retry["content"][0]["text"] == "error: request must be a non-empty string"
    assert _fetches(out)[0]["body"]["text"] == "text me"
    assert _of_kind(out, "entry")[0]["data"]["outcome"] == "filed"


def test_the_command_without_a_model_or_while_clarifying_starts_nothing(tmp_path: Path) -> None:
    out = _clarify(tmp_path, [], TEST_NO_MODEL="1")
    assert _of_kind(out, "model_call") == [] and _of_kind(out, "entry") == []
    assert _notices(out) == [
        {
            "kind": "notify",
            "message": "reef: no model to clarify with; pick one with /model, or use /reefine --direct",
            "type": "error",
        }
    ]
    # A second ask while the first is still running is refused; the first goes on.
    out = _clarify(tmp_path / "twice", [{"hang": True}], TEST_REPEAT="2")
    assert len(_of_kind(out, "model_call")) == 1
    assert _notices(out)[0] == {
        "kind": "notify",
        "message": "reef: still clarifying 'text me when you are blocked'; ask again once it is filed",
        "type": "warning",
    }
    # The release file is checked before the model is engaged: without it nothing is sent or asked.
    bare = _ask(
        tmp_path / "bare",
        _install_root(tmp_path / "bare", with_release_file=False),
        {"POST /reef/train": {"status": 200, "body": ACCEPTED}},
        direct=False,
    )
    assert _of_kind(bare, "model_call") == []
    (notice,) = bare["events"]
    assert notice["type"] == "error" and "nothing was sent" in notice["message"]


def test_the_command_files_as_is_with_the_flag_or_without_a_ui(tmp_path: Path) -> None:
    answers = {"POST /reef/train": {"status": 200, "body": ACCEPTED}}
    agent_dir = _install_root(tmp_path)
    flagged = _ask(tmp_path, agent_dir, answers, text="  text me  ", direct=True)
    headless = _ask(tmp_path, agent_dir, answers, text="text me", direct=False, TEST_HEADLESS="1")
    for out in (flagged, headless):
        assert out["error"] is None
        (request,) = _fetches(out)
        assert request["method"] == "POST" and request["body"]["text"] == "text me"
        assert _of_kind(out, "user_message") == []
        assert _notices(out) == [{"kind": "notify", "message": ACCEPTED_NOTICE, "type": "info"}]
        assert _of_kind(out, "status") == [{"kind": "status", "key": "reef", "text": "reef: request q-1 queued"}]


def test_ask_user_returns_the_chosen_option_or_the_typed_answer(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    out = _tool(
        tmp_path,
        agent_dir,
        "ask_user",
        QUESTIONS,
        TEST_SELECT=json.dumps(["SMS (recommended)", OTHER]),
        TEST_INPUT=json.dumps(["Slack"]),
    )
    assert out["error"] is None
    # The recommended option is filed as the option alone, without its mark.
    assert json.loads(out["result"]["content"][0]["text"]) == [
        {"question": "Which channel?", "answer": "SMS"},
        {"question": "When?", "answer": "Slack"},
    ]
    # Every question offers its options, the recommended one first and marked, then the free text answer and the
    # way out; a recommendation that names no option leaves the list as given.
    assert [event["options"] for event in _of_kind(out, "select")] == [
        ["SMS (recommended)", "Email", OTHER, CANCEL],
        ["Always", "Nights", "Weekends", OTHER, CANCEL],
    ]
    assert _of_kind(out, "input") == [{"kind": "input", "title": "When?", "placeholder": ""}]


@pytest.mark.parametrize(
    ("selects", "inputs", "asked", "typed"),
    [
        # Esc on the first question: no second question, and no free text fallback.
        ([None], [], 1, 0),
        # The explicit way out reads the same as Esc.
        ([CANCEL], [], 1, 0),
        # Esc on the second question stops there, keeping no partial answers.
        (["SMS", None], [], 2, 0),
        # Esc on the free text answer backs out of the request too, for one meaning of Esc throughout.
        ([OTHER], [None], 1, 1),
    ],
    ids=["escape", "cancel-option", "escape-later", "escape-the-input"],
)
def test_ask_user_cancels_the_whole_request_and_files_nothing(
    tmp_path: Path, selects: list, inputs: list, asked: int, typed: int
) -> None:
    """Esc is the way out of a clarification, not a skipped question: nothing is filed and the model is told so."""
    out = _tool(
        tmp_path,
        _install_root(tmp_path),
        "ask_user",
        QUESTIONS,
        TEST_SELECT=json.dumps(selects),
        TEST_INPUT=json.dumps(inputs),
    )
    assert out["error"] is None
    assert out["result"]["content"] == [
        {
            "type": "text",
            "text": "the user cancelled this harness request: do not file it, do not ask again, and say it was cancelled",
        }
    ]
    # The dialogs stop at the cancel: no later question is asked, and no answer is kept.
    assert len(_of_kind(out, "select")) == asked and len(_of_kind(out, "input")) == typed
    assert _notices(out) == [
        {"kind": "notify", "message": "reef: request cancelled; nothing was filed", "type": "info"}
    ]
    # Nothing is filed and nothing is watched: the request never reaches the service.
    assert _fetches(out) == [] and _of_kind(out, "status") == [] and _of_kind(out, "widget") == []


def test_ask_user_without_a_ui_tells_the_model_to_assume_and_say_so(tmp_path: Path) -> None:
    out = _tool(tmp_path, _install_root(tmp_path), "ask_user", QUESTIONS, TEST_HEADLESS="1")
    assert out["error"] is None
    assert out["result"]["content"] == [
        {
            "type": "text",
            "text": "no UI in this session: proceed with your best assumptions and list them in the request",
        }
    ]
    assert _of_kind(out, "select") == [] and _of_kind(out, "input") == []


def test_file_request_composes_the_text_posts_it_and_starts_the_watch(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    answers = {"POST /reef/train": {"status": 200, "body": ACCEPTED}}
    params = {
        "request": "text me when you are blocked",
        "clarifications": [{"question": "Which channel?", "answer": "SMS"}, {"question": "When?", "answer": "Nights"}],
    }
    out = _tool(tmp_path, agent_dir, "file_request", params, TEST_ANSWERS=json.dumps(answers), REEF_TOKEN="tok")
    assert out["error"] is None
    (request,) = _fetches(out)
    assert request["url"] == "http://reef:8900/reef/train" and request["method"] == "POST"
    assert request["headers"] == {
        "x-reef-scenario": "code-repair",
        "authorization": "Bearer tok",
        "content-type": "application/json",
    }
    assert request["body"].pop("client")["platform"] == sys.platform  # the machine rides with every request
    assert request["body"] == {
        "text": "text me when you are blocked\n\nClarifications:\n- Q: Which channel?\n  A: SMS\n- Q: When?\n  A: Nights",
        "session": "sess-1234",
        "release_id": "v1",
    }
    assert out["result"]["content"] == [{"type": "text", "text": FILED}]
    assert _of_kind(out, "status") == [{"kind": "status", "key": "reef", "text": "reef: request q-1 queued"}]
    # The filing is stored beside the release file until its report is delivered.
    (entry,) = _stored(tmp_path)
    assert entry["id"] == "q-1" and entry["text"] == request["body"]["text"]
    assert abs(entry["filed_at"] - time.time()) < 60
    # Without clarifications the text is the request alone; the service's cap bounds it.
    out = _tool(tmp_path, agent_dir, "file_request", {"request": "x" * 5000}, TEST_ANSWERS=json.dumps(answers))
    assert _fetches(out)[0]["body"]["text"] == "x" * 4000


def test_file_request_failures_throw_the_commands_messages(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    params = {"request": "text me", "clarifications": []}
    unreachable = _tool(tmp_path, agent_dir, "file_request", params)
    assert unreachable["error"].startswith("reef unreachable at http://reef:8900: ")
    refused = {
        "POST /reef/train": {"status": 400, "body": {"error": "training requests require training_mode='manual'"}}
    }
    out = _tool(tmp_path, agent_dir, "file_request", params, TEST_ANSWERS=json.dumps(refused))
    assert out["error"].startswith("reef refused the request (HTTP 400): ")
    assert "training_mode='manual'" in out["error"]
    bare = tmp_path / "bare"
    out = _tool(bare, _install_root(bare, with_release_file=False), "file_request", params)
    assert ".reef-harness-release" in out["error"] and "nothing was sent" in out["error"]
    assert _fetches(out) == []
    for failed in (unreachable, out):
        assert failed["result"] is None and _of_kind(failed, "status") == []
    # Nothing filed, nothing stored.
    assert not (tmp_path / REQUESTS_FILE).exists() and not (bare / REQUESTS_FILE).exists()


ASK = f"{LONG_TEXT[:57]}..."
REQUEST_ROW = {"training_request": {"id": "q-1", "text": LONG_TEXT}}
REVIEW = {
    "design": "A tool that texts you.",
    "review": {"result": "partial", "covered": ["texting"], "uncovered": []},
}
SELECTED_ROW = {
    "release_id": "rel-1111-selected",
    "parent_release_id": "rel-0000-creation",
    "operation": "training",
    "pending": False,
    "metrics": {
        "selected": True,
        **REQUEST_ROW,
        "proposal_notes": {
            **REVIEW,
            "review": {**REVIEW["review"], "uncovered": ["two way replies", "idle detection"]},
        },
    },
}
PENDING_ROW = {
    **SELECTED_ROW,
    "release_id": "rel-3333-pending",
    "pending": True,
    "metrics": {"selected": True, **REQUEST_ROW},
}
REJECTED_ROW = {
    **SELECTED_ROW,
    "release_id": "rel-0000-creation",
    "metrics": {
        "selected": False,
        "selection": {"policy": "floor", "reason": "candidate missed the floor on 1 of 1 tasks"},
        **REQUEST_ROW,
        "proposal_notes": REVIEW,
    },
}
SKIPPED_ROW = {**SELECTED_ROW, "release_id": "rel-0000-creation", "metrics": {"skipped": "no proposal", **REQUEST_ROW}}
# The proposer's own reason, as Reefine records it when its model call fails.
FAILURE = (
    "model call failed after 58.2 s (max_tokens=16384): model endpoint returned non-text content; a thinking model "
    "may have spent the reply budget on its reasoning, raise REEF_PROPOSER_MAX_TOKENS"
)
SKIPPED_FAILED_ROW = {**SKIPPED_ROW, "metrics": {**SKIPPED_ROW["metrics"], "proposal_notes": {"failure": FAILURE}}}
# The record route detects requests removed from storage.
RECORD_PATH = "/reef/scenarios/code-repair/records/q-1"
CREATION_ROW = RELEASES["releases"][0]


def _catalog_with(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "POST /reef/train": {"status": 200, "body": ACCEPTED},
        "GET /reef/harness/releases": {
            "status": 200,
            "body": _listed({"scenario": "code-repair", "releases": [CREATION_ROW, row]}),
        },
    }


PROGRESS_PATH = "/reef/harness/requests/q-1/progress"
#: A catalog that never carries the request, so the watch keeps polling and the spinner keeps drawing.
WAITING = {
    "POST /reef/train": {"status": 200, "body": ACCEPTED},
    "GET /reef/harness/releases": {
        "status": 200,
        "body": _listed({"scenario": "code-repair", "releases": [CREATION_ROW]}),
    },
}


def _widgets(out: dict[str, Any]) -> list[Any]:
    """What the spinner drew, in order; a cleared widget is None."""
    return [event["content"] for event in _of_kind(out, "widget")]


def _running(state: str, **rest: Any) -> dict[str, Any]:
    body = {
        "request_id": "q-1",
        "settled": False,
        "step": None,
        "state": state,
        "meaning": None,
        "started_at": None,
        "episodes_total": None,
        "step_record": None,
        **rest,
    }
    return {"status": 200, "body": body}


def test_the_spinner_sits_above_the_input_and_names_the_phase_the_service_reports(tmp_path: Path) -> None:
    """The step runs in the background: a widget above the editor, not a line the person has to read in the chat."""
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        {
            **WAITING,
            f"GET {PROGRESS_PATH}": _running("evaluating", episodes_total=2, step_record="/work/steps/1"),
            f"GET {RECORD_PATH}": {"status": 200, "body": {}},
        },
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="150",
    )
    assert out["error"] is None
    drawn = [content for content in _widgets(out) if content]
    assert drawn, "the spinner never drew"
    # One line while it is closed, above the input box, naming the phase in the person's words and the way in.
    assert all(len(content) == 1 for content in drawn)
    assert any("checking the harness" in content[0] for content in drawn)
    assert all("ctrl+q or /reefine to look in" in content[0] for content in drawn)
    # The line carries the request's page as a terminal hyperlink, so a click opens it where the terminal offers one.
    assert all(f"\x1b]8;;{REQUEST_PAGE}{KEY}\x1b\\open the page\x1b]8;;\x1b\\" in content[0] for content in drawn)
    # The frames turn, so the person sees the step is alive between the polls.
    assert len({content[0][0] for content in drawn}) > 1
    # The phase is the service's own, read from the route the page reads.
    assert [event["url"] for event in _fetches(out) if event["url"].endswith("/progress")]


def test_the_look_in_key_opens_the_spinner_in_place_and_closes_it_again(tmp_path: Path) -> None:
    """pi offers no click target for a widget, so the key the spinner names is how a person looks in."""
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        {
            **WAITING,
            f"GET {PROGRESS_PATH}": _running(
                "proposing",
                step_record="/work/steps/1",
                activity=[
                    {"at": 1.0, "kind": "model", "text": "asking glm"},
                    {"at": 2.0, "kind": "agent", "text": "write harness/extensions/away.ts"},
                    {"at": 3.0, "kind": "trial", "text": "trial 1 exited 1 in 40 s", "failed": True},
                ],
            ),
            f"GET {RECORD_PATH}": {"status": 200, "body": {}},
        },
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="240",
        TEST_SHORTCUT_AT_MS=json.dumps([80, 170]),
    )
    assert out["error"] is None
    # A plain ctrl+letter pi leaves free: a terminal without modified-key reporting drops the shift from
    # ctrl+shift+<letter>, so such a key would reach pi as its own binding (issue #541).
    assert out["shortcuts"] == ["ctrl+q"]
    opened = [content for content in _widgets(out) if content and len(content) > 1]
    assert opened, "the key never opened the spinner"
    panel = opened[-1]
    assert "ctrl+q or /reefine to close" in panel[0] and "writing the change" in panel[0]
    body = "\n".join(panel[1:])
    assert "asked: text me when you are blocked" in body and "request: q-1" in body
    assert "step record: /work/steps/1" in body
    # The proposer's latest moves, newest first, a failed one marked.
    moves = [line.strip() for line in panel[1:] if line.strip()[:2] in ("- ", "x ")]
    assert moves == [
        "x trial: trial 1 exited 1 in 40 s",
        "- agent: write harness/extensions/away.ts",
        "- model: asking glm",
    ]
    # The page link is the full detail, and the person is told the step keeps running without them.
    assert f"full detail: {REQUEST_PAGE}" in body
    assert "the step runs in the background; your input stays yours" in body
    # The second press closes it: the spinner is one line again.
    assert len(_widgets(out)[-1]) == 1


def test_the_look_in_key_says_so_when_no_step_is_running(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        _install_root(tmp_path),
        TEST_STEP="session_start",
        TEST_ANSWERS=json.dumps(CATALOG),
        TEST_WAIT_MS="60",
        TEST_SHORTCUT_AT_MS=json.dumps([20]),
    )
    assert out["error"] is None
    assert {"kind": "notify", "message": "reef: no harness request is running", "type": "info"} in _notices(out)
    assert _widgets(out) == []


@pytest.mark.parametrize(
    "progress",
    [{"status": 404, "body": {}}, {"status": 500, "body": {}}],
    ids=["no-such-route", "service-error"],
)
def test_the_spinner_falls_back_to_queued_when_the_service_reports_no_phase(
    tmp_path: Path, progress: dict[str, Any]
) -> None:
    """An older service has no progress route; the spinner still turns and the watch still settles."""
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        {**WAITING, f"GET {PROGRESS_PATH}": progress, f"GET {RECORD_PATH}": {"status": 200, "body": {}}},
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="120",
    )
    assert out["error"] is None
    drawn = [content for content in _widgets(out) if content]
    assert drawn and all("queued, waiting for a step" in content[0] for content in drawn)


def test_a_headless_session_draws_no_spinner(tmp_path: Path) -> None:
    """Without a UI there is no input box to sit above; the footer and the report carry the result as before."""
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        _catalog_with(SELECTED_ROW),
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="150",
        TEST_HEADLESS="1",
    )
    assert out["error"] is None
    # The widget is only ever cleared, never drawn: stopWatch clears both indicators whatever the mode.
    assert all(content is None for content in _widgets(out))
    assert _of_kind(out, "message")


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            SELECTED_ROW,
            f"reef: '{ASK}' is published as release rel-1111. Install when ready with /versions v1 install."
            " Details: /versions v1.\nNot covered: two way replies; idle detection",
        ),
        (
            PENDING_ROW,
            f"reef: '{ASK}' is ready as release rel-3333. This release changes an extension, so read it before "
            "it runs: /versions v1 opens the page, /versions v1 install serves it.",
        ),
        (
            REJECTED_ROW,
            f"reef: '{ASK}' did not pass the checks (candidate missed the floor on 1 of 1 tasks). Nothing changed; "
            "rephrase or split the request. Details: /versions v1.",
        ),
        (
            SKIPPED_ROW,
            f"reef: '{ASK}' produced no change (no proposal). Nothing changed. Details: /versions v1.",
        ),
        (
            SKIPPED_FAILED_ROW,
            f"reef: '{ASK}' produced no change (no proposal: {FAILURE}). Nothing changed. Details: /versions v1.",
        ),
    ],
    ids=["selected", "pending", "rejected", "skipped", "skipped-failure"],
)
def test_the_watch_reports_the_result_and_what_the_review_left_uncovered(
    tmp_path: Path, row: dict[str, Any], expected: str
) -> None:
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        _catalog_with(row),
        text=LONG_TEXT,
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="200",
    )
    assert out["error"] is None
    kinds = [event["kind"] for event in out["events"]]
    # The filing, then the watch: the catalog read that finds the row ends it, and the report follows.
    assert kinds[:3] == ["fetch", "notify", "status"]
    # The report is the message and the notice that follow it; an install the settle offers comes after them.
    report = kinds.index("message")
    assert kinds[report : report + 2] == ["message", "notify"]
    (catalog,) = [event for event in _fetches(out) if event["url"].endswith("/reef/harness/releases")]
    assert catalog["method"] == "GET" and catalog["url"] == "http://reef:8900/reef/harness/releases"
    statuses = _of_kind(out, "status")
    assert statuses[0] == {"kind": "status", "key": "reef", "text": "reef: request q-1 queued"}
    # The settled step clears both indicators: the footer line and the spinner above the input.
    assert statuses[-1] == {"kind": "status", "key": "reef", "text": None}
    assert _of_kind(out, "widget")[-1] == {"kind": "widget", "key": "reef-harness", "content": None}
    # The report is a custom message the chat renders and the session keeps, without a turn, then the notice.
    assert out["events"][report] == {
        "kind": "message",
        "message": {"customType": "reef-harness", "content": expected, "display": True},
        "options": {"triggerTurn": False},
    }
    assert out["events"][report + 1] == {"kind": "notify", "message": expected, "type": "info"}
    # The report delivered, the stored filing is dropped.
    assert _stored(tmp_path) == []


def test_the_watch_gives_up_after_its_cap_and_says_where_the_result_will_show(tmp_path: Path) -> None:
    # The catalog never carries the request: the clock jumps past 30 minutes after a few polls.
    answers = _catalog_with({**SKIPPED_ROW, "metrics": {"skipped": "no proposal"}})
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        answers,
        text=LONG_TEXT,
        REEF_HARNESS_WATCH_MS="10",
        TEST_CLOCK_SKEW_AFTER_MS="35",
        TEST_WAIT_MS="200",
    )
    kinds = [event["kind"] for event in out["events"]]
    # The cap clears both indicators, the footer then the spinner, and says where the result will show.
    assert kinds[:3] == ["fetch", "notify", "status"] and kinds[-3:] == ["status", "widget", "notify"]
    assert 2 <= kinds.count("fetch") <= 24  # a handful of polls (catalog, progress and record), then none past the cap
    # The record route answered nothing, so the footer stayed at queued until the watch cleared it.
    assert [event["text"] for event in _of_kind(out, "status")] == ["reef: request q-1 queued", None]
    assert out["events"][-3] == {"kind": "status", "key": "reef", "text": None}
    assert out["events"][-2] == {"kind": "widget", "key": "reef-harness", "content": None}
    assert out["events"][-1] == {
        "kind": "notify",
        "message": f"reef: no result yet for '{ASK}'; /versions shows it when it settles",
        "type": "warning",
    }
    # The filing stays stored past the cap: the next session start reports the result once the catalog has it.
    (entry,) = _stored(tmp_path)
    assert entry["id"] == "q-1" and entry["text"] == LONG_TEXT
    # A catalog read failing is one missed poll, never a notice: the polls go on.
    silent = _ask(
        tmp_path,
        _install_root(tmp_path / "silent"),
        {"POST /reef/train": {"status": 200, "body": ACCEPTED}},
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="80",
    )
    assert len(_fetches(silent)) >= 3
    assert _notices(silent) == [{"kind": "notify", "message": ACCEPTED_NOTICE, "type": "info"}]


def test_the_footer_says_queued_until_progress_reports_a_running_step_then_counts(tmp_path: Path) -> None:
    """The footer follows explicit progress, then counts elapsed time; record reads only detect disappearance."""
    other = {**SELECTED_ROW, "metrics": {"selected": True, "training_request": {"id": "q-other", "text": "x"}}}
    answers = {
        **_catalog_with(other),
        f"GET {RECORD_PATH}": {"status": 200, "body": {}},
        f"GET {PROGRESS_PATH}": [
            {"status": 404, "body": {"error": "not retained"}},
            {"status": 200, "body": {"agent_record_id": "q-1", "state": "queued"}},
            {"status": 200, "body": {"agent_record_id": "q-1", "state": "running"}},
        ],
    }
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        answers,
        text=LONG_TEXT,
        REEF_TOKEN="tok",
        REEF_HARNESS_WATCH_MS="10",
        TEST_CLOCK_SKEW_AFTER_MS="80",
        TEST_CLOCK_SKEW_MS="65000",
        TEST_WAIT_MS="250",
    )
    assert out["error"] is None
    # The footer is redrawn only when its text changes; the runner prints before its own shutdown clears it.
    assert [event["text"] for event in _of_kind(out, "status")] == [
        "reef: request q-1 queued",
        "reef: step for request q-1 running for 0m 00s",
        "reef: step for request q-1 running for 1m 05s",
    ]
    reads = [event for event in _fetches(out) if event["url"] == f"http://reef:8900{RECORD_PATH}"]
    assert len(reads) == 2  # one per poll until the step was seen running, none after
    assert reads[0]["method"] == "GET"
    assert reads[0]["headers"] == {"x-reef-scenario": "code-repair", "authorization": "Bearer tok"}
    assert len([event for event in _fetches(out) if event["url"].endswith("/reef/harness/releases")]) > 3
    assert _notices(out) == [{"kind": "notify", "message": ACCEPTED_NOTICE, "type": "info"}]


def test_session_shutdown_clears_the_watch(tmp_path: Path) -> None:
    answers = _catalog_with({**SKIPPED_ROW, "metrics": {"skipped": "no proposal"}})
    out = _ask(
        tmp_path,
        _install_root(tmp_path),
        answers,
        REEF_HARNESS_WATCH_MS="10",
        TEST_SHUTDOWN_AFTER_MS="50",
        TEST_WAIT_MS="200",
    )
    assert out["fetchesAtShutdown"] >= 2
    assert len(_fetches(out)) == out["fetchesAtShutdown"]  # no poll after the shutdown
    assert _of_kind(out, "status")[-1] == {"kind": "status", "key": "reef", "text": None}
    assert _notices(out) == [{"kind": "notify", "message": ACCEPTED_NOTICE, "type": "info"}]


def test_a_second_filing_replaces_the_first_watch(tmp_path: Path) -> None:
    out = _tool(
        tmp_path,
        _install_root(tmp_path),
        "file_request",
        {"request": LONG_TEXT},
        TEST_ANSWERS=json.dumps(_catalog_with(SKIPPED_ROW)),
        TEST_REPEAT="2",
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="150",
    )
    assert out["error"] is None
    # Two filings, one watch: the second clears the first's indicators, and one poll settles one notice.
    assert [event["kind"] for event in out["events"]][-2:] == ["message", "notify"]
    assert [event["method"] for event in _fetches(out)][:2] == ["POST", "POST"]
    assert [event["text"] for event in _of_kind(out, "status")] == [
        "reef: request q-1 queued",
        None,
        "reef: request q-1 queued",
        None,
    ]


def test_session_start_says_the_commands_exist_and_counts_the_releases_awaiting_review(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    first = "reef: /reefine <what it should do> asks for a harness change; /versions lists the versions."
    out = _run(tmp_path, agent_dir, TEST_STEP="session_start", TEST_ANSWERS=json.dumps(CATALOG))
    assert out["error"] is None
    review = "1 release(s) ready to install: /versions v3 (install with /versions v3 install)"
    assert _notices(out) == [{"kind": "notify", "message": f"{first}\n{review}", "type": "info"}]
    # Two pending rows list both steps; a promoted pending row no longer waits.
    two = {**RELEASES, "releases": [*RELEASES["releases"], {**PENDING_ROW, "release_id": "rel-5555-pending"}]}
    out = _run(
        tmp_path,
        agent_dir,
        TEST_STEP="session_start",
        TEST_ANSWERS=json.dumps({"GET /reef/harness/releases": {"status": 200, "body": two}}),
    )
    assert _notices(out)[0]["message"].splitlines()[1] == (
        "2 release(s) ready to install: /versions v3, v5 (install with /versions <version> install)"
    )
    promoted = {"GET /reef/harness/releases": {"status": 200, "body": AFTER_PROMOTE}}
    out = _run(tmp_path, agent_dir, TEST_STEP="session_start", TEST_ANSWERS=json.dumps(promoted))
    assert _notices(out) == [{"kind": "notify", "message": first, "type": "info"}]
    # The catalog read failing keeps the first line; without a UI nothing is said and nothing is read.
    out = _run(tmp_path, agent_dir, TEST_STEP="session_start")
    assert _notices(out) == [{"kind": "notify", "message": first, "type": "info"}]
    out = _run(tmp_path, agent_dir, TEST_STEP="session_start", TEST_ANSWERS=json.dumps(CATALOG), TEST_HEADLESS="1")
    assert out["events"] == []


def test_versions_keeps_design_and_review_on_the_detail_page(tmp_path: Path) -> None:
    """The listing shows summaries; full design and review notes belong to the step's page."""
    notes = {
        "design": "D" * 300,
        "review": {"result": "partial", "covered": ["x"], "uncovered": ["two way replies", "idle detection"]},
    }
    rows = [
        row if index != 1 else {**row, "metrics": {**row["metrics"], "proposal_notes": notes}}
        for index, row in enumerate(RELEASES["releases"])
    ]
    catalog = {"GET /reef/harness/releases": {"status": 200, "body": {**RELEASES, "releases": rows}}}
    listed = _versions(tmp_path, _install_root(tmp_path), catalog)
    baseline = _versions(tmp_path, tmp_path / "pi-agent", CATALOG)
    assert _notices(listed) == _notices(baseline)
    # The step's own dialog names the release and its result, not the notes.
    shown = _versions(tmp_path, _install_root(tmp_path / "one"), catalog, args="1")
    assert _dialogs(shown)[0]["message"] == "rel-1111-selected (selected, current)"


# -- the report that survives: the stored filings, the session start and the fetch deadline ---------------------


def test_a_filing_is_stored_beside_the_release_file_the_newest_ten_and_none_older_than_a_day(tmp_path: Path) -> None:
    """The file holds {id, text, filed_at} entries, newest last; a write keeps the newest ten and drops what is
    older than a day, so a request never reported does not stay forever, and a file that is not JSON starts over."""
    agent_dir = _install_root(tmp_path)
    now = time.time()
    old = [{"id": f"q-old-{index}", "text": f"request {index}", "filed_at": now - index} for index in range(12, 0, -1)]
    stale = {"id": "q-stale", "text": "long ago", "filed_at": now - 25 * 3600}
    (tmp_path / REQUESTS_FILE).write_text(json.dumps([stale, *old]), encoding="utf-8")
    other = {**SELECTED_ROW, "metrics": {"selected": True, "training_request": {"id": "q-other", "text": "x"}}}
    out = _ask(tmp_path, agent_dir, _catalog_with(other), text=LONG_TEXT)
    assert out["error"] is None
    stored = _stored(tmp_path)
    assert [entry["id"] for entry in stored] == [f"q-old-{index}" for index in range(9, 0, -1)] + ["q-1"]
    assert stored[-1]["text"] == LONG_TEXT and abs(stored[-1]["filed_at"] - now) < 60
    (tmp_path / REQUESTS_FILE).write_text("not json", encoding="utf-8")
    out = _ask(tmp_path, agent_dir, _catalog_with(other), text="text me")
    assert [(entry["id"], entry["text"]) for entry in _stored(tmp_path)] == [("q-1", "text me")]


def test_session_start_reports_a_stored_request_that_settled_and_re_arms_the_watch_for_one_still_running(
    tmp_path: Path,
) -> None:
    """A request filed before a restart, or settled while the person was away: its result is delivered at the
    next session start as the custom message and the notice, and dropped; one the catalog does not hold yet gets
    the watch again and stays stored. An entry older than a day is not reported, and a catalog that cannot be
    read still leaves the stored request with the watch."""
    agent_dir = _install_root(tmp_path)
    now = time.time()
    entries = [
        {"id": "q-1", "text": LONG_TEXT, "filed_at": now - 600},
        {"id": "q-9", "text": "log when blocked", "filed_at": now - 60},
        {"id": "q-stale", "text": "long ago", "filed_at": now - 25 * 3600},
    ]
    (tmp_path / REQUESTS_FILE).write_text(json.dumps(entries), encoding="utf-8")
    stale_row = {**SELECTED_ROW, "metrics": {"selected": True, "training_request": {"id": "q-stale", "text": "x"}}}
    rows = {"scenario": "code-repair", "releases": [CREATION_ROW, SELECTED_ROW, stale_row]}
    out = _run(
        tmp_path,
        agent_dir,
        TEST_STEP="session_start",
        TEST_ANSWERS=json.dumps({"GET /reef/harness/releases": {"status": 200, "body": rows}}),
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="60",
    )
    assert out["error"] is None
    first = "reef: /reefine <what it should do> asks for a harness change; /versions lists the versions."
    report = (
        f"reef: '{ASK}' is published as release rel-1111. Install when ready with /versions v1 install."
        " Details: /versions v1.\nNot covered: two way replies; idle detection"
    )
    # The still-running entry's record is read once (unanswered here: the runner knows no such route) before
    # the watch takes it, so a request the service no longer knows is dropped instead of watched.
    kinds = [event["kind"] for event in out["events"]]
    assert kinds[:6] == ["fetch", "notify", "message", "notify", "fetch", "status"]
    assert out["events"][1] == {"kind": "notify", "message": first, "type": "info"}
    assert out["events"][2] == {
        "kind": "message",
        "message": {"customType": "reef-harness", "content": report, "display": True},
        "options": {"triggerTurn": False},
    }
    assert out["events"][3] == {"kind": "notify", "message": report, "type": "info"}
    assert out["events"][4]["url"].endswith("/reef/scenarios/code-repair/records/q-9")
    assert out["events"][5] == {"kind": "status", "key": "reef", "text": "reef: request q-9 queued"}
    assert len(_of_kind(out, "message")) == 1  # the stale entry's row is in the catalog; it is not reported
    assert len([event for event in _fetches(out) if event["url"].endswith("/reef/harness/releases")]) >= 3
    assert [entry["id"] for entry in _stored(tmp_path)] == ["q-9"]
    # The catalog unreadable at the start: the first line stands, and the stored request gets the watch anyway.
    out = _run(tmp_path, agent_dir, TEST_STEP="session_start", REEF_HARNESS_WATCH_MS="10", TEST_WAIT_MS="40")
    assert _notices(out) == [{"kind": "notify", "message": first, "type": "info"}]
    assert _of_kind(out, "status") == [{"kind": "status", "key": "reef", "text": "reef: request q-9 queued"}]
    assert [entry["id"] for entry in _stored(tmp_path)] == ["q-9"]


def test_a_hung_read_ends_at_the_fetch_deadline_so_the_watch_goes_on_and_a_filing_answers(tmp_path: Path) -> None:
    """Every fetch carries an abort signal with a deadline: a catalog read that never answers costs one poll, not
    every later tick, and a filing that never answers reports reef unreachable instead of hanging the tool."""
    agent_dir = _install_root(tmp_path)
    out = _ask(
        tmp_path,
        agent_dir,
        {"POST /reef/train": {"status": 200, "body": ACCEPTED}},
        text=LONG_TEXT,
        TEST_HANG=json.dumps(["GET /reef/harness/releases"]),
        REEF_HARNESS_FETCH_MS="20",
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="200",
    )
    assert out["error"] is None
    reads = [event for event in _fetches(out) if event["url"].endswith("/reef/harness/releases")]
    assert len(reads) >= 3 and all(event["signal"] for event in _fetches(out))
    assert _notices(out) == [{"kind": "notify", "message": ACCEPTED_NOTICE, "type": "info"}]
    hung = _ask(
        tmp_path,
        agent_dir,
        {},
        TEST_HANG=json.dumps(["POST /reef/train"]),
        REEF_HARNESS_FETCH_MS="20",
    )
    assert _notices(hung) == [
        {"kind": "notify", "message": "reef unreachable at http://reef:8900: no answer within 20 ms", "type": "error"}
    ]
    assert _of_kind(hung, "status") == []


# -- the next step after a result: the install through the wrapper, and for a pending release the promote first --

INSTALL_REASON = f"Read the change first: http://reef:8900/reef/harness/releases/1/page?scenario=code-repair{KEY}"
INSTALLED_LINE = "Installed release rel-1111. Type /reload to load it now."
CHECK = "osascript -e 'display notification \"reef\"'"
# What the wrapper lists for the selected release: an env item with a prompt, a permission item with a check, and
# an item already met.
LISTING = {
    "release_id": "rel-1111-selected",
    "items": [
        {
            "name": "REEF_AWAY_PHONE",
            "kind": "env",
            "check": None,
            "prompt": "The phone number to text, with the country code",
            "met": False,
        },
        {"name": "notify", "kind": "permission", "check": CHECK, "prompt": None, "met": False},
        {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID", "prompt": None, "met": True},
    ],
}
NOTHING_TO_SET_UP = {"release_id": "rel-4444-promote", "items": []}


def _wrapper(path: Path) -> str:
    """A wrapper script at ``path``, as the install script writes it; the runner stubs what it answers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return str(path)


def _exec_answers(listing: dict[str, Any] = LISTING, **answers: Any) -> str:
    """TEST_EXEC for a wrapper whose ``setup --json`` prints ``listing`` and whose other calls exit 0 unless named."""
    return json.dumps({"setup --json": {"stdout": json.dumps(listing)}, **answers})


def _settle(
    tmp_path: Path, agent_dir: Path, row: dict[str, Any], answers: dict[str, Any] | None = None, **env: str
) -> list[dict[str, Any]]:
    """The events after the result's report, for a request whose step settled as ``row``."""
    out = _ask(
        tmp_path,
        agent_dir,
        {**_catalog_with(row), **PROMOTED, **(answers or {})},
        text=LONG_TEXT,
        REEF_HARNESS_WATCH_MS="10",
        TEST_WAIT_MS="200",
        **env,
    )
    assert out["error"] is None
    kinds = [event["kind"] for event in out["events"]]
    assert kinds[:3] == ["fetch", "notify", "status"]
    # The settle: the report, then the notice. What follows them is what the case reads.
    report = kinds.index("message")
    assert kinds[report : report + 2] == ["message", "notify"]
    return out["events"][report + 2 :]


def install_step(tmp_path: Path, agent_dir: Path, row: dict[str, object], **env: str) -> list[dict[str, object]]:
    """Run the install action explicitly, returning events after the catalog read."""
    out = _versions(tmp_path, agent_dir, _catalog_with(row), args="1 install", **env)
    assert out["error"] is None
    assert out["events"][0]["kind"] == "fetch"
    return out["events"][1:]


@pytest.mark.parametrize("row", [REJECTED_ROW, SKIPPED_ROW], ids=["rejected", "skipped"])
def test_versions_install_refuses_a_step_that_published_no_tree(tmp_path: Path, row: dict[str, object]) -> None:
    """A rejected or skipped step commits no tree of its own: its row carries the head's id, so installing it
    would install the version already there. A release merely held back from the head is not refused."""
    events = install_step(tmp_path, _install_root(tmp_path), row, TEST_CONFIRM="1")
    assert len(events) == 1
    assert events[0]["kind"] == "notify" and events[0]["type"] == "warning"
    assert "published no tree" in events[0]["message"]


def _exec_args(events: list[dict[str, Any]]) -> list[list[str]]:
    return [event["args"] for event in events if event["kind"] == "exec"]


def _said(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(event["message"], event["type"]) for event in events if event["kind"] == "notify"]


def test_a_selected_release_installs_through_the_wrapper_after_the_confirm_with_the_setup_loop(
    tmp_path: Path,
) -> None:
    """The explicit install command asks for confirmation; yes runs the wrapper's update pinned to the
    release, then the setup loop: each unmet item is asked once, an env value through the input (the prompt as
    its title) and handed over as one argument, a check through a confirm (the check as its message) and run by
    name, one line per item; a met item is not asked. The last line says how to load the installed tree."""
    agent_dir = _install_root(tmp_path)
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_INPUT=json.dumps(["+1 555 0100"]),
        TEST_EXEC=_exec_answers(),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert events == [
        {"kind": "confirm", "title": "Install release rel-1111 now?", "message": INSTALL_REASON},
        {"kind": "exec", "command": wrapper, "args": ["update", "--release", "rel-1111-selected"]},
        {"kind": "exec", "command": wrapper, "args": ["setup", "--json", "--release", "rel-1111-selected"]},
        {"kind": "input", "title": "The phone number to text, with the country code", "placeholder": ""},
        {
            "kind": "exec",
            "command": wrapper,
            "args": ["setup", "--set", "REEF_AWAY_PHONE=+1 555 0100", "--release", "rel-1111-selected"],
        },
        {"kind": "notify", "message": "reef: REEF_AWAY_PHONE set", "type": "info"},
        {"kind": "confirm", "title": "Run this check?", "message": CHECK},
        {"kind": "exec", "command": wrapper, "args": ["setup", "--run", "notify", "--release", "rel-1111-selected"]},
        {"kind": "notify", "message": "reef: notify met", "type": "info"},
        {"kind": "notify", "message": INSTALLED_LINE, "type": "info"},
    ]
    # Without the exported path the wrapper beside the release file serves; an item without a prompt is asked for
    # a value by name, and a check's prompt is its confirm's title.
    beside = _wrapper(tmp_path / "reef-pi")
    listing = {
        "release_id": "rel-1111-selected",
        "items": [
            {"name": "SMTP_HOST", "kind": "env", "check": None, "prompt": None, "met": False},
            {
                "name": "mail",
                "kind": "service",
                "check": "nc -z mail 25",
                "prompt": "Reach the mail host",
                "met": False,
            },
        ],
    }
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_INPUT=json.dumps(["mail"]),
        TEST_EXEC=_exec_answers(listing),
    )
    assert [event["command"] for event in events if event["kind"] == "exec"] == [beside] * 4
    assert events[3] == {"kind": "input", "title": "Value for SMTP_HOST", "placeholder": ""}
    assert events[6] == {"kind": "confirm", "title": "Reach the mail host", "message": "nc -z mail 25"}
    assert _said(events) == [("reef: SMTP_HOST set", "info"), ("reef: mail met", "info"), (INSTALLED_LINE, "info")]


def test_a_declined_install_names_the_commands_and_runs_nothing(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    events = install_step(tmp_path, agent_dir, SELECTED_ROW, TEST_EXEC=_exec_answers(), REEF_HARNESS_WRAPPER=wrapper)
    assert events == [
        {"kind": "confirm", "title": "Install release rel-1111 now?", "message": INSTALL_REASON},
        {"kind": "notify", "message": INSTALL_LATER, "type": "info"},
    ]


def test_a_declined_or_failed_setup_item_stays_unmet_and_is_named_at_the_end(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    # No value typed and a check declined: both skipped and named; the tree is installed all the same.
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM=json.dumps([True, False]),
        TEST_INPUT=json.dumps([""]),
        TEST_EXEC=_exec_answers(),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert _exec_args(events) == [
        ["update", "--release", "rel-1111-selected"],
        ["setup", "--json", "--release", "rel-1111-selected"],
    ]
    assert _said(events) == [
        ("reef: REEF_AWAY_PHONE skipped", "warning"),
        ("reef: notify skipped", "warning"),
        ("reef: still to set up: REEF_AWAY_PHONE, notify (reef-pi setup)", "warning"),
        (INSTALLED_LINE, "info"),
    ]
    # A value the wrapper refuses and a check that fails: not met, with the exit code.
    failing = {
        "setup --set": {"code": 2, "stderr": "no env item named REEF_AWAY_PHONE"},
        "setup --run notify": {"code": 1},
    }
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_INPUT=json.dumps(["x"]),
        TEST_EXEC=_exec_answers(**failing),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert _said(events) == [
        ("reef: REEF_AWAY_PHONE not met (exit 2)", "warning"),
        ("reef: notify not met (exit 1)", "warning"),
        ("reef: still to set up: REEF_AWAY_PHONE, notify (reef-pi setup)", "warning"),
        (INSTALLED_LINE, "info"),
    ]


def test_an_update_the_wrapper_refuses_runs_the_setup_loop_first_then_the_update_again(tmp_path: Path) -> None:
    """The wrapper exits 3 without installing while an item is unmet: the setup loop collects what it needs and the
    update runs again. A failed update stops with its stderr and no setup; a listing that fails is said and the
    loop stops, the installed tree still announced."""
    agent_dir = _install_root(tmp_path)
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    refused = {"code": 3, "stderr": "reef-pi update: rel-1111-selected requires setup first:\n  REEF_AWAY_PHONE (env)"}
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_INPUT=json.dumps(["+1 555 0100"]),
        TEST_EXEC=_exec_answers(update=[refused, {"code": 0}]),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert _exec_args(events) == [
        ["update", "--release", "rel-1111-selected"],
        ["setup", "--json", "--release", "rel-1111-selected"],
        ["setup", "--set", "REEF_AWAY_PHONE=+1 555 0100", "--release", "rel-1111-selected"],
        ["setup", "--run", "notify", "--release", "rel-1111-selected"],
        ["update", "--release", "rel-1111-selected"],
    ]
    assert _said(events)[-1] == (INSTALLED_LINE, "info")
    # Refused again after the loop left an item unmet: the wrapper's own words, and no install line.
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM=json.dumps([True, False]),
        TEST_INPUT=json.dumps([""]),
        TEST_EXEC=_exec_answers(update=refused),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert _exec_args(events)[-1] == ["update", "--release", "rel-1111-selected"]
    assert _said(events)[-1] == (f"reef: reef-pi update failed (exit 3): {refused['stderr']}", "error")
    failed = {"code": 1, "stderr": "curl: (7) Failed to connect"}
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_EXEC=_exec_answers(update=failed),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert _exec_args(events) == [["update", "--release", "rel-1111-selected"]]
    assert _said(events) == [("reef: reef-pi update failed (exit 1): curl: (7) Failed to connect", "error")]
    listing_failed = {
        "setup --json": {"code": 2, "stderr": "reef-pi setup: no release rel-1111-selected in the catalog"}
    }
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_EXEC=json.dumps(listing_failed),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert not any(event["kind"] == "input" for event in events)
    assert _said(events) == [
        ("reef-pi setup: no release rel-1111-selected in the catalog", "error"),
        (INSTALLED_LINE, "info"),
    ]


def test_without_a_wrapper_on_disk_the_install_names_the_commands(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    # An exported path that does not exist counts as none, and nothing sits beside the release file.
    events = install_step(
        tmp_path,
        agent_dir,
        SELECTED_ROW,
        TEST_CONFIRM="1",
        TEST_EXEC=_exec_answers(),
        REEF_HARNESS_WRAPPER=str(tmp_path / "gone" / "reef-pi"),
    )
    assert events == [
        {"kind": "confirm", "title": "Install release rel-1111 now?", "message": INSTALL_REASON},
        {"kind": "notify", "message": NO_WRAPPER, "type": "warning"},
    ]


@pytest.mark.parametrize("busy", ["0", "1"])
@pytest.mark.parametrize("headless", ["0", "1"])
def test_a_settle_offers_the_install_only_when_the_session_is_between_turns(
    tmp_path: Path, busy: str, headless: str
) -> None:
    """A win reaches the person who asked for it without them going looking: the settle offers its install. The
    dialog waits for a turn to end, so a busy session keeps the report's commands, and headless has no dialog."""
    agent_dir = _install_root(tmp_path)
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    for row in (SELECTED_ROW, PENDING_ROW):
        events = _settle(
            tmp_path,
            agent_dir,
            row,
            TEST_HEADLESS=headless,
            TEST_BUSY=busy,
            TEST_CONFIRM="1",
            TEST_EXEC=_exec_answers(NOTHING_TO_SET_UP),
            REEF_HARNESS_WRAPPER=wrapper,
        )
        if busy == "1" or headless == "1":
            assert events == []
            continue
        assert events[0]["kind"] == "confirm"
        assert events[0]["title"] == f"Install release {str(row['release_id'])[:8]} now?"
        # A pending release says what makes it different before the person answers.
        pending = row is PENDING_ROW
        assert ("runs in pi with your privileges" in events[0]["message"]) is pending
        # A pending release is served before it is installed; a published one installs straight through.
        assert [event["kind"] for event in events[1:]] == (
            ["fetch", "exec", "exec", "notify"] if pending else ["exec", "exec", "notify"]
        )
        if pending:
            assert events[1]["url"] == "http://reef:8900/reef/scenarios/code-repair/promote"
            assert events[1]["body"] == {"release_id": row["release_id"]}
        assert events[-1]["message"].endswith("Type /reload to load it now.")


def test_a_settle_that_published_no_tree_offers_no_install(tmp_path: Path) -> None:
    """A rejected or skipped step commits no tree of its own, so its report ends the settle."""
    agent_dir = _install_root(tmp_path)
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    for row in (REJECTED_ROW, SKIPPED_ROW):
        events = _settle(
            tmp_path,
            agent_dir,
            row,
            TEST_CONFIRM="1",
            TEST_EXEC=_exec_answers(),
            REEF_HARNESS_WRAPPER=wrapper,
        )
        assert events == []


def test_a_settle_whose_install_is_declined_runs_nothing(tmp_path: Path) -> None:
    """The offer is a question: declining it installs nothing, promotes nothing and names the commands."""
    events = _settle(
        tmp_path,
        _install_root(tmp_path),
        PENDING_ROW,
        TEST_EXEC=_exec_answers(),
        REEF_HARNESS_WRAPPER=_wrapper(tmp_path / "bin" / "reef-pi"),
    )
    assert [event["kind"] for event in events] == ["confirm", "notify"]
    assert events[1] == {"kind": "notify", "message": INSTALL_LATER, "type": "info"}


def test_a_request_the_service_no_longer_knows_ends_the_watch_with_one_notice_and_is_dropped(tmp_path: Path) -> None:
    """Two polls in a row that find no record of the request (a 404, the scenario reset under it) end the watch
    with one warning, clear the footer and drop the stored entry; a single 404 is a record not written yet."""
    agent_dir = _install_root(tmp_path)
    other = {**SELECTED_ROW, "metrics": {"selected": True, "training_request": {"id": "q-other", "text": "x"}}}
    answers = {
        **_catalog_with(other),
        f"GET {RECORD_PATH}": [{"status": 404, "body": {"error": "no record"}}] * 3,
    }
    out = _ask(tmp_path, agent_dir, answers, text=LONG_TEXT, REEF_HARNESS_WATCH_MS="10", TEST_WAIT_MS="200")
    assert out["error"] is None
    gone = "reef: request q-1 is no longer on the service (its scenario was reset); ask again with /reefine"
    assert [event for event in _notices(out) if event["type"] == "warning"] == [
        {"kind": "notify", "message": gone, "type": "warning"}
    ]
    assert [event["text"] for event in _of_kind(out, "status")][-1] is None
    assert len([call for call in _fetches(out) if call["url"].endswith(RECORD_PATH)]) == 2
    assert json.loads((tmp_path / REQUESTS_FILE).read_text(encoding="utf-8")) == []


def test_session_start_drops_a_stored_request_the_service_no_longer_knows(tmp_path: Path) -> None:
    """A stored request whose row is not in the catalog and whose record the service answers with 404 is said
    to be gone, once, and dropped instead of taking the watch."""
    agent_dir = _install_root(tmp_path)
    (tmp_path / REQUESTS_FILE).write_text(
        json.dumps([{"id": "q-9", "text": "log when blocked", "filed_at": time.time() - 60}]), encoding="utf-8"
    )
    answers = {
        "GET /reef/harness/releases": {
            "status": 200,
            "body": _listed({"scenario": "code-repair", "releases": [CREATION_ROW]}),
        },
        "GET /reef/scenarios/code-repair/records/q-9": {"status": 404, "body": {"error": "no record"}},
    }
    out = _run(tmp_path, agent_dir, TEST_STEP="session_start", TEST_ANSWERS=json.dumps(answers), TEST_WAIT_MS="60")
    assert out["error"] is None
    gone = "reef: request q-9 is no longer on the service (its scenario was reset); ask again with /reefine"
    assert {"kind": "notify", "message": gone, "type": "warning"} in _notices(out)
    assert _of_kind(out, "status") == []
    assert json.loads((tmp_path / REQUESTS_FILE).read_text(encoding="utf-8")) == []
