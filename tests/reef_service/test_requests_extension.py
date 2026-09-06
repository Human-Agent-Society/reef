"""The harness requests extension: reef-pi's /reef-harness command, run under node with stubs.

The asset registers nothing under ``PI_OFFLINE`` and no tools at all; the
command posts the request with the session id and the sidecar's release,
leaves inference receipts available for feedback, and reports durable acceptance.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ASSET = Path(__file__).parents[2] / "reef" / "harness" / "adapters" / "pi" / "requests.ts"
SKILL = Path(__file__).parents[2] / "reef" / "harness" / "adapters" / "pi" / "pi_extension_api.md"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

ACCEPTED = {"agent_record_id": "q-1", "scenario": "code-repair", "request_type": "train"}
# One runner for every case: it loads the asset with a stub pi, a stub ctx and a stub fetch, runs the command
# when TEST_STEP names it, and prints what the extension registered and every call it made.
RUNNER = """
import requests from "./requests.mjs";

const tools = {};
const commands = {};
const events = [];
const pi = {
  registerTool(definition) { tools[definition.name] = definition; },
  registerCommand(name, definition) { commands[name] = definition; },
  on(name) { events.push({ kind: "on", name }); },
  sendUserMessage(text, options) { events.push({ kind: "user_message", text, options: options ?? null }); },
  exec: async () => ({ stdout: "", stderr: "", code: 0, killed: false }),
};
const ctx = {
  hasUI: true,
  isIdle: () => true,
  ui: {
    confirm: async (title, message) => { events.push({ kind: "confirm", title, message }); return false; },
    notify: (message, type) => events.push({ kind: "notify", message, type }),
    select: async () => undefined,
  },
  sessionManager: { getSessionId: () => "sess-1234" },
};
const answers = JSON.parse(process.env.TEST_ANSWERS || "{}");
globalThis.fetch = async (url, init = {}) => {
  const method = init.method || "GET";
  events.push({ kind: "fetch", method, url, headers: init.headers ?? {}, body: init.body ? JSON.parse(init.body) : null });
  const answer = answers[`${method} ${new URL(url).pathname}`];
  if (!answer) throw new Error(`connection refused: ${url}`);
  return { ok: answer.status < 400, status: answer.status, json: async () => answer.body, text: async () => JSON.stringify(answer.body) };
};
requests(pi);
const out = { tools: Object.keys(tools), commands: Object.keys(commands), events, error: null };
try {
  if (process.env.TEST_STEP === "command") {
    await commands["reef-harness"].handler(process.env.TEST_ARGS || "", ctx);
  }
} catch (error) {
  out.error = error.message;
}
console.log(JSON.stringify(out));
""".strip()


def _install_root(tmp_path: Path, *, sidecar: bool = True) -> Path:
    """A pulled pi tree: the sidecar at the root and the models.json that points at the proxy in pi-agent."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    if sidecar:
        (tmp_path / ".reef-harness-release").write_text(json.dumps({"release_id": "v1"}), encoding="utf-8")
    models = {"providers": {"reef": {"api": "openai-completions", "baseUrl": "http://127.0.0.1:4567/v1"}}}
    (agent_dir / "models.json").write_text(json.dumps(models), encoding="utf-8")
    return agent_dir


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
    for name in ("PI_OFFLINE", "REEF_TOKEN"):
        if name not in env:
            full_env.pop(name, None)
    completed = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True, env=full_env)
    return json.loads(completed.stdout)


def _fetches(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in out["events"] if event["kind"] == "fetch"]


def _notices(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in out["events"] if event["kind"] == "notify"]


def _ask(
    tmp_path: Path, agent_dir: Path, answers: dict[str, Any], text: str = "text me when you are blocked", **env: str
) -> dict[str, Any]:
    return _run(tmp_path, agent_dir, TEST_STEP="command", TEST_ARGS=text, TEST_ANSWERS=json.dumps(answers), **env)


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


def test_the_extension_carries_no_tools_and_no_confirmation() -> None:
    text = ASSET.read_text(encoding="utf-8")
    assert "registerTool" not in text
    assert "typebox" not in text
    assert "ui.confirm" not in text


def test_offline_registers_nothing(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), PI_OFFLINE="1")
    assert out["tools"] == [] and out["commands"] == [] and out["events"] == []


def test_a_missing_service_url_registers_nothing(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), REEF_SERVICE_URL="")
    assert out["tools"] == [] and out["commands"] == []


def test_a_missing_scenario_registers_nothing(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), REEF_SCENARIO="")
    assert out["tools"] == [] and out["commands"] == []


def test_registers_the_command_and_no_tool(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path))
    assert out["tools"] == []
    assert out["commands"] == ["reef-harness"]
    assert out["events"] == []


def test_the_command_prints_usage_with_no_argument(tmp_path: Path) -> None:
    out = _run(tmp_path, _install_root(tmp_path), TEST_STEP="command", TEST_ARGS="   ")
    assert out["events"] == [
        {"kind": "notify", "message": "Usage: /reef-harness <what the harness should do>", "type": "warning"}
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
    assert request["body"] == {"text": "text me when you are blocked", "session": "sess-1234", "release_id": "v1"}
    assert _notices(out) == [{"kind": "notify", "message": "Training request q-1 accepted.", "type": "info"}]


def test_the_command_needs_no_capture_proxy_or_models_file(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    (agent_dir / "models.json").unlink()
    out = _ask(tmp_path, agent_dir, {"POST /reef/train": {"status": 200, "body": ACCEPTED}})
    assert out["error"] is None
    assert len(_fetches(out)) == 1
    assert _notices(out)[0]["message"] == "Training request q-1 accepted."


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


def test_the_command_without_the_sidecar_sends_nothing(tmp_path: Path) -> None:
    out = _ask(
        tmp_path,
        _install_root(tmp_path, sidecar=False),
        {"POST /reef/train": {"status": 200, "body": ACCEPTED}},
    )
    (notice,) = out["events"]
    assert notice["kind"] == "notify" and notice["type"] == "error"
    assert ".reef-harness-release" in notice["message"] and "nothing was sent" in notice["message"]


def test_the_command_falls_back_to_the_sidecar_beside_the_agent_dir(tmp_path: Path) -> None:
    answers = {"POST /reef/train": {"status": 200, "body": ACCEPTED}}
    out = _ask(tmp_path, _install_root(tmp_path), answers, REEF_HARNESS_DEST="")
    assert _fetches(out)[0]["body"]["release_id"] == "v1"
