"""The harness requests extension: reef-pi's /reef-harness and /reef-versions commands, run under node with stubs.

The asset registers nothing under ``PI_OFFLINE`` and no tools at all; the
ask command posts the request with the session id and the release file's release,
leaves inference receipts available for feedback, and reports durable
acceptance; the versions command lists the chain, prints a step's page and
promotes a pending release after a confirmation.
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
    confirm: async (title, message) => { events.push({ kind: "confirm", title, message }); return process.env.TEST_CONFIRM === "1"; },
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
  } else if (process.env.TEST_STEP === "versions") {
    await commands["reef-versions"].handler(process.env.TEST_ARGS || "", ctx);
  }
} catch (error) {
  out.error = error.message;
}
console.log(JSON.stringify(out));
""".strip()


def _install_root(tmp_path: Path, *, with_release_file: bool = True) -> Path:
    """A pulled pi tree: the release file at the root and the models.json that points at the proxy in pi-agent."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    if with_release_file:
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
    for name in ("PI_OFFLINE", "REEF_TOKEN", "TEST_CONFIRM"):
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
    # Asking confirms nothing; the one confirm guards the promote inside /reef-versions, registered after it.
    asking, versions, _ = text.partition('pi.registerCommand("reef-versions"')
    assert versions and "ui.confirm" not in asking


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
    assert out["commands"] == ["reef-harness", "reef-versions"]
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


# The catalog /reef-versions reads, oldest first: the creation row, a published win with a request, a rejected
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
CATALOG = {"GET /reef/harness/releases": {"status": 200, "body": RELEASES}}
PROMOTED = {"POST /reef/scenarios/code-repair/promote": {"status": 200, "body": {"release_id": "rel-4444-promote"}}}


def _versions(tmp_path: Path, agent_dir: Path, answers: dict[str, Any], args: str = "", **env: str) -> dict[str, Any]:
    return _run(tmp_path, agent_dir, TEST_STEP="versions", TEST_ARGS=args, TEST_ANSWERS=json.dumps(answers), **env)


def test_versions_lists_the_chain_oldest_first_one_line_per_row(tmp_path: Path) -> None:
    out = _versions(tmp_path, _install_root(tmp_path), CATALOG, REEF_TOKEN="tok")
    assert out["error"] is None
    (catalog,) = _fetches(out)
    assert catalog["method"] == "GET" and catalog["url"] == "http://reef:8900/reef/harness/releases"
    assert catalog["headers"] == {"x-reef-scenario": "code-repair", "authorization": "Bearer tok"}
    (notice,) = _notices(out)
    assert notice["type"] == "info"
    assert notice["message"].splitlines() == [
        "0  rel-0000  creation",
        f'1  rel-1111  selected  current  "{LONG_TEXT[:57]}..."',
        '2  rel-1111  rejected  "answer with care"',
        '3  rel-3333  pending  "log when blocked"',
        "4  rel-1111  skipped",
    ]


def test_versions_with_a_step_prints_the_tokens_its_row_carries(tmp_path: Path) -> None:
    out = _versions(tmp_path, _install_root(tmp_path), CATALOG, args="1", REEF_TOKEN="tok")
    (notice,) = _notices(out)
    lines = notice["message"].splitlines()
    assert lines[0] == "Harness step 1: rel-1111-selected (selected, current)"
    assert lines[-1] == "tokens: proposer 1200 in / 80 out, gate 950 in / 75 out"
    # A row without usage prints no token line.
    (tmp_path / "other").mkdir()
    out = _versions(tmp_path / "other", _install_root(tmp_path / "other"), CATALOG, args="4", REEF_TOKEN="tok")
    (notice,) = _notices(out)
    assert not any(line.startswith("tokens:") for line in notice["message"].splitlines())


def test_versions_with_a_step_prints_the_page_url_and_for_a_pending_release_the_promote_and_the_trial_install(
    tmp_path: Path,
) -> None:
    agent_dir = _install_root(tmp_path)
    out = _versions(tmp_path, agent_dir, CATALOG, args="3", REEF_TOKEN="tok")
    (notice,) = _notices(out)
    lines = notice["message"].splitlines()
    assert lines[0] == "Harness step 3: rel-3333-pending (pending)"
    assert lines[1] == "page: http://reef:8900/reef/harness/releases/3/page"
    auth = "-H 'x-reef-scenario: code-repair' -H \"Authorization: Bearer $REEF_TOKEN\" "
    assert (
        lines[2] == f"read it: curl -fsS {auth}'http://reef:8900/reef/harness/releases/3/page' > harness-step-3.html"
    )
    assert lines[3] == (
        f"promote: curl -fsS {auth}-X POST -H 'content-type: application/json' "
        "-d '{\"release_id\":\"rel-3333-pending\"}' 'http://reef:8900/reef/scenarios/code-repair/promote'"
    )
    assert lines[4] == "or from here: /reef-versions 3 promote"
    assert lines[5] == (
        f"trial install (replaces the tree at {tmp_path}): "
        f"curl -fsS {auth}'http://reef:8900/reef/harness/install?adapter=pi&release_id=rel-3333-pending'"
        f" | bash -s -- '{tmp_path}'"
    )
    # The way back rides beside the trial: the served head's own install, pinned by release id.
    assert lines[6] == (
        f"back to the head: curl -fsS {auth}'http://reef:8900/reef/harness/install?adapter=pi&release_id=rel-1111-selected'"
        f" | bash -s -- '{tmp_path}'"
    )
    assert len(lines) == 7
    # The token never leaves the environment: the printed commands name the variable, not the value.
    assert "tok" not in notice["message"].replace("$REEF_TOKEN", "")

    published = _versions(tmp_path, agent_dir, CATALOG, args="1")
    (notice,) = _notices(published)
    lines = notice["message"].splitlines()
    assert lines[0] == "Harness step 1: rel-1111-selected (selected, current)"
    assert (
        lines[2]
        == "read it: curl -fsS -H 'x-reef-scenario: code-repair' 'http://reef:8900/reef/harness/releases/1/page' > harness-step-1.html"
    )
    assert lines[3] == "tokens: proposer 1200 in / 80 out, gate 950 in / 75 out"
    assert len(lines) == 4 and "promote" not in notice["message"]


def test_versions_promotes_a_pending_release_after_the_confirm_and_not_without(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    confirmed = _versions(
        tmp_path,
        agent_dir,
        {**CATALOG, **PROMOTED},
        args="3 promote",
        TEST_CONFIRM="1",
        REEF_TOKEN="tok",
    )
    assert confirmed["error"] is None
    assert [event["kind"] for event in confirmed["events"]] == ["fetch", "confirm", "fetch", "notify"]
    prompt = confirmed["events"][1]
    assert prompt["title"] == "Promote harness step 3?"
    assert "rel-3333-pending" in prompt["message"] and "/reef/harness/releases/3/page" in prompt["message"]
    promote = confirmed["events"][2]
    assert promote["method"] == "POST" and promote["url"] == "http://reef:8900/reef/scenarios/code-repair/promote"
    assert promote["headers"] == {
        "x-reef-scenario": "code-repair",
        "authorization": "Bearer tok",
        "content-type": "application/json",
    }
    assert promote["body"] == {"release_id": "rel-3333-pending"}
    assert confirmed["events"][3] == {
        "kind": "notify",
        "message": "Promoted step 3: the head is now rel-4444-promote; the update notice offers it at the next session start.",
        "type": "info",
    }

    declined = _versions(tmp_path, agent_dir, {**CATALOG, **PROMOTED}, args="3 promote")
    assert [event["kind"] for event in declined["events"]] == ["fetch", "confirm", "notify"]
    assert declined["events"][2] == {"kind": "notify", "message": "step 3 not promoted", "type": "info"}

    not_pending = _versions(tmp_path, agent_dir, {**CATALOG, **PROMOTED}, args="1 promote", TEST_CONFIRM="1")
    assert [event["kind"] for event in not_pending["events"]] == ["fetch", "notify"]
    assert not_pending["events"][1] == {
        "kind": "notify",
        "message": "step 1 is not pending (selected); nothing to promote",
        "type": "warning",
    }


def test_versions_reports_an_unreachable_reef_as_one_notice(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    for args in ("", "3", "3 promote"):
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
    assert _notices(missing) == [
        {"kind": "notify", "message": "no step 9: the catalog holds steps 0 to 4", "type": "warning"}
    ]
    for args in ("three", "-1", "3 publish", "3 promote now"):
        out = _versions(tmp_path, agent_dir, CATALOG, args=args)
        assert _fetches(out) == []
        assert _notices(out) == [
            {"kind": "notify", "message": "Usage: /reef-versions [step] [promote]", "type": "warning"}
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


def test_versions_marks_the_served_head_current_and_never_the_pending_row(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    catalog = {"GET /reef/harness/releases": {"status": 200, "body": NEWEST_PENDING}}
    listed = _versions(tmp_path, agent_dir, catalog)
    (notice,) = _notices(listed)
    assert notice["message"].splitlines() == [
        "0  rel-0000  creation",
        '1  rel-1111  selected  current  "say when blocked"',
        "2  rel-1111  rejected",
        "3  rel-1111  rejected",
        "4  rel-4444  pending",
    ]
    head = _versions(tmp_path, agent_dir, catalog, args="1")
    assert _notices(head)[0]["message"].splitlines()[0] == "Harness step 1: rel-1111-selected (selected, current)"
    pending = _versions(tmp_path, agent_dir, catalog, args="4")
    lines = _notices(pending)[0]["message"].splitlines()
    assert lines[0] == "Harness step 4: rel-4444-pending (pending)"
    assert lines[6].startswith("back to the head: ") and "release_id=rel-1111-selected'" in lines[6]


def test_versions_reads_a_promoted_pending_row_as_promoted_and_offers_no_promote(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    catalog = {"GET /reef/harness/releases": {"status": 200, "body": AFTER_PROMOTE}}
    listed = _versions(tmp_path, agent_dir, catalog)
    (notice,) = _notices(listed)
    assert notice["message"].splitlines()[4:] == ["4  rel-4444  promoted at step 5", "5  rel-5555  promote  current"]
    shown = _versions(tmp_path, agent_dir, catalog, args="4")
    lines = _notices(shown)[0]["message"].splitlines()
    assert lines[0] == "Harness step 4: rel-4444-pending (promoted at step 5)"
    assert len(lines) == 3 and "promote" not in "\n".join(lines[1:]) and "install" not in "\n".join(lines)
    refused = _versions(tmp_path, agent_dir, {**catalog, **PROMOTED}, args="4 promote", TEST_CONFIRM="1")
    assert [event["kind"] for event in refused["events"]] == ["fetch", "notify"]
    assert refused["events"][1] == {
        "kind": "notify",
        "message": "step 4 is already promoted at step 5; nothing to promote",
        "type": "warning",
    }


def test_versions_takes_only_a_run_of_digits_as_the_step(tmp_path: Path) -> None:
    agent_dir = _install_root(tmp_path)
    for args in ("1e0", "0x1", "1.0", "+1", "1 promote extra"):
        out = _versions(tmp_path, agent_dir, CATALOG, args=args)
        assert _fetches(out) == []
        assert _notices(out) == [
            {"kind": "notify", "message": "Usage: /reef-versions [step] [promote]", "type": "warning"}
        ]
    leading_zero = _versions(tmp_path, agent_dir, CATALOG, args="01")
    assert (
        _notices(leading_zero)[0]["message"].splitlines()[0] == "Harness step 1: rel-1111-selected (selected, current)"
    )
