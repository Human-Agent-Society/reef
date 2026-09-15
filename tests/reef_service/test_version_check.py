"""The shipped update notice: seeded by config, rendered byte exact, bounded.

The notice is composition: ``evolution.version_check: true`` appends the
adapter's shipped ``code_extension`` entry to the seed, the same load and
render paths as every other node carry it, and adapters without a shipped
extension refuse boot with a config error naming them. With the ``reef-pi``
wrapper on disk the notice offers to set up an unmet head through it before
the update, and runs the update through it too, ending with the reload line.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.version_check import VERSION_CHECK_ENTRY_ID, version_check_entry
from reef.harness.tree.render import render_composition
from reef.recipe import RecipeConfigError
from reef.recipe.cordis import CordisRecipe
from reef.train.cordis_backend import CordisBackend
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

ASSET = Path(__file__).parents[2] / "reef" / "harness" / "adapters" / "pi" / "version_check.ts"


def _config(**evolution: object) -> dict[str, object]:
    return {
        "evolution": {
            "propose": lambda nodes, samples, model: None,
            "evaluate": lambda task, result: 0.0,
            "tasks": ["probe"],
            **evolution,
        }
    }


def test_version_check_seeds_the_shipped_extension_and_renders_it_byte_exact() -> None:
    recipe = CordisRecipe.from_environment({}, config=_config(version_check=True))
    entry = next(options for options in recipe.seed if options["id"] == VERSION_CHECK_ENTRY_ID)
    nodes = tuple((str(options["name"]), options["config"]) for options in recipe.seed)
    files = render_composition(nodes, get_adapter("pi"))
    assert files["pi-agent/extensions/reef-version-check.ts"] == ASSET.read_text(encoding="utf-8")
    assert entry["config"]["name"] == VERSION_CHECK_ENTRY_ID


def test_version_check_entry_passes_the_backends_seed_validation() -> None:
    CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, model: None),
        score_episode=resolve_episode_scorer(lambda task, result: 0.0),
        tasks=("probe",),
        models=ModelBinding(base_url="http://localhost:8000", model="demo-model"),
        seed=(version_check_entry("pi"),),
        binary="fake-pi",
    )


def test_version_check_refuses_an_adapter_without_a_shipped_extension() -> None:
    with pytest.raises(RecipeConfigError, match="'opencode' ships no version check extension"):
        CordisRecipe.from_environment({}, config=_config(adapter="opencode", version_check=True))


def test_version_check_must_be_a_boolean() -> None:
    with pytest.raises(RecipeConfigError, match="version_check must be a boolean"):
        CordisRecipe.from_environment({}, config=_config(version_check="yes"))


def test_version_check_off_by_default_seeds_nothing() -> None:
    recipe = CordisRecipe.from_environment({}, config=_config())
    assert not any(options.get("id") == VERSION_CHECK_ENTRY_ID for options in recipe.seed)


def test_the_notice_registers_a_startup_prompt_and_matches_the_install_route() -> None:
    text = ASSET.read_text(encoding="utf-8")
    assert 'pi.on("session_start"' in text
    assert "checked || process.env.PI_OFFLINE" in text
    assert 'const updateOption = `Update with ${wrapper ? "reef-pi update" : instruction}`' in text
    assert '[updateOption, "Skip"]' in text
    assert 'pi.exec(wrapper, ["update"])' in text
    assert 'pi.exec("bash"' in text
    assert "if (!ctx.hasUI)" in text
    assert "/reef/harness/install?adapter=pi" in text
    assert "| bash" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_parses_as_plain_javascript(tmp_path: Path) -> None:
    """The asset stays annotation-free by design (see its header comment), so
    a plain node parse is the check; TS syntax would fail here first."""
    module = tmp_path / "version_check.mjs"
    module.write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["node", "--check", str(module)], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize(
    ("choose_update", "expected_kinds"),
    [(True, ["select", "notify", "exec", "notify"]), (False, ["select"])],
)
def test_the_notice_prompts_before_start_and_honors_the_choice(
    tmp_path: Path, choose_update: bool, expected_kinds: list[str]
) -> None:
    module = tmp_path / "version_check.mjs"
    module.write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    (tmp_path / ".reef-harness-release").write_text(json.dumps({"release_id": "v1"}), encoding="utf-8")
    runner = tmp_path / "runner.mjs"
    runner.write_text(
        """
import versionCheck from "./version_check.mjs";

let sessionStart;
const events = [];
versionCheck({
  on(name, handler) {
    if (name === "session_start") sessionStart = handler;
  },
  exec: async (command, args) => {
    events.push({ kind: "exec", command, args });
    return { stdout: "", stderr: "", code: 0, killed: false };
  },
});
globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({ releases: [{ release_id: "v1" }, { release_id: "v2" }] }),
});
await sessionStart(
  { type: "session_start", reason: "startup" },
  {
    hasUI: true,
    ui: {
      select: async (title, options) => {
        events.push({ kind: "select", title, options });
        return process.env.TEST_CHOOSE_UPDATE === "1" ? options[0] : options[1];
      },
      notify: (message, type) => events.push({ kind: "notify", message, type }),
    },
  },
);
console.log(JSON.stringify(events));
""".strip(),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "REEF_SERVICE_URL": "http://reef:8900",
        "REEF_SCENARIO": "code-repair",
        "TEST_CHOOSE_UPDATE": "1" if choose_update else "0",
    }
    for name in ("PI_OFFLINE", "REEF_HARNESS_WRAPPER"):
        env.pop(name, None)

    completed = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True, env=env)

    events = json.loads(completed.stdout)
    assert [event["kind"] for event in events] == expected_kinds
    prompt = events[0]
    assert prompt["options"][0].startswith("Update with curl -fsS ")
    assert prompt["options"][1] == "Skip"
    assert "Current: v1" in prompt["title"]
    assert "Latest:  v2" in prompt["title"]
    if choose_update:
        execution = events[2]
        assert execution["command"] == "bash"
        assert "/reef/harness/install?adapter=pi" in execution["args"][1]
        assert "bash -s --" in execution["args"][1]
        # scenario, serviceUrl, token, destDir: destDir defaults to agentDir/..
        assert execution["args"][-4:-1] == ["code-repair", "http://reef:8900", ""]
        assert execution["args"][-1].endswith("pi-agent/..") or "/" in execution["args"][-1]
        assert events[3]["message"] == "Installed release v2. Type /reload to load it now."


def _notice(
    tmp_path: Path, releases: object, release_info: object, *, headless: bool = False, **env: str
) -> tuple[list, str]:
    """The UI events and stderr of one session start of the notice against ``releases`` with ``release_info`` on disk.

    ``env`` sets variables for the session, as a shell that exports what a release requires would, and the runner's
    knobs: TEST_CHOOSE_UPDATE picks the update in the select (else Skip); TEST_CONFIRM and TEST_INPUT answer the
    confirms and the inputs from JSON lists, in order, the last confirm repeating; TEST_EXEC maps a joined argv of
    a pi.exec call, or a prefix of it, to the {stdout, stderr, code} the stub answers (no key: exit 0, nothing
    printed); REEF_HARNESS_WRAPPER names a wrapper on disk, as run_agent exports it."""
    module = tmp_path / "version_check.mjs"
    module.write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "pi-agent").mkdir(exist_ok=True)
    runner = tmp_path / "runner.mjs"
    runner.write_text(
        """
import versionCheck from "./version_check.mjs";

let sessionStart;
const events = [];
const execAnswers = JSON.parse(process.env.TEST_EXEC || "{}");
const execAnswerFor = (args) => {
  const joined = args.join(" ");
  const keys = Object.keys(execAnswers).filter((key) => joined === key || joined.startsWith(`${key} `));
  const key = keys.sort((a, b) => b.length - a.length)[0];
  return { stdout: "", stderr: "", code: 0, killed: false, ...(key === undefined ? {} : execAnswers[key]) };
};
versionCheck({
  on(name, handler) {
    if (name === "session_start") sessionStart = handler;
  },
  exec: async (command, args) => {
    events.push({ kind: "exec", command, args });
    return execAnswerFor(args);
  },
});
const confirms = JSON.parse(process.env.TEST_CONFIRM || "[]");
const inputs = JSON.parse(process.env.TEST_INPUT || "[]");
globalThis.fetch = async () => ({ ok: true, json: async () => JSON.parse(process.env.TEST_RELEASES) });
await sessionStart(
  { type: "session_start", reason: "startup" },
  {
    hasUI: process.env.TEST_HEADLESS !== "1",
    ui: {
      select: async (title, options) => {
        events.push({ kind: "select", title, options });
        return process.env.TEST_CHOOSE_UPDATE === "1" ? options[0] : options[1];
      },
      confirm: async (title, message) => {
        events.push({ kind: "confirm", title, message });
        return (confirms.length > 1 ? confirms.shift() : confirms[0]) === true;
      },
      input: async (title, placeholder) => {
        events.push({ kind: "input", title, placeholder });
        return inputs.shift() ?? undefined;
      },
      notify: (message, type) => events.push({ kind: "notify", message, type }),
    },
  },
);
console.log(JSON.stringify(events));
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / ".reef-harness-release").write_text(json.dumps(release_info), encoding="utf-8")
    full_env = {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
        "REEF_SERVICE_URL": "http://reef:8900",
        "REEF_SCENARIO": "code-repair",
        "TEST_RELEASES": json.dumps({"releases": releases}),
        "TEST_HEADLESS": "1" if headless else "0",
        **env,
    }
    for name in (
        "PI_OFFLINE",
        "REEF_HARNESS_WRAPPER",
        "TEST_CHOOSE_UPDATE",
        "TEST_CONFIRM",
        "TEST_INPUT",
        "TEST_EXEC",
    ):
        if name not in env:
            full_env.pop(name, None)
    completed = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True, env=full_env)
    return json.loads(completed.stdout), completed.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_prints_the_setup_list_instead_of_the_update_while_an_item_is_unmet(tmp_path: Path) -> None:
    """While the head's ``training_request.requires`` has an item the release file's ``setup`` does not check off,
    the notice prints the setup list and never offers the install."""
    requires = [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"}, {"name": "notify", "kind": "permission"}]
    releases = [
        {"release_id": "v1", "pending": False},
        {
            "release_id": "v2",
            "pending": False,
            "metrics": {"training_request": {"text": "text me", "requires": requires}},
        },
        # A pending tail row is not the head: its items stay out of the setup list and the offer names v2.
        {
            "release_id": "v3",
            "pending": True,
            "metrics": {"training_request": {"requires": [{"name": "later", "kind": "env"}]}},
        },
    ]
    # One item checked off, one not: the setup list through the UI, no prompt.
    events, stderr = _notice(
        tmp_path, releases, {"release_id": "v1", "setup": [{"name": "TWILIO_SID", "checked_at": 1.0}]}
    )
    # The pending tail's review notice belongs to the requests extension's session-start line, not to this one.
    assert [event["kind"] for event in events] == ["notify"] and stderr == ""
    assert events[0]["type"] == "warning"
    assert events[0]["message"] == (
        "Reef harness update available (v2), but it requires setup first:\n"
        "  notify (permission)\n"
        "Run reef-pi setup, then start reef-pi again."
    )
    # Headless, nothing checked off: the whole list on stderr, nothing through the UI.
    events, stderr = _notice(tmp_path, releases, {"release_id": "v1"}, headless=True)
    assert events == []
    assert "  TWILIO_SID (env): TWILIO_SID\n  notify (permission)\nRun reef-pi setup" in stderr
    # Every item checked off: the update is offered, against v2.
    events, _ = _notice(
        tmp_path,
        releases,
        {"release_id": "v1", "setup": [{"name": "TWILIO_SID", "checked_at": 1}, {"name": "notify"}]},
    )
    assert [event["kind"] for event in events] == ["select"]
    assert "Latest:  v2" in events[0]["title"]
    # Already on the head: no offer, whatever the check offs say; the head requires TWILIO_SID, so a shell
    # without it hears that once (see test_the_notice_warns_once_per_unset_variable_the_installed_release_needs).
    events, _ = _notice(tmp_path, releases, {"release_id": "v2"})
    assert [event["type"] for event in events] == ["warning"] and "TWILIO_SID is not set" in events[0]["message"]
    assert _notice(tmp_path, releases, {"release_id": "v2"}, TWILIO_SID="AC1") == ([], "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_reads_the_chains_union_and_tolerates_a_bad_requires_or_setup_entry(tmp_path: Path) -> None:
    """The head needs every item over its chain, as the manifest lists it: a row whose requires is not a list
    adds nothing, a null setup entry is skipped, a check off whose recorded check is not the item's is unmet, and
    a promote row continues at the release it promoted."""
    item = {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"}
    releases = [
        {"release_id": "v1", "pending": False, "parent_release_id": None},
        {
            "release_id": "v2",
            "pending": False,
            "parent_release_id": "v1",
            "metrics": {"training_request": {"requires": [item]}},
        },
        {
            "release_id": "v3",
            "pending": False,
            "parent_release_id": "v2",
            "metrics": {"training_request": {"requires": "x"}},
        },
    ]
    stale = {"name": "TWILIO_SID", "checked_at": 1, "check": "OTHER"}
    events, _ = _notice(tmp_path, releases, {"release_id": "v1", "setup": [None, stale]})
    assert [event["kind"] for event in events] == ["notify"]
    assert events[0]["message"] == (
        "Reef harness update available (v3), but it requires setup first:\n"
        "  TWILIO_SID (env): TWILIO_SID\n"
        "Run reef-pi setup, then start reef-pi again."
    )
    events, _ = _notice(tmp_path, releases, {"release_id": "v1", "setup": [{**stale, "check": "TWILIO_SID"}]})
    assert [event["kind"] for event in events] == ["select"] and "Latest:  v3" in events[0]["title"]
    # The promoted head needs what the pending release it promoted named, on top of the chain's.
    releases += [
        {
            "release_id": "p1",
            "pending": True,
            "parent_release_id": "v3",
            "metrics": {"training_request": {"requires": [{"name": "notify", "kind": "permission"}]}},
        },
        {"release_id": "v4", "pending": False, "parent_release_id": "v3", "rollback_target_release_id": "p1"},
    ]
    events, _ = _notice(tmp_path, releases, {"release_id": "v3", "setup": [{**stale, "check": "TWILIO_SID"}]})
    assert [event["kind"] for event in events] == ["notify", "notify"]
    assert (
        events[0]["message"] == "reef: TWILIO_SID is not set; the installed harness needs it (reef-pi setup lists it)"
    )
    assert events[1]["message"].splitlines()[:2] == [
        "Reef harness update available (v4), but it requires setup first:",
        "  notify (permission)",
    ]
    # A catalog that is not a list is silence, never an error.
    assert _notice(tmp_path, "nope", {"release_id": "v1"}) == ([], "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_never_offers_a_pending_release(tmp_path: Path) -> None:
    """A release held for review is served to no session, so the head the notice
    offers is the newest row that is not pending: a pending tail behind the
    pinned head is silence here (the requests extension's session-start line
    names it), a newer row that is not pending is still offered, and a trial
    install of the pending release gets no offer until its promote."""
    release_info = {"release_id": "v1"}
    pending_tail = [{"release_id": "v1"}, {"release_id": "v2", "pending": True}]
    assert _notice(tmp_path, pending_tail, release_info) == ([], "")
    assert _notice(tmp_path, pending_tail, release_info, headless=True) == ([], "")
    # A catalog whose every row is pending has no head to offer.
    assert _notice(tmp_path, [{"release_id": "v2", "pending": True}], release_info) == ([], "")
    promoted_then_pending = [{"release_id": "v1"}, {"release_id": "v2"}, {"release_id": "v3", "pending": True}]
    events, stderr = _notice(tmp_path, promoted_then_pending, release_info)
    assert [event["kind"] for event in events] == ["select"] and stderr == ""
    assert "Current: v1" in events[0]["title"] and "Latest:  v2" in events[0]["title"]
    assert "v3" not in events[0]["title"]
    events, stderr = _notice(tmp_path, promoted_then_pending, release_info, headless=True)
    assert events == [] and "Latest:  v2" in stderr and "v3" not in stderr
    # A trial install of the pending release by id (?release_id=v3) is the person's choice: no offer to move back.
    trial = {"release_id": "v3"}
    assert _notice(tmp_path, promoted_then_pending, trial) == ([], "")
    assert _notice(tmp_path, promoted_then_pending, trial, headless=True) == ([], "")
    # Once a promote republishes the trial tree, the promoted head is offered to it.
    promoted = [*promoted_then_pending, {"release_id": "v4", "rollback_target_release_id": "v3"}]
    events, _ = _notice(tmp_path, promoted, trial)
    assert [event["kind"] for event in events] == ["select"]
    assert "Current: v3" in events[0]["title"] and "Latest:  v4" in events[0]["title"]
    # A null row is skipped, never an error; a release file that is not a record is silence.
    events, _ = _notice(tmp_path, [{"release_id": "v1"}, None, {"release_id": "v2"}], release_info)
    assert [event["kind"] for event in events] == ["select"] and "Latest:  v2" in events[0]["title"]
    assert _notice(tmp_path, promoted_then_pending, None) == ([], "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_warns_once_per_unset_variable_the_installed_release_needs(tmp_path: Path) -> None:
    """Before the head comparison, every ``env`` item over the installed release's chain whose variable (the check,
    else the name) this shell lacks gets one warning line; a set variable, a checked off one included, is silent,
    and the other kinds say nothing here. The update flow follows unchanged."""
    releases = [
        {"release_id": "v1", "pending": False, "parent_release_id": None},
        {
            "release_id": "v2",
            "pending": False,
            "parent_release_id": "v1",
            "metrics": {
                "training_request": {
                    "requires": [
                        {"name": "twilio.sid", "kind": "env", "check": "TWILIO_SID"},
                        {"name": "SMTP_HOST", "kind": "env"},
                        {"name": "notify", "kind": "permission", "check": "true"},
                    ]
                }
            },
        },
        {"release_id": "v3", "pending": False, "parent_release_id": "v2"},
    ]
    warning = "reef: {} is not set; the installed harness needs it (reef-pi setup lists it)"
    # On the head with both variables unset: two warnings, in the chain's order, and nothing else.
    events, stderr = _notice(tmp_path, releases, {"release_id": "v3", "setup": [{"name": "twilio.sid"}]})
    assert [(event["type"], event["message"]) for event in events] == [
        ("warning", warning.format("TWILIO_SID")),
        ("warning", warning.format("SMTP_HOST")),
    ]
    assert stderr == ""
    # One set: one warning; both set: silence. Headless, the warnings go to stderr.
    events, _ = _notice(tmp_path, releases, {"release_id": "v3"}, TWILIO_SID="AC1")
    assert [event["message"] for event in events] == [warning.format("SMTP_HOST")]
    assert _notice(tmp_path, releases, {"release_id": "v3"}, TWILIO_SID="AC1", SMTP_HOST="mail") == ([], "")
    events, stderr = _notice(tmp_path, releases, {"release_id": "v3"}, headless=True)
    assert events == [] and stderr.splitlines() == [warning.format("TWILIO_SID"), warning.format("SMTP_HOST")]
    # Behind the head with every item checked off, the warning still comes first (a check off records that the
    # variable was set once, not that this shell has it) and the offer follows unchanged.
    checked = [{"name": "twilio.sid"}, {"name": "SMTP_HOST"}, {"name": "notify"}]
    events, _ = _notice(tmp_path, releases, {"release_id": "v2", "setup": checked}, TWILIO_SID="AC1")
    assert [event["kind"] for event in events] == ["notify", "select"]
    assert events[0]["message"] == warning.format("SMTP_HOST") and "Latest:  v3" in events[1]["title"]
    # The installed release v1 requires nothing: silence about the environment, whatever v2 needs.
    events, _ = _notice(tmp_path, releases, {"release_id": "v1", "setup": [{"name": "notify"}]})
    assert [event["kind"] for event in events] == ["notify"] and "requires setup first" in events[0]["message"]


# -- the setup through the wrapper, then the update through it -------------------------------------------------

SETUP_REQUIRES = [
    {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID", "prompt": "Your Twilio account SID"},
    {"name": "notify", "kind": "permission", "check": "true"},
]
SETUP_RELEASES = [
    {"release_id": "v1", "pending": False},
    {"release_id": "v2", "pending": False, "metrics": {"training_request": {"requires": SETUP_REQUIRES}}},
]
# What the wrapper lists for v2 while nothing is checked off: every item unmet, the prompt beside it.
SETUP_LISTING = {"release_id": "v2", "items": [{"prompt": None, **item, "met": False} for item in SETUP_REQUIRES]}
SETUP_LIST = "  TWILIO_SID (env): TWILIO_SID\n  notify (permission): true"
SETUP_MESSAGE = (
    f"Reef harness update available (v2), but it requires setup first:\n{SETUP_LIST}\n"
    "Run reef-pi setup, then start reef-pi again."
)
UPDATE_TITLE = "Reef harness update available\n\nCurrent: v1\nLatest:  v2"
LISTED = {"setup --json --release v2": {"stdout": json.dumps(SETUP_LISTING)}}


def _wrapper(path: Path) -> str:
    """A wrapper script at ``path``, as the install script writes it; the runner stubs what it answers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return str(path)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_offers_the_setup_through_the_wrapper_then_the_update_through_it(tmp_path: Path) -> None:
    """With a wrapper on disk and a UI, an unmet head is offered as a setup first: each unmet item is asked once,
    an env value through the input (the prompt as its title) and handed over as one argument, a check through a
    confirm and run by name, one line per item. Then the update is offered, names the wrapper's update, runs it,
    and ends with the reload line. The wrapper is the exported path, else the one beside the release file."""
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    answers = {**LISTED, "update": {"stdout": "installed v2\n"}}
    events, stderr = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1"},
        TEST_CONFIRM=json.dumps([True, True]),
        TEST_INPUT=json.dumps(["AC1"]),
        TEST_CHOOSE_UPDATE="1",
        TEST_EXEC=json.dumps(answers),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert stderr == ""
    assert events == [
        {"kind": "confirm", "title": "Set up release v2 now?", "message": SETUP_LIST},
        {"kind": "exec", "command": wrapper, "args": ["setup", "--json", "--release", "v2"]},
        {"kind": "input", "title": "Your Twilio account SID", "placeholder": ""},
        {"kind": "exec", "command": wrapper, "args": ["setup", "--set", "TWILIO_SID=AC1", "--release", "v2"]},
        {"kind": "notify", "message": "reef: TWILIO_SID set", "type": "info"},
        {"kind": "confirm", "title": "Run this check?", "message": "true"},
        {"kind": "exec", "command": wrapper, "args": ["setup", "--run", "notify", "--release", "v2"]},
        {"kind": "notify", "message": "reef: notify met", "type": "info"},
        {"kind": "select", "title": UPDATE_TITLE, "options": ["Update with reef-pi update", "Skip"]},
        {"kind": "notify", "message": "Updating Reef harness...", "type": "info"},
        {"kind": "exec", "command": wrapper, "args": ["update"]},
        {"kind": "notify", "message": "Installed release v2. Type /reload to load it now.", "type": "info"},
    ]
    # Without the exported path the wrapper beside the release file serves; Skip at the offer runs no update.
    beside = _wrapper(tmp_path / "reef-pi")
    events, _ = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1"},
        TEST_CONFIRM=json.dumps([True]),
        TEST_INPUT=json.dumps(["AC1"]),
        TEST_EXEC=json.dumps(answers),
    )
    assert [event["command"] for event in events if event["kind"] == "exec"] == [beside] * 3
    assert [event["kind"] for event in events][-2:] == ["notify", "select"]
    (tmp_path / "reef-pi").unlink()
    # Nothing unmet with a wrapper: the offer alone, naming the wrapper's update; its failure is said as before.
    checked = [{"name": "TWILIO_SID", "check": "TWILIO_SID"}, {"name": "notify", "check": "true"}]
    events, _ = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1", "setup": checked},
        TEST_CHOOSE_UPDATE="1",
        TEST_EXEC=json.dumps({"update": {"code": 1, "stderr": "curl: (7) Failed to connect"}}),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert events == [
        {"kind": "select", "title": UPDATE_TITLE, "options": ["Update with reef-pi update", "Skip"]},
        {"kind": "notify", "message": "Updating Reef harness...", "type": "info"},
        {"kind": "exec", "command": wrapper, "args": ["update"]},
        {"kind": "notify", "message": "Reef harness update failed:\ncurl: (7) Failed to connect", "type": "error"},
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_declined_setup_prints_the_list_and_offers_no_update(tmp_path: Path) -> None:
    """No at the setup's confirm prints the list, as before, and offers no update; an item the loop left unmet is
    named and the update is not offered either, since the install would refuse; a listing that fails is said and
    stops the loop. Headless, a wrapper changes nothing: the list on stderr, nothing asked, nothing run."""
    wrapper = _wrapper(tmp_path / "bin" / "reef-pi")
    events, stderr = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1"},
        TEST_CONFIRM=json.dumps([False]),
        TEST_EXEC=json.dumps(LISTED),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert stderr == ""
    assert events == [
        {"kind": "confirm", "title": "Set up release v2 now?", "message": SETUP_LIST},
        {"kind": "notify", "message": SETUP_MESSAGE, "type": "warning"},
    ]
    events, _ = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1"},
        TEST_CONFIRM=json.dumps([True, False]),
        TEST_INPUT=json.dumps(["AC1"]),
        TEST_CHOOSE_UPDATE="1",
        TEST_EXEC=json.dumps(LISTED),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert [event["kind"] for event in events] == [
        "confirm",
        "exec",
        "input",
        "exec",
        "notify",
        "confirm",
        "notify",
        "notify",
    ]
    assert [event["message"] for event in events if event["kind"] == "notify"] == [
        "reef: TWILIO_SID set",
        "reef: notify skipped",
        "reef: still to set up: notify (reef-pi setup)",
    ]
    failed = {"setup --json": {"code": 2, "stderr": "reef-pi setup: no release v2 in the catalog"}}
    events, _ = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1"},
        TEST_CONFIRM=json.dumps([True]),
        TEST_CHOOSE_UPDATE="1",
        TEST_EXEC=json.dumps(failed),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert events[1:] == [
        {"kind": "exec", "command": wrapper, "args": ["setup", "--json", "--release", "v2"]},
        {"kind": "notify", "message": "reef-pi setup: no release v2 in the catalog", "type": "error"},
    ]
    events, stderr = _notice(
        tmp_path,
        SETUP_RELEASES,
        {"release_id": "v1"},
        headless=True,
        TEST_CONFIRM=json.dumps([True]),
        TEST_EXEC=json.dumps(LISTED),
        REEF_HARNESS_WRAPPER=wrapper,
    )
    assert events == [] and stderr.strip() == SETUP_MESSAGE
