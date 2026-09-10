"""The shipped update notice: seeded by config, rendered byte exact, bounded.

The notice is composition: ``evolution.version_check: true`` appends the
adapter's shipped ``code_extension`` entry to the seed, the same load and
render paths as every other node carry it, and adapters without a shipped
extension refuse boot with a config error naming them.
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
from reef.train.cordis_backend import CordisBackend, CordisRecipe
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
    assert "const updateOption = `Update with ${instruction}`" in text
    assert '[updateOption, "Skip"]' in text
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
    env.pop("PI_OFFLINE", None)

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
        assert events[3]["message"] == "Reef harness updated. Restart reef-pi to load it."


def _notice(tmp_path: Path, releases: object, release_info: object, *, headless: bool = False) -> tuple[list, str]:
    """The UI events and stderr of one session start of the notice against ``releases`` with ``release_info`` on disk."""
    module = tmp_path / "version_check.mjs"
    module.write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "pi-agent").mkdir(exist_ok=True)
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
  exec: async () => ({ stdout: "", stderr: "", code: 0, killed: false }),
});
globalThis.fetch = async () => ({ ok: true, json: async () => JSON.parse(process.env.TEST_RELEASES) });
await sessionStart(
  { type: "session_start", reason: "startup" },
  {
    hasUI: process.env.TEST_HEADLESS !== "1",
    ui: {
      select: async (title, options) => {
        events.push({ kind: "select", title, options });
        return options[1];
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
    env = {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
        "REEF_SERVICE_URL": "http://reef:8900",
        "REEF_SCENARIO": "code-repair",
        "TEST_RELEASES": json.dumps({"releases": releases}),
        "TEST_HEADLESS": "1" if headless else "0",
    }
    env.pop("PI_OFFLINE", None)
    completed = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True, env=env)
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
    # Already on the head: silence, whatever the check offs say.
    events, _ = _notice(tmp_path, releases, {"release_id": "v2"})
    assert events == []


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
    assert [event["kind"] for event in events] == ["notify"]
    assert events[0]["message"].splitlines()[:2] == [
        "Reef harness update available (v4), but it requires setup first:",
        "  notify (permission)",
    ]
    # A catalog that is not a list is silence, never an error.
    assert _notice(tmp_path, "nope", {"release_id": "v1"}) == ([], "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_never_offers_a_pending_release(tmp_path: Path) -> None:
    """A release held for review is served to no session, so the head the notice
    offers is the newest row that is not pending: a pending tail behind the
    pinned head is silence, a newer row that is not pending is still offered,
    and a trial install of the pending release gets no offer until its promote."""
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
