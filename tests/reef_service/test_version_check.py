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
from reef.harness.model_binding import ModelBinding
from reef.harness.render import render_composition
from reef.harness.version_check import VERSION_CHECK_ENTRY_ID, version_check_entry
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
