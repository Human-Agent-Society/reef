"""The shipped update notice: seeded by config, rendered byte exact, bounded.

The notice is composition: ``evolution.version_check: true`` appends the
adapter's shipped ``code_extension`` entry to the seed, the same load and
render paths as every other node carry it, and adapters without a shipped
extension refuse boot with a config error naming them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from recipes.harness_evolve import HarnessEvolveRecipe
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding
from reef.harness.render import render_composition
from reef.harness.version_check import VERSION_CHECK_ENTRY_ID, version_check_entry
from reef.recipe import RecipeConfigError
from reef.train.harness_backend import HarnessEvolveBackend
from reef.train.harness_backend.strategies import resolve_episode_scorer, resolve_proposer

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
    recipe = HarnessEvolveRecipe.from_environment({}, config=_config(version_check=True))
    entry = next(options for options in recipe.seed if options["id"] == VERSION_CHECK_ENTRY_ID)
    nodes = tuple((str(options["name"]), options["config"]) for options in recipe.seed)
    files = render_composition(nodes, get_adapter("pi"))
    assert files["pi-agent/extensions/reef-version-check.ts"] == ASSET.read_text(encoding="utf-8")
    assert entry["config"]["name"] == VERSION_CHECK_ENTRY_ID


def test_version_check_entry_passes_the_backends_seed_validation() -> None:
    HarnessEvolveBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, model: None),
        score_episode=resolve_episode_scorer(lambda task, result: 0.0),
        tasks=("probe",),
        models=ModelBinding(base_url="http://localhost:8000", model="demo-model"),
        seed=(version_check_entry("pi"),),
    )


def test_version_check_refuses_an_adapter_without_a_shipped_extension() -> None:
    with pytest.raises(RecipeConfigError, match="'opencode' ships no version check extension"):
        HarnessEvolveRecipe.from_environment({}, config=_config(adapter="opencode", version_check=True))


def test_version_check_must_be_a_boolean() -> None:
    with pytest.raises(RecipeConfigError, match="version_check must be a boolean"):
        HarnessEvolveRecipe.from_environment({}, config=_config(version_check="yes"))


def test_version_check_off_by_default_seeds_nothing() -> None:
    recipe = HarnessEvolveRecipe.from_environment({}, config=_config())
    assert not any(options.get("id") == VERSION_CHECK_ENTRY_ID for options in recipe.seed)


def test_the_notice_guards_come_first_and_the_install_command_matches_the_route() -> None:
    text = ASSET.read_text(encoding="utf-8")
    body = text.split("export default", 1)[1]
    assert body.splitlines()[1].strip() == "if (process.env.PI_OFFLINE) return; // hermetic episodes stay silent"
    assert "/reef/harness/install?adapter=pi" in text
    assert "| bash" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_notice_parses_as_plain_javascript(tmp_path: Path) -> None:
    """The asset stays annotation-free by design (see its header comment), so
    a plain node parse is the check; TS syntax would fail here first."""
    module = tmp_path / "version_check.mjs"
    module.write_text(ASSET.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["node", "--check", str(module)], check=True, capture_output=True)
