"""The shipped harness requests entries: seeded by config after the notice, rendered byte exact, reserved.

``evolution.requests: true`` appends the adapter's ``code_extension`` and
``skill`` entries to the seed; the same load and render paths as every
other node carry them, a pi release carries the rendered files alone (the
entries live in the commit log the service proposer reads), and adapters
without a shipped extension refuse boot naming them. The
assets are read through ``_ASSETS``, so these tests write placeholders and
point the module at them; the shipped files are checked when present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service.test_harness_recipe import MODEL, batch, make_binary

import reef.harness.episodes.requests as requests
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.episodes.requests import REQUESTS_ENTRY_ID, REQUESTS_SKILL_ID, request_entries
from reef.harness.episodes.version_check import VERSION_CHECK_ENTRY_ID, version_check_entry
from reef.harness.tree.render import render_composition
from reef.recipe import RecipeConfigError
from reef.train.cordis_backend import CordisBackend, CordisRecipe
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

EXTENSION = "export default function (pi) {\n  if (process.env.PI_OFFLINE) return;\n}\n"
SKILL = "---\nname: reef-pi-extension-api\ndescription: placeholder\n---\n# pi extension API\n"
SHIPPED = requests._ASSETS["pi"]


def _config(**evolution: object) -> dict[str, object]:
    return {
        "evolution": {
            "propose": lambda nodes, samples, model: None,
            "evaluate": lambda task, result: 0.0,
            "tasks": ["probe"],
            **evolution,
        }
    }


@pytest.fixture
def placeholders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    extension = tmp_path / "requests.ts"
    extension.write_text(EXTENSION, encoding="utf-8")
    skill = tmp_path / "pi_extension_api.md"
    skill.write_text(SKILL, encoding="utf-8")
    monkeypatch.setitem(requests._ASSETS, "pi", (extension, skill))
    return extension, skill


def test_requests_seeds_the_extension_and_the_skill_after_the_notice(placeholders: tuple[Path, Path]) -> None:
    recipe = CordisRecipe.from_environment({}, config=_config(version_check=True, requests=True))
    assert [options["id"] for options in recipe.seed] == [
        VERSION_CHECK_ENTRY_ID,
        REQUESTS_ENTRY_ID,
        REQUESTS_SKILL_ID,
    ]
    extension, skill = recipe.seed[1], recipe.seed[2]
    assert extension["name"] == "code_extension" and extension["config"]["name"] == REQUESTS_ENTRY_ID
    assert skill["name"] == "skill" and skill["config"]["name"] == REQUESTS_SKILL_ID
    nodes = tuple((str(options["name"]), options["config"]) for options in recipe.seed)
    files = render_composition(nodes, get_adapter("pi"))
    assert files["pi-agent/extensions/reef-requests.ts"] == EXTENSION
    assert files["pi-agent/skills/reef-pi-extension-api/SKILL.md"] == SKILL
    # The base release is the rendered files alone: pi declares no entries list, the extension reads none.
    base = recipe.base_artifact_files()
    assert base is not None and "pi-agent/tree.json" not in base
    assert base["pi-agent/extensions/reef-requests.ts"] == EXTENSION


def test_requests_without_the_notice_seeds_the_two_entries_alone(placeholders: tuple[Path, Path]) -> None:
    recipe = CordisRecipe.from_environment({}, config=_config(requests=True))
    assert [options["id"] for options in recipe.seed] == [REQUESTS_ENTRY_ID, REQUESTS_SKILL_ID]


def test_request_entries_pass_the_backends_seed_validation_and_a_recovered_state_boots(
    tmp_path: Path, placeholders: tuple[Path, Path]
) -> None:
    seed = (version_check_entry("pi"), *request_entries("pi"))
    backend = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, models: None),
        score_episode=resolve_episode_scorer(lambda task, result: 0.0),
        tasks=("probe",),
        models=MODEL,
        seed=seed,
        binary=str(make_binary(tmp_path)),
    )
    entries = [dict(entry) for entry in seed]
    rendered = backend._render_for_episode(entries)
    assert "pi-agent/tree.json" not in rendered and rendered["pi-agent/extensions/reef-requests.ts"] == EXTENSION
    # A recovered state carrying reef's own entries meets the admission gate and steps on.
    result = backend.prepare_step(batch(), {"steps": 1, "entries": entries}, 0)
    assert result.outcome == "skip" and result.metrics["skipped"] == "no proposal"
    assert [entry["id"] for entry in result.state["entries"]] == [
        VERSION_CHECK_ENTRY_ID,
        REQUESTS_ENTRY_ID,
        REQUESTS_SKILL_ID,
    ]


def test_requests_refuses_an_adapter_without_a_shipped_extension(placeholders: tuple[Path, Path]) -> None:
    with pytest.raises(RecipeConfigError, match="'opencode' ships no requests extension"):
        CordisRecipe.from_environment({}, config=_config(adapter="opencode", requests=True))
    with pytest.raises(DescriptorError, match="'native' ships no requests extension"):
        request_entries("native")


def test_requests_must_be_a_boolean(placeholders: tuple[Path, Path]) -> None:
    with pytest.raises(RecipeConfigError, match="requests must be a boolean"):
        CordisRecipe.from_environment({}, config=_config(requests="yes"))


def test_requests_off_by_default_seeds_nothing(placeholders: tuple[Path, Path]) -> None:
    recipe = CordisRecipe.from_environment({}, config=_config())
    assert not any(options.get("id") in (REQUESTS_ENTRY_ID, REQUESTS_SKILL_ID) for options in recipe.seed)


def test_a_missing_asset_refuses_boot_naming_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extension = tmp_path / "requests.ts"
    extension.write_text(EXTENSION, encoding="utf-8")
    monkeypatch.setitem(requests._ASSETS, "pi", (extension, tmp_path / "pi_extension_api.md"))
    with pytest.raises(DescriptorError, match=r"'pi' requests asset pi_extension_api\.md cannot be read"):
        request_entries("pi")
    with pytest.raises(RecipeConfigError, match=r"requests asset pi_extension_api\.md cannot be read"):
        CordisRecipe.from_environment({}, config=_config(requests=True))


@pytest.mark.skipif(not all(asset.is_file() for asset in SHIPPED), reason="the shipped requests assets are absent")
def test_the_shipped_assets_render_byte_exact() -> None:
    recipe = CordisRecipe.from_environment({}, config=_config(requests=True))
    nodes = tuple((str(options["name"]), options["config"]) for options in recipe.seed)
    files = render_composition(nodes, get_adapter("pi"))
    extension, skill = SHIPPED
    assert files["pi-agent/extensions/reef-requests.ts"] == extension.read_text(encoding="utf-8")
    assert files["pi-agent/skills/reef-pi-extension-api/SKILL.md"] == skill.read_text(encoding="utf-8")
