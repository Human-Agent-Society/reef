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
from reef_service.test_harness_recipe import MODEL, batch, make_binary, runtime

import reef.harness.episodes.requests as requests
from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError
from reef.harness.episodes.requests import (
    REQUESTS_COMMAND,
    REQUESTS_ENTRY_ID,
    REQUESTS_SKILL_ID,
    command_text,
    request_entries,
)
from reef.harness.episodes.version_check import VERSION_CHECK_ENTRY_ID, version_check_entry
from reef.harness.tree.render import render_composition
from reef.recipe import RecipeConfigError
from reef.recipe.cordis import CordisRecipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.cordis_backend import CordisBackend
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
    with pytest.raises(RecipeConfigError, match="'terminus' ships no requests extension"):
        CordisRecipe.from_environment({}, config=_config(adapter="terminus", requests=True))
    with pytest.raises(DescriptorError, match="'native' ships no requests extension"):
        request_entries("native")


@pytest.mark.parametrize(
    ("adapter", "typed", "request_words", "wrapper", "timeout"),
    [
        ("claude", "/reefine", 'The request is "$ARGUMENTS".', "reef-claude", "give it 150000"),
        ("codex", "$reefine", "the text after $reefine", '"$REEF_HARNESS_WRAPPER"', "wait for it to finish"),
        ("opencode", "/reefine", 'The request is "$ARGUMENTS".', '"$REEF_HARNESS_WRAPPER"', "give it 150000"),
        ("hermes", "/reefine", "alongside the skill invocation:", '"$REEF_HARNESS_WRAPPER"', "give it 150."),
        ("dsh", "/reefine", "the text after /reefine", '"$REEF_HARNESS_WRAPPER"', "timeoutMs 150000"),
    ],
)
def test_an_adapter_without_an_extension_seeds_one_reefine_command_the_wrapper_answers(
    adapter: str, typed: str, request_words: str, wrapper: str, timeout: str
) -> None:
    """Off pi the requests entry is one agent_command, ``reefine``, rendered where the adapter keeps its commands:
    it names the typed request, files it with the wrapper's evolve in the form the harness's permission check lets
    through, the request in single quotes, then waits in pieces the harness's shell tool allows."""
    recipe = CordisRecipe.from_environment({}, config=_config(adapter=adapter, requests=True))
    (options,) = [options for options in recipe.seed if options.get("id") == REQUESTS_ENTRY_ID]
    assert options["name"] == "agent_command" and options["config"]["name"] == REQUESTS_COMMAND
    descriptor = get_adapter(adapter)
    files = render_composition(_nodes([options]), descriptor)
    text = files[descriptor.node_paths["agent_command"].format(name=REQUESTS_COMMAND)]
    assert f"The person typed {typed} " in text and request_words in text
    assert f"{wrapper} evolve '<the request>'" in text and f"`{wrapper} wait <request id> --timeout 100`" in text
    assert timeout in text and "Its --timeout counts seconds." in text
    assert f"reef-{adapter} setup" in text and "{" not in text
    assert not any(options.get("id") == REQUESTS_SKILL_ID for options in recipe.seed)


def test_the_claude_command_pre_approves_only_the_wrapper_and_opencode_runs_it_as_the_build_agent() -> None:
    """Claude Code refuses an allowed-tools rule on a variable and approves a named one; opencode would run the
    command with a mode agent that has no bash."""
    claude = command_text("claude")
    assert claude.startswith("---\ndescription: Ask Reef to change this harness\n")
    assert "allowed-tools: Bash(reef-claude evolve:*), Bash(reef-claude wait:*)\n---\n" in claude
    assert "agent: build\n---\n" in command_text("opencode")
    # The quirks modules write the frontmatter hermes and dsh read; codex is told to write the path itself.
    assert not command_text("hermes").startswith("---") and not command_text("dsh").startswith("---")
    assert 'sandbox_permissions to "require_escalated"' in command_text("codex")
    # Codex's "don't ask again" writes a rule for that exact command: the text promises no more than that.
    assert "lets that same command run again without asking" in command_text("codex")
    assert "keeps it for the session" not in command_text("codex")


@pytest.mark.parametrize("adapter", ["claude", "codex", "opencode", "hermes", "dsh"])
def test_the_command_checks_it_runs_under_the_wrapper_except_where_the_check_costs_an_approval(adapter: str) -> None:
    """The check is one more shell call: on Claude Code it asks the person to approve a variable expansion, and the
    command exists only in the tree reef-claude runs, so there it is left out."""
    checked = "When REEF_HARNESS_WRAPPER is not set, this session was not started through" in command_text(adapter)
    assert checked is (adapter != "claude")


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


UPGRADED = "export default function (pi) {\n  if (process.env.PI_OFFLINE) return;\n  // upgraded\n}\n"


def _pi_backend(tmp_path: Path, seed: tuple) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, models: None),
        score_episode=resolve_episode_scorer(lambda task, result: 0.0),
        tasks=("probe",),
        models=MODEL,
        seed=seed,
        binary=str(make_binary(tmp_path)),
    )


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    return root


def test_a_served_tree_with_older_reef_entries_is_republished_with_the_shipped_ones(
    tmp_path: Path, placeholders: tuple[Path, Path]
) -> None:
    """The comparison is by rendered file: a stale entry is replaced where it sits, a missing one is appended, and
    the entries a proposal added stay; a tree that already carries the shipped files needs nothing."""
    extension, _ = placeholders
    rules = {"id": "r1", "name": "rules", "config": {"text": "proposed rules"}}
    old_entries = [request_entries("pi")[0], rules]
    served = _write_tree(tmp_path / "served", render_composition(_nodes(old_entries), get_adapter("pi")))
    extension.write_text(UPGRADED, encoding="utf-8")
    backend = _pi_backend(tmp_path, request_entries("pi"))

    result = backend.shipped_content_update({"steps": 3, "entries": old_entries}, served)
    assert result is not None and not result.pending
    assert result.metrics == {"shipped_content_update": {"entries": [REQUESTS_ENTRY_ID, REQUESTS_SKILL_ID]}}
    assert [entry["id"] for entry in result.state["entries"]] == [REQUESTS_ENTRY_ID, "r1", REQUESTS_SKILL_ID]
    assert result.state["steps"] == 3 and result.state["entries"][1] == rules
    tree = result.artifact.local_path
    assert (tree / "pi-agent/extensions/reef-requests.ts").read_text(encoding="utf-8") == UPGRADED
    assert (tree / "pi-agent/skills/reef-pi-extension-api/SKILL.md").read_text(encoding="utf-8") == SKILL
    assert "proposed rules" in (tree / "pi-agent/AGENTS.md").read_text(encoding="utf-8")

    # The republished tree carries the shipped files, so a second look finds nothing to do.
    assert backend.shipped_content_update(dict(result.state), tree) is None


def _nodes(entries: list[dict]) -> tuple[tuple[str, object], ...]:
    return tuple((str(entry["name"]), entry["config"]) for entry in entries)


def test_a_scenario_opened_by_an_upgraded_reef_commits_the_shipped_entries_once(
    tmp_path: Path, placeholders: tuple[Path, Path]
) -> None:
    """A restart after the assets changed publishes one training commit that serves them; the next restart finds the
    served tree current and commits nothing."""
    extension, _ = placeholders
    base = tmp_path / "base"
    _write_tree(base, render_composition(_nodes(list(request_entries("pi"))), get_adapter("pi")))
    factory = InMemoryRepositoryBackend.factory(base, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"

    def open_scenario() -> tuple[list[dict], str, list[dict]]:
        built = CordisRecipe(
            resolve_proposer(lambda nodes, samples, models: None),
            resolve_episode_scorer(lambda task, result: 0.0),
            ("probe",),
            binary=str(make_binary(tmp_path)),
            seed=request_entries("pi"),
            runtime=runtime(),
        )
        dispatcher = Dispatcher(
            built, factory, agent_record_dir=agent_record_dir, scenario_storage=SQLiteScenarioStorage(agent_record_dir)
        )
        try:
            scenario = dispatcher.get_or_create_scenario("upgrade")
            assert scenario is not None
            head = scenario.repository.resolve(scenario.current_artifact_ref()).local_path
            served = (head / "pi-agent/extensions/reef-requests.ts").read_text(encoding="utf-8")
            return list(scenario.releases()), served, list(scenario.trainer.state["entries"])
        finally:
            dispatcher.close()

    releases, served, _ = open_scenario()
    assert [row["operation"] for row in releases] == ["creation"] and served == EXTENSION

    extension.write_text(UPGRADED, encoding="utf-8")
    releases, served, entries = open_scenario()
    assert served == UPGRADED and entries[0]["config"]["code"] == UPGRADED
    # Newest first: the update is a served training commit on top of the creation release, held for no review.
    assert [(row["operation"], row["current"], row["pending"]) for row in releases] == [
        ("training", True, False),
        ("creation", False, False),
    ]
    assert releases[0]["metrics"] == {"shipped_content_update": {"entries": [REQUESTS_ENTRY_ID]}}

    releases_again, served, _ = open_scenario()
    assert served == UPGRADED and len(releases_again) == len(releases)
