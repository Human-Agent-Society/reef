from __future__ import annotations

import pytest
from tests.support._strategies import resolve_episode_scorer, resolve_proposer

from reef.artifact import Artifact, ArtifactNotFound, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.inference.http import InferenceProxyRuntime
from reef.recipe import Recipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.service.app import RequestService
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import Surface

#: The files pi reads for a tree that holds only ``rules.md`` with the text ``rule``: its config files are empty.
PI_RULES_RENDER = {"pi-agent/AGENTS.md": "rule\n", "pi-agent/models.json": "{}\n", "pi-agent/settings.json": "{}\n"}
#: A pi harness evolution recipe: what its surface serves is the subject here, so it never builds a backend.
PI_RECIPE = HarnessEvolutionRecipe(
    resolve_proposer(lambda context: None), resolve_episode_scorer(lambda task, result: 0.0), ("t",)
)


class _HarnessRecipe(Recipe):
    """A pull-delivery recipe that serves the harness surface harness evolution builds."""

    def build_surface(self, scenario: str) -> Surface:
        return PI_RECIPE.build_surface(scenario)


def _service(tmp_path, *, recipe, rules: str | None) -> RequestService:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    if rules is not None:
        (bootstrap / "rules.md").write_text(rules, encoding="utf-8")
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(None),
    )
    dispatcher.get_or_create_scenario("delivery")
    return RequestService(dispatcher)


def test_the_manifest_serves_the_release_rendered_or_as_its_source_tree(tmp_path) -> None:
    service = _service(tmp_path, recipe=_HarnessRecipe(), rules="rule")
    rendered = service.harness_manifest({"x-reef-scenario": "delivery"})
    assert rendered["files"] == PI_RULES_RENDER
    assert rendered["release_id"]
    source = service.harness_manifest({"x-reef-scenario": "delivery"}, form="source")
    assert source["files"] == {"rules.md": "rule"} and source["release_id"] == rendered["release_id"]
    with pytest.raises(ValueError, match="form must be 'rendered' or 'source'"):
        service.harness_manifest({"x-reef-scenario": "delivery"}, form="tree")


def test_current_files_rejects_a_weights_scenario_without_materializing(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, recipe=Recipe(), rules="x")

    def fail_materialize(ref):
        raise AssertionError("weights artifact must not be materialized for a files read")

    monkeypatch.setattr(
        service.dispatcher.get_or_create_scenario("delivery").repository, "materialize", fail_materialize
    )
    with pytest.raises(ArtifactNotFound, match="the deployment's recipe carries no harness surface"):
        service.harness_manifest({"x-reef-scenario": "delivery"})


def test_current_files_never_creates_a_scenario(tmp_path) -> None:
    service = _service(tmp_path, recipe=_HarnessRecipe(), rules="x")
    with pytest.raises(ArtifactNotFound):
        service.harness_manifest({"x-reef-scenario": "unknown"})


def test_manifest_error_distinguishes_a_scenario_without_a_published_tree(tmp_path) -> None:
    # A scenario exists and its recipe has a harness surface, but its release holds no tree yet. The message must
    # tell the operator that nothing has been published, not that the surface is missing.
    service = _service(tmp_path, recipe=_HarnessRecipe(), rules=None)
    with pytest.raises(ArtifactNotFound, match="holds no harness tree"):
        service.harness_manifest({"x-reef-scenario": "delivery"})


def test_harness_surface_renders_the_source_tree_for_its_adapter(tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "rules.md").write_text("rule", encoding="utf-8")
    surface = PI_RECIPE.build_surface("s")
    assert surface.files is not None
    assert surface.files.read_files(Artifact.local(root)) == PI_RULES_RENDER
    empty = tmp_path / "empty"
    empty.mkdir()
    assert surface.files.read_files(Artifact.local(empty)) is None


def test_harness_surface_ignores_repository_bookkeeping(tmp_path) -> None:
    """Backend files at the tree root are neither served nor rendered.

    The git-lfs backend commits reef-artifact.json and .gitattributes into
    every version; treating them as harness content would fail every tree
    pulled from a durable repository, since neither is a path of the layout.
    """
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "rules.md").write_text("rule", encoding="utf-8")
    (root / "reef-artifact.json").write_text("{}", encoding="utf-8")
    (root / ".gitattributes").write_text("* text", encoding="utf-8")
    surface = PI_RECIPE.build_surface("s")
    assert surface.files is not None
    assert surface.files.read_files(Artifact.local(root)) == PI_RULES_RENDER


class ClaudeHarnessRecipe(_HarnessRecipe):
    @property
    def harness_adapter(self) -> str | None:
        return "claude"


def test_scenario_list_names_the_harness_adapter(tmp_path) -> None:
    (tmp_path / "harness").mkdir()
    (tmp_path / "weights").mkdir()
    harness = _service(tmp_path / "harness", recipe=ClaudeHarnessRecipe(), rules="x").dispatcher
    weights = _service(tmp_path / "weights", recipe=Recipe(), rules="x").dispatcher
    assert [(row["scenario"], row["adapter"]) for row in harness.list_scenarios()] == [("delivery", "claude")]
    assert all("adapter" not in row for row in weights.list_scenarios())
    # A registered scenario that is not in memory, as while it loads or after its preload failed, names it too.
    harness.get_or_create_scenario("waiting")
    harness.registry.remove("waiting")
    assert [(row["scenario"], row["loaded"], row["adapter"]) for row in harness.list_scenarios()] == [
        ("delivery", True, "claude"),
        ("waiting", False, "claude"),
    ]


def test_harness_recipes_name_their_adapter(tmp_path, monkeypatch) -> None:
    (tmp_path / "demo_adapter_surface.py").write_text(
        "from reef.evaluation.harness import EpisodeScorer\n"
        "from reef.train.harness_backend.contract import Proposer\n\n"
        "class Propose(Proposer):\n"
        "    def propose(self, context):\n        return None\n\n"
        "class Evaluate(EpisodeScorer):\n"
        "    def score(self, task, result):\n        return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = {
        "model": {"path": "small"},
        "evolution": {
            "propose": "demo_adapter_surface:Propose",
            "evaluate": "demo_adapter_surface:Evaluate",
            "tasks": ["t"],
            "adapter": "claude",
        },
    }
    built = HarnessEvolutionRecipe.from_environment(
        {}, config=config, runtime=InferenceProxyRuntime(model_path="small", base_url="http://up")
    )
    assert built.harness_adapter == "claude"
    assert Recipe().harness_adapter is None
