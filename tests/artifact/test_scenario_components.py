from __future__ import annotations

import pytest

from reef.artifact import Artifact
from reef.artifact.components import ENTRIES_FILE, AcceptAnyArtifact, ComponentSpec, ScenarioComponents, TextFileTree
from reef.recipe.base import Recipe


def test_recipe_default_components_are_empty() -> None:
    components = Recipe().build_components("s")
    assert components.names == ()
    assert components.loaded_component is None
    assert components.files_component is None


def test_a_weight_kind_marks_the_component_the_runtime_loads() -> None:
    components = ScenarioComponents(
        {"notes": ComponentSpec(files=TextFileTree()), "weights": ComponentSpec(kind="model")}
    )
    assert components.loaded_component == "weights"
    assert components.files_component == "notes"
    assert ComponentSpec(kind="model_adapter").loaded
    assert not ComponentSpec(kind="harness_plugin").loaded
    assert not ComponentSpec().loaded


def test_a_component_refuses_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="component kind"):
        ComponentSpec(kind="weights")  # type: ignore[arg-type]


def test_default_component_validator_accepts_anything() -> None:
    validator = ComponentSpec().validator
    assert isinstance(validator, AcceptAnyArtifact)
    validator.validate(object())  # type: ignore[arg-type]
    # A record-only scenario has no components, so nothing to admit.
    Recipe().build_components("s").validate(object())  # type: ignore[arg-type]


def test_text_file_tree_ignores_repository_bookkeeping(tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "rules.md").write_text("rule", encoding="utf-8")
    (root / "reef-artifact.json").write_text("{}", encoding="utf-8")
    (root / ".gitattributes").write_text("* text", encoding="utf-8")
    assert TextFileTree().read_files(Artifact.local(root)) == {"rules.md": "rule"}


def test_a_client_pull_leaves_the_entries_file_of_a_harness_artifact_out(tmp_path) -> None:
    tree = tmp_path / "harness"
    tree.mkdir()
    (tree / "AGENTS.md").write_text("be brief\n", encoding="utf-8")
    (tree / ENTRIES_FILE).write_text('[{"id": "rules"}]', encoding="utf-8")

    assert TextFileTree().read_files(Artifact.local(tree)) == {"AGENTS.md": "be brief\n"}
