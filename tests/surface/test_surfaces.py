from __future__ import annotations

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.sync import create_harness_surface
from reef.recipe.base import Recipe
from reef.surface import ComponentSurface, Surface, TextFileTree, WeightLoader, create_weight_surface
from reef.surface.base import AcceptAnyArtifact


def test_recipe_default_surface_is_noop() -> None:
    surface = Recipe().build_surface("s")
    assert type(surface) is Surface
    assert surface.loader is None
    assert surface.files is None


def test_surface_factories_return_composed_surface_instances() -> None:
    assert type(create_weight_surface(kind="model")) is Surface
    assert type(create_harness_surface(get_adapter("pi"))) is Surface


def test_each_built_in_surface_declares_the_kind_of_its_component() -> None:
    def kinds(surface: Surface) -> dict[str, str | None]:
        return {name: component.kind for name, component in surface.components.items()}

    assert kinds(create_weight_surface(kind="model")) == {"weights": "model"}
    assert kinds(create_weight_surface(kind="model_adapter")) == {"weights": "model_adapter"}
    assert kinds(create_harness_surface(get_adapter("pi"))) == {"harness": "harness_plugin"}
    assert kinds(Surface(loader=WeightLoader())) == {"release": None}, "a surface may declare no kind"
    with pytest.raises(ValueError, match="model or a model_adapter"):
        create_weight_surface(kind="harness_plugin")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="component kind"):
        ComponentSurface(kind="weights")  # type: ignore[arg-type]


def test_default_component_validator_accepts_anything() -> None:
    validator = ComponentSurface().validator
    assert isinstance(validator, AcceptAnyArtifact)
    validator.validate(object())  # type: ignore[arg-type]
    # A record-only surface has no components, so nothing to admit.
    Recipe().build_surface("s").validate(object())  # type: ignore[arg-type]


def test_text_file_tree_ignores_repository_bookkeeping(tmp_path) -> None:
    from reef.artifact import Artifact

    root = tmp_path / "artifact"
    root.mkdir()
    (root / "rules.md").write_text("rule", encoding="utf-8")
    (root / "reef-artifact.json").write_text("{}", encoding="utf-8")
    (root / ".gitattributes").write_text("* text", encoding="utf-8")
    assert TextFileTree().read_files(Artifact.local(root)) == {"rules.md": "rule"}


def test_a_harness_artifact_holds_its_entries_and_a_client_pull_leaves_them_out(tmp_path) -> None:
    from reef.artifact import Artifact, DirectoryFiles
    from reef.surface.files import ENTRIES_FILE, TextFileTree, entries_file, read_entries

    entries = ({"id": "rules", "name": "rules", "config": {"text": "be brief"}},)
    tree = tmp_path / "harness"
    tree.mkdir()
    (tree / "AGENTS.md").write_text("be brief\n", encoding="utf-8")
    for relative, text in entries_file(entries).items():
        (tree / relative).write_text(text, encoding="utf-8")

    assert read_entries(DirectoryFiles(tree)) == entries
    assert TextFileTree().read_files(Artifact.local(tree)) == {"AGENTS.md": "be brief\n"}
    (tree / ENTRIES_FILE).write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a list of objects"):
        read_entries(DirectoryFiles(tree))
    (tree / ENTRIES_FILE).unlink()
    with pytest.raises(ValueError, match="no readable"):
        read_entries(DirectoryFiles(tree))
