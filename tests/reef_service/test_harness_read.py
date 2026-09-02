from __future__ import annotations

import pytest

from reef.artifact import ArtifactNotFound, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.service.app import RequestService
from reef.surface import Surface, create_harness_surface


class _HarnessRecipe(Recipe):
    """A pull-delivery recipe, as harness evolution builds it."""

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface()


def _service(tmp_path, *, recipe, skill_text: str | None) -> RequestService:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    if skill_text is not None:
        (bootstrap / "skills").mkdir()
        (bootstrap / "skills" / "SKILL.md").write_text(skill_text, encoding="utf-8")
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=None,
    )
    dispatcher.get_or_create_scenario("delivery")
    return RequestService(dispatcher)


def test_current_files_returns_the_manifest(tmp_path) -> None:
    service = _service(tmp_path, recipe=_HarnessRecipe(), skill_text="Always check units.")
    manifest = service.harness_manifest({"x-reef-scenario": "delivery"})
    assert manifest["files"] == {"skills/SKILL.md": "Always check units."}
    assert manifest["release_id"]


def test_current_files_rejects_a_weights_scenario_without_materializing(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path, recipe=Recipe(), skill_text="x")

    def fail_materialize(ref):
        raise AssertionError("weights artifact must not be materialized for a files read")

    monkeypatch.setattr(
        service.dispatcher.get_or_create_scenario("delivery").repository, "materialize", fail_materialize
    )
    with pytest.raises(ArtifactNotFound, match="the deployment's recipe carries no harness surface"):
        service.harness_manifest({"x-reef-scenario": "delivery"})


def test_current_files_never_creates_a_scenario(tmp_path) -> None:
    service = _service(tmp_path, recipe=_HarnessRecipe(), skill_text="x")
    with pytest.raises(ArtifactNotFound):
        service.harness_manifest({"x-reef-scenario": "unknown"})


def test_manifest_error_distinguishes_a_scenario_without_a_published_composition(tmp_path) -> None:
    # A scenario exists and its recipe has a harness surface, but no files
    # have been published yet. The message must tell the operator that the
    # trainer has not published, not that the surface is missing.
    service = _service(tmp_path, recipe=_HarnessRecipe(), skill_text=None)
    with pytest.raises(ArtifactNotFound, match="no harness composition has been published yet"):
        service.harness_manifest({"x-reef-scenario": "delivery"})


def test_harness_surface_reads_the_file_tree(tmp_path) -> None:
    from reef.artifact import Artifact

    d = tmp_path / "artifact" / "skills"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("rule", encoding="utf-8")
    surface = create_harness_surface()
    assert surface.files is not None
    assert surface.files.read_files(Artifact.local(d.parent)) == {"skills/SKILL.md": "rule"}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert surface.files.read_files(Artifact.local(empty)) is None


def test_harness_surface_ignores_repository_bookkeeping(tmp_path) -> None:
    """Backend files at the tree root are neither served nor validated.

    The git-lfs backend commits reef-artifact.json and .gitattributes into
    every version; treating them as harness content would fail every tree
    pulled from a durable repository.
    """
    from reef.artifact import Artifact

    root = tmp_path / "artifact"
    (root / "skills").mkdir(parents=True)
    (root / "skills" / "SKILL.md").write_text("rule", encoding="utf-8")
    (root / "reef-artifact.json").write_text("{}", encoding="utf-8")
    (root / ".gitattributes").write_text("* text", encoding="utf-8")
    surface = create_harness_surface()
    assert surface.files is not None
    assert surface.files.read_files(Artifact.local(root)) == {"skills/SKILL.md": "rule"}
