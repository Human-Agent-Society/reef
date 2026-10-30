from __future__ import annotations

import pytest

from reef.core.errors import ReefError
from reef.recipe.base import Recipe
from reef.surface import (
    AdapterWeightRuntime,
    ComponentSurface,
    SkillLayer,
    Surface,
    WeightLoader,
    WeightRuntime,
    create_harness_surface,
    create_skill_surface,
    create_weight_surface,
)
from reef.surface.base import AcceptAnyArtifact
from reef.surface.skills import SkillValidator
from reef.surface.weights import reported_runtime_load_id, reported_runtime_load_spans


class _PullOnlyModule(SkillLayer):
    """Test stand-in for a pull-only layer module: never transforms."""

    layer = "skills"


def test_recipe_default_surface_is_noop() -> None:
    surface = Recipe().build_surface("s")
    assert type(surface) is Surface
    assert surface.loader is None
    assert surface.files is None


def test_surface_factories_return_composed_surface_instances() -> None:
    assert type(create_weight_surface(kind="model")) is Surface
    assert type(create_harness_surface()) is Surface
    assert type(create_skill_surface([_PullOnlyModule()])) is Surface


def test_each_built_in_surface_declares_the_kind_of_its_component() -> None:
    def kinds(surface: Surface) -> dict[str, str | None]:
        return {name: component.kind for name, component in surface.components.items()}

    assert kinds(create_weight_surface(kind="model")) == {"weights": "model"}
    assert kinds(create_weight_surface(kind="model_adapter")) == {"weights": "model_adapter"}
    assert kinds(create_harness_surface()) == {"harness": "harness_plugin"}
    assert kinds(create_skill_surface([_PullOnlyModule()])) == {"skills": "harness_plugin"}
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


def _skill_artifact(tmp_path, text: str = "Always check units."):
    from reef.artifact import Artifact

    d = tmp_path / "skill-artifact" / "skills"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return Artifact.local(d.parent)


def test_skill_validator_rejects_empty_and_stray_files(tmp_path) -> None:
    from reef.artifact import Artifact

    validator = SkillValidator((_PullOnlyModule(),))
    d = tmp_path / "bad-artifact"
    d.mkdir()
    with pytest.raises(ReefError):
        validator.validate(Artifact.local(d))
    (d / "skills").mkdir()
    (d / "skills" / "SKILL.md").write_text("  ", encoding="utf-8")
    with pytest.raises(ReefError):
        validator.validate(Artifact.local(d))
    (d / "skills" / "SKILL.md").write_text("Real content.", encoding="utf-8")
    validator.validate(Artifact.local(d))
    (d / "stray.md").write_text("outside every layer", encoding="utf-8")
    with pytest.raises(ReefError):
        validator.validate(Artifact.local(d))


def test_skill_validator_ignores_repository_bookkeeping(tmp_path) -> None:
    from reef.artifact import Artifact

    root = tmp_path / "artifact"
    (root / "skills").mkdir(parents=True)
    (root / "skills" / "SKILL.md").write_text("rule", encoding="utf-8")
    (root / "reef-artifact.json").write_text("{}", encoding="utf-8")
    (root / ".gitattributes").write_text("* text", encoding="utf-8")
    layer = _PullOnlyModule()
    surface = create_skill_surface([layer])
    assert surface.files is not None
    assert surface.files.read_files(Artifact.local(root)) == {"skills/SKILL.md": "rule"}
    SkillValidator((layer,)).validate(Artifact.local(root))


def test_the_reported_version_is_read_from_every_response_shape() -> None:
    assert reported_runtime_load_id({"metadata": {"runtime_load_id": "wv-7"}}) == "wv-7"
    # Native /generate shape: top-level meta_info.
    assert reported_runtime_load_id({"meta_info": {"runtime_load_id": "wv-7"}}) == "wv-7"
    # OpenAI chat shape: per-choice meta_info.
    assert reported_runtime_load_id({"choices": [{"meta_info": {"runtime_load_id": "wv-7"}}]}) == "wv-7"
    assert reported_runtime_load_id({"choices": [{"message": {"content": "hi"}}]}) is None


def test_mixed_token_versions_from_an_in_place_update_are_valid_spans() -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": "engine:7"}}],
        "training": {
            "response_length": 3,
            "runtime_load_id": None,
            "runtime_load_spans": [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 3, "runtime_load_id": "engine:7"},
            ],
        },
    }

    assert [span.runtime_load_id for span in reported_runtime_load_spans(response)] == ["engine:6", "engine:7"]


def test_the_reported_version_of_an_anthropic_response_is_its_final_training_span() -> None:
    response = {
        "type": "message",
        "content": [{"type": "text", "text": "done"}],
        "training": {
            "response_length": 2,
            "runtime_load_id": None,
            "runtime_load_spans": [
                {"start": 0, "end": 1, "runtime_load_id": "engine:6"},
                {"start": 1, "end": 2, "runtime_load_id": "engine:7"},
            ],
        },
    }

    assert reported_runtime_load_id(response) == "engine:7"


@pytest.mark.parametrize(
    "runtime_load_ids",
    [
        ("engine:6", "engine:7", "engine:6"),
        ("engine:7", "engine:6"),
        ("engine:6", "replacement:7"),
    ],
)
def test_impossible_runtime_load_histories_are_rejected(runtime_load_ids) -> None:
    response = {
        "choices": [{"meta_info": {"runtime_load_id": runtime_load_ids[-1]}}],
        "training": {
            "response_length": len(runtime_load_ids),
            "runtime_load_spans": [
                {"start": index, "end": index + 1, "runtime_load_id": runtime_load_id}
                for index, runtime_load_id in enumerate(runtime_load_ids)
            ],
        },
    }

    with pytest.raises(ValueError, match=r"advance monotonically|return to an earlier"):
        reported_runtime_load_spans(response)


class _StubTrainingRuntime(WeightRuntime):
    """Satisfies WeightRuntime; only the version probe does anything."""

    def __init__(self, served: str | None, published: str | None = None):
        self._served = served
        self._published = published

    @property
    def base_url(self) -> str:
        return "http://engine.invalid"

    def current_runtime_load_id(self) -> str | None:
        return self._published

    def activate_checkpoint(self, artifact, *, scenario=None) -> str:
        return artifact.ref.release_id


class _StubAdapterRuntime(_StubTrainingRuntime, AdapterWeightRuntime):
    """Serves one adapter version per scenario, from ``adapters``."""

    def __init__(self, served: str | None, adapters: dict[str, str]):
        super().__init__(served)
        self._adapters = adapters

    def serving_adapter_runtime_load_id(self, scenario):
        return self._adapters.get(scenario)


def test_a_durable_release_serves_under_the_version_that_reef_acknowledged(tmp_path) -> None:
    from reef.artifact import Artifact

    # A commit, a rollback or a restart loaded the release under a version that the runtime names.
    release = Artifact.local(tmp_path)
    runtime = _StubTrainingRuntime("e:6", published="e:5")
    assert create_weight_surface(kind="model").served_runtime_load_id(release, runtime) == "e:5"
    # A runtime that serves no weights leaves the served version unknown.
    assert create_weight_surface(kind="model").served_runtime_load_id(release, None) is None
    # A per-scenario adapter serves a durable release under the version the engine holds for the scenario.
    adapters = _StubAdapterRuntime("e:6", {"math": "e:4"})
    surface = create_weight_surface(scenario="math", runtime=adapters, kind="model_adapter")
    assert surface.served_runtime_load_id(release, adapters) == "e:4"
    # An engine that holds no adapter for the scenario leaves the served version unknown.
    assert surface.served_runtime_load_id(release, _StubAdapterRuntime("e:6", {})) is None


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
