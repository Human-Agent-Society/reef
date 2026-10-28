"""Multi-component releases: the manifest, component views, per-component surfaces, and commits."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reef.artifact import Artifact, ArtifactNotFound, ArtifactPublicationError, InMemoryRepositoryBackend, Repository
from reef.artifact.artifact import ArtifactRef, ArtifactValidator
from reef.artifact.composite import compose_release
from reef.core import RequestType
from reef.core.activation import ActivationRule
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, ReleaseComponents, release_components
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate
from reef.recipe.base import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.service.request_service import RequestService
from reef.service.wire import parse_request_headers
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import (
    ArtifactLoader,
    ComponentSurface,
    InferenceHooks,
    InferenceLease,
    LeasingInferenceHooks,
    ServingRuntime,
    Surface,
    TextFileTree,
    WeightLoader,
)
from reef.surface.base import AcceptAnyArtifact
from reef.train import ComponentTrainer
from reef.train.types import TrainStepResult

WEIGHTS = "weights"
HARNESS = "harness"
#: The changed kinds of a step held for review: the test recipe's activation rule does not activate it.
HELD = frozenset({"held"})


def _tree(root: Path, files: Mapping[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return root


class _RecordingActivator(ArtifactLoader):
    """Record which component content each lifecycle call received."""

    def __init__(self) -> None:
        self.activated: list[str] = []

    def activate(self, artifact: Artifact, runtime: ServingRuntime | None, *, source: Artifact | None = None) -> str:
        self.activated.append(artifact.ref.content_id)
        return artifact.ref.release_id


class _RecordingValidator(ArtifactValidator):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def validate(self, artifact: Artifact) -> None:
        self.seen.append(artifact.ref.content_id)


@pytest.mark.unit
def test_manifest_derives_one_content_id_per_combination() -> None:
    single = ReleaseComponents({WEIGHTS: ComponentEntry("w1")})
    assert single.single
    assert single.content_id == "w1"
    assert single.relative_path(WEIGHTS) == Path(".")

    both = ReleaseComponents({WEIGHTS: ComponentEntry("w1"), HARNESS: ComponentEntry("h1", {"note": "seed"})})
    assert not both.single
    assert both.names == (WEIGHTS, HARNESS)
    assert both.content_id.startswith("composite:")
    reordered = ReleaseComponents({HARNESS: ComponentEntry("h1"), WEIGHTS: ComponentEntry("w1")})
    assert reordered.content_id == both.content_id
    assert both.relative_path(HARNESS) == Path(HARNESS)
    assert ReleaseComponents.from_dict(both.to_dict()) == both
    assert release_components({COMPONENTS_METADATA_KEY: both.to_dict()}) == both
    assert release_components(None) is None
    assert release_components({"runtime_load_id": "inc:1"}) is None

    with pytest.raises(KeyError):
        both.relative_path("config")
    with pytest.raises(ValueError, match="directory name"):
        ReleaseComponents({"../escape": ComponentEntry("x")})
    with pytest.raises(ValueError, match="at least one"):
        ReleaseComponents({})


@pytest.mark.unit
def test_component_views_of_a_composed_release(tmp_path: Path) -> None:
    weights = Artifact.local(
        _tree(tmp_path / "w", {"adapter_config.json": "{}"}), metadata={"runtime_load_id": "inc:3"}
    )
    harness = Artifact.local(_tree(tmp_path / "h", {"AGENTS.md": "rules"}))
    composed = compose_release({WEIGHTS: weights, HARNESS: harness}, directory=tmp_path / "release")

    assert composed.components is not None
    assert composed.components.names == (WEIGHTS, HARNESS)
    assert composed.ref.content_id.startswith("composite:")
    view = composed.component(WEIGHTS)
    assert view.local_path == tmp_path / "release" / WEIGHTS
    assert view.ref.content_id == weights.ref.content_id
    assert view.ref.release_id == composed.ref.release_id
    assert dict(view.metadata) == {"runtime_load_id": "inc:3"}
    assert TextFileTree().read_files(composed.component(HARNESS)) == {"AGENTS.md": "rules"}
    with pytest.raises(ArtifactNotFound):
        composed.component("config")

    # A release without a manifest is its own only component, whatever the caller names.
    assert harness.component(HARNESS) is harness
    with pytest.raises(ArtifactPublicationError, match="at least two"):
        compose_release({HARNESS: harness}, directory=tmp_path / "single")


@pytest.mark.unit
def test_materialized_releases_keep_their_manifest_and_composed_staging_carries_components(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    _tree(initial / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(initial / HARNESS, {"AGENTS.md": "seed"})
    backend = InMemoryRepositoryBackend("agent", initial, root=tmp_path / "repository")
    base = backend.resolve_release()
    manifest = ReleaseComponents({WEIGHTS: ComponentEntry("w0"), HARNESS: ComponentEntry("h0")})
    fork = backend.fork(base.release_id, metadata={COMPONENTS_METADATA_KEY: manifest.to_dict()})
    repository = Repository(
        backend, base, current_artifact=fork, checkpoint_artifact=fork, local_dir=tmp_path / "local"
    )

    materialized = repository.materialize(fork)
    assert materialized.components == manifest
    assert TextFileTree().read_files(materialized.component(HARNESS)) == {"AGENTS.md": "seed"}

    evolved = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "evolved"}))
    staged = repository.stage_composed(1, {WEIGHTS: materialized.component(WEIGHTS), HARNESS: evolved}, parent=fork)
    published = repository.publish(staged, expected_parent=fork, metadata=staged.metadata)
    again = repository.materialize(published)
    assert again.components is not None
    assert again.components.entries[WEIGHTS].content_id == "w0"
    assert again.components.entries[HARNESS].content_id == evolved.ref.content_id
    assert again.ref.content_id == staged.ref.content_id
    assert TextFileTree().read_files(again.component(HARNESS)) == {"AGENTS.md": "evolved"}
    assert TextFileTree().read_files(again.component(WEIGHTS)) == {"adapter_config.json": "{}"}


@pytest.mark.unit
def test_surface_routes_each_capability_to_its_component(tmp_path: Path) -> None:
    activator = _RecordingActivator()
    validator = _RecordingValidator()
    surface = Surface(
        components={
            WEIGHTS: ComponentSurface(validator=validator, loader=activator),
            HARNESS: ComponentSurface(files=TextFileTree()),
        }
    )
    assert surface.names == (WEIGHTS, HARNESS)
    assert not surface.single
    assert surface.loader_component == WEIGHTS
    assert surface.files_component == HARNESS
    assert surface.loader is activator

    weights = Artifact.local(_tree(tmp_path / "w", {"adapter_config.json": "{}"}))
    harness = Artifact.local(_tree(tmp_path / "h", {"AGENTS.md": "rules"}))
    first = compose_release({WEIGHTS: weights, HARNESS: harness}, directory=tmp_path / "r1")
    assert surface.files is not None
    assert surface.files.read_files(first) == {"AGENTS.md": "rules"}

    surface.validate(first)
    assert validator.seen == [weights.ref.content_id]
    surface.activate(first, None)
    assert activator.activated == [weights.ref.content_id]

    # A release that keeps the served weights does not activate them again.
    evolved = Artifact.local(_tree(tmp_path / "h2", {"AGENTS.md": "evolved"}))
    second = compose_release({WEIGHTS: weights, HARNESS: evolved}, directory=tmp_path / "r2")
    assert surface.component_changed(second, first, HARNESS)
    assert not surface.component_changed(second, first, WEIGHTS)
    surface.activate(second, None, previous=first)
    assert activator.activated == [weights.ref.content_id]
    # A release without a manifest counts as changed.
    assert surface.component_changed(second, harness, WEIGHTS)
    surface.activate(second, None, previous=harness)
    assert activator.activated == [weights.ref.content_id, weights.ref.content_id]


@pytest.mark.unit
def test_surface_rejects_ambiguous_component_sets() -> None:
    with pytest.raises(ValueError, match="at most one component loads"):
        Surface(
            components={"a": ComponentSurface(loader=WeightLoader()), "b": ComponentSurface(loader=WeightLoader())}
        )
    with pytest.raises(ValueError, match="file tree"):
        Surface(components={"a": ComponentSurface(files=TextFileTree()), "b": ComponentSurface(files=TextFileTree())})
    with pytest.raises(ValueError, match="directory name"):
        Surface(components={"a/b": ComponentSurface()})


class _TaggingHooks(InferenceHooks):
    """Append the component's file to the request so order is observable."""

    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]:
        files = TextFileTree().read_files(artifact) or {}
        return {**request, "seen": [*request.get("seen", []), *sorted(files)]}

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None:
        return None


@dataclass
class _Lease(InferenceLease):
    released: list[str]
    content_id: str

    def release(self) -> None:
        self.released.append(self.content_id)


class _LeasingHooks(LeasingInferenceHooks):
    def __init__(self) -> None:
        self.released: list[str] = []

    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]:
        return {**request, "seen": [*request.get("seen", []), "lease"]}

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None:
        return None

    def begin_request(self, artifact: Artifact, path: str) -> InferenceLease:
        return _Lease(self.released, artifact.ref.content_id)


@pytest.mark.unit
def test_surface_chains_inference_hooks_and_leases_in_component_order(tmp_path: Path) -> None:
    leasing = _LeasingHooks()
    surface = Surface(
        components={
            "skills": ComponentSurface(inference=_TaggingHooks()),
            WEIGHTS: ComponentSurface(inference=leasing),
        }
    )
    hooks = surface.inference
    assert isinstance(hooks, LeasingInferenceHooks)
    skills = Artifact.local(_tree(tmp_path / "s", {"SKILL.md": "text"}))
    weights = Artifact.local(_tree(tmp_path / "w", {"adapter_config.json": "{}"}))
    composed = compose_release({"skills": skills, WEIGHTS: weights}, directory=tmp_path / "release")

    prepared = hooks.prepare_request(composed, "/v1/chat/completions", {"messages": []})
    assert prepared["seen"] == ["SKILL.md", "lease"]
    hooks.begin_request(composed, "/v1/chat/completions").release()
    assert leasing.released == [weights.ref.content_id]

    # A flat surface hands out its component's hooks unchanged.
    assert Surface(components={WEIGHTS: ComponentSurface(inference=leasing)}).inference is leasing


class _TwoComponentRecipe(Recipe):
    """A record-only recipe whose scenario serves weights and a harness tree through one trainer."""

    activator = _RecordingActivator()

    def activation_rule(self) -> ActivationRule:
        return AlwaysActivate(HELD)

    def build_surface(self, scenario: str) -> Surface:
        return Surface(
            components={
                WEIGHTS: ComponentSurface(loader=self.activator),
                HARNESS: ComponentSurface(files=TextFileTree()),
            }
        )

    def build_trainers(
        self, scenario, records, *, surface, algorithm_states, experiment_logger=None, task_storage=None
    ):
        # One trainer, bound to the harness, publishes whichever component a step names.
        return (
            ComponentTrainer(
                HARNESS,
                self.build(
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(HARNESS),
                    experiment_logger=experiment_logger,
                ),
            ),
        )


@pytest.mark.unit
def test_inference_hands_the_handler_the_loaded_component(tmp_path: Path) -> None:
    """A handler that reads the tree must see the weights, not a release root of component directories."""
    initial = tmp_path / "initial"
    _tree(initial / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(initial / HARNESS, {"AGENTS.md": "seed"})
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        scenario.commit(TrainStepResult(state={}, artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "x"}))))
        service = RequestService(dispatcher)
        prepared = service._prepare_inference(
            parse_request_headers({"x-reef-scenario": "agent"}, RequestType.INFERENCE), _NullHandler(), None
        )
        release = prepared.artifact.materialize()
        assert prepared.served.ref.release_id == release.ref.release_id
        assert prepared.served.local_path == (release.local_path or Path()) / WEIGHTS
        assert TextFileTree().read_files(prepared.served) == {"adapter_config.json": "{}"}
    finally:
        dispatcher.close()


class _NullHandler(InferenceHandler):
    async def inference(self, artifact, path, payload):
        raise NotImplementedError

    async def inference_stream(self, artifact, path, payload):
        raise NotImplementedError


@pytest.mark.unit
def test_multi_component_scenario_commits_one_component_and_carries_the_rest(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    _tree(initial / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(initial / HARNESS, {"AGENTS.md": "seed"})
    recipe = _TwoComponentRecipe()
    activator = recipe.activator
    activator.activated.clear()
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        surface = scenario.surface
        base = scenario.repository.materialize(scenario.current_artifact_ref())
        assert base.components is not None
        assert base.components.names == (WEIGHTS, HARNESS)
        base_weights = base.components.entries[WEIGHTS].content_id
        assert activator.activated == [base_weights]
        assert surface.files is not None
        assert surface.files.read_files(base) == {"AGENTS.md": "seed"}

        # A harness step carries the weights forward and does not reload them.
        evolved = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "evolved"}))
        scenario.commit(TrainStepResult(state={}, artifact=evolved, component=HARNESS))
        after_harness = scenario.repository.materialize(scenario.current_artifact_ref())
        assert after_harness.components is not None
        assert after_harness.components.entries[WEIGHTS].content_id == base_weights
        assert after_harness.components.entries[HARNESS].content_id == evolved.ref.content_id
        assert surface.files.read_files(after_harness) == {"AGENTS.md": "evolved"}
        assert TextFileTree().read_files(after_harness.component(WEIGHTS)) == {"adapter_config.json": "{}"}
        assert activator.activated == [base_weights]

        # A weights step activates the new weights and keeps the evolved harness.
        trained = Artifact.local(
            _tree(tmp_path / "w1", {"adapter_config.json": '{"r": 8}'}), metadata={"runtime_load_id": "inc:1"}
        )
        scenario.commit(TrainStepResult(state={}, artifact=trained, component=WEIGHTS))
        after_weights = scenario.repository.materialize(scenario.current_artifact_ref())
        assert after_weights.components is not None
        assert after_weights.components.entries[WEIGHTS].content_id == trained.ref.content_id
        assert dict(after_weights.component(WEIGHTS).metadata) == {"runtime_load_id": "inc:1"}
        assert surface.files.read_files(after_weights) == {"AGENTS.md": "evolved"}
        assert activator.activated[0] == base_weights
        assert set(activator.activated[1:]) == {trained.ref.content_id}
        # The record names the harness trainer, but the tree did not change: no new harness head.
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        assert service.harness_head(headers) == after_harness.ref.release_id
        assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [
            base.ref.release_id,
            after_harness.ref.release_id,
        ]

        # A publication naming no component replaces the committing trainer's own; unknown names are refused.
        scenario.commit(TrainStepResult(state={}, artifact=Artifact.local(_tree(tmp_path / "h2", {"AGENTS.md": "x"}))))
        assert surface.files.read_files(scenario.repository.materialize(scenario.current_artifact_ref())) == {
            "AGENTS.md": "x"
        }
        with pytest.raises(ReefError, match="serves no component"):
            scenario.commit(TrainStepResult(state={}, artifact=evolved, component="config"))

        # Activating the harness-only step's harness binds it to the weights served now: no weights reload.
        served_before = len(activator.activated)
        tip = scenario.repository.checkpoint_artifact
        assert tip is not None
        activated = scenario.activate(f"{after_harness.ref.release_id}:{HARNESS}", HARNESS)
        assert (activated.artifact_id, activated.content_id) == (
            f"{after_harness.ref.release_id}:{HARNESS}",
            evolved.ref.content_id,
        )
        served = scenario.repository.materialize(scenario.current_artifact_ref())
        assert served.ref.release_id == activated.release_id
        assert served.components is not None
        assert served.components.entries[WEIGHTS].content_id == trained.ref.content_id
        assert surface.files.read_files(served) == {"AGENTS.md": "evolved"}
        assert activator.activated[served_before:] == []
        assert service.harness_head(headers) == after_harness.ref.release_id

        # Activating the base's weights loads them and keeps the served harness; the chain tip stays. The creation
        # carried the base artifact forward, so its artifact_id is the base release's.
        base_weights_id = base.components.artifact_ids(base.ref.release_id)[WEIGHTS]
        assert base_weights_id == f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}"
        trained_weights = scenario.artifact_status()[WEIGHTS]["training_start"]
        scenario.activate(base_weights_id, WEIGHTS)
        # The activation moves the weights' active artifact; their training start stays on the trained weights.
        assert scenario.artifact_status()[WEIGHTS] == {"active": base_weights_id, "training_start": trained_weights}
        assert trained_weights != base_weights_id
        served = scenario.repository.materialize(scenario.current_artifact_ref())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].content_id == base_weights
        assert surface.files.read_files(served) == {"AGENTS.md": "evolved"}
        assert activator.activated[served_before:] == [base_weights]
        assert scenario.repository.checkpoint_artifact == tip

        # A harness step held for review is made on the chain tip; activating its harness binds it to the weights
        # served now.
        held = Artifact.local(_tree(tmp_path / "h3", {"AGENTS.md": "held"}))
        scenario.commit(TrainStepResult(state={}, artifact=held, component=HARNESS, changed_kinds=HELD))
        assert surface.files.read_files(scenario.repository.materialize(scenario.current_artifact_ref())) == {
            "AGENTS.md": "evolved"
        }
        held_record = scenario.store.history()[-1]
        held_ref = held_record.artifact_ref
        assert held_ref.parent_release_id == tip.release_id
        # The held artifact is the chain tip, the parent of the next publication, but does not serve.
        assert scenario.repository.checkpoint_artifact == held_ref and not held_record.activated
        scenario.activate(f"{held_ref.release_id}:{HARNESS}", HARNESS)
        served = scenario.repository.materialize(scenario.current_artifact_ref())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].content_id == base_weights
        assert surface.files.read_files(served) == {"AGENTS.md": "held"}
        assert service.harness_head(headers) == held_ref.release_id

        # Weights held for review are activated as weights, under the tree served by then.
        held_weights = Artifact.local(
            _tree(tmp_path / "w3", {"adapter_config.json": '{"r": 32}'}), metadata={"runtime_load_id": "inc:3"}
        )
        scenario.commit(TrainStepResult(state={}, artifact=held_weights, component=WEIGHTS, changed_kinds=HELD))
        held_weights_release = scenario.store.history()[-1].artifact_ref.release_id
        newest = Artifact.local(_tree(tmp_path / "h4", {"AGENTS.md": "newest"}))
        scenario.commit(TrainStepResult(state={}, artifact=newest, component=HARNESS))
        newest_release = scenario.current_artifact_ref().release_id
        scenario.activate(f"{held_weights_release}:{WEIGHTS}", WEIGHTS)
        served = scenario.repository.materialize(scenario.current_artifact_ref())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].content_id == held_weights.ref.content_id
        assert surface.files.read_files(served) == {"AGENTS.md": "newest"}
        assert activator.activated[-1] == held_weights.ref.content_id
        # That activation changed no tree: the harness head and catalog stay at the newest harness step.
        assert service.harness_head(headers) == newest_release
        assert service.harness_manifest(headers)["release_id"] == newest_release
        catalog = service.harness_releases(headers)["releases"]
        assert catalog[-1]["release_id"] == newest_release
        assert held_weights_release not in {row["release_id"] for row in catalog}
        # The artifact listing marks each component's active artifact.
        active = {row["component"]: row["artifact_id"] for row in scenario.artifacts() if row["active"]}
        assert active == {WEIGHTS: f"{held_weights_release}:{WEIGHTS}", HARNESS: f"{newest_release}:{HARNESS}"}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_held_release_activates_without_the_manifest_of_its_parent(tmp_path: Path) -> None:
    """An activation reads the active release and the artifact's release only: the creation's bytes may be gone."""
    initial = tmp_path / "initial"
    _tree(initial / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(initial / HARNESS, {"AGENTS.md": "seed"})
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        surface = scenario.surface
        assert surface.files is not None
        creation = scenario.repository.materialize(scenario.current_artifact_ref())
        held = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "held"}))
        scenario.commit(TrainStepResult(state={}, artifact=held, component=HARNESS, changed_kinds=HELD))
        held_release = scenario.store.history()[-1].artifact_ref.release_id
        later = Artifact.local(
            _tree(tmp_path / "w1", {"adapter_config.json": '{"r": 8}'}), metadata={"runtime_load_id": "inc:1"}
        )
        scenario.commit(TrainStepResult(state={}, artifact=later, component=WEIGHTS))
        assert creation.local_path is not None
        shutil.rmtree(creation.local_path)
        scenario.activate(f"{held_release}:{HARNESS}", HARNESS)
        served = scenario.repository.materialize(scenario.current_artifact_ref())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].content_id == later.ref.content_id
        assert surface.files.read_files(served) == {"AGENTS.md": "held"}
        assert scenario.scenario_step == 3
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_activation_after_a_rejected_harness_step_adds_no_harness_release(tmp_path: Path) -> None:
    """The tree a rejected step served is the last one named; activating the served artifact changes nothing."""
    initial = tmp_path / "initial"
    _tree(initial / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(initial / HARNESS, {"AGENTS.md": "seed"})
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        scenario.commit(
            TrainStepResult(
                state={}, artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "one"})), component=HARNESS
            )
        )
        first_harness = scenario.current_artifact_ref().release_id
        weights = Artifact.local(
            _tree(tmp_path / "w1", {"adapter_config.json": '{"r": 8}'}), metadata={"runtime_load_id": "inc:1"}
        )
        scenario.commit(TrainStepResult(state={}, artifact=weights, component=WEIGHTS))
        scenario.commit(TrainStepResult(state={}, metrics={"selected": False}), component=HARNESS)
        steps = scenario.scenario_step
        # The first harness artifact serves already: activating it changes nothing.
        assert scenario.activate(f"{first_harness}:{HARNESS}", HARNESS).artifact_id == f"{first_harness}:{HARNESS}"
        assert scenario.scenario_step == steps
        rows, head = RequestService.harness_lineage(scenario)
        assert head == first_harness
        # Newest first: the rejected step's row (named by the first harness release), that release, the creation.
        assert [row["operation"] for row in rows] == ["training", "training", "creation"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_activation_refused_by_validation_changes_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    initial = tmp_path / "initial"
    _tree(initial / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(initial / HARNESS, {"AGENTS.md": "seed"})
    staged_root = tmp_path / "staged"
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=staged_root,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        held = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "held"}))
        scenario.commit(TrainStepResult(state={}, artifact=held, component=HARNESS, changed_kinds=HELD))
        held_release = scenario.store.history()[-1].artifact_ref.release_id
        served = scenario.current_artifact_ref()
        before = {path for path in staged_root.rglob("*") if path.is_dir()}

        def refuse(self: ArtifactValidator, artifact: Artifact) -> None:
            raise ReefError("refused by the validator")

        # An activation runs the release's own check, as its publication did.
        monkeypatch.setattr(AcceptAnyArtifact, "validate", refuse)
        with pytest.raises(ReefError, match="refused by the validator"):
            scenario.activate(f"{held_release}:{HARNESS}", HARNESS)
        assert {path for path in staged_root.rglob("*") if path.is_dir()} == before
        assert scenario.scenario_step == 1
        assert scenario.current_artifact_ref() == served
    finally:
        dispatcher.close()


def _seedless_release(tmp_path: Path) -> Artifact:
    release = tmp_path / "release"
    (release / "harness").mkdir(parents=True)
    (release / "harness" / "AGENTS.md").write_text("seed", encoding="utf-8")
    manifest = {
        "harness": {"content_id": "content:harness", "metadata": {}},
        "weights": {"content_id": "content:weights", "metadata": {}},
    }
    return Artifact(
        ArtifactRef("composite:x", "rel-1", None),
        None,
        local_path=release,
        metadata={COMPONENTS_METADATA_KEY: manifest},
    )


@pytest.mark.unit
def test_two_readers_of_a_seedless_component_both_get_its_directory(tmp_path: Path, monkeypatch: Any) -> None:
    """Two inference threads read one release after a harness commit: both find the weights leaf missing and
    the second must not fail on the directory the first made."""
    artifact = _seedless_release(tmp_path)
    barrier = threading.Barrier(2)
    original = Path.mkdir

    def paced(self: Path, *args: Any, **kwargs: Any) -> None:
        # Both threads are past the existence check when they get here.
        barrier.wait(timeout=5)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", paced)
    errors: list[BaseException] = []

    def read() -> None:
        try:
            artifact.component("weights")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert [repr(error) for error in errors] == []
    assert artifact.component("weights").local_path == tmp_path / "release" / "weights"
