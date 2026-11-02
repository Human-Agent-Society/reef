"""Multi-component releases: the manifest, component views, per-component surfaces, and commits."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime
from tests.support.serving import wait_for_serving

from reef.artifact import (
    Artifact,
    ArtifactNotFound,
    ArtifactPublicationError,
    DefaultStart,
    InMemoryRepositoryBackend,
    Repository,
)
from reef.artifact.artifact import ArtifactRef, ArtifactValidator
from reef.artifact.composite import compose_release
from reef.core.activation import ActivationRule
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, ReleaseComponents, release_components
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate
from reef.observability import NullExperimentTracker
from reef.recipe.base import Recipe
from reef.scenario.factory import ScenarioFactory
from reef.service.request_service import RequestService
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import ArtifactLoader, ComponentSurface, ServingRuntime, Surface, TextFileTree, WeightLoader
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
    root = tmp_path / "root"
    _tree(root / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(root / HARNESS, {"AGENTS.md": "seed"})
    backend = InMemoryRepositoryBackend("agent", directory=tmp_path / "repository")
    manifest = ReleaseComponents({WEIGHTS: ComponentEntry("w0"), HARNESS: ComponentEntry("h0")})
    root_ref = backend.register(
        Artifact.local(root, content_id=manifest.content_id), metadata={COMPONENTS_METADATA_KEY: manifest.to_dict()}
    )
    repository = Repository(
        backend, root_ref, current_artifact=root_ref, checkpoint_artifact=root_ref, local_dir=tmp_path / "local"
    )

    materialized = repository.materialize(root_ref)
    assert materialized.components == manifest
    assert TextFileTree().read_files(materialized.component(HARNESS)) == {"AGENTS.md": "seed"}

    evolved = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "evolved"}))
    staged = repository.stage_composed(
        1, {WEIGHTS: materialized.component(WEIGHTS), HARNESS: evolved}, parent=root_ref
    )
    published = repository.publish(staged, expected_parent=root_ref, metadata=staged.metadata)
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

    def default_start(self) -> DefaultStart:
        return DefaultStart(files={f"{WEIGHTS}/adapter_config.json": "{}", f"{HARNESS}/AGENTS.md": "seed"})

    def build_trainers(
        self,
        scenario,
        records,
        *,
        surface,
        algorithm_states,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
        root=None,
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
                    root=root,
                ),
            ),
        )


@pytest.mark.unit
def test_multi_component_scenario_commits_one_component_and_carries_the_rest(tmp_path: Path) -> None:
    recipe = _TwoComponentRecipe(runtime=StubInferenceRuntime())
    activator = recipe.activator
    activator.activated.clear()
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        surface = scenario.surface
        base = scenario.repository.materialize(scenario.current_artifact_ref())
        assert base.components is not None
        assert base.components.names == (WEIGHTS, HARNESS)
        base_weights = base.components.entries[WEIGHTS].content_id
        wait_for_serving(dispatcher, "agent")
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
        wait_for_serving(dispatcher, "agent")
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
        wait_for_serving(dispatcher, "agent")
        assert activator.activated[0] == base_weights
        assert set(activator.activated[1:]) == {trained.ref.content_id}
        # The record names the harness trainer, but the tree did not change: no new harness head.
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        assert service.harness_release_id(scenario) == after_harness.ref.release_id
        assert [row["artifact_id"] for row in service.harness_artifacts(headers)["artifacts"]] == [
            base.component(HARNESS).artifact_id,
            after_harness.component(HARNESS).artifact_id,
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
        wait_for_serving(dispatcher, "agent")
        assert activator.activated[served_before:] == []
        assert service.harness_release_id(scenario) == after_harness.ref.release_id

        # Activating the root's weights loads them and keeps the served harness; the chain tip stays. The root
        # release made its artifacts, so their artifact_id is the root release's.
        base_weights_id = base.components.artifact_ids(base.ref.release_id)[WEIGHTS]
        assert base_weights_id == f"{scenario.repository.start_artifact.release_id}:{WEIGHTS}"
        trained_weights = scenario.artifact_status()[WEIGHTS]["training_start"]
        scenario.activate(base_weights_id, WEIGHTS)
        # The activation moves the weights' active artifact; their training start stays on the trained weights.
        assert scenario.artifact_status()[WEIGHTS] == {"active": base_weights_id, "training_start": trained_weights}
        assert trained_weights != base_weights_id
        served = scenario.repository.materialize(scenario.current_artifact_ref())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].content_id == base_weights
        assert surface.files.read_files(served) == {"AGENTS.md": "evolved"}
        wait_for_serving(dispatcher, "agent")
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
        assert service.harness_release_id(scenario) == held_ref.release_id

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
        wait_for_serving(dispatcher, "agent")
        assert activator.activated[-1] == held_weights.ref.content_id
        # That activation changed no tree: the harness head and catalog stay at the newest harness step.
        assert service.harness_release_id(scenario) == newest_release
        assert service.harness_manifest(headers)["artifact_id"] == f"{newest_release}:{HARNESS}"
        catalog = service.harness_artifacts(headers)["artifacts"]
        assert catalog[-1]["artifact_id"] == f"{newest_release}:{HARNESS}"
        assert f"{held_weights_release}:{HARNESS}" not in {row["artifact_id"] for row in catalog}
        # The artifact listing marks each component's active artifact.
        active = {row["component"]: row["artifact_id"] for row in scenario.artifacts() if row["active"]}
        assert active == {WEIGHTS: f"{held_weights_release}:{WEIGHTS}", HARNESS: f"{newest_release}:{HARNESS}"}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_selections_restore_and_recover_without_publishing_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    repositories = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    with closing(SQLiteScenarioStorage(tmp_path / "store")) as storage:
        factory = ScenarioFactory(
            _TwoComponentRecipe(),
            repositories,
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
            activates_runtime=False,
        )
        with closing(factory.load_or_create("agent")) as scenario:
            initial = {name: state["active"] for name, state in scenario.artifact_status().items()}
            scenario.commit(
                TrainStepResult(
                    state={}, artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "one"})), component=HARNESS
                )
            )
            after_first = {name: state["active"] for name, state in scenario.artifact_status().items()}
            scenario.commit(
                TrainStepResult(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "w1", {"adapter_config.json": '{"r":8}'})),
                    component=WEIGHTS,
                )
            )
            weights_id = scenario.artifact_status()[WEIGHTS]["active"]
            scenario.commit(
                TrainStepResult(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "h2", {"AGENTS.md": "held"})),
                    component=HARNESS,
                    changed_kinds=HELD,
                )
            )
            starts = {name: state["training_start"] for name, state in scenario.artifact_status().items()}
            checkpoint = scenario.repository.require_checkpoint_artifact()
            artifacts = {row["artifact_id"] for row in scenario.artifacts()}

            def refuse_publication(*args: Any, **kwargs: Any) -> None:
                pytest.fail("selecting existing artifacts must not publish storage")

            publish = scenario.repository.backend.publish
            monkeypatch.setattr(scenario.repository.backend, "publish", refuse_publication)
            scenario.activate(initial[HARNESS], HARNESS)
            assert scenario.artifact_status()[WEIGHTS]["active"] == weights_id
            scenario.serve(after_first[HARNESS], HARNESS)
            assert scenario.restore(1) == after_first
            assert {name: state["training_start"] for name, state in scenario.artifact_status().items()} == starts
            assert {row["artifact_id"] for row in scenario.artifacts()} == artifacts
            assert scenario.repository.require_checkpoint_artifact() == checkpoint
            scenario.commit(TrainStepResult(state={}))
            active = scenario.current_artifact_ref()
            assert active.release_id.startswith("selection:")

        # The next process has no materialized views. Only operation records and original artifact bytes survive.
        shutil.rmtree(tmp_path / "staged")
        with closing(factory.load_or_create("agent")) as recovered:
            assert recovered.current_artifact_ref() == active
            assert recovered.repository.require_checkpoint_artifact() == checkpoint
            assert {name: state["active"] for name, state in recovered.artifact_status().items()} == after_first
            assert {name: state["training_start"] for name, state in recovered.artifact_status().items()} == starts
            served = recovered.served_beside()[after_first[HARNESS]]
            beside = recovered.repository.resolve(served.release)
            assert beside.component(WEIGHTS).artifact_id == weights_id
            assert beside.component(HARNESS).artifact_id == after_first[HARNESS]
            assert [record.operation for record in recovered.store.history()][-4:] == [
                "activate",
                "serve",
                "restore",
                "training",
            ]
            # A later held checkpoint stores the current selection independently of the operation history.
            monkeypatch.setattr(recovered.repository.backend, "publish", publish)
            recovered.commit(
                TrainStepResult(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "h3", {"AGENTS.md": "also held"})),
                    component=HARNESS,
                    changed_kinds=HELD,
                )
            )
        shutil.rmtree(tmp_path / "staged")
        with closing(SQLiteScenarioStorage(tmp_path / "empty-store")) as empty_storage:
            checkpoint_factory = ScenarioFactory(
                _TwoComponentRecipe(),
                repositories,
                scenario_storage=empty_storage,
                experiment_tracker=NullExperimentTracker(),
                activates_runtime=False,
            )
            with closing(checkpoint_factory.load_or_create("agent")) as recovered:
                assert recovered.current_artifact_ref() == active
                assert {name: state["active"] for name, state in recovered.artifact_status().items()} == after_first
                view = recovered.repository.resolve(active)
                assert view.component(WEIGHTS).artifact_id == after_first[WEIGHTS]
                assert view.component(HARNESS).artifact_id == after_first[HARNESS]
                with pytest.raises(ValueError, match="earlier operation history is unavailable"):
                    recovered.restore(recovered.scenario_step)
                # The creation remains a known selection even when the intervening operations were lost.
                assert recovered.restore(0) == initial


@pytest.mark.unit
def test_concurrent_first_readers_share_one_selection_view(tmp_path: Path, monkeypatch: Any) -> None:
    root = compose_release(
        {
            WEIGHTS: Artifact.local(_tree(tmp_path / "weights", {"adapter_config.json": "{}"})),
            HARNESS: Artifact.local(_tree(tmp_path / "harness", {"AGENTS.md": "seed"})),
        },
        directory=tmp_path / "composed",
    )
    backend = InMemoryRepositoryBackend("agent", directory=tmp_path / "repository")
    initial = backend.register(root, metadata=root.metadata)
    repository = Repository(backend, initial, current_artifact=initial)
    selection = repository.selection(repository.resolve(initial), (WEIGHTS, HARNESS))
    reference = repository.select(selection).ref
    recovered = Repository(backend, initial, current_artifact=initial)
    recovered.remember_selection(reference, selection)
    entered, duplicate, resume, second_started = (threading.Event() for _ in range(4))

    def compose_once(components: Mapping[str, Artifact], *, directory: Path) -> Artifact:
        if entered.is_set():
            duplicate.set()
        entered.set()
        if not resume.wait(timeout=5):
            raise TimeoutError("test did not release composition")
        return compose_release(components, directory=directory)

    def second_reader() -> Artifact:
        second_started.set()
        return recovered.resolve(reference)

    monkeypatch.setattr("reef.artifact.repository.compose_release", compose_once)
    with ThreadPoolExecutor(max_workers=2) as readers:
        first = readers.submit(recovered.resolve, reference)
        try:
            assert entered.wait(timeout=5)
            second = readers.submit(second_reader)
            assert second_started.wait(timeout=5)
            assert not duplicate.wait(timeout=0.05)
            # Head compare-and-swap is independent of a reader's blocked filesystem work.
            recovered.activate(initial, expected=initial)
        finally:
            resume.set()
        first_view, second_view = first.result(timeout=5), second.result(timeout=5)
    assert first_view is second_view
    assert first_view.component(HARNESS).local_path.joinpath("AGENTS.md").read_text() == "seed"


@pytest.mark.unit
def test_restoring_a_selection_does_not_readmit_unchanged_component_artifacts(tmp_path: Path) -> None:
    class TrainedWeights(ArtifactValidator):
        """A training admission check that refuses the scenario's default seed."""

        def validate(self, artifact: Artifact) -> None:
            if artifact.materialize().local_path.joinpath("adapter_config.json").read_text() == "{}":
                raise ReefError("default weights are not a trained candidate")

    class RestoreRecipe(_TwoComponentRecipe):
        """A harness trainer whose unchanged default weights do not pass training admission."""

        def build_surface(self, scenario: str) -> Surface:
            return Surface(
                components={
                    WEIGHTS: ComponentSurface(validator=TrainedWeights()),
                    HARNESS: ComponentSurface(files=TextFileTree()),
                }
            )

    with closing(SQLiteScenarioStorage(tmp_path / "store")) as storage:
        factory = ScenarioFactory(
            RestoreRecipe(),
            InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
            activates_runtime=False,
        )
        with closing(factory.load_or_create("agent")) as scenario:
            for step in (1, 2):
                scenario.commit(
                    TrainStepResult(
                        state={},
                        component=HARNESS,
                        artifact=Artifact.local(_tree(tmp_path / f"h{step}", {"AGENTS.md": f"harness {step}"})),
                    )
                )
            selected = scenario.restore(1)
            view = scenario.repository.resolve(scenario.current_artifact_ref())
            assert view.component(HARNESS).local_path.joinpath("AGENTS.md").read_text() == "harness 1"
            assert selected[WEIGHTS] == scenario.artifact_status()[WEIGHTS]["active"]


@pytest.mark.unit
def test_selection_recovers_after_record_append_before_head_movement(tmp_path: Path, monkeypatch: Any) -> None:
    repositories = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    with closing(SQLiteScenarioStorage(tmp_path / "store")) as storage:
        factory = ScenarioFactory(
            _TwoComponentRecipe(),
            repositories,
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
            activates_runtime=False,
        )
        with closing(factory.load_or_create("agent")) as scenario:
            scenario.commit(
                TrainStepResult(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "held"})),
                    component=HARNESS,
                    changed_kinds=HELD,
                )
            )
            checkpoint = scenario.repository.require_checkpoint_artifact()
            selected_id = f"{checkpoint.release_id}:{HARNESS}"
            previous = scenario.current_artifact_ref()

            def interrupted(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("process stopped after append")

            monkeypatch.setattr(scenario.repository, "activate", interrupted)
            with pytest.raises(RuntimeError, match="process stopped"):
                scenario.activate(selected_id, HARNESS)
            assert scenario.current_artifact_ref() == previous
            assert scenario.store.history()[-1].selection is not None
        with closing(factory.load_or_create("agent")) as recovered:
            assert recovered.scenario_step == 2
            assert recovered.artifact_status()[HARNESS]["active"] == selected_id
            assert recovered.repository.require_checkpoint_artifact() == checkpoint
            assert (
                recovered.repository.resolve(recovered.current_artifact_ref()).component(HARNESS).artifact_id
                == selected_id
            )


@pytest.mark.unit
def test_a_held_release_activates_without_the_manifest_of_its_parent(tmp_path: Path) -> None:
    """An activation reads the active release and the artifact's release only: the creation's bytes may be gone."""
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
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
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
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
    staged_root = tmp_path / "staged"
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=staged_root,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
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
