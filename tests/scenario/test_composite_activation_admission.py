"""Activation, new training starts and inference admission in a composite whose weights load into a runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support.component_trainers import HARNESS, WEIGHTS, ComponentJobs, TextHarnessRecipe, WeightsJobComponent
from tests.support.component_trainers import _dispatcher as two_trainer_dispatcher
from tests.support.component_trainers import _records
from tests.support.component_trainers import _scenario as two_trainer_scenario
from tests.support.component_trainers import (
    checked_commit,
    component_files,
    component_id,
    finish_job,
    seeded_start,
    two_trainer_recipe,
    wait_for_job_check,
)
from tests.support.runtime_stubs import StubInferenceRuntime
from tests.support.samples_service import make_samples
from tests.support.serving import wait_for_serving

from reef.artifact.components import ArtifactLayout, ComponentSpec, ScenarioComponents, TextFileTree
from reef.artifact.files import ArtifactFiles
from reef.artifact.sources import REFERENCE_FILE, DirectorySource, HuggingFaceRevision
from reef.core.errors import CommitPendingError, ReefError, TrainingStartRefusedError
from reef.dispatcher import Dispatcher
from reef.dispatcher.errors import local_error_source
from reef.evaluation.harness import EvaluationSettings
from reef.execution.workers import SERVICE_COMPONENTS
from reef.recipe import CompositeRecipe
from reef.scenario import Scenario
from reef.train.harness_backend.settings import EvolutionSettings


class RefusingImports(WeightsJobComponent):
    """A training component that cannot start a job from the weights of a published snapshot."""

    def check_import_start(self, artifact_dir: Path) -> None:
        raise ValueError("cannot train from these weights")


def _dispatcher(
    tmp_path: Path, recipe: CompositeRecipe | None = None, backends: Mapping[str, ComponentJobs] | None = None
) -> Dispatcher:
    """A dispatcher over the two trainer recipe, or over ``recipe``, whose weights load into a stub runtime; the
    inference worker serves each selection. ``backends`` replace the components' ``ComponentJobs``."""
    dispatcher, _ = two_trainer_dispatcher(
        tmp_path,
        recipe=recipe,
        backends=backends,
        # Loading a scenario wakes its workers: held, they leave every step to the test.
        hold_local_cycles=True,
        # No samples worker: the test makes the samples and commits each step itself.
        components=frozenset(SERVICE_COMPONENTS) - {"samples"},
    )
    return dispatcher


def _runtime(dispatcher: Dispatcher) -> StubInferenceRuntime:
    """The stub runtime of the dispatcher's deployment."""
    runtime = dispatcher.live.runtime
    assert isinstance(runtime, StubInferenceRuntime)
    return runtime


def _scenario(dispatcher: Dispatcher) -> Scenario:
    """Scenario ``agent``, created, once its start artifact serves: an automatic job waits for it."""
    scenario = two_trainer_scenario(dispatcher)
    wait_for_serving(dispatcher, "agent")
    return scenario


def _step(scenario: Scenario, component: str, step: int) -> str:
    for record in _records(step, scenario.serving_artifact_id()):
        scenario.records.append(record)
    make_samples(scenario)
    result = checked_commit(scenario, component)
    assert result is not None
    scenario.commit(result, component=component)
    finish_job(scenario, component)
    return scenario.current_snapshot().revision


def _job_step(scenario, component: str):
    """The next prepared step of ``component``'s job loop; the training worker runs each job on its own thread."""
    deadline = time.monotonic() + 10
    while (result := scenario.next_commit(component)) is None:
        assert time.monotonic() < deadline, "the training job did not move"
        time.sleep(0.01)
    return result


@pytest.mark.unit
def test_a_local_cycle_clears_only_the_errors_it_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The weights worker's failure stays in the status while a harness cycle runs after the wake."""
    dispatcher = _dispatcher(tmp_path)
    # The test runs each cycle itself; a wake starts no competing worker.
    monkeypatch.setattr(dispatcher.local_workers, "start", lambda scenario, component=None: None)
    dispatcher.local_workers.open()
    try:
        scenario = _scenario(dispatcher)
        for record in _records(1, scenario.serving_artifact_id()):
            scenario.records.append(record)
        make_samples(scenario)
        # The training service checks the harness job first; the cycle below then commits the job's submission.
        wait_for_job_check(scenario, HARNESS)
        dispatcher.errors.record("agent", "RuntimeError: training job away", local_error_source(WEIGHTS))
        assert dispatcher.local_workers.process_step("agent", HARNESS) is True
        assert dispatcher.training_status.training_errors() == ["agent: RuntimeError: training job away"]
        dispatcher.errors.record("agent", "harness away", local_error_source(HARNESS))
        assert sorted(dispatcher.training_status.training_errors()) == [
            "agent: RuntimeError: training job away",
            "agent: harness away",
        ]
        finish_job(scenario, HARNESS)
        for record in _records(2, scenario.serving_artifact_id()):
            scenario.records.append(record)
        make_samples(scenario)
        wait_for_job_check(scenario, HARNESS)
        # The harness cycle clears its own error; the weights worker's stays until that worker clears it.
        assert dispatcher.local_workers.process_step("agent", HARNESS) is True
        assert dispatcher.training_status.training_errors() == ["agent: RuntimeError: training job away"]
        dispatcher.errors.clear("agent", local_error_source(WEIGHTS))
        assert dispatcher.training_status.training_errors() == []
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_activating_a_held_harness_artifact_leaves_held_admission_closed(tmp_path: Path) -> None:
    """The activation binds the held harness to the weights that serve now, so it loads nothing."""
    harness = ComponentJobs(HARNESS, tmp_path / "candidates")
    dispatcher = _dispatcher(tmp_path, backends={HARNESS: harness})
    runtime = _runtime(dispatcher)
    try:
        scenario = _scenario(dispatcher)
        harness.hold_next = True
        _step(scenario, HARNESS, 1)
        held = next(row for row in scenario.artifacts() if row["operation"] == "training")
        assert held["component"] == HARNESS and not held["active"]
        runtime.close_admission()
        runtime.served = "before"
        activated = scenario.activate(held["artifact_id"], HARNESS)
        assert activated.artifact_id == held["artifact_id"]
        # The activation records a selection of existing artifacts and materializes it locally.
        assert scenario.store.history()[-1].selection is not None
        assert not scenario.store.history()[-1].made_artifact
        assert component_files(scenario, scenario.current_snapshot()) == {
            WEIGHTS: "weights seed",
            HARNESS: "harness step 1",
        }
        # The inference worker serves the new selection: the weights that serve stay loaded.
        wait_for_serving(dispatcher, "agent")
        assert not runtime.inference_admission_status["open"]
        assert runtime.served == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_harness_only_activation_leaves_held_admission_closed(tmp_path: Path) -> None:
    """An activation that changes no loaded weights neither resumes admission nor calls a version served."""
    dispatcher = _dispatcher(tmp_path)
    runtime = _runtime(dispatcher)
    try:
        scenario = _scenario(dispatcher)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        # Another publication holds admission: published, not yet committed.
        runtime.close_admission()
        runtime.served = "before"
        scenario.activate(component_id(scenario, first, HARNESS), HARNESS)
        assert component_files(scenario, scenario.current_snapshot())[HARNESS] == "harness step 1"
        # The inference worker serves the new selection: the weights that serve stay loaded.
        wait_for_serving(dispatcher, "agent")
        assert not runtime.inference_admission_status["open"]
        assert runtime.served == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_failed_activation_record_leaves_the_active_artifact_and_a_retry_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights load first; an append that fails leaves the log, and so the active artifact, as they were."""
    dispatcher = _dispatcher(tmp_path)
    runtime = _runtime(dispatcher)
    try:
        scenario = _scenario(dispatcher)
        trained = _step(scenario, WEIGHTS, 1)
        append_commit = scenario.store.append_commit

        def store_away(**keywords):
            raise OSError("commit storage away")

        monkeypatch.setattr(scenario.store, "append_commit", store_away)
        with pytest.raises(OSError, match="commit storage away"):
            scenario.activate(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
        assert scenario.current_snapshot().revision == trained
        assert scenario.store.history()[-1].operation == "training"
        monkeypatch.setattr(scenario.store, "append_commit", append_commit)

        activated = scenario.activate(
            component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS
        )
        assert scenario.artifact_status()[WEIGHTS]["active"] == activated.artifact_id
        assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights seed"
        record = scenario.store.history()[-1]
        assert (record.operation, record.component, record.checkpoint) == ("activate", WEIGHTS, False)
        assert record.components is not None and WEIGHTS in record.components
        assert runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_weights_start_the_training_component_refuses_is_refused_before_anything_changes(tmp_path: Path) -> None:
    """The training component imports the snapshot as the next job's start; one it cannot train from refuses the
    new start before any record lands, and serving stays as it was."""
    dispatcher, _ = two_trainer_dispatcher(
        tmp_path, training=RefusingImports(tmp_path / "checkpoints"), hold_local_cycles=True
    )
    try:
        scenario = two_trainer_scenario(dispatcher, 1)
        make_samples(scenario)
        base = scenario.current_snapshot().revision
        for _ in range(3):
            result = _job_step(scenario, WEIGHTS)
            scenario.commit(result, component=WEIGHTS)
        served = scenario.current_snapshot().revision
        steps = scenario.commit_number
        assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights job 1"
        with pytest.raises(TrainingStartRefusedError, match="cannot train from these weights"):
            scenario.train_from(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
        with pytest.raises(ValueError, match="name the component"):
            scenario.train_from(base)
        assert scenario.current_snapshot().revision == served
        assert scenario.commit_number == steps
        assert dispatcher.live.runtime is not None
        assert dispatcher.live.runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_activation_and_a_new_start_wait_while_the_weight_trainer_has_a_prepared_job_commit(tmp_path: Path) -> None:
    """A job's checkpoint decision is prepared without a reserved batch; both operations still wait for its commit."""
    dispatcher, _ = two_trainer_dispatcher(
        tmp_path, training=WeightsJobComponent(tmp_path / "checkpoints"), hold_local_cycles=True
    )
    try:
        scenario = two_trainer_scenario(dispatcher, 1, 2)
        make_samples(scenario)
        base = scenario.current_snapshot().revision
        # The first job's submission, checkpoint and end; then the second job's submission.
        for _ in range(4):
            result = _job_step(scenario, WEIGHTS)
            scenario.commit(result, component=WEIGHTS)
        served = scenario.current_snapshot().revision
        assert served != base
        decided = _job_step(scenario, WEIGHTS)
        assert decided.metrics["training_job"]["event"] == "checkpoint"
        trainer = scenario.trainer_for(WEIGHTS)
        assert trainer.pending_job_data is None and trainer.has_pending_commit
        with pytest.raises(CommitPendingError, match="waits for its commit"):
            scenario.activate(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
        with pytest.raises(CommitPendingError, match="waits for its commit"):
            scenario.train_from(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
        assert scenario.current_snapshot().revision == served
        scenario.commit(decided, component=WEIGHTS)
        assert not trainer.has_pending_commit
        scenario.activate(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
        assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights seed"
    finally:
        dispatcher.close()


def store(tmp_path: Path) -> Path:
    """A new empty directory for an import to store into."""
    stores = tmp_path / "stores"
    stores.mkdir(exist_ok=True)
    directory = stores / str(len(list(stores.iterdir())))
    directory.mkdir()
    return directory


class HarnessText(ArtifactLayout):
    """The harness layout of the two trainer recipe: one ``harness.txt``."""

    def check(self, files: ArtifactFiles) -> None:
        if files.paths() != (f"{HARNESS}.txt",):
            raise ValueError(f"a harness artifact holds only {HARNESS}.txt")


@dataclass(frozen=True)
class _ImportingHarnessRecipe(TextHarnessRecipe):
    """The text harness recipe whose component takes imports in the layout ``HarnessText``."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents({HARNESS: ComponentSpec(files=TextFileTree(), layout=HarnessText())})


def importing_recipe() -> CompositeRecipe:
    """The two trainer recipe whose harness takes imports; its weights take none."""
    harness = _ImportingHarnessRecipe(
        evolution=EvolutionSettings(), evaluation=EvaluationSettings(policy="none"), start=seeded_start(HARNESS)
    )
    return two_trainer_recipe(harness=harness)


class PinnedSource(DirectorySource):
    """A directory whose source keeps it fixed, as a Hugging Face commit does: it offers a reference."""

    @property
    def reference(self):
        return {REFERENCE_FILE: json.dumps(HuggingFaceRevision("org/harness", "0" * 40).to_dict()).encode()}


@pytest.mark.unit
def test_a_component_whose_readers_resolve_no_reference_imports_a_copy(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path, importing_recipe())
    imported_dir = tmp_path / "imported"
    imported_dir.mkdir()
    (imported_dir / f"{HARNESS}.txt").write_text("harness imported", encoding="utf-8")
    try:
        scenario = _scenario(dispatcher)
        _step(scenario, WEIGHTS, 1)
        assert not HarnessText().resolves_references
        imported = scenario.import_artifact(
            PinnedSource(imported_dir, source_version="0" * 40, link=False),
            HARNESS,
            source={"uri": f"hf://org/harness@{'0' * 40}", "version": "0" * 40},
            directory=store(tmp_path),
        )
        # The harness's readers read its files, so the import stores them, not the source's reference.
        scenario.activate(imported.artifact_id, HARNESS)
        assert component_files(scenario, scenario.current_snapshot())[HARNESS] == "harness imported"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_import_binds_one_component_on_the_active_selection_and_activates_nothing(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path, importing_recipe())
    imported_dir = tmp_path / "imported"
    imported_dir.mkdir()
    (imported_dir / f"{HARNESS}.txt").write_text("harness imported", encoding="utf-8")
    try:
        scenario = _scenario(dispatcher)
        _step(scenario, WEIGHTS, 1)
        served = scenario.current_snapshot()
        tip = scenario.repository.checkpoint_artifact

        source = {"uri": imported_dir.as_uri(), "version": ""}
        imported = scenario.import_artifact(
            DirectorySource(imported_dir, source_version="", link=False),
            HARNESS,
            source=source,
            directory=store(tmp_path),
        )

        assert imported.artifact_id in scenario.store.history()[-1].components.values()
        assert scenario.current_snapshot() == served
        assert scenario.repository.checkpoint_artifact == tip
        record = scenario.store.history()[-1]
        assert (record.operation, record.component, record.checkpoint) == ("import", HARNESS, True)
        weights_id = scenario.artifact_status()[WEIGHTS]["active"]
        assert record.components == {HARNESS: imported.artifact_id, WEIGHTS: weights_id}
        rows = [row for row in scenario.artifacts() if row["operation"] == "import"]
        assert [(row["artifact_id"], row["component"], row["active"]) for row in rows] == [
            (imported.artifact_id, HARNESS, False)
        ]
        with pytest.raises(ValueError, match="name the component"):
            scenario.import_artifact(
                DirectorySource(imported_dir, source_version="", link=False),
                None,
                source=source,
                directory=store(tmp_path),
            )
        with pytest.raises(ValueError, match="takes no imported snapshots"):
            scenario.import_artifact(
                DirectorySource(imported_dir, source_version="", link=False),
                WEIGHTS,
                source=source,
                directory=store(tmp_path),
            )
        (imported_dir / "extra.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="holds only"):
            scenario.import_artifact(
                DirectorySource(imported_dir, source_version="", link=False),
                HARNESS,
                source=source,
                directory=store(tmp_path),
            )

        # Activating it binds the imported harness to the weights that serve now.
        scenario.activate(imported.artifact_id, HARNESS)
        assert component_files(scenario, scenario.current_snapshot()) == {
            WEIGHTS: "weights step 1",
            HARNESS: "harness imported",
        }
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_served_harness_binds_the_weights_served_now_and_leaves_the_active_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A serve records its component selection without publishing artifacts or moving a head."""
    dispatcher = _dispatcher(tmp_path)
    runtime = _runtime(dispatcher)
    try:
        scenario = _scenario(dispatcher)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        active = scenario.current_snapshot()
        tip = scenario.repository.checkpoint_artifact
        artifact_id = component_id(scenario, first, HARNESS)
        with pytest.raises(ReefError, match="answers only from the snapshot it has loaded"):
            scenario.serve(artifact_id, HARNESS)
        # A runtime that keeps two artifacts loaded loads weights only, so it cannot answer for the weights that the
        # harness artifact binds once other weights are active.
        with monkeypatch.context() as two_places:
            two_places.setattr(StubInferenceRuntime, "served_artifact_capacity", property(lambda self: 2))
            with pytest.raises(ReefError, match=f"only from the artifacts of component {WEIGHTS!r}"):
                scenario.serve(artifact_id, HARNESS)

        runtime.answers_each_artifact = True
        served = scenario.serve(artifact_id, HARNESS)
        assert served.artifact_id == artifact_id
        selection = scenario.served_beside()[artifact_id].selection
        ref = scenario.repository.select(selection).ref
        assert component_files(scenario, ref) == {WEIGHTS: "weights seed", HARNESS: "harness step 1"}
        assert scenario.current_snapshot() == active
        assert scenario.repository.checkpoint_artifact == tip
        record = scenario.store.history()[-1]
        assert (record.operation, record.component, record.components[HARNESS]) == ("serve", HARNESS, artifact_id)

        stopped = scenario.stop_serving(artifact_id)
        assert stopped.selection == selection
        assert scenario.store.history()[-1].components == record.components
        assert artifact_id not in scenario.served_beside()
    finally:
        dispatcher.close()
