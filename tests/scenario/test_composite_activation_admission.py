"""Activation, new training starts and inference admission in a composite whose weights load into a runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from tests.support.component_trainers import HARNESS, WEIGHTS, WeightsJobComponent, _ComponentBackend
from tests.support.component_trainers import _dispatcher as two_trainer_dispatcher
from tests.support.component_trainers import _records
from tests.support.component_trainers import _scenario as two_trainer_scenario
from tests.support.component_trainers import _TwoTrainerRecipe, component_files, stop_workers
from tests.support.runtime_stubs import StubInferenceRuntime

from reef.artifact import InMemoryRepositoryBackend
from reef.core.errors import CommitPending, TrainingStartRefused
from reef.dispatcher import Dispatcher
from reef.dispatcher.errors import local_error_source
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.surface.weights import WeightInferenceHooks, WeightLoader


@dataclass(frozen=True)
class _LoadedRecipe(_TwoTrainerRecipe):
    """The two trainer recipe whose weights component loads into a runtime, as a real weight surface does."""

    def build_surface(self, scenario: str) -> Surface:
        return Surface(
            components={
                WEIGHTS: ComponentSurface(loader=WeightLoader(), inference=WeightInferenceHooks()),
                HARNESS: ComponentSurface(files=TextFileTree()),
            }
        )


class RefusingImports(WeightsJobComponent):
    """A training component that cannot start a job from the weights of a published release."""

    def import_start(self, release: Path) -> str | None:
        raise ValueError("cannot train from these weights")


def _dispatcher(tmp_path: Path) -> Dispatcher:
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    records = tmp_path / "records"
    return Dispatcher(
        _LoadedRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(records),
    )


def _scenario(dispatcher: Dispatcher, runtime: StubInferenceRuntime):
    """The scenario with its inference runtime bound on the committer alone, after the creation activated the base."""
    scenario = dispatcher.get_or_create_scenario("agent")
    # Loading the scenario woke its workers: stopped, they leave every step to the test.
    stop_workers(dispatcher, "agent")
    binding = replace(scenario._binding, runtime=runtime)
    scenario._binding = binding
    scenario._committer._binding = binding
    return scenario


def _step(scenario, component: str, step: int) -> str:
    for record in _records(step):
        scenario.records.append(record)
    result = scenario.prepare_training_step(component)
    assert result is not None
    scenario.commit(result, component=component)
    return scenario.current_artifact_ref().release_id


@pytest.mark.unit
def test_a_local_cycle_clears_only_the_errors_it_recorded(tmp_path: Path) -> None:
    """The weights worker's failure stays in the status while a harness cycle runs after the wake."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        for record in _records(1):
            scenario.records.append(record)
        dispatcher.errors.record("agent", "RuntimeError: training job away", local_error_source(WEIGHTS))
        assert dispatcher.local_workers.process_step("agent", HARNESS) is True
        assert dispatcher.training_status.training_errors() == ["agent: RuntimeError: training job away"]
        dispatcher.errors.record("agent", "harness away", local_error_source(HARNESS))
        assert sorted(dispatcher.training_status.training_errors()) == [
            "agent: RuntimeError: training job away",
            "agent: harness away",
        ]
        for record in _records(2):
            scenario.records.append(record)
        # The harness cycle clears its own error; the weights worker's stays until that worker clears it.
        assert dispatcher.local_workers.process_step("agent", HARNESS) is True
        assert dispatcher.training_status.training_errors() == ["agent: RuntimeError: training job away"]
        dispatcher.errors.clear("agent", local_error_source(WEIGHTS))
        assert dispatcher.training_status.training_errors() == []
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_activating_a_held_harness_release_leaves_held_admission_closed(tmp_path: Path) -> None:
    """The activation binds the held harness to the weights that serve now, so it loads nothing."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        dispatcher.recipe.backends[HARNESS].hold_next = True
        _step(scenario, HARNESS, 1)
        held = next(row for row in scenario.artifacts() if row["operation"] == "training")
        assert held["component"] == HARNESS and not held["active"]
        runtime.close_admission()
        runtime.served = "before"
        activated = scenario.activate(held["artifact_id"], HARNESS)
        assert activated.artifact_id == f"{held['release_id']}:{HARNESS}"
        # The activation publishes the combination: the held harness on the weights served now.
        assert activated.release_id != held["release_id"]
        assert component_files(scenario, scenario.current_artifact_ref()) == {
            WEIGHTS: "weights seed",
            HARNESS: "harness step 1",
        }
        assert not runtime.inference_admission_status["open"]
        assert runtime.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_harness_only_activation_leaves_held_admission_closed(tmp_path: Path) -> None:
    """An activation that changes no loaded weights neither resumes admission nor calls a version served."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        # Another publication holds admission: published, not yet committed.
        runtime.close_admission()
        runtime.served = "before"
        scenario.activate(f"{first}:{HARNESS}", HARNESS)
        assert component_files(scenario, scenario.current_artifact_ref())[HARNESS] == "harness step 1"
        assert not runtime.inference_admission_status["open"]
        assert runtime.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_failed_activation_record_leaves_the_active_artifact_and_a_retry_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weights load first; an append that fails leaves the log, and so the active artifact, as they were."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        trained = _step(scenario, WEIGHTS, 1)
        commit_step = scenario.store.commit_step

        def store_away(**keywords):
            raise OSError("commit storage away")

        monkeypatch.setattr(scenario.store, "commit_step", store_away)
        with pytest.raises(OSError, match="commit storage away"):
            scenario.activate(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
        assert scenario.current_artifact_ref().release_id == trained
        assert scenario.store.history()[-1].operation == "training"
        monkeypatch.setattr(scenario.store, "commit_step", commit_step)

        activated = scenario.activate(
            f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS
        )
        assert scenario.current_artifact_ref().release_id == activated.release_id
        assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights seed"
        record = scenario.store.history()[-1]
        assert (record.operation, record.component, record.checkpoint) == ("activate", WEIGHTS, False)
        assert record.components is not None and WEIGHTS in record.components
        assert runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_weights_start_the_training_component_refuses_is_refused_before_anything_changes(tmp_path: Path) -> None:
    """The training component imports the release as the next job's start; one it cannot train from refuses the
    new start before any record lands, and serving stays as it was."""
    dispatcher, _ = two_trainer_dispatcher(
        tmp_path, training=RefusingImports(tmp_path / "checkpoints"), hold_local_cycles=True
    )
    try:
        scenario = two_trainer_scenario(dispatcher, 1)
        base = scenario.current_artifact_ref().release_id
        for _ in range(3):
            result = scenario.prepare_training_step(WEIGHTS)
            assert result is not None
            scenario.commit(result, component=WEIGHTS)
        served = scenario.current_artifact_ref().release_id
        steps = scenario.scenario_step
        assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights job 1"
        with pytest.raises(TrainingStartRefused, match="cannot train from these weights"):
            scenario.train_from(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
        with pytest.raises(ValueError, match="name the component"):
            scenario.train_from(base)
        assert scenario.current_artifact_ref().release_id == served
        assert scenario.scenario_step == steps
        assert dispatcher.recipe.runtime is not None
        assert dispatcher.recipe.runtime.inference_admission_status["open"]
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
        base = scenario.current_artifact_ref().release_id
        # The first job's submission, checkpoint and end; then the second job's submission.
        for _ in range(4):
            result = scenario.prepare_training_step(WEIGHTS)
            assert result is not None
            scenario.commit(result, component=WEIGHTS)
        served = scenario.current_artifact_ref().release_id
        assert served != base
        decided = scenario.prepare_training_step(WEIGHTS)
        assert decided is not None and decided.metrics["training_job"]["event"] == "checkpoint"
        trainer = scenario.trainer_for(WEIGHTS)
        assert trainer.pending_job_data is None and trainer.has_pending_commit
        with pytest.raises(CommitPending, match="waits for its commit"):
            scenario.activate(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
        with pytest.raises(CommitPending, match="waits for its commit"):
            scenario.train_from(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
        assert scenario.current_artifact_ref().release_id == served
        scenario.commit(decided, component=WEIGHTS)
        assert not trainer.has_pending_commit
        scenario.activate(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
        assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights seed"
    finally:
        dispatcher.close()
