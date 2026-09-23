"""Rollback and inference admission in a composite whose weights load into a runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime
from reef_service.test_component_trainers import (
    HARNESS,
    WEIGHTS,
    _component_files,
    _ComponentBackend,
    _records,
    _TwoTrainerRecipe,
)

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.release_chain import ReleaseNotRestorable
from reef.dispatcher import Dispatcher, local_error_source
from reef.runtime.interfaces import TrainingRuntime
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


class _RestoringTraining(StubTrainingRuntime):
    """A training runtime whose restore works, with a serving side that restores too."""

    def restore_serving_checkpoint(self, artifact):
        return "restored"


class _NoRestoreTraining(StubTrainingRuntime):
    """A training runtime without checkpoint restore, as the Slime executor runtime is."""

    @property
    def supports_checkpoint_restore(self):
        return False

    restore_checkpoint = TrainingRuntime.restore_checkpoint  # type: ignore[assignment]


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
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
    )


def _scenario(dispatcher: Dispatcher, training: StubTrainingRuntime):
    """The scenario with its runtimes bound on the committer alone: the dispatcher's training thread stays out."""
    scenario = dispatcher.get_or_create_scenario("agent")
    binding = replace(scenario._binding, runtime=training.inference, training_runtime=training)
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


def _fail_next_install(scenario) -> None:
    chain = scenario._committer._artifacts
    install = chain.install_checkpoint
    failed = {"done": False}

    def fail_once(ref, *, expected, expected_checkpoint):
        if not failed["done"]:
            failed["done"] = True
            raise RuntimeError("artifact backend away")
        return install(ref, expected=expected, expected_checkpoint=expected_checkpoint)

    chain.install_checkpoint = fail_once


@pytest.mark.unit
def test_a_local_cycle_stands_aside_while_admission_is_closed_and_runs_once_it_reopens(tmp_path: Path) -> None:
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        for record in _records(1):
            scenario.records.append(record)
        training.inference.pause_admission()
        assert dispatcher._process_local_backend_step("agent", HARNESS) is False
        assert dispatcher._recipe.backends[HARNESS].prepared == 0
        assert ("agent", HARNESS) in dispatcher._training.stood_aside
        training.inference.resume_admission()
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        assert dispatcher._recipe.backends[HARNESS].prepared == 1
        assert ("agent", HARNESS) not in dispatcher._training.stood_aside
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_local_cycle_clears_only_the_errors_it_recorded(tmp_path: Path) -> None:
    """The training thread's failure stays in the status while a harness cycle runs after the wake."""
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        for record in _records(1):
            scenario.records.append(record)
        dispatcher._record_training_error("agent", "RuntimeError: training marker is UPDATING_WEIGHTS")
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        assert dispatcher._training_errors() == ["agent: RuntimeError: training marker is UPDATING_WEIGHTS"]
        dispatcher._record_training_error("agent", "harness away", source=local_error_source(HARNESS))
        for record in _records(2):
            scenario.records.append(record)
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        assert dispatcher._training_errors() == []
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_local_cycle_that_waited_for_a_job_stands_aside_when_the_job_left_admission_closed(tmp_path: Path) -> None:
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        for record in _records(1):
            scenario.records.append(record)
        outcomes: list[bool] = []
        with dispatcher._local_cycle_lock("agent"):
            worker = threading.Thread(
                target=lambda: outcomes.append(dispatcher._process_local_backend_step("agent", HARNESS))
            )
            worker.start()
            worker.join(0.5)
            assert worker.is_alive()  # past the first admission check, waiting for the lock
            training.inference.pause_admission()
        worker.join(10)
        assert outcomes == [False]
        assert dispatcher._recipe.backends[HARNESS].prepared == 0
        assert ("agent", HARNESS) in dispatcher._training.stood_aside
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_retry_of_a_promote_over_the_creation_leaves_held_admission_closed(tmp_path: Path) -> None:
    """The release served before the promote is the creation: its manifest names the weights the promote kept."""
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        dispatcher._recipe.backends[HARNESS].hold_next = True
        _step(scenario, HARNESS, 1)
        held = next(row["release_id"] for row in scenario.releases() if row.get("pending"))
        training.inference.pause_admission()
        training.inference._current_runtime_load_id = "before"
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(held, operation="promote")
        scenario.rollback(held, operation="promote")
        assert not training.inference.inference_admission_status["open"]
        assert training.inference.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_harness_only_rollback_leaves_held_admission_closed(tmp_path: Path) -> None:
    """A rollback that restores nothing neither resumes admission nor calls a version served."""
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        # A dispatched weight job holds admission: published, not yet committed.
        training.inference.pause_admission()
        training.inference._current_runtime_load_id = "before"
        scenario.rollback(first)
        assert not training.inference.inference_admission_status["open"]
        assert training.inference.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_recorded_retry_of_a_harness_only_rollback_leaves_held_admission_closed(tmp_path: Path) -> None:
    """The first attempt records the rollback and fails after; the retry has nothing to resume either."""
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        training.inference.pause_admission()
        training.inference._current_runtime_load_id = "before"
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(first)
        records = scenario.store.history()
        assert records[-1].operation == "rollback" and records[-1].rollback_target_release_id == first
        assert not training.inference.inference_admission_status["open"]

        assert scenario.rollback(first) == records[-1].artifact_ref
        assert _component_files(scenario, scenario.current_artifact_ref())[HARNESS] == "harness step 1"
        assert not training.inference.inference_admission_status["open"]
        assert training.inference.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_rejected_step_before_the_rollback_does_not_make_its_retry_look_like_a_restore(tmp_path: Path) -> None:
    """The release served before the rollback is the newest published one, not a rejected step's empty record."""
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        dispatcher._recipe.backends[HARNESS].reject_next = True
        _step(scenario, HARNESS, 3)
        assert scenario.store.history()[-1].components is None
        training.inference.pause_admission()
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(first)
        scenario.rollback(first)
        assert not training.inference.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_composite_without_loaded_weights_never_resumes_admission_on_a_recorded_retry(tmp_path: Path) -> None:
    training = _RestoringTraining()
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    dispatcher = Dispatcher(
        _TwoTrainerRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = _scenario(dispatcher, training)
        first = _step(scenario, WEIGHTS, 1)
        _step(scenario, WEIGHTS, 2)
        training.inference.pause_admission()
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(first)
        scenario.rollback(first)
        assert not training.inference.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_recorded_retry_of_a_weights_rollback_reopens_the_admission_it_closed(tmp_path: Path) -> None:
    """The attempt that restored the weights closed admission; its retry installs the head and reopens it."""
    training = _RestoringTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        base = scenario.current_artifact_ref().release_id
        _step(scenario, WEIGHTS, 1)
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(base)
        assert not training.inference.inference_admission_status["open"]
        assert scenario.rollback(base) == scenario.store.history()[-1].artifact_ref
        assert training.inference.inference_admission_status["open"]
        assert _component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights seed"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_weights_rollback_without_checkpoint_restore_is_refused_before_admission_closes(tmp_path: Path) -> None:
    training = _NoRestoreTraining()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, training)
        base = scenario.current_artifact_ref().release_id
        served = _step(scenario, WEIGHTS, 1)
        with pytest.raises(ReleaseNotRestorable, match="cannot restore training weights"):
            scenario.rollback(base)
        assert scenario.current_artifact_ref().release_id == served
        assert training.inference.inference_admission_status["open"]
    finally:
        dispatcher.close()
