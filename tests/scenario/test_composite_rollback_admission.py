"""Rollback and inference admission in a composite whose weights load into a runtime."""

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
from reef.artifact.release_chain import ReleaseNotRestorable
from reef.core.errors import ReefError
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
def test_the_retry_of_a_promote_over_the_creation_leaves_held_admission_closed(tmp_path: Path) -> None:
    """The release served before the promote is the creation: its manifest names the weights the promote kept."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        dispatcher.recipe.backends[HARNESS].hold_next = True
        _step(scenario, HARNESS, 1)
        held = next(row["release_id"] for row in scenario.releases() if row.get("pending"))
        runtime.close_admission()
        runtime.served = "before"
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(held, operation="promote")
        scenario.rollback(held, operation="promote")
        assert not runtime.inference_admission_status["open"]
        assert runtime.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_harness_only_rollback_leaves_held_admission_closed(tmp_path: Path) -> None:
    """A rollback that restores nothing neither resumes admission nor calls a version served."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        # Another publication holds admission: published, not yet committed.
        runtime.close_admission()
        runtime.served = "before"
        scenario.rollback(first)
        assert not runtime.inference_admission_status["open"]
        assert runtime.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_recorded_retry_of_a_harness_only_rollback_leaves_held_admission_closed(tmp_path: Path) -> None:
    """The first attempt records the rollback and fails after; the retry has nothing to resume either."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        runtime.close_admission()
        runtime.served = "before"
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(first)
        records = scenario.store.history()
        assert records[-1].operation == "rollback" and records[-1].rollback_target_release_id == first
        assert not runtime.inference_admission_status["open"]

        assert scenario.rollback(first) == records[-1].artifact_ref
        assert component_files(scenario, scenario.current_artifact_ref())[HARNESS] == "harness step 1"
        assert not runtime.inference_admission_status["open"]
        assert runtime.current_runtime_load_id() == "before"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_rejected_step_before_the_rollback_does_not_make_its_retry_look_like_a_restore(tmp_path: Path) -> None:
    """The release served before the rollback is the newest published one, not a rejected step's empty record."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        first = _step(scenario, HARNESS, 1)
        _step(scenario, HARNESS, 2)
        dispatcher.recipe.backends[HARNESS].reject_next = True
        _step(scenario, HARNESS, 3)
        assert scenario.store.history()[-1].components is None
        runtime.close_admission()
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(first)
        scenario.rollback(first)
        assert not runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_composite_without_loaded_weights_never_resumes_admission_on_a_recorded_retry(tmp_path: Path) -> None:
    runtime = StubInferenceRuntime()
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    dispatcher = Dispatcher(
        _TwoTrainerRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = _scenario(dispatcher, runtime)
        first = _step(scenario, WEIGHTS, 1)
        _step(scenario, WEIGHTS, 2)
        runtime.close_admission()
        _fail_next_install(scenario)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(first)
        scenario.rollback(first)
        assert not runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_recorded_retry_of_a_weights_rollback_installs_the_restored_head(tmp_path: Path) -> None:
    """The attempt that restored the weights records the rollback and fails after; its retry installs the head."""
    runtime = StubInferenceRuntime()
    dispatcher = _dispatcher(tmp_path)
    try:
        scenario = _scenario(dispatcher, runtime)
        base = scenario.current_artifact_ref().release_id
        _step(scenario, WEIGHTS, 1)
        _fail_next_install(scenario)
        served_before = len(runtime.activated)
        with pytest.raises(RuntimeError, match="artifact backend away"):
            scenario.rollback(base)
        assert len(runtime.activated) == served_before + 1
        assert scenario.rollback(base) == scenario.store.history()[-1].artifact_ref
        assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights seed"
        assert runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_weights_rollback_the_training_component_refuses_is_refused_before_admission_closes(tmp_path: Path) -> None:
    """The training component imports a rolled-back release as the next job's start; one it cannot train from
    refuses the rollback before the weights load and admission closes."""
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
        assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights job 1"
        with pytest.raises(ReleaseNotRestorable, match="cannot train from these weights"):
            scenario.rollback(base)
        assert scenario.current_artifact_ref().release_id == served
        assert dispatcher.recipe.runtime is not None
        assert dispatcher.recipe.runtime.inference_admission_status["open"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_rollback_is_refused_while_the_weight_trainer_has_a_prepared_job_commit(tmp_path: Path) -> None:
    """A job's checkpoint decision is prepared without a reserved batch; the rollback still waits for its commit."""
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
        with pytest.raises(ReefError, match="pending commit"):
            scenario.rollback(base)
        assert scenario.current_artifact_ref().release_id == served
        scenario.commit(decided, component=WEIGHTS)
        assert not trainer.has_pending_commit
        scenario.rollback(base)
        assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights seed"
    finally:
        dispatcher.close()
