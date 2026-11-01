"""Idle scenarios unload and load again. Admission limits the scenarios that a service registers and loads."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Event, Thread

import pytest
from tests.support._threshold_processor import threshold_pipeline
from tests.support.runtime_stubs import StubTrainingComponent

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import ScenarioCapacityFull, ScenarioLimitReached
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.scenario import Scenario
from reef.scenario.admission import LoadAdmission, LoadedScenario, ScenarioAdmission, ScenarioLimits
from reef.storage.scenario import ScenarioSettings
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import Trainer
from reef.train.jobs import JobLoop


class StepJobs(StubTrainingComponent):
    """A training component whose ``n``-th job saves one checkpoint holding ``model.txt`` = ``step <n>``. With
    ``block``, a job waits in ``submit``, so the trainer's cycle waits with it."""

    def __init__(self, artifact_dir: Path) -> None:
        super().__init__(artifact_dir, name="steps", shares_scenarios=True)
        self.block = False
        self.started = Event()
        self.resume = Event()

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        if self.block:
            self.started.set()
            assert self.resume.wait(10), "the test did not release the cycle"
        step = len(self.submitted)
        path = self.checkpoint_root / f"{job.data.data_id.replace(':', '-')}-{step}"
        path.mkdir(parents=True)
        (path / "model.txt").write_text(f"step {step}", encoding="utf-8")
        return (SavedCheckpoint(0, str(path), str(path)),)


@dataclass(frozen=True)
class StepRecipe(Recipe):
    backend: StepJobs | None = None

    def build(
        self,
        scenario,
        records,
        *,
        algorithm_state=None,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
        root=None,
    ):
        return Trainer.build(
            scenario,
            records,
            # A trainer that trains nothing has no job size, so its pipeline keeps every sample.
            pipeline_factory=partial(threshold_pipeline, job_samples=None if self.backend is None else 1),
            job_loop=None if self.backend is None else JobLoop(self.backend, scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            root=root,
        )


def make_dispatcher(
    tmp_path: Path,
    *,
    backend: StepJobs | None = None,
    admission: ScenarioAdmission | None = None,
    idle_unload_seconds: float | None = None,
    storage: SQLiteScenarioStorage | None = None,
) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    (initial / "model.txt").write_text("initial", encoding="utf-8")
    return Dispatcher(
        StepRecipe(backend=backend),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=storage if storage is not None else SQLiteScenarioStorage(tmp_path / "records"),
        scenario_admission=admission,
        scenario_idle_unload_seconds=idle_unload_seconds,
        coordination=memory_coordination(),
    )


def scored_pair(scenario: str, index: int) -> tuple[AgentRecord, AgentRecord]:
    inference = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
        agent_record_id=f"i{index}",
    )
    report = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"i{index}"]},
        agent_record_id=f"r{index}",
        references=(f"i{index}",),
    )
    return inference, report


def loaded_names(dispatcher: Dispatcher) -> list[str]:
    return [row["scenario"] for row in dispatcher.list_scenarios() if row["loaded"]]


def wait_for(condition, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition did not become true"
        time.sleep(0.01)


def trained(dispatcher: Dispatcher, name: str, step: int) -> Scenario:
    """Accept one scored pair, and wait until the local worker commits the job of ``step``: its submission, its
    checkpoint and its end, scenario step ``3 * step``."""
    current = dispatcher.get_or_create_scenario(name)
    assert current is not None
    dispatcher.accept_records(scored_pair(name, step))
    wait_for(lambda: current.scenario_step == 3 * step and not current.trainer.holds_untrained_records())
    return current


@pytest.mark.unit
def test_idle_scenario_unloads_and_loads_again_with_its_history(tmp_path) -> None:
    dispatcher = make_dispatcher(tmp_path, backend=StepJobs(tmp_path / "candidates"))
    try:
        before = trained(dispatcher, "math", 1)
        releases = before.releases()

        # The samples of the step's records come from the samples queue, so the scenario is idle a moment later.
        wait_for(lambda: dispatcher.unload_if_idle("math"))
        assert loaded_names(dispatcher) == []
        assert [row["scenario"] for row in dispatcher.list_scenarios()] == ["math"]
        assert dispatcher.build_training_status()["loaded_scenarios"] == 0

        after = dispatcher.get_or_create_scenario("math")
        assert after is not None and after is not before
        assert after.scenario_step == 3
        assert after.releases() == releases
        assert [row["agent_record_id"] for row in dispatcher.read_records("math")["records"]] == ["i1", "r1"]
        # The reloaded scenario trains on, with a new local worker.
        trained(dispatcher, "math", 2)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_idle_unload_runs_after_the_configured_idle_time(tmp_path) -> None:
    dispatcher = make_dispatcher(tmp_path, idle_unload_seconds=0.05)
    try:
        dispatcher.accept_record(scored_pair("math", 1)[0])
        wait_for(lambda: loaded_names(dispatcher) == [])
        assert dispatcher.get_or_create_scenario("math", allow_implicit_creation=False) is not None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_scenario_with_untrained_records_stays_loaded(tmp_path) -> None:
    dispatcher = make_dispatcher(tmp_path, backend=StepJobs(tmp_path / "candidates"))
    try:
        current = dispatcher.get_or_create_scenario("math")
        assert current is not None
        # The rows go around the dispatcher. No worker reads them, so they stay untrained.
        for item in scored_pair("math", 1):
            current.records.append(item)

        assert not dispatcher.unload_if_idle("math")
        assert loaded_names(dispatcher) == ["math"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_scenario_with_a_running_cycle_stays_loaded(tmp_path) -> None:
    backend = StepJobs(tmp_path / "candidates")
    dispatcher = make_dispatcher(tmp_path, backend=backend)
    try:
        backend.block = True
        dispatcher.accept_records(scored_pair("math", 1))
        assert backend.started.wait(10)

        assert not dispatcher.unload_if_idle("math")
    finally:
        backend.resume.set()
        dispatcher.close()


@pytest.mark.unit
def test_scenario_with_a_request_in_flight_stays_loaded(tmp_path) -> None:
    dispatcher = make_dispatcher(tmp_path)
    try:
        current = dispatcher.get_or_create_scenario("math")
        assert current is not None
        measurement = current.operations.start("ingest/write")
        assert not dispatcher.unload_if_idle("math")
        measurement.finish(succeeded=True)
        assert dispatcher.unload_if_idle("math")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_load_limit_unloads_the_least_recently_used_idle_scenario(tmp_path) -> None:
    dispatcher = make_dispatcher(tmp_path, admission=ScenarioLimits(max_loaded_scenarios=2))
    try:
        dispatcher.get_or_create_scenario("a")
        dispatcher.get_or_create_scenario("b")
        dispatcher.get_or_create_scenario("a")  # Now the last use of "b" is older than that of "a".

        dispatcher.get_or_create_scenario("c")

        assert loaded_names(dispatcher) == ["a", "c"]
        assert [row["scenario"] for row in dispatcher.list_scenarios()] == ["a", "b", "c"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_full_service_refuses_a_load_when_no_scenario_is_idle(tmp_path) -> None:
    dispatcher = make_dispatcher(tmp_path, admission=ScenarioLimits(max_loaded_scenarios=1))
    try:
        busy = dispatcher.get_or_create_scenario("a")
        assert busy is not None
        measurement = busy.operations.start("ingest/write")

        with pytest.raises(ScenarioCapacityFull) as refused:
            dispatcher.get_or_create_scenario("b")

        assert refused.value.retry_after_seconds > 0
        assert loaded_names(dispatcher) == ["a"]
        measurement.finish(succeeded=True)
        assert dispatcher.get_or_create_scenario("b") is not None
        assert loaded_names(dispatcher) == ["b"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_concurrent_loads_share_the_load_limit(tmp_path) -> None:
    class SlowLoads(ScenarioLimits):
        """The built-in admission, with a delay. The delay lets a second load run at the same time as the first."""

        def admit_load(self, scenario: str, *, loaded: tuple[LoadedScenario, ...]) -> LoadAdmission:
            admission = super().admit_load(scenario, loaded=loaded)
            time.sleep(0.05)
            return admission

    dispatcher = make_dispatcher(tmp_path, admission=SlowLoads(max_loaded_scenarios=1))

    def load(name: str) -> None:
        # Timing decides which load is refused. The test checks only the limit.
        with suppress(ScenarioCapacityFull):
            dispatcher.get_or_create_scenario(name)

    try:
        threads = [Thread(target=load, args=(name,)) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(loaded_names(dispatcher)) == 1
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_registration_limit_refuses_a_new_scenario_and_stores_nothing(tmp_path) -> None:
    storage = SQLiteScenarioStorage(tmp_path / "records")
    dispatcher = make_dispatcher(tmp_path, admission=ScenarioLimits(max_scenarios=1), storage=storage)
    try:
        dispatcher.get_or_create_scenario("a")
        assert dispatcher.unload_if_idle("a")

        # A registered scenario loads again. The limit counts registrations, not loads.
        assert dispatcher.get_or_create_scenario("a") is not None
        with pytest.raises(ScenarioLimitReached, match=r"reef\.max_scenarios"):
            dispatcher.get_or_create_scenario("b")
        with pytest.raises(ScenarioLimitReached):
            dispatcher.create_scenario("b")

        assert not dispatcher.has_scenario("b")
        assert storage.read_settings("b") == ScenarioSettings()
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_loaded_scenarios"):
        ScenarioLimits(max_loaded_scenarios=0)
    with pytest.raises(ValueError, match="max_scenarios"):
        ScenarioLimits(max_scenarios=0)


@pytest.mark.unit
@pytest.mark.parametrize("held_lock", ["registry", "cycle"])
def test_unload_returns_without_waiting_for_scenario_locks(tmp_path, held_lock) -> None:
    dispatcher = make_dispatcher(tmp_path)
    try:
        dispatcher.get_or_create_scenario("math")
        if held_lock == "registry":
            lock = dispatcher.registry.lock_for("math")
        else:
            lock = dispatcher.local_workers.cycle_lock("math")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with lock:
                assert pool.submit(dispatcher.unload_if_idle, "math").result(timeout=2) is False
                assert loaded_names(dispatcher) == ["math"]
            assert pool.submit(dispatcher.unload_if_idle, "math").result(timeout=2) is True
        assert loaded_names(dispatcher) == []
    finally:
        dispatcher.close()
