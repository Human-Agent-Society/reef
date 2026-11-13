"""One trainer per snapshot component: independent workers that meet at the commit boundary."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Self

import pytest
from aiohttp.test_utils import TestServer
from tests.support.component_trainers import (
    HARNESS,
    MAX_ARTIFACT_LAG,
    WEIGHTS,
    ComponentJobs,
    JudgedAtOnce,
    Judgment,
    WeightsJobComponent,
    WeightsRecipe,
    _append_records,
    _dispatcher,
    _RecordingTracker,
    _records,
    _repository,
    _scenario,
    checked_commit,
    component_files,
    component_id,
    component_stages,
    finish_job,
    harness_recipe,
    seeded_start,
    stop_workers,
    two_trainer_recipe,
    wait_for_job_check,
)
from tests.support.polling import became_true
from tests.support.records import memory_records, report
from tests.support.runtime_stubs import StubInferenceRuntime, harness_live_objects, job_sized_stages
from tests.support.samples_service import make_samples
from tests.support.serving import wait_for_serving
from tests.support.threshold_processor import ThresholdJobData, ThresholdSamples
from tests.support.trajectories import policy_trajectory

from reef.artifact import Artifact, ArtifactNotFoundError, DefaultStart, SnapshotRef
from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
from reef.artifact.repository import Repository
from reef.core import AgentRecord, RequestType
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError
from reef.core.requirements import required_by
from reef.core.snapshot import RootSource
from reef.dispatcher import Dispatcher
from reef.dispatcher.samples import LocalSampleProducers
from reef.execution.jobs import SupportsReceipt
from reef.execution.workers import SERVICE_COMPONENTS
from reef.pipeline_context import PipelineContext
from reef.recipe import CompositeRecipe, Recipe
from reef.record2sample import GroupDecision, JudgedSamples, RecordJudging, ReportContext, ReportGrouping
from reef.record2sample.assembly import SampleAssemblySettings
from reef.record2sample.service import SAMPLES_QUEUE, Selection
from reef.record2sample.staleness import ServedArtifacts, lags_behind
from reef.record2sample.traces import TraceSamples
from reef.sample2job.traces import TraceJobData
from reef.scenario import Scenario
from reef.scenario.live_objects import LiveObjects
from reef.scenario.scenario import validate_component_trainers
from reef.service.app import create_app
from reef.service.request_service import RequestService
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    CommitHistoryError,
    CommitRecord,
    parse_scenario_metadata,
    scenario_metadata_for,
)
from reef.storage.samples import consumed_sample_id, stage_item_id
from reef.train import ComponentTrainer, Trainer
from reef.train.component import JobStatus, SnapshotStart, TrainingJob
from reef.train.job_queue import CHECKS_QUEUE, JOBS_QUEUE, TrainingCheckWorker
from reef.train.jobs import JobLoopState
from reef.train.pipeline import PipelineSpec, TrainingPipeline
from reef.train.types import JobData, Sample


class _HeldJobData(ThresholdJobData):
    """Batches like ``ThresholdJobData``, and holds or fails the trainer's cycle that prepares a job.

    ``job_parameters`` runs in the trainer's cycle, under the scenario's cycle lock and outside the trainer's own
    lock. It counts its calls in ``attempts``. With ``blocks``, it sets ``evaluating`` and waits for ``release``, as a step of minutes would.
    With ``fails``, it then raises ``RuntimeError("proposer away")``, so the cycle fails. The ``held_cycle`` fixture
    resets the class attributes and sets ``release`` at teardown.
    """

    blocks = False
    fails = False
    attempts = 0
    evaluating = threading.Event()
    release = threading.Event()

    def job_parameters(self, data: JobData) -> Mapping[str, Any]:
        held = type(self)
        held.attempts += 1
        if held.blocks:
            held.evaluating.set()
            assert held.release.wait(30)
        if held.fails:
            raise RuntimeError("proposer away")
        return super().job_parameters(data)


@pytest.fixture
def held_cycle() -> Iterator[type[_HeldJobData]]:
    """``_HeldJobData`` with its class attributes as new, for one test."""
    _HeldJobData.blocks = False
    _HeldJobData.fails = False
    _HeldJobData.attempts = 0
    _HeldJobData.evaluating = threading.Event()
    _HeldJobData.release = threading.Event()
    yield _HeldJobData
    _HeldJobData.release.set()


def held_recipe(component: str) -> CompositeRecipe:
    """The two trainer recipe whose ``component`` trainer builds its job data with ``_HeldJobData``."""
    stages = component_stages(sample2job=_HeldJobData)
    if component == WEIGHTS:
        return two_trainer_recipe(weights=WeightsRecipe(pipeline_stages=stages))
    return two_trainer_recipe(harness=harness_recipe(stages))


def listed_revisions(scenario: Scenario) -> list[str]:
    """The storage revision of each harness catalog row, oldest first."""
    return [row["artifact_snapshot"].revision for row in reversed(scenario.harness_snapshot_rows())]


agent_report = partial(report, scenario="agent", score=0.5)


class _Dispatchers:
    """Opens a test's dispatchers over its directory and closes each at teardown, letting held backends go first."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.opened: list[tuple[Dispatcher, Mapping[str, ComponentJobs]]] = []

    def open(self, **options: Any) -> tuple[Dispatcher, dict[str, ComponentJobs]]:
        """``_dispatcher`` over the test's directory, closed at teardown."""
        dispatcher, backends = _dispatcher(self.tmp_path, **options)
        self.opened.append((dispatcher, backends))
        return dispatcher, backends

    def close(self) -> None:
        for dispatcher, backends in reversed(self.opened):
            for backend in backends.values():
                if isinstance(backend, ComponentJobs):
                    backend.unblock()
            dispatcher.close()


@pytest.fixture
def dispatchers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Dispatchers]:
    # A trainer's worker polls a job that runs, or that waits for its check, at this interval.
    monkeypatch.setattr("reef.dispatcher.local_workers.POLL_SECONDS", 0.05)
    opened = _Dispatchers(tmp_path)
    yield opened
    opened.close()


def settle_samples(dispatcher: Dispatcher, name: str = "agent") -> None:
    """Wait until the samples worker ended every record selection of scenario ``name``, such as the one that a load
    asks for. A selection wakes the trainers' workers, so stop them again: the test, not a worker, decides when a
    trainer prepares."""
    assert became_true(lambda: all(item.final for item in dispatcher.queue.items(SAMPLES_QUEUE, scenario=name)))
    stop_workers(dispatcher, name)


def loaded(dispatcher: Dispatcher, *steps: int) -> Scenario:
    """Load the scenario and wait for its serving and sample selections before the test prepares training."""
    scenario = _scenario(dispatcher)
    if scenario.runtime is not None and scenario.runtime.serves_requests:
        wait_for_serving(dispatcher, "agent")
    settle_samples(dispatcher)
    _append_records(scenario, *steps)
    make_samples(scenario)
    return scenario


def reload_agent(dispatcher: Dispatcher) -> Scenario:
    """Reload scenario ``agent``, once the samples worker ended the selection that the load asked for."""
    rebuilt = dispatcher.registry.reload("agent")
    settle_samples(dispatcher)
    return rebuilt


def submitted_jobs(training: WeightsJobComponent, count: int) -> list[TrainingJob]:
    """The jobs that ``training`` took, once the training worker gave it ``count`` of them from the jobs queue."""
    assert became_true(lambda: len(training.submitted) == count)
    return training.submitted


def training_components(scenario: Scenario) -> list[str]:
    """The component of each training job that ended, newest first: one per step."""
    return [
        row["component"]
        for row in scenario.snapshot_queries.snapshot_rows(scenario.commit_number)
        if row["operation"] == "training"
        and (row.get("metrics") or {}).get("training_job", {}).get("event") in ("finished", "failed")
    ]


def append_commit(
    scenario: Scenario,
    component: str,
    *,
    checkpoint_metrics: Mapping[str, Any] | None = None,
    make: bool = True,
) -> list[tuple[str, ...]] | None:
    """Make the samples of the new records, as the samples service does, then commit ``component``'s next job: its
    submission, and for a component whose job ends in ``submit`` (``ComponentJobs``) its checkpoint and its end too;
    the sources of its batch, or None when nothing was ready. ``checkpoint_metrics`` join the checkpoint's commit;
    ``make`` False trains the samples that are stored already."""
    if make:
        make_samples(scenario)
    result = checked_commit(scenario, component)
    if result is None:
        return None
    batch = scenario.trainer_for(component).pending_job_data
    assert batch is not None
    sources = [tuple(item.source_agent_record_ids) for item in batch.samples]
    scenario.commit(result, component=component)
    finish_job(scenario, component, checkpoint_metrics)
    return sources


def wait_for_jobs(dispatcher: Dispatcher, training: WeightsJobComponent, count: int) -> list[TrainingJob]:
    """The jobs that ``training`` ran, once the training worker took ``count`` of them from the jobs queue and ended
    their items, as the weights trainer then reads them."""

    def ended() -> bool:
        items = [dispatcher.queue.read(JOBS_QUEUE, job.job_id) for job in training.submitted]
        return len(items) == count and all(item is not None and item.final for item in items)

    assert became_true(ended)
    return training.submitted


def settle_weights_job(scenario: Scenario) -> None:
    """Commit the running weights job's checkpoint decisions and its end, as the weights worker does once the training
    worker reported them in the jobs queue. Then wait until the selected artifact serves: the next automatic job
    waits for it."""
    while scenario.trainer_for(WEIGHTS).job_running:
        deadline = time.monotonic() + 10
        while (result := scenario.next_commit(WEIGHTS)) is None:
            assert time.monotonic() < deadline, "the training worker did not report the running job"
            time.sleep(0.02)
        scenario.commit(result, component=WEIGHTS)
    assert became_true(lambda: scenario.served_artifact_id() == scenario.serving_artifact_id())


@pytest.mark.unit
def test_component_trainers_take_turns_on_one_snapshot_chain(dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert [component_trainer.component for component_trainer in scenario.trainers] == [WEIGHTS, HARNESS]
    assert scenario.trainer_for(HARNESS) is scenario.trainers[1].trainer

    assert append_commit(scenario, HARNESS) is not None
    assert component_files(scenario, scenario.current_snapshot()) == {
        WEIGHTS: "weights seed",
        HARNESS: "harness step 1",
    }
    assert append_commit(scenario, WEIGHTS) is not None
    assert component_files(scenario, scenario.current_snapshot()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness step 1",
    }
    assert backends[WEIGHTS].prepared == backends[HARNESS].prepared == 1

    # Each step is one job: its submission, its checkpoint and its end.
    records = scenario.store.history()
    assert [record.component for record in records] == [HARNESS] * 3 + [WEIGHTS] * 3
    # A commit retires no row: each trainer consumed the same rows, and its own sample of them, at its job's
    # submission.
    assert scenario.records.count("agent") == 2
    assert records[0].consumed_ids == {"i1", "r1", consumed_sample_id(stage_item_id("agent", f"{HARNESS}:r1:0"))}
    assert records[3].consumed_ids == {"i1", "r1", consumed_sample_id(stage_item_id("agent", f"{WEIGHTS}:r1:0"))}

    # Each trainer recovers from its own commits.
    reloaded = reload_agent(dispatcher)
    for component in (WEIGHTS, HARNESS):
        recovered = JobLoopState.from_dict(reloaded.trainer_for(component).state)
        assert recovered.finished_jobs == 1 and recovered.jobs == ()
    assert reloaded.commit_number == 6
    _append_records(reloaded, 2)
    make_samples(reloaded)
    assert checked_commit(reloaded, HARNESS) is not None


def submission(scenario: Scenario, component: str) -> CommitRecord:
    """The newest commit of ``scenario`` that submitted a job of ``component``: the one that consumed its rows."""
    return next(
        record
        for record in reversed(scenario.store.history())
        if record.component == component and job_event(record) == "submitted"
    )


def drain(dispatcher: Dispatcher, component: str) -> None:
    """Run ``component``'s local cycles until one commits nothing and no job of the component runs or waits for its
    check: the cycles commit its job's submission, checkpoint and end, as the training worker reports them."""
    deadline = time.monotonic() + 10
    while True:
        committed = dispatcher.local_workers.process_step("agent", component)
        scenario = dispatcher.registry.get_optional("agent")
        assert scenario is not None
        trainer = scenario.trainer_for(component)
        if not committed:
            if not trainer.job_running and not trainer.job_check_waits:
                return
            assert time.monotonic() < deadline, "the training worker did not report the job"
            time.sleep(0.01)


def job_event(record: CommitRecord) -> str | None:
    """The training job event a commit records: submitted, checkpoint, finished or failed; None for no job."""
    metrics = record.metrics or {}
    return metrics["training_job"]["event"] if "training_job" in metrics else None


@pytest.mark.unit
def test_a_weights_job_trainer_and_the_harness_trainer_share_one_snapshot_chain(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The weights trainer commits its job's submission, its checkpoint and its end; the harness commits between them
    on the same chain. Only the submission consumes the job's rows, and a reload recovers each trainer from its own
    commits: the next job starts from the last checkpoint."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert [component_trainer.component for component_trainer in scenario.trainers] == [WEIGHTS, HARNESS]
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    [job] = wait_for_jobs(dispatcher, training, 1)
    assert [item.source_agent_record_ids for item in job.read_data().samples] == [("i1", "r1")]
    assert append_commit(scenario, HARNESS) == [("i1", "r1")]
    for _ in range(2):
        wait_for_serving(dispatcher, "agent")
        result = scenario.next_commit(WEIGHTS)
        assert result is not None
        scenario.commit(result, component=WEIGHTS)
    assert scenario.next_commit(WEIGHTS) is None
    assert component_files(scenario, scenario.current_snapshot()) == {
        WEIGHTS: "weights job 1",
        HARNESS: "harness step 1",
    }

    records = scenario.store.history()
    assert [(record.component, job_event(record)) for record in records] == [
        (WEIGHTS, "submitted"),
        (HARNESS, "submitted"),
        (HARNESS, "checkpoint"),
        (HARNESS, "finished"),
        (WEIGHTS, "checkpoint"),
        (WEIGHTS, "finished"),
    ]
    assert records[0].consumed_ids == {"i1", "r1", consumed_sample_id(stage_item_id("agent", f"{WEIGHTS}:r1:0"))}
    assert records[1].consumed_ids == {"i1", "r1", consumed_sample_id(stage_item_id("agent", f"{HARNESS}:r1:0"))}
    assert all(record.consumed_ids == frozenset() for record in records[2:])
    assert {record.training_job_id for record in records if record.component == WEIGHTS} == {job.job_id}

    reloaded = reload_agent(dispatcher)
    state = JobLoopState.from_dict(reloaded.trainer_for(WEIGHTS).state)
    assert (state.jobs, state.finished_jobs, state.start) == ((), 1, str(tmp_path / "checkpoints" / "1"))
    assert JobLoopState.from_dict(reloaded.trainer_for(HARNESS).state).finished_jobs == 1
    _append_records(reloaded, 2)
    assert append_commit(reloaded, WEIGHTS) == [("i2", "r2")]
    assert wait_for_jobs(dispatcher, training, 2)[-1].start == state.start


@pytest.mark.unit
def test_a_weights_job_waits_for_the_harness_cycle_of_its_scenario(
    tmp_path: Path, dispatchers: _Dispatchers, held_cycle: type[_HeldJobData]
) -> None:
    """A scenario runs one step at a time: the weights worker submits nothing while the harness cycle evaluates, and
    the status still reads. Once the cycle ends, the worker submits the job and commits its checkpoint and its end."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(recipe=held_recipe(HARNESS), training=training)
    scenario = loaded(dispatcher, 1)
    # The training service checked the harness job, so the cycle that the test holds is the one that submits it.
    wait_for_job_check(scenario, HARNESS)
    held_cycle.blocks = True
    harness = threading.Thread(target=lambda: drain(dispatcher, HARNESS))
    harness.start()
    assert held_cycle.evaluating.wait(10)
    dispatcher.local_workers.start("agent", WEIGHTS)
    time.sleep(0.2)
    assert training.submitted == []
    assert scenario.trainer_for(WEIGHTS).pending_job_data is None
    assert dispatcher.build_training_status()["scenarios"]["agent"]["commit_number"] == 0
    held_cycle.release.set()
    harness.join(10)
    assert not harness.is_alive()
    assert became_true(lambda: scenario.commit_number == 6)
    events = [(record.component, job_event(record)) for record in scenario.store.history()]
    # The harness job was submitted first; the weights job's commits may come between the harness job's own.
    assert events[0] == (HARNESS, "submitted")
    for component in (HARNESS, WEIGHTS):
        assert [event for name, event in events if name == component] == ["submitted", "checkpoint", "finished"]
    assert len(training.submitted) == 1


@pytest.mark.unit
def test_a_reload_leaves_the_harness_cycle_in_flight_its_instance(
    monkeypatch: Any, dispatchers: _Dispatchers, held_cycle: type[_HeldJobData]
) -> None:
    """A reload rebuilds the scenario at once; the instance a harness cycle evaluates on stays open until the cycle
    ends, and the cycle then looks again on the rebuilt instance instead of failing."""
    dispatcher, _ = dispatchers.open(recipe=held_recipe(HARNESS))
    scenario = loaded(dispatcher, 1)
    # The training service checked the job, so the cycle that the test holds is the one that submits it.
    wait_for_job_check(scenario, HARNESS)
    held_cycle.blocks = True
    closed_under_the_cycle: list[bool] = []
    close = scenario.close

    def observed_close() -> None:
        closed_under_the_cycle.append(not held_cycle.release.is_set())
        close()

    monkeypatch.setattr(scenario, "close", observed_close)
    outcome: list[bool] = []
    errors: list[BaseException] = []

    def cycle() -> None:
        try:
            outcome.append(dispatcher.local_workers.process_step("agent", HARNESS))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=cycle)
    worker.start()
    assert held_cycle.evaluating.wait(10)
    reloading = threading.Thread(target=lambda: dispatcher.registry.reload("agent"))
    reloading.start()
    reloading.join(5)
    # The reload does not wait for the cycle.
    assert not reloading.is_alive()
    rebuilt = dispatcher.registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    held_cycle.release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert errors == [] and outcome == [True]
    assert closed_under_the_cycle == [False]
    assert dispatcher.build_training_status()["error"] is None
    # The rows the cycle held train on the rebuilt instance.
    drain(dispatcher, HARNESS)
    assert training_components(rebuilt) == [HARNESS]


@pytest.mark.unit
def test_a_harness_backlog_is_kept_for_the_next_start_when_the_service_stops(
    tmp_path: Path, dispatchers: _Dispatchers, held_cycle: type[_HeldJobData]
) -> None:
    """A stopping service takes its routes away first, so a harness cycle's model calls through them fail and would
    come back as a skip that consumes the batch: the cycle ending then commits nothing and no further cycle starts."""
    held_cycle.blocks = True
    # One repository for both starts: the job of the second start trains from a snapshot of the first.
    factory = _repository(tmp_path)
    dispatcher, stopped = dispatchers.open(recipe=held_recipe(HARNESS), backend_factory=factory)
    loaded(dispatcher, 1, 2)
    dispatcher.local_workers.start("agent", HARNESS)
    assert held_cycle.evaluating.wait(10)
    dispatcher.stop_local_cycles()
    closing = threading.Thread(target=dispatcher.close)
    closing.start()
    held_cycle.release.set()
    closing.join(10)
    assert not closing.is_alive()
    assert held_cycle.attempts == 1 and stopped[HARNESS].prepared == 0
    # The next start finds both batches unread and trains them, with no new record to wake the harness.
    restarted, backends = dispatchers.open(backend_factory=factory)
    # What a restart does for every registration a repository lists (this in-memory one lists none).
    restarted.preload_scenarios(("agent",))
    scenario = _scenario(restarted)
    assert became_true(lambda: training_components(scenario).count(HARNESS) == 2)
    assert backends[HARNESS].prepared == 2


@pytest.mark.unit
def test_held_local_cycles_run_nothing_until_the_service_answers(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A service whose recipe calls itself holds its local cycles while it starts: a harness step woken by the
    preload or by a record would fail its model calls into a skip that consumes its batch."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = _scenario(dispatcher, 1)
    dispatcher.preload_scenarios(("agent",))
    for record in _records(2, scenario.serving_artifact_id()):
        dispatcher.accept_record(record)
    time.sleep(0.2)
    assert backends[HARNESS].prepared == 0
    dispatcher.open_local_cycles()
    assert became_true(lambda: training_components(scenario).count(HARNESS) == 2)
    assert backends[HARNESS].prepared == 2


@pytest.mark.unit
def test_the_app_opens_held_local_cycles_once_it_listens(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    _scenario(dispatcher, 1)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_app(dispatcher, open_local_cycles_at=f"http://127.0.0.1:{port}")

    async def serve() -> None:
        server = TestServer(app, host="127.0.0.1", port=port)
        await server.start_server()
        try:
            # Waited for off the loop, which has to answer the app's own probe.
            await asyncio.to_thread(became_true, lambda: backends[HARNESS].prepared > 0)
        finally:
            await server.close()

    asyncio.run(serve())
    assert backends[HARNESS].prepared == 1


@pytest.mark.unit
@pytest.mark.parametrize("cycle_fails", [False, True])
def test_an_instance_parked_while_its_cycle_lock_is_held_closes_when_the_holder_lets_go(
    dispatchers: _Dispatchers, cycle_fails: bool
) -> None:
    """A reload lands while a cycle of the scenario holds its cycle lock: the instance it replaced closes when the
    cycle ends, also when the cycle fails, not at shutdown."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher)
    closed: list[bool] = []
    close = scenario.close

    def observed_close() -> None:
        closed.append(True)
        close()

    scenario.close = observed_close  # type: ignore[method-assign]
    with (
        pytest.raises(RuntimeError, match="cycle failed") if cycle_fails else nullcontext(),
        dispatcher.local_workers.cycle("agent"),
    ):
        dispatcher.registry.reload("agent")
        assert closed == []
        if cycle_fails:
            raise RuntimeError("cycle failed")
    assert closed == [True]
    with dispatcher.local_workers.lock:
        assert dispatcher.local_workers.scenarios["agent"].replaced == []


@pytest.mark.unit
def test_a_failed_request_of_a_scenario_that_is_gone_starts_no_worker(
    monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """Workers started for a deleted name would record errors that nothing clears."""
    dispatcher, _ = dispatchers.open()
    old = _scenario(dispatcher)
    dispatcher.delete_scenario("agent")
    monkeypatch.setattr(old.trainer_for(HARNESS), "fail_pending_request", lambda error: True)
    dispatcher.local_workers.fail_request(old, RuntimeError("rollout failed"), HARNESS)
    with dispatcher.local_workers.lock:
        assert dispatcher.local_workers.workers == {}


@pytest.mark.unit
def test_a_failing_local_cycle_runs_once_per_wake_and_keeps_its_error_in_the_status(
    dispatchers: _Dispatchers, held_cycle: type[_HeldJobData]
) -> None:
    """A cycle that fails costs one attempt per wake, never a loop of reloads, and the failure stays visible."""
    away = held_cycle
    away.fails = True
    dispatcher, _ = dispatchers.open(recipe=held_recipe(HARNESS))
    # Both rows first, then one wake: an accept wakes the worker per row, which would be two wakes.
    loaded(dispatcher, 1)
    dispatcher.local_workers.start("agent", HARNESS)
    assert became_true(lambda: dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away")
    assert away.attempts == 1
    # Another wake: the worker looks again, once.
    dispatcher.local_workers.start("agent", HARNESS)
    assert became_true(
        lambda: away.attempts > 1
        and dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"
    )
    assert away.attempts == 2


@pytest.mark.unit
def test_a_sibling_waiting_at_the_cycle_lock_looks_again_after_a_reload_under_it(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers, held_cycle: type[_HeldJobData]
) -> None:
    """The failing cycle rebuilds the scenario at once, under the lock; the sibling that read the old instance looks again
    on the new one instead of sleeping with its rows unread."""
    # The weights cycle waits for the test to let it go, then fails.
    away = held_cycle
    away.blocks = away.fails = True
    harness = ComponentJobs(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(recipe=held_recipe(WEIGHTS), backends={HARNESS: harness})
    scenario = loaded(dispatcher, 1)
    # The reload after the failure wakes the workers, and a woken weights worker would fail and reload again under
    # the rest of the test; the test runs each cycle itself.
    monkeypatch.setattr(dispatcher.local_workers, "start", lambda scenario, component=None: None)
    failures: list[BaseException] = []

    def fail() -> None:
        try:
            dispatcher.local_workers.process_step("agent", WEIGHTS)
        except RuntimeError as exc:
            failures.append(exc)

    failing = threading.Thread(target=fail)
    failing.start()
    assert away.evaluating.wait(10)
    # The sibling reads the instance, then reaches the cycle lock the failing cycle holds.
    reached = threading.Event()
    cycle_lock = dispatcher.local_workers.cycle_lock
    outcome: list[bool] = []
    sibling = threading.Thread(target=lambda: outcome.append(dispatcher.local_workers.process_step("agent", HARNESS)))

    def observed(name: str):
        lock = cycle_lock(name)
        if threading.current_thread() is sibling:
            reached.set()
        return lock

    monkeypatch.setattr(dispatcher.local_workers, "cycle_lock", observed)
    sibling.start()
    assert reached.wait(10)
    away.release.set()
    failing.join(10)
    sibling.join(10)
    assert [str(error) for error in failures] == ["proposer away"]
    rebuilt = dispatcher.registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert outcome == [True] and harness.prepared == 0
    # Looking again on the rebuilt instance commits the rows that were there all along.
    drain(dispatcher, HARNESS)
    assert training_components(rebuilt) == [HARNESS]


@dataclass(frozen=True)
class _NamedComponentRecipe(Recipe):
    """A recipe serving the one component ``component``, which its trainer does not evolve."""

    component: str = ""

    def default_start(self) -> DefaultStart:
        return seeded_start(self.component)

    def build_components(self, scenario: str) -> ScenarioComponents:
        spec = ComponentSpec(files=TextFileTree()) if self.component == HARNESS else ComponentSpec()
        return ScenarioComponents(components={self.component: spec})


def named_components_recipe(names: tuple[str, ...]) -> CompositeRecipe:
    """A composite recipe serving the named components."""
    return CompositeRecipe(components={name: _NamedComponentRecipe(component=name) for name in names})


@pytest.mark.unit
def test_a_recipe_serving_fewer_components_than_registered_is_refused(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A step carries forward only the components the recipe serves; one it does not serve would leave every snapshot."""
    names = ("weights", "harness", "config")
    factory = _repository(tmp_path)
    first, _ = dispatchers.open(recipe=named_components_recipe(names), live=LiveObjects(), backend_factory=factory)
    assert first.get_or_create_scenario("agent") is not None
    first.close()
    second, _ = dispatchers.open(
        recipe=named_components_recipe(("weights", "harness")), live=LiveObjects(), backend_factory=factory
    )
    with pytest.raises(ReefError, match=r"does not serve \['config'\]"):
        second.get_or_create_scenario("agent")


@pytest.mark.unit
def test_a_weights_job_commits_while_the_harness_cycle_fails(
    tmp_path: Path, dispatchers: _Dispatchers, held_cycle: type[_HeldJobData]
) -> None:
    """The harness cycle fails and the failure shows; the weights job still lands. Then the harness's rows are
    stale, since the job's checkpoint serves, so the harness stops trying them.

    Each wake before then costs one attempt, and the job's commits and queue changes each wake the scenario, so
    the count of attempts depends on how the wakes group."""
    away = held_cycle
    away.fails = True
    training = HeldJobs(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(recipe=held_recipe(HARNESS), training=training)
    scenario = _scenario(dispatcher)
    for record in _records(1, scenario.serving_artifact_id()):
        dispatcher.accept_record(record)
    assert became_true(lambda: dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away")
    training.finish(submitted_jobs(training, 1)[0])
    assert became_true(
        lambda: "finished" in [job_event(row) for row in _scenario(dispatcher).store.history()],
        timeout_seconds=15,
    )
    # A wake that was already due can still land after the job finishes, so the test waits for the count to hold
    # still for a second; a harness that keeps trying never lets it.
    assert away.attempts >= 1
    held_since, attempts = time.monotonic(), away.attempts

    def settled() -> bool:
        nonlocal held_since, attempts
        if away.attempts != attempts:
            held_since, attempts = time.monotonic(), away.attempts
        return time.monotonic() - held_since >= 1.0

    assert became_true(settled)


@pytest.mark.unit
@pytest.mark.parametrize("select_before_commit", [False, True])
def test_a_second_report_on_a_trained_inference_is_settled_after_a_restart(
    dispatchers: _Dispatchers, select_before_commit: bool
) -> None:
    """The restarted harness settles a second report on a trained source; weights keep independent consumption."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    scenario.records.append(agent_report("r1b", "i1"))
    if select_before_commit:
        make_samples(scenario)
    # The job trains r1a alone; r1b references the same source and makes no new sample.
    assert append_commit(scenario, HARNESS, make=False) is not None
    assert scenario.records.count("agent") == 3
    assert scenario.next_commit(HARNESS) is None
    rebuilt = reload_agent(dispatcher)
    make_samples(rebuilt)
    assert rebuilt.next_commit(HARNESS) is None
    # Selection before the commit can persist the skipped report as consumed.
    harness = rebuilt.trainer_for(HARNESS)
    assert "r1b" in harness.skipped_record_ids | harness.consumed_record_ids
    assert "r1b" not in rebuilt.trainer_for(WEIGHTS).consumed_record_ids


@pytest.mark.unit
def test_a_report_settled_between_two_commits_stays_settled_after_a_restart(dispatchers: _Dispatchers) -> None:
    """r1b arrived after commit A trained i1 and before commit B: the replay settles it at A's watermark, as it was."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, HARNESS) is not None
    scenario.records.append(agent_report("r1b", "i1"))
    assert scenario.next_commit(HARNESS) is None
    _append_records(scenario, 2)
    assert append_commit(scenario, HARNESS) is not None
    rebuilt = reload_agent(dispatcher)
    assert rebuilt.next_commit(HARNESS) is None
    # Settled, not live: commit B names it as consumed, so the rebuilt harness skips it.
    assert "r1b" in rebuilt.trainer_for(HARNESS).consumed_record_ids


@pytest.mark.unit
def test_a_report_on_an_inference_the_weights_job_trained_is_skipped_not_raised(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The weights job's submission holds i1 while the harness commits and r1b arrives; the submission's commit
    consumes i1 with r1b unread by the weights trainer. Every later read skips r1b, on the live path and after a
    restart."""
    dispatcher, _ = dispatchers.open(training=WeightsJobComponent(tmp_path / "checkpoints"), hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    submitted = checked_commit(scenario, WEIGHTS)
    assert submitted is not None
    assert append_commit(scenario, HARNESS) is not None
    scenario.records.append(agent_report("r1b", "i1"))
    assert scenario.next_commit(HARNESS) is None
    scenario.commit(submitted, component=WEIGHTS)
    settle_weights_job(scenario)
    wait_for_serving(dispatcher, "agent")
    make_samples(scenario)
    assert scenario.records.read("agent", "i1") is not None
    assert scenario.next_commit(WEIGHTS) is None
    weights = scenario.trainer_for(WEIGHTS)
    # The samples worker can skip the report before the job's final commit consumes the skipped ID.
    assert "r1b" in weights.skipped_record_ids | weights.consumed_record_ids
    _append_records(scenario, 2)
    assert append_commit(scenario, HARNESS) is not None
    rebuilt = reload_agent(dispatcher)
    assert rebuilt.next_commit(HARNESS) is None
    assert append_commit(rebuilt, WEIGHTS) == [("i2", "r2")]


class _SlotGrouping(ReportGrouping):
    """Reports batch by the group and the retry slot named in their metadata; a group is ready at two slots."""

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload["metadata"]
        return metadata["group"], metadata["slot"]

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.INCOMPLETE


class _SlottedSamples(ThresholdSamples):
    """The threshold stage with ``_SlotGrouping``."""

    def __init__(self, context: PipelineContext, settings: SampleAssemblySettings | None = None) -> None:
        super().__init__(context, settings, grouping=_SlotGrouping())


def slotted_recipe() -> CompositeRecipe:
    """The two trainer recipe whose harness batches reports by group and retry slot."""
    return two_trainer_recipe(harness=harness_recipe(component_stages(record2sample=_SlottedSamples)))


def slotted_report(record_id: str, reference: str, group: str, slot: int) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [reference], "metadata": {"group": group, "slot": slot}},
        agent_record_id=record_id,
        references=(reference,),
    )


@pytest.mark.unit
def test_a_retry_settled_for_a_taken_slot_stays_settled_after_a_restart(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """r1dup retried slot 0 while r1 held it, so the harness settled it; the harness commit names r1dup as consumed
    with the rows it trained. The replay skips r1 and r1dup alike, so the rebuilt harness keeps r1dup out of G."""
    dispatcher, _ = dispatchers.open(recipe=slotted_recipe(), hold_local_cycles=True)
    scenario = loaded(dispatcher)
    for step in (1, 2, 3):
        scenario.records.append(_records(step, scenario.serving_artifact_id())[0])
    scenario.records.append(slotted_report("r1", "i1", "G", 0))
    scenario.records.append(slotted_report("r1dup", "i3", "G", 0))
    scenario.records.append(slotted_report("r2", "i2", "G", 1))
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    assert append_commit(scenario, HARNESS) == [("i1", "r1"), ("i2", "r2")]
    assert scenario.records.read("agent", "r1dup") is not None
    rebuilt = reload_agent(dispatcher)
    rebuilt.records.append(_records(4, rebuilt.serving_artifact_id())[0])
    rebuilt.records.append(slotted_report("r4", "i4", "G", 2))
    # Live, slot 0 was never r1dup's: G holds r4 alone and waits.
    assert rebuilt.next_commit(HARNESS) is None
    assert "r1dup" in submission(rebuilt, HARNESS).consumed_ids


def fail_first_record_of(monkeypatch: pytest.MonkeyPatch, scenario: Scenario, component: str) -> list[bool]:
    """Make ``component``'s next commit record fail once, after its trainer acknowledged the batch."""
    committer = scenario._committer
    original = committer._append_commit_record
    failed: list[bool] = []

    def append(**kwargs: Any) -> Any:
        if kwargs.get("component") == component and not failed:
            failed.append(True)
            raise OSError("transient store write error")
        return original(**kwargs)

    monkeypatch.setattr(committer, "_append_commit_record", append)
    return failed


@pytest.mark.unit
def test_a_harness_commit_that_fails_leaves_its_rows_to_the_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """The harness acknowledges its batch before its record is durable. When that commit fails, the scenario
    reloads at once, and the weights job submitted after it leaves the harness's rows stored. The harness's sample
    of them is an item of the scenario, and the harness consumed the rows when it saved it: the rebuilt harness
    trains the sample once."""
    dispatcher, _ = dispatchers.open(training=WeightsJobComponent(tmp_path / "checkpoints"))
    # Recovery reloads wake workers; this test drives each cycle explicitly.
    monkeypatch.setattr(dispatcher.local_workers, "start", lambda scenario, component=None: None)
    scenario = loaded(dispatcher, 1)
    fail_first_record_of(monkeypatch, scenario, HARNESS)
    with pytest.raises(OSError, match="transient store write error"):
        drain(dispatcher, HARNESS)
    rebuilt = dispatcher.registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert append_commit(rebuilt, WEIGHTS) == [("i1", "r1")]
    assert rebuilt.records.read("agent", "i1") is not None and rebuilt.records.read("agent", "r1") is not None
    drain(dispatcher, HARNESS)
    current = dispatcher.registry.get_optional("agent")
    assert current is not None
    history = current.store.history()
    # A job's submission consumes its rows; its checkpoint and its end consume none.
    submissions = [record for record in history if job_event(record) == "submitted"]
    assert [sorted(record.consumed_ids) for record in submissions if record.component == HARNESS] == [
        ["i1", "r1", consumed_sample_id(stage_item_id("agent", f"{HARNESS}:r1:0"))]
    ]
    # Likewise, the weights trainer's sample was stored before the reload: its submission names it and its rows.
    assert [sorted(record.consumed_ids) for record in submissions if record.component == WEIGHTS] == [
        ["i1", "r1", consumed_sample_id(stage_item_id("agent", f"{WEIGHTS}:r1:0"))]
    ]


@pytest.mark.unit
def test_a_weights_commit_that_fails_leaves_its_batch_to_the_weights_trainer_after_a_harness_commit(
    monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """The reverse: a weights commit that fails after its acknowledgment, then a harness commit before the weights
    trainer's reload. The harness commit leaves the rows stored, so the rebuilt weights trainer trains the batch its
    failed commit had acknowledged."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, HARNESS) == [("i1", "r1")]
    _append_records(scenario, 2)
    fail_first_record_of(monkeypatch, scenario, WEIGHTS)
    with pytest.raises(OSError, match="transient store write error"):
        append_commit(scenario, WEIGHTS)
    assert append_commit(scenario, HARNESS) == [("i2", "r2")]
    assert scenario.records.read("agent", "i1") is not None and scenario.records.read("agent", "r1") is not None
    rebuilt = reload_agent(dispatcher)
    assert append_commit(rebuilt, WEIGHTS) == [("i1", "r1")]


class _RetiringRetriesJudging(RecordJudging):
    """A turn is judged once its successor arrives, and a retry of a turn already seen is retired, as OpenClaw-RL
    retires a client's duplicate turn."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def observe(self, item: AgentRecord, now: float, stage: JudgedSamples) -> None:
        if item.payload.get("completes"):
            stage.dispatch(Judgment(item.payload["completes"]))
        if item.payload["turn"] in self.seen:
            stage.retire(item.agent_record_id)
            return
        self.seen.add(item.payload["turn"])
        stage.track(item)

    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        return job


class _RetiringRetriesSamples(JudgedSamples):
    """One-token stub samples for every judged turn of a ``_RetiringRetriesJudging``, each judged at the next poll."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(_RetiringRetriesJudging(), JudgedAtOnce())

    def is_stale(self, samples: Sequence[Sample], served: ServedArtifacts) -> bool:
        return lags_behind(samples, served, MAX_ARTIFACT_LAG)

    def make_sample(self, record: AgentRecord, judgment: Judgment) -> Any:
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            artifact_id=record.payload["artifact_id"],
        )


class RetiringRetriesJobData(ThresholdJobData):
    """Batches the judged turns of ``_RetiringRetriesJudging``."""

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


def computed_weights_recipe() -> CompositeRecipe:
    """The two trainer recipe whose weights trainer computes its own feedback: it trains the judged turns of
    ``_RetiringRetriesJudging``."""
    stages = component_stages(_RetiringRetriesSamples, RetiringRetriesJobData, record2sample_config={})
    return two_trainer_recipe(weights=WeightsRecipe(pipeline_stages=stages))


def turn_record(scenario: Scenario, record_id: str, turn: str, completes: str | None = None) -> AgentRecord:
    """The record of one turn, answered by the artifact that ``scenario`` serves now."""
    payload: dict[str, Any] = {
        "turn": turn,
        "tokens": [1, 2],
        "loss_mask": [0, 1],
        "rollout_log_probs": [-0.2],
        "artifact_id": scenario.serving_artifact_id(),
    }
    if completes:
        payload["completes"] = completes
    return AgentRecord.create(
        scenario="agent", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


@pytest.mark.unit
@pytest.mark.parametrize("reload", [False, True])
def test_a_retry_a_computed_processor_retired_never_trains_after_a_reload(
    tmp_path: Path, reload: bool, dispatchers: _Dispatchers
) -> None:
    """A retries turn t1 after A trained; the weights processor retires the retry, and its next commit names the
    retry as consumed. A reload keeps it out, so a later turn that completes the retry finds nothing to judge, as on
    the live path."""
    dispatcher, _ = dispatchers.open(recipe=computed_weights_recipe(), hold_local_cycles=True)
    scenario = loaded(dispatcher)
    scenario.records.append(turn_record(scenario, "A", "t1"))
    scenario.records.append(turn_record(scenario, "B", "t2", completes="A"))
    assert append_commit(scenario, WEIGHTS) == [("A",)]
    scenario.records.append(turn_record(scenario, "A-retry", "t1"))
    scenario.records.append(turn_record(scenario, "C", "t3", completes="B"))
    assert append_commit(scenario, WEIGHTS) == [("B",)]
    assert "A-retry" in submission(scenario, WEIGHTS).consumed_ids
    if reload:
        scenario = reload_agent(dispatcher)
    scenario.records.append(turn_record(scenario, "E", "t4", completes="A-retry"))
    assert append_commit(scenario, WEIGHTS) is None


@pytest.mark.unit
def test_a_run_report_on_an_inference_both_trainers_trained_is_skipped_by_both(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """run references i1 and i2 and is stored before the weights job's submission commits i1, which the harness
    trained already. Each trainer then skips run, since one of its sources trained, instead of raising on it."""
    dispatcher, _ = dispatchers.open(training=WeightsJobComponent(tmp_path / "checkpoints"), hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    scenario.records.append(_records(2, scenario.serving_artifact_id())[0])
    submitted = checked_commit(scenario, WEIGHTS)
    assert submitted is not None
    assert append_commit(scenario, HARNESS) is not None
    scenario.records.append(agent_report("run", "i1", "i2"))
    scenario.commit(submitted, component=WEIGHTS)
    settle_weights_job(scenario)
    wait_for_serving(dispatcher, "agent")
    make_samples(scenario)
    assert scenario.next_commit(HARNESS) is None
    assert scenario.next_commit(WEIGHTS) is None
    for component in (WEIGHTS, HARNESS):
        trainer = scenario.trainer_for(component)
        assert "run" in trainer.skipped_record_ids | trainer.consumed_record_ids, component


@pytest.mark.unit
def test_an_activation_of_the_creation_after_a_rejected_first_step(dispatchers: _Dispatchers) -> None:
    """A rejected step records the creation without a checkpoint; the creation still has its own bytes."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher)
    creation = scenario.current_snapshot().revision
    backends[HARNESS].reject_next = True
    for step in (1, 2):
        _append_records(scenario, step)
        assert append_commit(scenario, HARNESS) is not None
    assert scenario.current_snapshot().revision != creation
    assert (
        next(
            row
            for row in scenario.snapshot_queries.snapshot_rows(scenario.commit_number)
            if row["operation"] == "creation"
        )["restorable"]
        is True
    )
    scenario.activate(component_id(scenario, scenario.repository.start_artifact.revision, HARNESS), HARNESS)
    assert component_files(scenario, scenario.current_snapshot())[HARNESS] == "harness seed"


@pytest.mark.unit
def test_a_job_submitted_on_an_instance_reloaded_under_it_is_the_same_job_on_the_new_instance(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The old instance's submission is never committed. The rebuilt trainer reserves the same rows, whose job has
    the same ID, so the component runs it once and the new instance commits it."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    old = loaded(dispatcher, 1)
    assert checked_commit(old, WEIGHTS) is not None
    [job] = submitted_jobs(training, 1)
    new = reload_agent(dispatcher)
    assert new is not old
    assert new.store.history() == ()
    assert append_commit(new, WEIGHTS) == [("i1", "r1")]
    assert training.submitted == [job]
    assert new.store.history()[-1].training_job_id == job.job_id


@pytest.mark.unit
def test_training_mode_switches_every_trainer_or_none(dispatchers: _Dispatchers) -> None:
    """A mode one component cannot run is refused before any component switches."""
    dispatcher, _ = dispatchers.open(auto_only_components=frozenset({HARNESS}))
    scenario = _scenario(dispatcher)
    with pytest.raises(NotImplementedError, match=r"harness \(AutoOnlyThresholdJobData\)"):
        scenario.set_training_mode("manual")
    assert [component_trainer.trainer.pipeline.training_mode for component_trainer in scenario.trainers] == [
        "auto",
        "auto",
    ]
    with pytest.raises(ValueError, match="training_mode must be 'auto' or 'manual'"):
        scenario.set_training_mode("hybrid")


@dataclass(frozen=True)
class _FlatStartRecipe(Recipe):
    """A recipe that declares the weights and the harness component itself and starts a scenario from ``files``. A
    composite recipe keeps each component's start under the component's directory, so only such a recipe can keep
    a file outside them."""

    files: Mapping[str, str] = field(default_factory=dict)

    def default_start(self) -> DefaultStart:
        return DefaultStart(files=dict(self.files))

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(components={WEIGHTS: ComponentSpec(), HARNESS: ComponentSpec(files=TextFileTree())})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layout", "outside"),
    [
        # Files at the root of the default start belong to no component and would be carried forward by none.
        ({"harness.txt": "harness seed"}, "harness.txt"),
        # A component keeps a directory; a file of that name would fail on every component view.
        ({f"{HARNESS}/{HARNESS}.txt": "harness seed", WEIGHTS: "not a directory"}, WEIGHTS),
    ],
)
def test_composite_registration_refuses_files_outside_its_components(
    tmp_path: Path, dispatchers: _Dispatchers, layout: dict[str, str], outside: str
) -> None:
    dispatcher, _ = dispatchers.open(recipe=_FlatStartRecipe(files=layout), live=LiveObjects())
    with pytest.raises(ReefError, match=rf"keeps \['{outside}'\] outside its components"):
        dispatcher.get_or_create_scenario("agent")
    # A refused root registers nothing.
    assert not dispatcher.has_scenario("agent")


@pytest.mark.unit
def test_composite_registration_starts_an_unseeded_component_empty(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A default start that seeds only the harness starts the weights component empty; it fills at its first step."""
    unseeded = ComponentJobs(WEIGHTS, tmp_path / "candidates", seeded=False)
    dispatcher, _ = dispatchers.open(hold_local_cycles=True, backends={WEIGHTS: unseeded})
    scenario = loaded(dispatcher)
    base = scenario.repository.materialize(scenario.current_snapshot())
    assert base.components is not None and base.components.names == (WEIGHTS, HARNESS)
    assert TextFileTree().read_files(base.component(WEIGHTS)) is None
    _append_records(scenario, 1)
    assert append_commit(scenario, WEIGHTS) is not None
    assert component_files(scenario, scenario.current_snapshot()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness seed",
    }


@pytest.mark.unit
def test_composite_recovery_refuses_a_registration_without_a_component_manifest(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A scenario registered by a recipe serving one component is not read as a composed one."""
    backend_factory = _repository(tmp_path)
    backend = backend_factory("agent")
    (tmp_path / "flat-root").mkdir()
    # Registration as a recipe serving one component writes it: no component manifest.
    backend.register(
        Artifact.local(tmp_path / "flat-root"),
        metadata={SCENARIO_METADATA_KEY: scenario_metadata_for(name="agent", source=RootSource("default"))},
    )
    dispatcher, _ = dispatchers.open(backend_factory=backend_factory)
    with pytest.raises(ReefError, match="registered without a component manifest"):
        dispatcher.get_or_create_scenario("agent")


@pytest.mark.unit
def test_the_samples_service_of_another_process_follows_the_commits_of_control(dispatchers: _Dispatchers) -> None:
    """Control commits from the sample store, and its stage reads no record. The stage of the other process ends the
    records, control's commit names them, and that stage then reads the job's end and releases them."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    control = loaded(dispatcher)
    samples_process = dispatcher.registry._scenario_factory.load_or_create("agent")
    try:
        ended: list[str] = []
        pipeline = samples_process.trainer_for(WEIGHTS).pipeline
        original = pipeline.job_ended
        pipeline.job_ended = lambda outcome: (ended.append(outcome.event), original(outcome))[1]
        stage = pipeline.record2sample
        control_stage = control.trainer_for(WEIGHTS).pipeline.record2sample
        _append_records(control, 1)
        samples_process.produce_samples(WEIGHTS)
        assert {"i1", "r1"} <= stage.releasable_record_ids()
        assert control_stage.releasable_record_ids() == frozenset()

        assert append_commit(control, WEIGHTS, make=False) == [("i1", "r1")]
        assert {"i1", "r1"} <= submission(control, WEIGHTS).consumed_ids
        assert control_stage.releasable_record_ids() == frozenset()
        assert ended == [], "the stage reads a job's end in the samples service's step"

        samples_process.produce_samples(WEIGHTS)
        assert ended == ["finished"]
        assert not {"i1", "r1"} & stage.releasable_record_ids()
        samples_process.produce_samples(WEIGHTS)
        assert ended == ["finished"], "a job's end is read once"
    finally:
        samples_process.close()


@pytest.mark.unit
def test_rows_one_trainer_consumed_before_a_restart_still_train_the_other(dispatchers: _Dispatchers) -> None:
    """A restart between two trainers' commits of the same rows: the second still trains them, the first never again."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, HARNESS) is not None

    reloaded = reload_agent(dispatcher)
    assert reloaded.next_commit(HARNESS) is None
    assert append_commit(reloaded, WEIGHTS) is not None
    # The weights trainer's stage ended the rows and saved its sample of them before the restart: the store keeps
    # both, so the commit names them.
    assert submission(reloaded, WEIGHTS).consumed_ids == {
        "i1",
        "r1",
        consumed_sample_id(stage_item_id("agent", f"{WEIGHTS}:r1:0")),
    }
    assert reloaded.records.count("agent") == 2


@pytest.mark.unit
def test_a_weights_step_marks_the_older_weights_samples_stale_and_the_harness_still_trains_them(
    dispatchers: _Dispatchers,
) -> None:
    """Weights and harness train from one stream of records. The weights stage is on-policy; the harness stage reads
    traces, which never go stale. Once a weights step activates new weights, the samples service marks the weights'
    sample of a row that the old weights answered stale, and the harness still trains that row."""
    weights = WeightsRecipe(pipeline_stages=component_stages(record2sample_config={}))
    harness = harness_recipe(job_sized_stages(PipelineSpec(TraceSamples, TraceJobData)))
    recipe = two_trainer_recipe(weights=weights, harness=harness)
    dispatcher, _ = dispatchers.open(recipe=recipe, hold_local_cycles=True)
    scenario = loaded(dispatcher, 1, 2)
    root = scenario.serving_artifact_id()
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    assert scenario.serving_artifact_id() != root
    assert became_true(lambda: scenario.served_artifact_id() == scenario.serving_artifact_id())

    make_samples(scenario)
    stale = {stored.sample_id: stored.stale for stored in scenario.trainer_for(WEIGHTS).pipeline.store.store.page()}
    assert stale[stage_item_id("agent", f"{WEIGHTS}:r2:0")], "the old weights answered row 2"
    assert not stale[stage_item_id("agent", f"{HARNESS}:r2:0")], "a trace never goes stale"
    assert append_commit(scenario, WEIGHTS) is None
    assert append_commit(scenario, HARNESS) is not None
    assert append_commit(scenario, HARNESS) is not None
    assert scenario.trainer_for(HARNESS).state["finished_jobs"] == 2


class NoSelectionProgress(Selection):
    """A selection whose progress nobody reads."""

    def progress(self, records_read: int, samples_created: int) -> None:
        return


def on_policy_weights(dispatchers: _Dispatchers, **options: Any) -> tuple[Dispatcher, Scenario]:
    """A dispatcher whose weights trainer is on-policy and that runs no samples service: the test makes the samples.
    Its scenario serves, and holds no row yet."""
    weights = WeightsRecipe(pipeline_stages=component_stages(record2sample_config={}))
    dispatcher, _ = dispatchers.open(
        recipe=two_trainer_recipe(weights=weights),
        hold_local_cycles=True,
        components=frozenset(SERVICE_COMPONENTS) - {"samples"},
        **options,
    )
    scenario = _scenario(dispatcher)
    wait_for_serving(dispatcher, "agent")
    return dispatcher, scenario


@pytest.mark.unit
def test_a_rollback_to_an_artifact_judged_before_waits_for_a_new_judgment(dispatchers: _Dispatchers) -> None:
    """A serves; a weights step activates B; row 2, answered by B, is judged fresh. An activation of A makes the
    history A, B, A: row 2 is one artifact behind. No automatic job takes it before the samples service judges the
    new history, and that judgment marks it stale."""
    dispatcher, scenario = on_policy_weights(dispatchers)
    root = scenario.serving_artifact_id()
    _append_records(scenario, 1)
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    _append_records(scenario, 2)
    make_samples(scenario)
    pipeline = scenario.trainer_for(WEIGHTS).pipeline
    row_two = stage_item_id("agent", f"{WEIGHTS}:r2:0")
    assert [stored.stale for stored in pipeline.store.read([row_two])] == [False]

    dispatcher.activate("agent", root, WEIGHTS)
    assert became_true(lambda: scenario.served_artifact_id() == scenario.serving_artifact_id())
    assert not pipeline.judged(scenario.served_artifacts())
    assert checked_commit(scenario, WEIGHTS) is None, "no job before the new judgment"
    make_samples(scenario)
    assert [stored.stale for stored in pipeline.store.read([row_two])] == [True]
    assert checked_commit(scenario, WEIGHTS) is None, "row 2 is stale and nothing else waits"


@pytest.mark.unit
def test_a_samples_service_in_its_own_process_judges_control_activations(dispatchers: _Dispatchers) -> None:
    """Control and a samples-only process share the storage. After a weights commit in control activates new
    weights, the samples process judges the new history without a reload, and control's next automatic job runs."""
    repository = _repository(dispatchers.tmp_path)
    _, scenario = on_policy_weights(dispatchers, backend_factory=repository)
    weights = WeightsRecipe(pipeline_stages=component_stages(record2sample_config={}))
    samples_process, _ = dispatchers.open(
        recipe=two_trainer_recipe(weights=weights),
        hold_local_cycles=True,
        components=frozenset({"samples"}),
        backend_factory=repository,
    )
    producers = LocalSampleProducers(samples_process.registry, None)

    def produce() -> None:
        producers.produce("agent", WEIGHTS, NoSelectionProgress("selection"))

    _append_records(scenario, 1)
    produce()
    assert append_commit(scenario, WEIGHTS, make=False) == [("i1", "r1")]
    assert became_true(lambda: scenario.served_artifact_id() == scenario.serving_artifact_id())
    pipeline = scenario.trainer_for(WEIGHTS).pipeline
    assert not pipeline.judged(scenario.served_artifacts())
    _append_records(scenario, 2)
    produce()
    assert pipeline.judged(scenario.served_artifacts())
    assert append_commit(scenario, WEIGHTS, make=False) == [("i2", "r2")]


@pytest.mark.unit
def test_a_weights_step_is_no_new_harness_head(dispatchers: _Dispatchers) -> None:
    """Clients pull the tree: a step that carried it forward unchanged is not announced or listed as a snapshot."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    assert append_commit(scenario, HARNESS) is not None
    harness_snapshot = scenario.current_snapshot().revision
    assert append_commit(scenario, WEIGHTS) is not None
    assert scenario.current_snapshot().revision != harness_snapshot
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, harness_snapshot, HARNESS)
    manifest = service.harness_manifest(headers)
    assert manifest["artifact_id"] == component_id(scenario, harness_snapshot, HARNESS)
    assert manifest["files"] == {"harness.txt": "harness step 1"}
    catalog = service.harness_artifacts(headers)["artifacts"]
    assert [row["operation"] for row in catalog] == ["creation", "training"]
    # The root snapshot made the creation's artifacts; the harness step made its own.
    base = scenario.repository.start_artifact.revision
    assert [row["artifact_id"] for row in catalog] == [
        component_id(scenario, base, HARNESS),
        component_id(scenario, harness_snapshot, HARNESS),
    ]
    assert [row["active"] for row in catalog] == [False, True]


@pytest.mark.unit
def test_each_components_step_reaches_the_tracker_under_its_name(dispatchers: _Dispatchers) -> None:
    """The event the dispatcher records for a step names the trainer that made it."""
    tracker = _RecordingTracker()
    dispatcher, _ = dispatchers.open(experiment_tracker=tracker, hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, HARNESS) is not None
    assert append_commit(scenario, WEIGHTS) is not None
    dispatcher.registry.experiment_reports.flush()
    # Each job's submission, checkpoint and end is one event, named by its trainer's component and training component.
    assert [event.context.component for event in tracker.events] == [HARNESS] * 3 + [WEIGHTS] * 3
    assert [event.context.backend for event in tracker.events] == [HARNESS] * 3 + [WEIGHTS] * 3
    assert [event.context.commit_number for event in tracker.events] == [1, 2, 3, 4, 5, 6]
    assert [row.get("component") for row in scenario.snapshot_queries.snapshot_rows(scenario.commit_number)] == [
        WEIGHTS
    ] * 3 + [HARNESS] * 3 + [None]


@pytest.mark.unit
def test_the_harness_catalog_is_one_lineage_of_trees(dispatchers: _Dispatchers) -> None:
    """A rejected step, an unlisted weights step and an activation never break the chain or list a new row."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    creation = scenario.current_snapshot().revision
    _append_records(scenario, 1)
    requires = {"training_request": {"id": "r1", "requires": [{"name": "TOKEN", "kind": "env"}]}}
    assert append_commit(scenario, HARNESS, checkpoint_metrics=requires) is not None
    h1 = scenario.current_snapshot().revision
    assert append_commit(scenario, WEIGHTS) is not None
    w1 = scenario.current_snapshot().revision

    # A rejected harness step is listed for its page, but it published nothing and names no head.
    _append_records(scenario, 2)
    backends[HARNESS].reject_next = True
    assert append_commit(scenario, HARNESS) is not None
    assert not any(record.made_artifact for record in scenario.store.history()[-2:])
    assert scenario.current_snapshot().revision == w1
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, h1, HARNESS)
    catalog = service.harness_artifacts(headers)["artifacts"]
    # The rejected row is named by the listed snapshot it ran on, so a client's poll agrees with the head;
    # like a flat scenario's rejected row, it lists the head's parent, not itself.
    assert listed_revisions(scenario) == [creation, h1, h1]
    assert catalog[-1]["metrics"]["selected"] is False
    assert "ran on" in service.harness_artifact_page(headers, 2).lower()
    assert service.harness_manifest(headers, form="source")["artifact_id"] == component_id(scenario, h1, HARNESS)

    # The next harness artifact descends from h1 in the listed chain; the weights artifact it was published
    # on stays under another name, and the requires chain walks the listed rows.
    assert append_commit(scenario, WEIGHTS) is not None
    _append_records(scenario, 3)
    assert append_commit(scenario, HARNESS) is not None
    h2 = scenario.current_snapshot().revision
    catalog = service.harness_artifacts(headers)["artifacts"]
    assert listed_revisions(scenario) == [creation, h1, h1, h2]
    assert required_by(catalog, component_id(scenario, h2, HARNESS)) == [{"name": "TOKEN", "kind": "env"}]
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, h2, HARNESS)
    page = service.harness_artifact_page(headers, 3)
    assert component_id(scenario, h2, HARNESS) in page
    assert h2 not in page

    # Activating h1's harness serves h1's tree again on the weights served now: the activation publishes that
    # combination, which the catalog does not list, and the head is the listed snapshot h1.
    scenario.activate(component_id(scenario, h1, HARNESS), HARNESS)
    activated_view = scenario.current_snapshot().revision
    assert activated_view not in (h1, h2)
    catalog = service.harness_artifacts(headers)["artifacts"]
    assert listed_revisions(scenario) == [creation, h1, h1, h2]
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, h1, HARNESS)
    assert required_by(catalog, component_id(scenario, h1, HARNESS)) == [{"name": "TOKEN", "kind": "env"}]

    # Weights held for review are activated as weights: the served tree stays, so the head stays a listed
    # snapshot and names no weights artifact or combination.
    _append_records(scenario, 4)
    backends[WEIGHTS].hold_next = True
    assert append_commit(scenario, WEIGHTS) is not None
    checkpoint = next(record for record in reversed(scenario.store.history()) if record.made_artifact)
    assert checkpoint.activated is False
    held = checkpoint.snapshot.revision
    _append_records(scenario, 5)
    assert append_commit(scenario, HARNESS) is not None
    h3 = scenario.current_snapshot().revision
    scenario.activate(component_id(scenario, held, WEIGHTS), WEIGHTS)
    assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights step 3"
    catalog = service.harness_artifacts(headers)["artifacts"]
    assert listed_revisions(scenario) == [creation, h1, h1, h2, h3]
    assert service.harness_manifest(headers, form="source")["artifact_id"] == component_id(scenario, h3, HARNESS)
    assert required_by(catalog, component_id(scenario, h3, HARNESS)) == [{"name": "TOKEN", "kind": "env"}]


@pytest.mark.unit
def test_a_training_step_carries_the_active_artifact_and_keeps_an_activated_component(
    dispatchers: _Dispatchers,
) -> None:
    """After an older weights artifact is activated, a harness step carries the active weights, not the chain tip's."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, WEIGHTS) is not None
    assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights step 1"
    scenario.activate(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
    wait_for_serving(dispatcher, "agent")
    assert append_commit(scenario, HARNESS) is not None
    assert component_files(scenario, scenario.current_snapshot()) == {
        WEIGHTS: "weights seed",
        HARNESS: "harness step 1",
    }
    active = {row["component"]: row["artifact_id"] for row in scenario.artifacts() if row["active"]}
    assert active == {
        WEIGHTS: component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS),
        HARNESS: component_id(scenario, scenario.current_snapshot().revision, HARNESS),
    }


@pytest.mark.unit
def test_the_composite_harness_catalog_marks_one_active_row_and_a_client_activates_a_row(
    dispatchers: _Dispatchers,
) -> None:
    """Each catalog row names its harness artifact; exactly one row is active, and posting a row's artifact_id and
    component as the ``served`` mapping of the update route makes that row the active one."""
    from aiohttp.test_utils import TestClient

    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, HARNESS) is not None
    assert append_commit(scenario, WEIGHTS) is not None
    _append_records(scenario, 2)
    assert append_commit(scenario, HARNESS) is not None
    headers = {"x-reef-scenario": "agent"}

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            rows = (await (await client.get("/reef/harness/artifacts", headers=headers)).json())["artifacts"]
            assert [row["active"] for row in rows] == [False, False, True]
            assert all(row["component"] == HARNESS and row["activated"] for row in rows)
            first = rows[1]
            body = {"served": {first["artifact_id"]: 100}, "component": first["component"]}
            before = (await (await client.get("/reef/scenarios/agent/settings")).json())["served"]
            assert before[HARNESS] == {rows[2]["artifact_id"]: 100} and set(before) == {HARNESS, WEIGHTS}
            # A scenario of two components needs the component, and the artifact must be one of that component.
            unnamed = await client.post("/reef/scenarios/agent/update", json={"served": body["served"]})
            assert unnamed.status == 400 and "name the component" in await unnamed.text()
            # The next automatic job is one trainer's, so a scenario of several components names the component.
            unnamed = await client.get("/reef/scenarios/agent/next-job")
            assert unnamed.status == 400 and "name one with component" in await unnamed.text()
            assert (await client.get(f"/reef/scenarios/agent/next-job?component={WEIGHTS}")).status == 200
            other = await client.post("/reef/scenarios/agent/update", json={**body, "component": WEIGHTS})
            assert other.status == 404
            assert (await (await client.get("/reef/scenarios/agent/settings")).json())["served"] == before
            answer = await client.post("/reef/scenarios/agent/update", json=body)
            assert answer.status == 200, await answer.text()
            assert (await answer.json())["served"] == {first["artifact_id"]: 100}
            rows = (await (await client.get("/reef/harness/artifacts", headers=headers)).json())["artifacts"]
            assert [row["artifact_id"] for row in rows if row["active"]] == [first["artifact_id"]]
            after = (await (await client.get("/reef/scenarios/agent/settings")).json())["served"]
            assert after == {**before, HARNESS: {first["artifact_id"]: 100}}
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_creation_manifest_read_that_failed_is_asked_again(monkeypatch: Any, dispatchers: _Dispatchers) -> None:
    """One failed read of the creation artifact must not hide the creation's manifest for the process lifetime."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher)
    wait_for_serving(dispatcher, "agent")
    # The serving of the weights read the creation's manifest; the test reads it as a process that has not.
    scenario.snapshot_queries.creation_components_read = False
    original = Repository.snapshot
    calls = {"failed": 0}

    reader = threading.current_thread()

    def fail_once(self: Repository, ref: SnapshotRef) -> Artifact:
        # The serving worker reads snapshots too; only the test's own read of the manifest fails.
        if calls["failed"] == 0 and threading.current_thread() is reader:
            calls["failed"] += 1
            raise ArtifactNotFoundError("the remote is away")
        return original(self, ref)

    monkeypatch.setattr(Repository, "snapshot", fail_once)
    assert scenario.snapshot_queries.creation_components(0) is None
    # Not asked again at the same step (a page polls every few seconds; a remote read is a fetch)...
    assert scenario.snapshot_queries.creation_components(0) is None
    assert calls["failed"] == 1
    # ...but asked again after the next commit.
    components = scenario.snapshot_queries.creation_components(1)
    assert components is not None and set(components) == {WEIGHTS, HARNESS}
    assert calls["failed"] == 1
    assert scenario.snapshot_queries.creation_components(0) is not None


@pytest.mark.unit
def test_an_activation_of_weights_held_at_step_one_is_no_harness_artifact(dispatchers: _Dispatchers) -> None:
    """The creation artifact has no record; its manifest still says the activated tree did not change."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    backends[WEIGHTS].hold_next = True
    scenario = loaded(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    creation = scenario.current_snapshot().revision
    _append_records(scenario, 1)
    assert append_commit(scenario, WEIGHTS) is not None
    # The job's checkpoint changed the held kind, so it made an artifact that does not serve.
    checkpoint = next(record for record in reversed(scenario.store.history()) if record.made_artifact)
    assert checkpoint.activated is False
    held = checkpoint.snapshot.revision
    scenario.activate(component_id(scenario, held, WEIGHTS), WEIGHTS)
    assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights step 1"
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, creation, HARNESS)
    assert listed_revisions(scenario) == [creation]


@pytest.mark.unit
def test_an_activation_under_the_same_tree_is_no_new_harness_head(dispatchers: _Dispatchers) -> None:
    """An activation that changes only the weights names the same harness artifact; none adds a row."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = loaded(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    base = scenario.current_snapshot().revision
    _append_records(scenario, 1)
    assert append_commit(scenario, HARNESS) is not None
    harness_snapshot = scenario.current_snapshot().revision
    assert append_commit(scenario, WEIGHTS) is not None
    first_weights = scenario.current_snapshot().revision
    _append_records(scenario, 2)
    assert append_commit(scenario, WEIGHTS) is not None

    # Back to the first weights under the same tree: nothing to pull.
    scenario.activate(component_id(scenario, first_weights, WEIGHTS), WEIGHTS)
    assert component_files(scenario, scenario.current_snapshot())[WEIGHTS] == "weights step 1"
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, harness_snapshot, HARNESS)
    assert listed_revisions(scenario) == [base, harness_snapshot]

    # Back to the root's harness: the seed tree is served again, named by the root snapshot.
    scenario.activate(component_id(scenario, scenario.repository.start_artifact.revision, HARNESS), HARNESS)
    assert service.harness_manifest(headers)["artifact_id"] == component_id(scenario, base, HARNESS)
    assert service.harness_manifest(headers)["files"] == {"harness.txt": "harness seed"}
    assert listed_revisions(scenario) == [base, harness_snapshot]


@pytest.mark.unit
def test_adopted_checkpoint_is_attributed_to_the_trainer_that_made_it(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A checkpoint read back from the artifact head names its component, so a lost commit history resets no trainer."""
    backend_factory = _repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        records_dir=tmp_path / "records-1", backend_factory=backend_factory, hold_local_cycles=True
    )
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, HARNESS) is not None
    head = scenario.repository.materialize(scenario.current_snapshot())
    checkpoint = head.metadata[SCENARIO_METADATA_KEY]
    assert isinstance(checkpoint, Mapping)
    assert checkpoint["component"] == HARNESS
    _, _, _, adopted = parse_scenario_metadata(checkpoint, checkpoint_head=head.ref)
    assert adopted is not None and adopted.component == HARNESS
    dispatcher.close()

    # The record store is gone; the checkpoint is adopted from the head and belongs to the harness trainer.
    recovered, _ = dispatchers.open(records_dir=tmp_path / "records-2", backend_factory=backend_factory)
    scenario = recovered.get_or_create_scenario("agent")
    # The checkpoint restores the running job; its worker must commit the end before inspecting final state.
    assert became_true(lambda: scenario.commit_number == 3)
    harness = JobLoopState.from_dict(scenario.trainer_for(HARNESS).state)
    assert harness.finished_jobs == 1 and harness.jobs == ()
    assert JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state).finished_jobs == 0
    last = scenario.last_commit_for(HARNESS)
    assert last is not None and last.commit_number == 3 and last.component == HARNESS
    assert scenario.last_commit_for(WEIGHTS) is None


@pytest.mark.unit
def test_a_lone_trainer_names_no_component_on_record(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A scenario of one trainer writes what it wrote before components: its commit record and the checkpoint's
    scenario metadata name no component."""
    # The harness recipe alone, serving one component: the shape of every shipped recipe.
    live = harness_live_objects(ComponentJobs(HARNESS, tmp_path / "candidates"))
    dispatcher, _ = dispatchers.open(recipe=harness_recipe(), live=live, hold_local_cycles=True)
    scenario = loaded(dispatcher, 1, 2)
    assert append_commit(scenario, HARNESS) is not None
    record = scenario.store.history()[-1]
    assert (record.component, record.components) == (None, None)
    checkpoint = scenario.repository.materialize(scenario.current_snapshot()).metadata[SCENARIO_METADATA_KEY]
    assert isinstance(checkpoint, Mapping)
    assert "component" not in checkpoint


@pytest.mark.unit
def test_dispatcher_runs_one_worker_per_component(dispatchers: _Dispatchers) -> None:
    dispatcher, _ = dispatchers.open()
    served = _scenario(dispatcher).serving_artifact_id()
    for step in (1, 2):
        for record in _records(step, served):
            dispatcher.accept_record(record)

    def both_trained_twice() -> bool:
        ended = training_components(_scenario(dispatcher))
        return ended.count(WEIGHTS) == ended.count(HARNESS) == 2

    assert became_true(both_trained_twice, timeout_seconds=20), "both component workers did not train twice"
    current = _scenario(dispatcher)
    assert component_files(current, current.current_snapshot()) == {
        WEIGHTS: "weights step 2",
        HARNESS: "harness step 2",
    }
    assert dispatcher.build_training_status()["error"] is None


@pytest.mark.unit
def test_commit_record_carries_its_component() -> None:
    record = CommitRecord(
        scenario="agent",
        commit_number=3,
        snapshot=SnapshotRef("r3", "r2"),
        checkpoint=True,
        algorithm_state={"steps": 1},
        high_water_sequence=2,
        high_water_offset=2,
        component=HARNESS,
    )
    encoded = record.to_dict()
    assert encoded["component"] == HARNESS
    assert CommitRecord.from_dict(encoded) == record
    assert "component" not in CommitRecord.from_dict({**encoded, "component": None}).to_dict()
    start = CommitRecord(
        scenario="agent",
        commit_number=4,
        snapshot=SnapshotRef("r1", None),
        checkpoint=False,
        algorithm_state=None,
        high_water_sequence=0,
        high_water_offset=0,
        operation="train_from",
        component=HARNESS,
    )
    assert CommitRecord.from_dict(start.to_dict()) == start
    with pytest.raises(CommitHistoryError, match="operation 'activate' names a component"):
        replace(start, operation="activate")


@pytest.mark.unit
def test_component_trainers_must_match_the_components() -> None:
    def trainer() -> Trainer:
        return Trainer.build("agent", memory_records("agent"), pipeline_factory=TrainingPipeline)

    flat = ScenarioComponents(components={HARNESS: ComponentSpec(files=TextFileTree())})
    composed = ScenarioComponents(components={WEIGHTS: ComponentSpec(), HARNESS: ComponentSpec(files=TextFileTree())})
    harness_only = (ComponentTrainer(HARNESS, trainer()),)
    assert validate_component_trainers(harness_only, flat) == harness_only
    assert validate_component_trainers(harness_only, composed) == harness_only
    named = (ComponentTrainer(WEIGHTS, trainer()), ComponentTrainer(HARNESS, trainer()))
    assert validate_component_trainers(named, composed) == named
    records_only = (ComponentTrainer(RECORDS_COMPONENT, trainer()),)
    assert validate_component_trainers(records_only, ScenarioComponents()) == records_only
    with pytest.raises(ReefError, match="serving no component"):
        validate_component_trainers(harness_only, ScenarioComponents())
    with pytest.raises(ValueError, match="non-empty"):
        ComponentTrainer("", trainer())
    with pytest.raises(ReefError, match="does not serve"):
        validate_component_trainers((ComponentTrainer("config", trainer()),), composed)
    with pytest.raises(ReefError, match="distinct"):
        validate_component_trainers(
            (ComponentTrainer(HARNESS, trainer()), ComponentTrainer(HARNESS, trainer())), composed
        )


@pytest.mark.unit
def test_service_start_notification_cannot_reopen_stopping_workers(dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = _scenario(dispatcher, 1)
    dispatcher.stop_local_cycles()
    dispatcher.open_local_cycles()
    assert dispatcher.local_workers.process_step("agent", HARNESS) is False
    assert backends[HARNESS].prepared == 0
    assert scenario.commit_number == 0
    dispatcher.close()
    dispatcher.local_workers.start("late", HARNESS)
    assert ("late", HARNESS) not in dispatcher.local_workers.workers


class HeldJobs(WeightsJobComponent):
    """The weights job component whose jobs run until the test finishes them: ``submit`` leaves a job running."""

    def submit(self, job: TrainingJob) -> None:
        if job.job_id in self.statuses:
            return
        self.submitted.append(job)
        self.statuses[job.job_id] = JobStatus(job.job_id, "running")

    def finish(self, job: TrainingJob) -> None:
        """End ``job`` with the checkpoint the stub saves for it."""
        self.statuses[job.job_id] = JobStatus(job.job_id, "finished", self.train(job))


def job_events(scenario: Scenario) -> list[str | None]:
    return [job_event(record) for record in scenario.store.history() if record.component == WEIGHTS]


@pytest.mark.unit
def test_dispatcher_runs_the_weights_jobs_and_the_harness_on_their_own_workers(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """Each trainer has its worker: the weights worker submits a job per batch and commits its checkpoint and end,
    and each job starts from the checkpoint of the job before it."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = loaded(dispatcher)
    # Both trainers must observe the base weights before a job can replace them, so their samples stay trainable.
    for component in (WEIGHTS, HARNESS):
        assert scenario.next_commit(component) is None
    served = scenario.serving_artifact_id()
    for step in (1, 2):
        for record in _records(step, served):
            dispatcher.accept_record(record)
    dispatcher.open_local_cycles()

    def both_trained_twice() -> bool:
        ended = training_components(_scenario(dispatcher))
        return ended.count(WEIGHTS) == ended.count(HARNESS) == 2

    assert became_true(both_trained_twice, timeout_seconds=20), "both component workers did not train twice"
    current = _scenario(dispatcher)
    assert component_files(current, current.current_snapshot()) == {
        WEIGHTS: "weights job 2",
        HARNESS: "harness step 2",
    }
    first, second = training.submitted
    assert second.start == training.statuses[first.job_id].checkpoints[-1].path
    assert dispatcher.build_training_status()["error"] is None


@pytest.mark.unit
def test_a_manual_weights_trainer_submits_the_samples_that_its_request_names(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """In manual mode a weights stage trains only for a training request: the job carries the samples that
    the request names and the request, and the submission consumes both."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = loaded(dispatcher)
    scenario.set_training_mode("manual")
    _append_records(scenario, 1)
    make_samples(scenario)
    # The samples service stores the step's sample, but without a training request nothing trains.
    assert scenario.next_commit(WEIGHTS) is None
    trainer = scenario.trainer_for(WEIGHTS)
    held = tuple(stored.sample_id for stored in trainer.pipeline.store.page())
    # In a scenario of several components, a sample's name names its component first.
    assert held == (stage_item_id("agent", f"{WEIGHTS}:r1:0"),)
    dispatcher.accept_request(
        "agent",
        {"text": "learn the tool", "session": "session-1", "artifact_id": "artifact-1", "samples": list(held)},
        request_id="t1",
    )
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    [job] = submitted_jobs(training, 1)
    assert job.data_id == "agent:request:t1"
    assert job.request_id == "t1"
    assert {"t1", "i1", "r1"} <= scenario.store.history()[-1].consumed_ids
    settle_weights_job(scenario)
    assert job_events(scenario) == ["submitted", "checkpoint", "finished"]


@pytest.mark.unit
def test_a_restart_polls_the_running_weights_job_and_commits_its_checkpoint(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The running job is in the weights trainer's committed state, also after the harness moved the commit number.
    The restarted trainer polls that job instead of reserving its rows again, and commits the checkpoint that the
    job saved meanwhile."""
    training = HeldJobs(tmp_path / "checkpoints")
    factory = _repository(tmp_path)
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True, backend_factory=factory)
    scenario = loaded(dispatcher, 1, 2)
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    [job] = submitted_jobs(training, 1)
    assert scenario.next_commit(WEIGHTS) is None
    assert append_commit(scenario, HARNESS) == [("i1", "r1")]
    assert append_commit(scenario, HARNESS) == [("i2", "r2")]
    dispatcher.close()

    training.finish(job)
    restarted, _ = dispatchers.open(training=training, backend_factory=factory)
    # What a restart does for every registration a repository lists: the load wakes the weights worker.
    restarted.preload_scenarios(("agent",))
    scenario = _scenario(restarted)
    assert became_true(lambda: job_events(scenario)[:3] == ["submitted", "checkpoint", "finished"])
    assert training.submitted[0] == job
    # The commit history keeps the artifacts that served before the restart. The base answered the rows behind the
    # job, and it is within the limit of the stage, so they train in the next job, which starts from the checkpoint.
    component = scenario.served_component()
    assert component is not None
    served = scenario.snapshot_queries.served_history(scenario.commit_number, component, scenario.components.names)
    assert len(served) == 2 and served[-1] == scenario.serving_artifact_id()
    assert became_true(lambda: len(training.submitted) == 2)
    assert [item.source_agent_record_ids for item in training.submitted[1].read_data().samples] == [("i2", "r2")]
    assert training.submitted[1].start == str(tmp_path / "checkpoints" / "1")


@pytest.mark.unit
def test_a_restarted_training_worker_runs_again_the_job_that_the_stopped_one_gave_back(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A stopped training worker gives its running job back to the jobs queue. The restarted worker takes it and
    submits it to its component, which runs it again from its latest checkpoint; a component that lost the job does
    not end it."""
    factory = _repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        training=HeldJobs(tmp_path / "checkpoints"), hold_local_cycles=True, backend_factory=factory
    )
    assert append_commit(loaded(dispatcher, 1), WEIGHTS) == [("i1", "r1")]
    dispatcher.close()

    other = HeldJobs(tmp_path / "other-checkpoints")
    restarted, _ = dispatchers.open(training=other, backend_factory=factory)
    restarted.preload_scenarios(("agent",))
    scenario = _scenario(restarted)
    assert became_true(lambda: len(other.submitted) == 1)
    assert job_events(scenario) == ["submitted"]
    state = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state)
    assert [job.job_id for job in state.jobs] == [other.submitted[0].job_id]


@pytest.mark.unit
def test_a_weights_start_cancels_the_running_job_and_moves_the_next_jobs_start(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A new start imports the snapshot into the training component and cancels the running job. The next job
    starts from the imported snapshot, not from the last checkpoint, and serving does not change."""
    training = HeldJobs(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    training.finish(submitted_jobs(training, 1)[0])
    settle_weights_job(scenario)
    _append_records(scenario, 2)
    assert append_commit(scenario, WEIGHTS) == [("i2", "r2")]
    running = submitted_jobs(training, 2)[-1]
    assert running.start == str(tmp_path / "checkpoints" / "1")

    served = scenario.current_snapshot()
    scenario.train_from(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
    # The training worker stops the job on the component once it sees the cancelled item.
    assert became_true(lambda: training.statuses[running.job_id].error == "cancelled")
    # Control names the snapshot and imports nothing: the training service imports it for the next job.
    assert training.imported == []
    assert scenario.current_snapshot() == served
    assert component_files(scenario, served)[WEIGHTS] == "weights job 1"
    # The harness trainer's start is its own: it does not move.
    assert scenario.store.history()[-1].component == WEIGHTS
    state = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state)
    assert state.jobs == () and SnapshotStart.from_text(state.start) == SnapshotStart(
        scenario.repository.start_artifact.revision
    )
    _append_records(scenario, 3)
    assert append_commit(scenario, WEIGHTS) == [("i3", "r3")]
    assert submitted_jobs(training, 3)[-1].start == str(training.imported[-1])


@pytest.mark.unit
@pytest.mark.parametrize("deleted", ["after the cycle", "inside the cycle"])
def test_a_job_that_was_submitted_and_not_committed_is_cancelled_when_its_scenario_is_deleted(
    tmp_path: Path, dispatchers: _Dispatchers, deleted: str
) -> None:
    """A trainer adds a job to the queue before the commit that records it. A delete of the scenario in between
    finds the job in no trainer's state: the job must still not run, or it would store a checkpoint and its
    record files for a scenario that is gone."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    factory = _repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        training=training,
        hold_local_cycles=True,
        backend_factory=factory,
        components=frozenset(SERVICE_COMPONENTS) - {"training"},
    )
    scenario = loaded(dispatcher, 1)
    # The training service answers the job's check and runs no job, so the job waits in the queue.
    checks = TrainingCheckWorker(
        training, dispatcher.queue, dispatcher.job_directory_store, factory, worker_id="checks", poll_seconds=0.01
    )
    checks.start()
    try:
        if deleted == "after the cycle":
            # The cycle submitted the job and stopped before its commit.
            assert checked_commit(scenario, WEIGHTS) is not None
            (job,) = dispatcher.queue.items(JOBS_QUEUE)
            assert job.state == "queued" and scenario.trainer_for(WEIGHTS).state["jobs"] == []
            dispatcher.delete_scenario("agent")
        else:
            # The delete ends while the cycle is about to add the job to the queue.
            queued = scenario.trainer_for(WEIGHTS).job_loop_required().component
            submit = queued.submit

            def delete_then_submit(job: TrainingJob) -> None:
                dispatcher.delete_scenario("agent")
                assert dispatcher.queue.items(JOBS_QUEUE) == ()
                submit(job)

            wait_for_job_check(scenario, WEIGHTS)
            queued.submit = delete_then_submit  # type: ignore[method-assign]
            dispatcher.open_local_cycles()
            dispatcher.local_workers.process_step("agent", WEIGHTS)
            (job,) = dispatcher.queue.items(JOBS_QUEUE)
    finally:
        checks.stop()
    cancelled = dispatcher.queue.read(JOBS_QUEUE, job.id)
    assert cancelled.state == "failed" and cancelled.error in ("the scenario was deleted", "cancelled")
    assert training.submitted == []


@pytest.mark.unit
def test_a_job_that_waits_for_its_check_holds_nothing_of_the_scenario(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """No training service answers. The job's data is not reserved between two polls, so the scenario's other
    operations run, and nothing is consumed. Once the check is answered, the next poll submits the job."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    factory = _repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        training=training,
        hold_local_cycles=True,
        backend_factory=factory,
        components=frozenset(SERVICE_COMPONENTS) - {"training"},
    )
    scenario = loaded(dispatcher, 1)
    trainer = scenario.trainer_for(WEIGHTS)
    for _ in range(3):
        assert scenario.next_commit(WEIGHTS) is None
    # The worker polls for the answer at its usual interval, and the trainer holds no reservation meanwhile.
    assert trainer.job_check_waits and trainer.needs_polling() and not trainer.has_pending_commit
    [check] = dispatcher.queue.items(CHECKS_QUEUE)
    assert check.state == "queued"

    root = scenario.artifact_status()[WEIGHTS]["training_start"]
    # Each of these raised CommitPendingError while a waiting job kept its data reserved.
    scenario.activate(root, WEIGHTS)
    scenario.train_from(root, WEIGHTS)
    assert scenario.store.history()[-1].operation == "train_from"
    assert all(record.training_job_id is None for record in scenario.store.history())
    # The job now starts from another start, so it is another job with a check of its own. The samples service
    # judges the samples against the activation first, as it does after each commit.
    make_samples(scenario)
    assert scenario.next_commit(WEIGHTS) is None
    first, second = dispatcher.queue.items(CHECKS_QUEUE)
    assert first.id == check.id and second.state == "queued"

    # The training service comes up and answers; the next poll submits the job with the samples that waited.
    checks = TrainingCheckWorker(
        training, dispatcher.queue, dispatcher.job_directory_store, factory, worker_id="checks", poll_seconds=0.01
    )
    checks.start()
    try:
        assert append_commit(scenario, WEIGHTS, make=False) == [("i1", "r1")]
    finally:
        checks.stop()
    assert not trainer.job_check_waits
    [queued] = dispatcher.queue.items(JOBS_QUEUE)
    assert SnapshotStart.from_text(queued.payload["start"]) == SnapshotStart(
        scenario.repository.start_artifact.revision
    )


@pytest.mark.unit
def test_a_check_that_could_not_run_consumes_no_sample(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """The training service cannot run the check. The trainer's cycle fails with an error that is no refusal, no
    commit is made, and the samples train once the check runs."""

    class Unreachable(WeightsJobComponent):
        reachable = False

        def check_job(self, job: TrainingJob) -> None:
            if not self.reachable:
                raise ConnectionError("the artifact store did not answer")

    training = Unreachable(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = loaded(dispatcher, 1)
    commits = scenario.commit_number
    with pytest.raises(RuntimeError, match=r"so nothing was decided about it: .*the artifact store did not answer"):
        checked_commit(scenario, WEIGHTS)
    assert scenario.commit_number == commits and training.submitted == []
    assert scenario.trainer_for(WEIGHTS).holds_untrained_records()
    training.reachable = True
    assert append_commit(scenario, WEIGHTS, make=False) == [("i1", "r1")]


@pytest.mark.unit
def test_a_restart_before_the_trainers_next_commit_imports_its_start_again(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The process stops after the ``train_from`` record and before the trainer's next commit records the start.
    The trainer's state recovers from its earlier commit, so the restarted scenario names the snapshot again."""
    training = HeldJobs(tmp_path / "checkpoints")
    factory = _repository(tmp_path)
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True, backend_factory=factory)
    scenario = loaded(dispatcher, 1)
    assert append_commit(scenario, WEIGHTS) == [("i1", "r1")]
    training.finish(submitted_jobs(training, 1)[0])
    settle_weights_job(scenario)
    served = scenario.current_snapshot()
    scenario.train_from(component_id(scenario, scenario.repository.start_artifact.revision, WEIGHTS), WEIGHTS)
    named = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state).start
    assert SnapshotStart.from_text(named) == SnapshotStart(scenario.repository.start_artifact.revision)
    dispatcher.close()

    restarted, _ = dispatchers.open(training=training, hold_local_cycles=True, backend_factory=factory)
    runtime = restarted.live.runtime
    assert isinstance(runtime, StubInferenceRuntime)
    # The previous owner's routes can remain until recovery starts. Wait for the new runtime to load its weights.
    assert became_true(lambda: runtime.served == "w1")
    scenario = loaded(restarted, 2)
    assert scenario.current_snapshot() == served
    state = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state)
    assert state.start == named and training.imported == []
    assert append_commit(scenario, WEIGHTS) == [("i2", "r2")]
    assert submitted_jobs(training, 2)[-1].start == str(training.imported[-1])


@pytest.mark.unit
def test_deleting_a_scenario_cancels_its_running_training_job(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """Nothing would read the job's checkpoints once its scenario is gone, so the delete cancels the job."""
    training = HeldJobs(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    assert append_commit(loaded(dispatcher, 1), WEIGHTS) == [("i1", "r1")]
    [job] = submitted_jobs(training, 1)
    dispatcher.delete_scenario("agent")
    # The training worker stops the job on the component once it sees the cancelled item.
    assert became_true(lambda: training.statuses[job.job_id].error == "cancelled")
    assert not dispatcher.registry.has("agent")
