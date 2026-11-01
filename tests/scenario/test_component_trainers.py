"""One trainer per release component: independent workers that meet at the commit boundary."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestServer
from tests.support._threshold_processor import threshold_pipeline
from tests.support._trajectories import policy_trajectory
from tests.support.component_trainers import (
    HARNESS,
    WEIGHTS,
    ComponentJobs,
    JudgedAtOnce,
    Judgment,
    WeightsJobComponent,
    _append_records,
    _dispatcher,
    _local_pair,
    _RecordingTracker,
    _records,
    _scenario,
    _seeded_repository,
    _TwoTrainerRecipe,
    component_files,
    finish_job,
    stop_workers,
)
from tests.support.samples_service import make_samples

from reef.artifact import Artifact, ArtifactNotFound, ArtifactRef
from reef.artifact.release_chain import ArtifactReleaseChain
from reef.core import AgentRecord, RequestType
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError
from reef.core.requirements import required_by
from reef.dispatcher import Dispatcher
from reef.execution.jobs import SupportsReceipt
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.record2sample import GroupDecision, JudgedSamples, RecordJudging, ReportContext, ReportGrouping
from reef.record2sample.service import SAMPLES_QUEUE
from reef.runtime.jobs import JobStatus, TrainingJob
from reef.sample2job import Sample2Job
from reef.scenario import Scenario
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
from reef.storage.samples import consumed_sample_id
from reef.storage.sqlite import SQLiteRecordStore
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.train import ComponentTrainer, Trainer
from reef.train.job_queue import JOBS_QUEUE
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, Sample


class _SlowBackend(ComponentJobs):
    """The same component whose step waits for the test to let it go, as a minutes long rollout run would."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.evaluating = threading.Event()
        self.release = threading.Event()

    def unblock(self) -> None:
        self.release.set()

    def step(self, job):
        self.evaluating.set()
        assert self.release.wait(30)
        return super().step(job)


class _AwayBackend(ComponentJobs):
    """A component whose method is away: every step fails, so no job is recorded."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.attempts = 0

    def step(self, job):
        self.attempts += 1
        raise RuntimeError("proposer away")


class _FlakyBackend(ComponentJobs):
    """A component whose step fails once, then answers."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.failures_left = 1

    def step(self, job):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("proposer away once")
        return super().step(job)


def _report(record_id: str, *references: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": list(references)},
        agent_record_id=record_id,
        references=references,
    )


def _wait_for(condition: Callable[[], bool], timeout_seconds: float = 10) -> bool:
    """Whether ``condition`` held within the timeout, checked every few milliseconds."""
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


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
                backend.unblock()
            dispatcher.close()


@pytest.fixture
def dispatchers(tmp_path: Path) -> Iterator[_Dispatchers]:
    opened = _Dispatchers(tmp_path)
    yield opened
    opened.close()


def _settle_samples(dispatcher: Dispatcher, name: str = "agent") -> None:
    """Wait until the samples worker ended every record selection of scenario ``name``, such as the one that a load
    asks for. A selection wakes the trainers' workers, so stop them again: the test, not a worker, decides when a
    trainer prepares."""
    assert _wait_for(lambda: all(item.final for item in dispatcher.queue.items(SAMPLES_QUEUE, scenario=name)))
    stop_workers(dispatcher, name)


def _loaded(dispatcher: Dispatcher, *steps: int) -> Scenario:
    """``_scenario`` once the samples worker ended the selection that the load asked for, with each step's rows
    appended after it and their samples made, as the samples service makes them."""
    scenario = _scenario(dispatcher)
    _settle_samples(dispatcher)
    _append_records(scenario, *steps)
    make_samples(scenario)
    return scenario


def _reload(dispatcher: Dispatcher) -> Scenario:
    """Reload scenario ``agent``, once the samples worker ended the selection that the load asked for."""
    rebuilt = dispatcher.registry.reload("agent")
    _settle_samples(dispatcher)
    return rebuilt


def _submitted(training: WeightsJobComponent, count: int) -> list[TrainingJob]:
    """The jobs that ``training`` took, once the training worker gave it ``count`` of them from the jobs queue."""
    assert _wait_for(lambda: len(training.submitted) == count)
    return training.submitted


def _training_components(scenario: Scenario) -> list[str]:
    """The component of each training job that ended, newest first: one per step."""
    return [
        row["component"]
        for row in scenario.releases()
        if row["operation"] == "training"
        and (row.get("metrics") or {}).get("training_job", {}).get("event") in ("finished", "failed")
    ]


def _commit_step(
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
    result = scenario.prepare_training_step(component)
    if result is None:
        return None
    batch = scenario.trainer_for(component).pending_job_data
    assert batch is not None
    sources = [tuple(item.source_agent_record_ids) for item in batch.samples]
    scenario.commit(result, component=component)
    finish_job(scenario, component, checkpoint_metrics)
    return sources


def _wait_for_jobs(dispatcher: Dispatcher, training: WeightsJobComponent, count: int) -> list[TrainingJob]:
    """The jobs that ``training`` ran, once the training worker took ``count`` of them from the jobs queue and ended
    their items, as the weights trainer then reads them."""

    def ended() -> bool:
        items = [dispatcher.queue.get(JOBS_QUEUE, job.job_id) for job in training.submitted]
        return len(items) == count and all(item is not None and item.final for item in items)

    assert _wait_for(ended)
    return training.submitted


def _settle_weights_job(scenario: Scenario) -> None:
    """Commit the running weights job's checkpoint decisions and its end, as the weights worker does once the training
    worker reported them in the jobs queue."""
    while scenario.trainer_for(WEIGHTS).job_running:
        deadline = time.monotonic() + 10
        while (result := scenario.prepare_training_step(WEIGHTS)) is None:
            assert time.monotonic() < deadline, "the training worker did not report the running job"
            time.sleep(0.02)
        scenario.commit(result, component=WEIGHTS)


@pytest.mark.unit
def test_component_trainers_take_turns_on_one_release_chain(dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert [component_trainer.component for component_trainer in scenario.trainers] == [WEIGHTS, HARNESS]
    assert scenario.trainer_for(HARNESS) is scenario.trainers[1].trainer

    assert _commit_step(scenario, HARNESS) is not None
    assert component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights seed",
        HARNESS: "harness step 1",
    }
    assert _commit_step(scenario, WEIGHTS) is not None
    assert component_files(scenario, scenario.current_artifact_ref()) == {
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
    assert records[0].consumed_ids == {"i1", "r1", consumed_sample_id(f"{HARNESS}:r1:0")}
    assert records[3].consumed_ids == {"i1", "r1", consumed_sample_id(f"{WEIGHTS}:r1:0")}

    # Each trainer recovers from its own commits.
    reloaded = _reload(dispatcher)
    for component in (WEIGHTS, HARNESS):
        recovered = JobLoopState.from_dict(reloaded.trainer_for(component).state)
        assert recovered.finished_jobs == 1 and recovered.job is None
    assert reloaded.scenario_step == 6
    _append_records(reloaded, 2)
    make_samples(reloaded)
    assert reloaded.prepare_training_step(HARNESS) is not None


def _submission(scenario: Scenario, component: str) -> CommitRecord:
    """The newest commit of ``scenario`` that submitted a job of ``component``: the one that consumed its rows."""
    return next(
        record
        for record in reversed(scenario.store.history())
        if record.component == component and _job_event(record) == "submitted"
    )


def _drain(dispatcher: Dispatcher, component: str) -> None:
    """Run ``component``'s local cycles until one commits nothing: its job's submission, checkpoint and end."""
    while dispatcher.local_workers.process_step("agent", component):
        pass


def _job_event(record: CommitRecord) -> str | None:
    """The training job event a commit records: submitted, checkpoint, finished or failed; None for no job."""
    metrics = record.metrics or {}
    return metrics["training_job"]["event"] if "training_job" in metrics else None


@pytest.mark.unit
def test_a_weights_job_trainer_and_the_harness_trainer_share_one_release_chain(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The weights trainer commits its job's submission, its checkpoint and its end; the harness commits between them
    on the same chain. Only the submission consumes the job's rows, and a reload recovers each trainer from its own
    commits: the next job starts from the last checkpoint."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert [component_trainer.component for component_trainer in scenario.trainers] == [WEIGHTS, HARNESS]
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    [job] = _wait_for_jobs(dispatcher, training, 1)
    assert [item.source_agent_record_ids for item in job.data.samples] == [("i1", "r1")]
    assert _commit_step(scenario, HARNESS) == [("i1", "r1")]
    for _ in range(2):
        result = scenario.prepare_training_step(WEIGHTS)
        assert result is not None
        scenario.commit(result, component=WEIGHTS)
    assert scenario.prepare_training_step(WEIGHTS) is None
    assert component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights job 1",
        HARNESS: "harness step 1",
    }

    records = scenario.store.history()
    assert [(record.component, _job_event(record)) for record in records] == [
        (WEIGHTS, "submitted"),
        (HARNESS, "submitted"),
        (HARNESS, "checkpoint"),
        (HARNESS, "finished"),
        (WEIGHTS, "checkpoint"),
        (WEIGHTS, "finished"),
    ]
    assert records[0].consumed_ids == {"i1", "r1", consumed_sample_id(f"{WEIGHTS}:r1:0")}
    assert records[1].consumed_ids == {"i1", "r1", consumed_sample_id(f"{HARNESS}:r1:0")}
    assert all(record.consumed_ids == frozenset() for record in records[2:])
    assert {record.training_job_id for record in records if record.component == WEIGHTS} == {job.job_id}

    reloaded = _reload(dispatcher)
    state = JobLoopState.from_dict(reloaded.trainer_for(WEIGHTS).state)
    assert (state.job, state.finished_jobs, state.start) == (None, 1, str(tmp_path / "checkpoints" / "1"))
    assert JobLoopState.from_dict(reloaded.trainer_for(HARNESS).state).finished_jobs == 1
    _append_records(reloaded, 2)
    assert _commit_step(reloaded, WEIGHTS) == [("i2", "r2")]
    assert _wait_for_jobs(dispatcher, training, 2)[-1].start == state.start


@pytest.mark.unit
def test_a_weights_job_waits_for_the_harness_cycle_of_its_scenario(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A scenario runs one step at a time: the weights worker submits nothing while the harness cycle evaluates, and
    the status still reads. Once the cycle ends, the worker submits the job and commits its checkpoint and its end."""
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(backends={HARNESS: slow}, training=training)
    scenario = _loaded(dispatcher, 1)
    harness = threading.Thread(target=lambda: _drain(dispatcher, HARNESS))
    harness.start()
    assert slow.evaluating.wait(10)
    dispatcher.local_workers.start("agent", WEIGHTS)
    time.sleep(0.2)
    assert training.submitted == []
    assert scenario.trainer_for(WEIGHTS).pending_job_data is None
    assert dispatcher.build_training_status()["scenarios"]["agent"]["scenario_step"] == 0
    slow.release.set()
    harness.join(10)
    assert not harness.is_alive()
    assert _wait_for(lambda: scenario.scenario_step == 6)
    events = [(record.component, _job_event(record)) for record in scenario.store.history()]
    # The harness job was submitted first; the weights job's commits may come between the harness job's own.
    assert events[0] == (HARNESS, "submitted")
    for component in (HARNESS, WEIGHTS):
        assert [event for name, event in events if name == component] == ["submitted", "checkpoint", "finished"]
    assert len(training.submitted) == 1


@pytest.mark.unit
def test_a_reload_leaves_the_harness_cycle_in_flight_its_instance(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """A reload rebuilds the scenario at once; the instance a harness cycle evaluates on stays open until the cycle
    ends, and the cycle then looks again on the rebuilt instance instead of failing."""
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(
        backends={WEIGHTS: ComponentJobs(WEIGHTS, tmp_path / "candidates"), HARNESS: slow}
    )
    scenario = _loaded(dispatcher, 1)
    closed_under_the_cycle: list[bool] = []
    close = scenario.close

    def observed_close() -> None:
        closed_under_the_cycle.append(not slow.release.is_set())
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
    assert slow.evaluating.wait(10)
    reloading = threading.Thread(target=lambda: dispatcher.local_workers.reload("agent", scenario))
    reloading.start()
    reloading.join(5)
    # The reload does not wait for the cycle.
    assert not reloading.is_alive()
    rebuilt = dispatcher.registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    slow.release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert errors == [] and outcome == [True]
    assert closed_under_the_cycle == [False]
    assert dispatcher.build_training_status()["error"] is None
    # The rows the cycle held train on the rebuilt instance.
    assert dispatcher.local_workers.process_step("agent", HARNESS) is True
    _drain(dispatcher, HARNESS)
    assert _training_components(rebuilt) == [HARNESS]


@pytest.mark.unit
def test_a_harness_backlog_is_kept_for_the_next_start_when_the_service_stops(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A stopping service takes its routes away first, so a harness cycle's model calls through them fail and would
    come back as a skip that consumes the batch: the cycle ending then commits nothing and no further cycle starts."""
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(
        backends={WEIGHTS: ComponentJobs(WEIGHTS, tmp_path / "candidates"), HARNESS: slow}
    )
    _loaded(dispatcher, 1, 2)
    dispatcher.local_workers.start("agent", HARNESS)
    assert slow.evaluating.wait(10)
    dispatcher.stop_local_cycles()
    closing = threading.Thread(target=dispatcher.close)
    closing.start()
    slow.release.set()
    closing.join(10)
    assert not closing.is_alive()
    assert slow.prepared == 1
    # The next start finds both batches unread and trains them, with no new record to wake the harness.
    restarted, backends = dispatchers.open()
    # What a restart does for every registration a repository lists (this in-memory one lists none).
    restarted.preload_scenarios(("agent",))
    scenario = _scenario(restarted)
    assert _wait_for(lambda: _training_components(scenario).count(HARNESS) == 2)
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
    assert _wait_for(lambda: _training_components(scenario).count(HARNESS) == 2)
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
            await asyncio.to_thread(_wait_for, lambda: backends[HARNESS].prepared > 0)
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
def test_a_failed_instruction_of_a_scenario_that_is_gone_starts_no_worker(
    monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """Workers started for a deleted name would record errors that nothing clears."""
    dispatcher, _ = dispatchers.open()
    old = _scenario(dispatcher)
    dispatcher.delete_scenario("agent")
    monkeypatch.setattr(old.trainer_for(HARNESS), "fail_pending_instruction", lambda error: True)
    dispatcher.local_workers.fail_instruction(old, RuntimeError("rollout failed"), HARNESS)
    with dispatcher.local_workers.lock:
        assert dispatcher.local_workers.workers == {}


@pytest.mark.unit
def test_a_failing_local_cycle_runs_once_per_wake_and_keeps_its_error_in_the_status(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A backend that is away costs one attempt per wake, never a loop of reloads, and the failure stays visible."""
    away = _AwayBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(
        backends={WEIGHTS: ComponentJobs(WEIGHTS, tmp_path / "candidates"), HARNESS: away}
    )
    # Both rows first, then one wake: an accept wakes the worker per row, which would be two wakes.
    _loaded(dispatcher, 1)
    dispatcher.local_workers.start("agent", HARNESS)
    _wait_for(lambda: away.attempts > 0)
    time.sleep(0.2)
    assert away.attempts == 1
    assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"
    # Another wake: the worker looks again, once.
    dispatcher.local_workers.start("agent", HARNESS)
    _wait_for(lambda: away.attempts > 1)
    time.sleep(0.2)
    assert away.attempts == 2
    assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"


class _BlockingAwayBackend(ComponentJobs):
    """A component whose step waits for the test to let it go, then fails."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.entered = threading.Event()
        self.go = threading.Event()

    def unblock(self) -> None:
        self.go.set()

    def step(self, job):
        self.entered.set()
        assert self.go.wait(30)
        raise RuntimeError("proposer away")


@pytest.mark.unit
def test_a_sibling_waiting_at_the_cycle_lock_looks_again_after_a_reload_under_it(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """The failing cycle rebuilds the scenario at once, under the lock; the sibling that read the old instance looks again
    on the new one instead of sleeping with its rows unread."""
    away = _BlockingAwayBackend(WEIGHTS, tmp_path / "candidates")
    harness = ComponentJobs(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends={WEIGHTS: away, HARNESS: harness})
    scenario = _loaded(dispatcher, 1)
    failures: list[BaseException] = []

    def fail() -> None:
        try:
            dispatcher.local_workers.process_step("agent", WEIGHTS)
        except RuntimeError as exc:
            failures.append(exc)

    failing = threading.Thread(target=fail)
    failing.start()
    assert away.entered.wait(10)
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
    away.go.set()
    failing.join(10)
    sibling.join(10)
    assert [str(error) for error in failures] == ["proposer away"]
    rebuilt = dispatcher.registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert outcome == [True] and harness.prepared == 0
    # Looking again on the rebuilt instance commits the rows that were there all along.
    assert dispatcher.local_workers.process_step("agent", HARNESS) is True
    _drain(dispatcher, HARNESS)
    assert _training_components(rebuilt) == [HARNESS]


@dataclass(frozen=True)
class _NamedComponentsRecipe(Recipe):
    """A recipe serving the named components, each evolved by its own local backend."""

    names: tuple[str, ...] = ()
    artifact_dir: Path = Path(".")

    def build_surface(self, scenario: str) -> Surface:
        return Surface(
            components={
                name: ComponentSurface(files=TextFileTree()) if name == HARNESS else ComponentSurface()
                for name in self.names
            }
        )

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
    ):
        return tuple(
            ComponentTrainer(
                name,
                Trainer.build(
                    scenario,
                    records,
                    pipeline_factory=threshold_pipeline,
                    job_loop=JobLoop(ComponentJobs(name, self.artifact_dir), scenario=scenario),
                    algorithm_state=algorithm_states.get(name),
                    experiment_logger=experiment_logger,
                    default_job_samples=1,
                ),
            )
            for name in self.names
        )


@pytest.mark.unit
def test_a_recipe_serving_fewer_components_than_registered_is_refused(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A step carries forward only the components the recipe serves; one it does not serve would leave every release."""
    names = ("weights", "harness", "config")
    factory = _seeded_repository(tmp_path, names)
    first, _ = dispatchers.open(
        recipe=_NamedComponentsRecipe(names=names, artifact_dir=tmp_path / "candidates"), backend_factory=factory
    )
    assert first.get_or_create_scenario("agent") is not None
    first.close()
    second, _ = dispatchers.open(
        recipe=_NamedComponentsRecipe(names=("weights", "harness"), artifact_dir=tmp_path / "candidates"),
        backend_factory=factory,
    )
    with pytest.raises(ReefError, match=r"does not serve \['config'\]"):
        second.get_or_create_scenario("agent")


@pytest.mark.unit
def test_a_weights_job_commits_while_the_harness_backend_is_away(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """The harness cycle fails and the failure shows; the weights job still lands. Then the harness's rows are
    stale, since the job's checkpoint serves, so the harness stops trying them.

    Each wake before then costs one attempt, and the job's commits and queue changes each wake the scenario, so
    the count of attempts depends on how the wakes group."""
    away = _AwayBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends={HARNESS: away}, training=WeightsJobComponent(tmp_path / "checkpoints"))
    scenario = _scenario(dispatcher)
    for record in _records(1, scenario.serving_artifact_id()):
        dispatcher.accept_record(record)
    assert _wait_for(lambda: dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away")
    assert _wait_for(
        lambda: "finished" in [_job_event(row) for row in _scenario(dispatcher).store.history()],
        timeout_seconds=15,
    )
    time.sleep(0.2)
    attempts = away.attempts
    assert attempts >= 1
    time.sleep(0.5)
    assert away.attempts == attempts


@pytest.mark.unit
def test_a_second_report_on_a_trained_inference_is_settled_after_a_restart(dispatchers: _Dispatchers) -> None:
    """The harness trained on i1 through r1a while the weights trainer still holds i1; after a restart r1b is not resolved against it."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    scenario.records.append(_report("r1b", "i1"))
    # The job trains the sample of r1a alone: r1b has no sample yet.
    assert _commit_step(scenario, HARNESS, make=False) is not None
    assert scenario.records.count("agent") == 3
    assert scenario.prepare_training_step(HARNESS) is None
    rebuilt = _reload(dispatcher)
    make_samples(rebuilt)
    assert rebuilt.prepare_training_step(HARNESS) is None
    # Its source trained, so the rebuilt harness skips it; the weights trainer still reads it.
    assert "r1b" in rebuilt.trainer_for(HARNESS).skipped_record_ids
    assert "r1b" not in rebuilt.trainer_for(WEIGHTS).consumed_record_ids


@pytest.mark.unit
def test_a_report_settled_between_two_commits_stays_settled_after_a_restart(dispatchers: _Dispatchers) -> None:
    """r1b arrived after commit A trained i1 and before commit B: the replay settles it at A's watermark, as it was."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None
    scenario.records.append(_report("r1b", "i1"))
    assert scenario.prepare_training_step(HARNESS) is None
    _append_records(scenario, 2)
    assert _commit_step(scenario, HARNESS) is not None
    rebuilt = _reload(dispatcher)
    assert rebuilt.prepare_training_step(HARNESS) is None
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
    scenario = _loaded(dispatcher, 1)
    submitted = scenario.prepare_training_step(WEIGHTS)
    assert submitted is not None
    assert _commit_step(scenario, HARNESS) is not None
    scenario.records.append(_report("r1b", "i1"))
    assert scenario.prepare_training_step(HARNESS) is None
    scenario.commit(submitted, component=WEIGHTS)
    _settle_weights_job(scenario)
    make_samples(scenario)
    assert scenario.records.get("agent", "i1") is not None
    assert scenario.prepare_training_step(WEIGHTS) is None
    assert "r1b" in scenario.trainer_for(WEIGHTS).skipped_record_ids
    _append_records(scenario, 2)
    assert _commit_step(scenario, HARNESS) is not None
    rebuilt = _reload(dispatcher)
    assert rebuilt.prepare_training_step(HARNESS) is None
    assert _commit_step(rebuilt, WEIGHTS) == [("i2", "r2")]


class _SlotGrouping(ReportGrouping):
    """Reports batch by the group and the retry slot named in their metadata; a group is ready at two slots."""

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload["metadata"]
        return metadata["group"], metadata["slot"]

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.INCOMPLETE


@dataclass(frozen=True)
class _SlottedRecipe(_TwoTrainerRecipe):
    """The two trainer recipe whose harness batches reports by group and retry slot."""

    def pipeline_for(self, component: str, context: PipelineContext) -> TrainingPipeline:
        if component == HARNESS:
            # The threshold pipeline with ``_SlotGrouping``.
            return threshold_pipeline(context, grouping=_SlotGrouping())
        return threshold_pipeline(context)


def _slotted_report(record_id: str, reference: str, group: str, slot: int) -> AgentRecord:
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
    dispatcher, _ = dispatchers.open(recipe=_SlottedRecipe(backends=_local_pair(tmp_path)), hold_local_cycles=True)
    scenario = _loaded(dispatcher)
    for step in (1, 2, 3):
        scenario.records.append(_records(step, scenario.serving_artifact_id())[0])
    scenario.records.append(_slotted_report("r1", "i1", "G", 0))
    scenario.records.append(_slotted_report("r1dup", "i3", "G", 0))
    scenario.records.append(_slotted_report("r2", "i2", "G", 1))
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    assert _commit_step(scenario, HARNESS) == [("i1", "r1"), ("i2", "r2")]
    assert scenario.records.get("agent", "r1dup") is not None
    rebuilt = _reload(dispatcher)
    rebuilt.records.append(_records(4, rebuilt.serving_artifact_id())[0])
    rebuilt.records.append(_slotted_report("r4", "i4", "G", 2))
    # Live, slot 0 was never r1dup's: G holds r4 alone and waits.
    assert rebuilt.prepare_training_step(HARNESS) is None
    assert "r1dup" in _submission(rebuilt, HARNESS).consumed_ids


def _fail_first_record_of(monkeypatch: pytest.MonkeyPatch, scenario: Scenario, component: str) -> list[bool]:
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
    of them is in the sample store, which consumed the rows for the harness: the rebuilt harness trains the sample
    once."""
    dispatcher, _ = dispatchers.open(training=WeightsJobComponent(tmp_path / "checkpoints"))
    scenario = _loaded(dispatcher, 1)
    _fail_first_record_of(monkeypatch, scenario, HARNESS)
    with pytest.raises(OSError, match="transient store write error"):
        dispatcher.local_workers.process_step("agent", HARNESS)
    rebuilt = dispatcher.registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert _commit_step(rebuilt, WEIGHTS) == [("i1", "r1")]
    assert rebuilt.records.get("agent", "i1") is not None and rebuilt.records.get("agent", "r1") is not None
    for _ in range(3):
        dispatcher.local_workers.process_step("agent", HARNESS)
    current = dispatcher.registry.get_optional("agent")
    assert current is not None
    history = current.store.history()
    # A job's submission consumes its rows; its checkpoint and its end consume none.
    submissions = [record for record in history if _job_event(record) == "submitted"]
    assert [sorted(record.consumed_ids) for record in submissions if record.component == HARNESS] == [
        [consumed_sample_id(f"{HARNESS}:r1:0")]
    ]
    # Likewise, the weights trainer's sample was stored before the reload: its submission names the sample.
    assert [sorted(record.consumed_ids) for record in submissions if record.component == WEIGHTS] == [
        [consumed_sample_id(f"{WEIGHTS}:r1:0")]
    ]


@pytest.mark.unit
def test_a_weights_commit_that_fails_leaves_its_batch_to_the_weights_trainer_after_a_harness_commit(
    monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """The reverse: a weights commit that fails after its acknowledgment, then a harness commit before the weights
    trainer's reload. The harness commit leaves the rows stored, so the rebuilt weights trainer trains the batch its
    failed commit had acknowledged."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) == [("i1", "r1")]
    _append_records(scenario, 2)
    _fail_first_record_of(monkeypatch, scenario, WEIGHTS)
    with pytest.raises(OSError, match="transient store write error"):
        _commit_step(scenario, WEIGHTS)
    assert _commit_step(scenario, HARNESS) == [("i2", "r2")]
    assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
    rebuilt = _reload(dispatcher)
    assert _commit_step(rebuilt, WEIGHTS) == [("i1", "r1")]


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
    """One-token stub samples for every judged turn."""

    def make_sample(self, record: AgentRecord, judgment: Judgment) -> Any:
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
        )


class RetiringRetriesJobData(Sample2Job):
    """Batches the judged turns of ``_RetiringRetriesJudging``."""

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


@dataclass(frozen=True)
class _ComputedWeightsRecipe(_TwoTrainerRecipe):
    """The two trainer recipe whose weights trainer computes its own feedback."""

    def pipeline_for(self, component: str, context: PipelineContext) -> TrainingPipeline:
        if component == WEIGHTS:
            # Trains the judged turns of ``_RetiringRetriesJudging``.
            return TrainingPipeline(
                context,
                record2sample=_RetiringRetriesSamples(_RetiringRetriesJudging(), JudgedAtOnce()),
                sample2job=RetiringRetriesJobData(context, None, None),
            )
        return threshold_pipeline(context)


def _turn(record_id: str, turn: str, completes: str | None = None) -> AgentRecord:
    payload: dict[str, Any] = {"turn": turn, "tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
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
    dispatcher, _ = dispatchers.open(
        recipe=_ComputedWeightsRecipe(backends=_local_pair(tmp_path)), hold_local_cycles=True
    )
    scenario = _loaded(dispatcher)
    scenario.records.append(_turn("A", "t1"))
    scenario.records.append(_turn("B", "t2", completes="A"))
    assert _commit_step(scenario, WEIGHTS) == [("A",)]
    scenario.records.append(_turn("A-retry", "t1"))
    scenario.records.append(_turn("C", "t3", completes="B"))
    assert _commit_step(scenario, WEIGHTS) == [("B",)]
    assert "A-retry" in _submission(scenario, WEIGHTS).consumed_ids
    if reload:
        scenario = _reload(dispatcher)
    scenario.records.append(_turn("E", "t4", completes="A-retry"))
    assert _commit_step(scenario, WEIGHTS) is None


@pytest.mark.unit
def test_a_run_report_on_an_inference_both_trainers_trained_is_skipped_by_both(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """run references i1 and i2 and is stored before the weights job's submission commits i1, which the harness
    trained already. Each trainer then skips run, since one of its sources trained, instead of raising on it."""
    dispatcher, _ = dispatchers.open(training=WeightsJobComponent(tmp_path / "checkpoints"), hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    scenario.records.append(_records(2, scenario.serving_artifact_id())[0])
    submitted = scenario.prepare_training_step(WEIGHTS)
    assert submitted is not None
    assert _commit_step(scenario, HARNESS) is not None
    scenario.records.append(_report("run", "i1", "i2"))
    scenario.commit(submitted, component=WEIGHTS)
    _settle_weights_job(scenario)
    make_samples(scenario)
    assert scenario.prepare_training_step(HARNESS) is None
    assert scenario.prepare_training_step(WEIGHTS) is None
    for component in (WEIGHTS, HARNESS):
        assert "run" in scenario.trainer_for(component).skipped_record_ids, component


@pytest.mark.unit
def test_an_activation_of_the_creation_after_a_rejected_first_step(dispatchers: _Dispatchers) -> None:
    """A rejected step records the creation without a checkpoint; the creation still has its own bytes."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher)
    creation = scenario.current_artifact_ref().release_id
    backends[HARNESS].reject_next = True
    for step in (1, 2):
        _append_records(scenario, step)
        assert _commit_step(scenario, HARNESS) is not None
    assert scenario.current_artifact_ref().release_id != creation
    assert next(row for row in scenario.releases() if row["operation"] == "creation")["restorable"] is True
    scenario.activate(f"{scenario.repository.backend.resolve_release('base').release_id}:{HARNESS}", HARNESS)
    assert component_files(scenario, scenario.current_artifact_ref())[HARNESS] == "harness seed"


@pytest.mark.unit
def test_a_job_submitted_on_an_instance_reloaded_under_it_is_the_same_job_on_the_new_instance(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The old instance's submission is never committed. The rebuilt trainer reserves the same rows, whose job has
    the same ID, so the component runs it once and the new instance commits it."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    old = _loaded(dispatcher, 1)
    assert old.prepare_training_step(WEIGHTS) is not None
    [job] = _submitted(training, 1)
    new = _reload(dispatcher)
    assert new is not old
    assert new.store.history() == ()
    assert _commit_step(new, WEIGHTS) == [("i1", "r1")]
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layout", "outside"),
    [
        # Files at the root of the base belong to no component and would be carried forward by none.
        ({"harness.txt": "harness seed"}, "harness.txt"),
        # A component keeps a directory; a file of that name would fail on every component view.
        ({f"{HARNESS}/{HARNESS}.txt": "harness seed", WEIGHTS: "not a directory"}, WEIGHTS),
    ],
)
def test_composite_registration_refuses_files_outside_its_components(
    tmp_path: Path, dispatchers: _Dispatchers, layout: dict[str, str], outside: str
) -> None:
    for relative, text in layout.items():
        (tmp_path / "initial" / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "initial" / relative).write_text(text, encoding="utf-8")
    dispatcher, _ = dispatchers.open()
    with pytest.raises(ReefError, match=rf"keeps \['{outside}'\] outside its components"):
        dispatcher.get_or_create_scenario("agent")


@pytest.mark.unit
def test_composite_registration_starts_an_unseeded_component_empty(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A fresh repository seeds only the harness; the weights component starts empty and fills at its first step."""
    _seeded_repository(tmp_path, (HARNESS,))
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher)
    base = scenario.repository.materialize(scenario.current_artifact_ref())
    assert base.components is not None and base.components.names == (WEIGHTS, HARNESS)
    assert TextFileTree().read_files(base.component(WEIGHTS)) is None
    _append_records(scenario, 1)
    assert _commit_step(scenario, WEIGHTS) is not None
    assert component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness seed",
    }


@pytest.mark.unit
def test_composite_recovery_refuses_a_registration_without_a_component_manifest(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A scenario registered by a recipe serving one component is not read as a composed one."""
    backend_factory = _seeded_repository(tmp_path)
    backend = backend_factory("agent")
    selected = backend.resolve_release(None)
    # Registration as written before releases named their components.
    backend.fork(
        selected.release_id,
        metadata={SCENARIO_METADATA_KEY: scenario_metadata_for(name="agent", base_artifact=selected)},
    )
    dispatcher, _ = dispatchers.open(backend_factory=backend_factory)
    with pytest.raises(ReefError, match="registered without a component manifest"):
        dispatcher.get_or_create_scenario("agent")


@pytest.mark.unit
def test_rows_one_trainer_consumed_before_a_restart_still_train_the_other(dispatchers: _Dispatchers) -> None:
    """A restart between two trainers' commits of the same rows: the second still trains them, the first never again."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None

    reloaded = _reload(dispatcher)
    assert reloaded.prepare_training_step(HARNESS) is None
    assert _commit_step(reloaded, WEIGHTS) is not None
    # The weights trainer's sample of the rows was stored before the restart, and the sample store consumed the rows
    # for it: the commit names the sample.
    assert _submission(reloaded, WEIGHTS).consumed_ids == {consumed_sample_id(f"{WEIGHTS}:r1:0")}
    assert reloaded.records.count("agent") == 2


@pytest.mark.unit
def test_a_weights_step_is_no_new_harness_head(dispatchers: _Dispatchers) -> None:
    """Clients pull the tree: a step that carried it forward unchanged is not announced or listed as a release."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    assert _commit_step(scenario, HARNESS) is not None
    harness_release = scenario.current_artifact_ref().release_id
    assert _commit_step(scenario, WEIGHTS) is not None
    assert scenario.current_artifact_ref().release_id != harness_release
    assert service.harness_release_id(scenario) == harness_release
    manifest = service.harness_manifest(headers)
    assert manifest["release_id"] == harness_release
    assert manifest["files"] == {"harness.txt": "harness step 1"}
    catalog = service.harness_releases(headers)["releases"]
    assert [row["operation"] for row in catalog] == ["creation", "training"]
    # The creation carries the base artifact; the harness step made its own.
    base = scenario.repository.backend.resolve_release("base").release_id
    assert [row["artifact_id"] for row in catalog] == [f"{base}:{HARNESS}", f"{harness_release}:{HARNESS}"]
    assert [row["active"] for row in catalog] == [False, True]


@pytest.mark.unit
def test_each_components_step_reaches_the_tracker_under_its_name(dispatchers: _Dispatchers) -> None:
    """The event the dispatcher records for a step names the trainer that made it."""
    tracker = _RecordingTracker()
    dispatcher, _ = dispatchers.open(experiment_tracker=tracker, hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None
    assert _commit_step(scenario, WEIGHTS) is not None
    dispatcher.registry.experiment_reports.flush()
    # Each job's submission, checkpoint and end is one event, named by its trainer's component and training component.
    assert [event.context.component for event in tracker.events] == [HARNESS] * 3 + [WEIGHTS] * 3
    assert [event.context.backend for event in tracker.events] == [HARNESS] * 3 + [WEIGHTS] * 3
    assert [event.context.step for event in tracker.events] == [1, 2, 3, 4, 5, 6]
    assert [row.get("component") for row in scenario.releases()] == [WEIGHTS] * 3 + [HARNESS] * 3 + [None]


@pytest.mark.unit
def test_the_harness_catalog_is_one_lineage_of_trees(dispatchers: _Dispatchers) -> None:
    """A rejected step, an unlisted weights step and an activation never break the chain or list a new row."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    creation = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    requires = {"training_request": {"id": "r1", "requires": [{"name": "TOKEN", "kind": "env"}]}}
    assert _commit_step(scenario, HARNESS, checkpoint_metrics=requires) is not None
    h1 = scenario.current_artifact_ref().release_id
    assert _commit_step(scenario, WEIGHTS) is not None
    w1 = scenario.current_artifact_ref().release_id

    # A rejected harness step is listed for its page, but it published nothing and names no head.
    _append_records(scenario, 2)
    backends[HARNESS].reject_next = True
    assert _commit_step(scenario, HARNESS) is not None
    assert not any(record.made_artifact for record in scenario.store.history()[-2:])
    assert scenario.current_artifact_ref().release_id == w1
    assert service.harness_release_id(scenario) == h1
    assert service.harness_manifest(headers)["release_id"] == h1
    catalog = service.harness_releases(headers)["releases"]
    # The rejected row is named by the listed release it ran on, so a client's poll agrees with the head;
    # like a flat scenario's rejected row, it lists the head's parent, not itself.
    assert [row["release_id"] for row in catalog] == [creation, h1, h1]
    assert catalog[-1]["metrics"]["selected"] is False
    assert catalog[-1]["composed_release_id"] == w1
    assert catalog[-1]["parent_release_id"] == catalog[1]["parent_release_id"] == creation
    assert "ran on" in service.harness_release_page(headers, 2).lower()
    assert service.harness_manifest(headers, form="source")["release_id"] == h1

    # The next harness release descends from h1 in the listed chain; the weights release it was published
    # on stays under another name, and the requires chain walks the listed rows.
    assert _commit_step(scenario, WEIGHTS) is not None
    w2 = scenario.current_artifact_ref().release_id
    _append_records(scenario, 3)
    assert _commit_step(scenario, HARNESS) is not None
    h2 = scenario.current_artifact_ref().release_id
    catalog = service.harness_releases(headers)["releases"]
    assert [row["release_id"] for row in catalog] == [creation, h1, h1, h2]
    assert catalog[-1]["parent_release_id"] == h1
    assert catalog[-1]["composed_parent_release_id"] == w2
    assert required_by(catalog, h2) == [{"name": "TOKEN", "kind": "env"}]
    assert service.harness_release_id(scenario) == h2
    page = service.harness_release_page(headers, 3)
    assert h2 in page

    # Activating h1's harness serves h1's tree again on the weights served now: the activation publishes that
    # combination, which the catalog does not list, and the head is the listed release h1.
    activated = scenario.activate(f"{h1}:{HARNESS}", HARNESS)
    assert activated.release_id not in (h1, h2)
    catalog = service.harness_releases(headers)["releases"]
    assert [row["release_id"] for row in catalog] == [creation, h1, h1, h2]
    assert service.harness_release_id(scenario) == h1
    assert service.harness_manifest(headers)["release_id"] == h1
    assert required_by(catalog, h1) == [{"name": "TOKEN", "kind": "env"}]

    # Weights held for review are activated as weights: the served tree stays, so the head stays a listed
    # release and names no weights release or combination.
    _append_records(scenario, 4)
    backends[WEIGHTS].hold_next = True
    assert _commit_step(scenario, WEIGHTS) is not None
    checkpoint = next(record for record in reversed(scenario.store.history()) if record.made_artifact)
    assert checkpoint.activated is False
    held = checkpoint.artifact_ref.release_id
    _append_records(scenario, 5)
    assert _commit_step(scenario, HARNESS) is not None
    h3 = scenario.current_artifact_ref().release_id
    combined = scenario.activate(f"{held}:{WEIGHTS}", WEIGHTS).release_id
    assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 3"
    catalog = service.harness_releases(headers)["releases"]
    assert [row["release_id"] for row in catalog] == [creation, h1, h1, h2, h3]
    assert (
        service.harness_manifest(headers, form="source")["release_id"]
        == h3
        not in (w1, w2, held, combined, activated.release_id)
    )
    assert required_by(catalog, h3) == [{"name": "TOKEN", "kind": "env"}]


@pytest.mark.unit
def test_a_training_step_carries_the_active_release_and_keeps_an_activated_component(
    dispatchers: _Dispatchers,
) -> None:
    """After an older weights artifact is activated, a harness step carries the active weights, not the chain tip's."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, WEIGHTS) is not None
    assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 1"
    scenario.activate(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
    assert _commit_step(scenario, HARNESS) is not None
    assert component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights seed",
        HARNESS: "harness step 1",
    }
    active = {row["component"]: row["artifact_id"] for row in scenario.artifacts() if row["active"]}
    assert active == {
        WEIGHTS: f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}",
        HARNESS: f"{scenario.current_artifact_ref().release_id}:{HARNESS}",
    }


@pytest.mark.unit
def test_the_composite_harness_catalog_marks_one_active_row_and_a_client_activates_a_row(
    dispatchers: _Dispatchers,
) -> None:
    """Each catalog row names its harness artifact; exactly one row is active, and posting a row's artifact_id and
    component to the activate route makes that row the active one."""
    from aiohttp.test_utils import TestClient

    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None
    assert _commit_step(scenario, WEIGHTS) is not None
    _append_records(scenario, 2)
    assert _commit_step(scenario, HARNESS) is not None
    headers = {"x-reef-scenario": "agent"}

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            rows = (await (await client.get("/reef/harness/releases", headers=headers)).json())["releases"]
            assert [row["active"] for row in rows] == [False, False, True]
            assert all(row["component"] == HARNESS and row["activated"] for row in rows)
            first = rows[1]
            body = {"artifact_id": first["artifact_id"], "component": first["component"]}
            answer = await client.post("/reef/scenarios/agent/activate", json=body)
            assert answer.status == 200, await answer.text()
            assert (await answer.json())["artifact_id"] == first["artifact_id"]
            rows = (await (await client.get("/reef/harness/releases", headers=headers)).json())["releases"]
            assert [row["artifact_id"] for row in rows if row["active"]] == [first["artifact_id"]]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_creation_manifest_read_that_failed_is_asked_again(monkeypatch: Any, dispatchers: _Dispatchers) -> None:
    """One failed read of the creation artifact must not hide the creation's manifest for the process lifetime."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher)
    original = ArtifactReleaseChain.resolve
    calls = {"failed": 0}

    def fail_once(self: ArtifactReleaseChain, ref: ArtifactRef) -> Artifact:
        if calls["failed"] == 0:
            calls["failed"] += 1
            raise ArtifactNotFound("the remote is away")
        return original(self, ref)

    monkeypatch.setattr(ArtifactReleaseChain, "resolve", fail_once)
    assert scenario.release_queries.creation_components(0) is None
    # Not asked again at the same step (a page polls every few seconds; a remote read is a fetch)...
    assert scenario.release_queries.creation_components(0) is None
    assert calls["failed"] == 1
    # ...but asked again after the next commit.
    components = scenario.release_queries.creation_components(1)
    assert components is not None and set(components) == {WEIGHTS, HARNESS}
    assert calls["failed"] == 1
    assert scenario.release_queries.creation_components(0) is not None


@pytest.mark.unit
def test_an_activation_of_weights_held_at_step_one_is_no_harness_release(dispatchers: _Dispatchers) -> None:
    """The creation artifact has no record; its manifest still says the activated tree did not change."""
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    backends[WEIGHTS].hold_next = True
    scenario = _loaded(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    creation = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    assert _commit_step(scenario, WEIGHTS) is not None
    # The job's checkpoint changed the held kind, so it made an artifact that does not serve.
    checkpoint = next(record for record in reversed(scenario.store.history()) if record.made_artifact)
    assert checkpoint.activated is False
    held = checkpoint.artifact_ref.release_id
    scenario.activate(f"{held}:{WEIGHTS}", WEIGHTS)
    assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 1"
    assert service.harness_release_id(scenario) == creation
    assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [creation]


@pytest.mark.unit
def test_an_activation_under_the_same_tree_is_no_new_harness_head(dispatchers: _Dispatchers) -> None:
    """An activation that changes only the weights names the same harness release; none adds a row."""
    dispatcher, _ = dispatchers.open(hold_local_cycles=True)
    scenario = _loaded(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    base = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    assert _commit_step(scenario, HARNESS) is not None
    harness_release = scenario.current_artifact_ref().release_id
    assert _commit_step(scenario, WEIGHTS) is not None
    first_weights = scenario.current_artifact_ref().release_id
    _append_records(scenario, 2)
    assert _commit_step(scenario, WEIGHTS) is not None

    # Back to the first weights under the same tree: nothing to pull.
    scenario.activate(f"{first_weights}:{WEIGHTS}", WEIGHTS)
    assert component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 1"
    assert service.harness_release_id(scenario) == harness_release
    assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [base, harness_release]

    # Back to the base's harness: the seed tree is served again, named by the base release.
    scenario.activate(f"{scenario.repository.backend.resolve_release('base').release_id}:{HARNESS}", HARNESS)
    assert service.harness_release_id(scenario) == base
    assert service.harness_manifest(headers)["release_id"] == base
    assert service.harness_manifest(headers)["files"] == {"harness.txt": "harness seed"}
    assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [base, harness_release]


@pytest.mark.unit
def test_adopted_checkpoint_is_attributed_to_the_trainer_that_made_it(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A checkpoint read back from the artifact head names its component, so a lost commit history resets no trainer."""
    backend_factory = _seeded_repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        records_dir=tmp_path / "records-1", backend_factory=backend_factory, hold_local_cycles=True
    )
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None
    head = scenario.repository.materialize(scenario.current_artifact_ref())
    checkpoint = head.metadata[SCENARIO_METADATA_KEY]
    assert isinstance(checkpoint, Mapping)
    assert checkpoint["component"] == HARNESS
    _, _, adopted = parse_scenario_metadata(checkpoint, checkpoint_head=head.ref)
    assert adopted is not None and adopted.component == HARNESS
    dispatcher.close()

    # The record store is gone; the checkpoint is adopted from the head and belongs to the harness trainer.
    recovered, _ = dispatchers.open(records_dir=tmp_path / "records-2", backend_factory=backend_factory)
    scenario = _scenario(recovered)
    # The head names the last commit of the harness job: its end, after its submission and its checkpoint.
    assert scenario.scenario_step == 3
    harness = JobLoopState.from_dict(scenario.trainer_for(HARNESS).state)
    assert harness.finished_jobs == 1 and harness.job is None
    assert JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state).finished_jobs == 0
    last = scenario.last_commit_for(HARNESS)
    assert last is not None and last.step == 3 and last.component == HARNESS
    assert scenario.last_commit_for(WEIGHTS) is None


@dataclass(frozen=True)
class _OneTrainerRecipe(_TwoTrainerRecipe):
    """The harness trainer alone, serving one component: the shape of every shipped recipe."""

    def build_surface(self, scenario: str) -> Surface:
        return Surface(components={HARNESS: ComponentSurface(files=TextFileTree())})


@pytest.mark.unit
def test_a_lone_trainer_names_no_component_on_record(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A scenario of one trainer writes what it wrote before components: its commit record and the checkpoint's
    scenario metadata name no component."""
    recipe = _OneTrainerRecipe(backends={HARNESS: ComponentJobs(HARNESS, tmp_path / "candidates")})
    dispatcher, _ = dispatchers.open(
        recipe=recipe, backend_factory=_seeded_repository(tmp_path, (HARNESS,)), hold_local_cycles=True
    )
    scenario = _loaded(dispatcher, 1, 2)
    assert _commit_step(scenario, HARNESS) is not None
    record = scenario.store.history()[-1]
    assert (record.component, record.components) == (None, None)
    checkpoint = scenario.repository.materialize(scenario.current_artifact_ref()).metadata[SCENARIO_METADATA_KEY]
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
        ended = _training_components(_scenario(dispatcher))
        return ended.count(WEIGHTS) == ended.count(HARNESS) == 2

    assert _wait_for(both_trained_twice, timeout_seconds=20), "both component workers did not train twice"
    current = _scenario(dispatcher)
    assert component_files(current, current.current_artifact_ref()) == {
        WEIGHTS: "weights step 2",
        HARNESS: "harness step 2",
    }
    assert dispatcher.build_training_status()["error"] is None


@pytest.mark.unit
def test_commit_record_carries_its_component() -> None:
    record = CommitRecord(
        scenario="agent",
        step=3,
        artifact_ref=ArtifactRef("c", "r3", "r2"),
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
        step=4,
        artifact_ref=ArtifactRef("c", "r1", None),
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
def test_component_trainers_must_match_the_surface() -> None:
    def trainer() -> Trainer:
        return Trainer.build("agent", SQLiteRecordStore(), pipeline_factory=TrainingPipeline)

    flat = Surface(components={HARNESS: ComponentSurface(files=TextFileTree())})
    composed = Surface(components={WEIGHTS: ComponentSurface(), HARNESS: ComponentSurface(files=TextFileTree())})
    harness_only = (ComponentTrainer(HARNESS, trainer()),)
    assert validate_component_trainers(harness_only, flat) == harness_only
    assert validate_component_trainers(harness_only, composed) == harness_only
    named = (ComponentTrainer(WEIGHTS, trainer()), ComponentTrainer(HARNESS, trainer()))
    assert validate_component_trainers(named, composed) == named
    records_only = (ComponentTrainer(RECORDS_COMPONENT, trainer()),)
    assert validate_component_trainers(records_only, Surface()) == records_only
    with pytest.raises(ReefError, match="serving no component"):
        validate_component_trainers(harness_only, Surface())
    with pytest.raises(ValueError, match="non-empty"):
        ComponentTrainer("", trainer())
    with pytest.raises(ReefError, match="does not serve"):
        validate_component_trainers((ComponentTrainer("config", trainer()),), composed)
    with pytest.raises(ReefError, match="distinct"):
        validate_component_trainers(
            (ComponentTrainer(HARNESS, trainer()), ComponentTrainer(HARNESS, trainer())), composed
        )


@pytest.mark.unit
def test_a_surface_built_with_the_flat_keywords_serves_one_component() -> None:
    """The surface main documents, Surface(inference=..., files=...), is one component; dataclasses.replace of a
    capability replaces it on that component; a surface of several components takes them per component."""
    import dataclasses

    from reef.surface.base import FLAT_COMPONENT

    tree = TextFileTree()
    surface = Surface(files=tree)
    assert surface.names == (FLAT_COMPONENT,) and surface.single and surface.files is tree
    named = Surface(components={HARNESS: ComponentSurface()})
    replaced = dataclasses.replace(named, files=tree)
    assert replaced.names == (HARNESS,) and replaced.components[HARNESS].files is tree
    with pytest.raises(ValueError, match="per component"):
        Surface(components={WEIGHTS: ComponentSurface(), HARNESS: ComponentSurface()}, files=tree)


@pytest.mark.unit
def test_service_start_notification_cannot_reopen_stopping_workers(dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open(hold_local_cycles=True)
    scenario = _scenario(dispatcher, 1)
    dispatcher.stop_local_cycles()
    dispatcher.open_local_cycles()
    assert dispatcher.local_workers.process_step("agent", HARNESS) is False
    assert backends[HARNESS].prepared == 0
    assert scenario.scenario_step == 0
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


def _job_events(scenario: Scenario) -> list[str | None]:
    return [_job_event(record) for record in scenario.store.history() if record.component == WEIGHTS]


@pytest.mark.unit
def test_dispatcher_runs_the_weights_jobs_and_the_harness_on_their_own_workers(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """Each trainer has its worker: the weights worker submits a job per batch and commits its checkpoint and end,
    and each job starts from the checkpoint of the job before it."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training)
    served = _scenario(dispatcher).serving_artifact_id()
    for step in (1, 2):
        for record in _records(step, served):
            dispatcher.accept_record(record)

    def both_trained_twice() -> bool:
        ended = _training_components(_scenario(dispatcher))
        return ended.count(WEIGHTS) == ended.count(HARNESS) == 2

    assert _wait_for(both_trained_twice, timeout_seconds=20), "both component workers did not train twice"
    current = _scenario(dispatcher)
    assert component_files(current, current.current_artifact_ref()) == {
        WEIGHTS: "weights job 2",
        HARNESS: "harness step 2",
    }
    first, second = training.submitted
    assert second.start == training.statuses[first.job_id].checkpoints[-1].path
    assert dispatcher.build_training_status()["error"] is None


@pytest.mark.unit
def test_a_manual_weights_trainer_waits_for_inserted_samples_and_submits_them_with_the_instruction(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """In manual mode a weights stage takes an instruction as a request to train: the job waits until a client
    inserts samples into the trainer's manual buffer, then carries them and the instruction, and the submission
    consumes both and empties the buffer."""
    training = WeightsJobComponent(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = _loaded(dispatcher)
    scenario.set_training_mode("manual")
    dispatcher.accept_instruction(
        "agent", {"text": "learn the tool", "session": "session-1", "release_id": "release-1"}, request_id="t1"
    )
    assert scenario.prepare_training_step(WEIGHTS) is None
    _append_records(scenario, 1)
    make_samples(scenario)
    # The samples service stores the step's sample, but no sample joins the manual buffer by itself: the instruction
    # waits.
    assert scenario.prepare_training_step(WEIGHTS) is None
    trainer = scenario.trainer_for(WEIGHTS)
    held = tuple(trainer.pipeline.held)
    # In a scenario of several components, a sample's ID names its component first.
    assert held == (f"{WEIGHTS}:r1:0",)
    trainer.change_buffer(held, insert=True)
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    assert trainer.buffer_listing()["samples"] == []
    [job] = _submitted(training, 1)
    assert job.data.data_id == "agent:instruction:t1"
    assert job.data.request is not None and job.data.request.id == "t1"
    assert {"t1", "i1", "r1"} <= scenario.store.history()[-1].consumed_ids
    _settle_weights_job(scenario)
    assert _job_events(scenario) == ["submitted", "checkpoint", "finished"]


@pytest.mark.unit
def test_a_restart_polls_the_running_weights_job_and_commits_its_checkpoint(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The running job is in the weights trainer's committed state, also after the harness moved the scenario step.
    The restarted trainer polls that job instead of reserving its rows again, and commits the checkpoint that the
    job saved meanwhile."""
    training = HeldJobs(tmp_path / "checkpoints")
    factory = _seeded_repository(tmp_path)
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True, backend_factory=factory)
    scenario = _loaded(dispatcher, 1, 2)
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    [job] = _submitted(training, 1)
    assert scenario.prepare_training_step(WEIGHTS) is None
    assert _commit_step(scenario, HARNESS) == [("i1", "r1")]
    assert _commit_step(scenario, HARNESS) == [("i2", "r2")]
    dispatcher.close()

    training.finish(job)
    restarted, _ = dispatchers.open(training=training, backend_factory=factory)
    # What a restart does for every registration a repository lists: the load wakes the weights worker.
    restarted.preload_scenarios(("agent",))
    scenario = _scenario(restarted)
    assert _wait_for(lambda: _job_events(scenario)[:3] == ["submitted", "checkpoint", "finished"])
    assert training.submitted[0] == job
    # The base answered the rows behind the job, so they are stale once the job's checkpoint serves. Rows that the
    # checkpoint answers train in the next job, which starts from that checkpoint.
    for record in _records(3, scenario.serving_artifact_id()):
        restarted.accept_record(record)
    assert _wait_for(lambda: len(training.submitted) == 2)
    assert [item.source_agent_record_ids for item in training.submitted[1].data.samples] == [("i3", "r3")]
    assert training.submitted[1].start == str(tmp_path / "checkpoints" / "1")


@pytest.mark.unit
def test_a_restarted_training_worker_runs_again_the_job_that_the_stopped_one_gave_back(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A stopped training worker gives its running job back to the jobs queue. The restarted worker takes it and
    submits it to its component, which runs it again from its latest checkpoint; a component that lost the job does
    not end it."""
    factory = _seeded_repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        training=HeldJobs(tmp_path / "checkpoints"), hold_local_cycles=True, backend_factory=factory
    )
    assert _commit_step(_loaded(dispatcher, 1), WEIGHTS) == [("i1", "r1")]
    dispatcher.close()

    other = HeldJobs(tmp_path / "other-checkpoints")
    restarted, _ = dispatchers.open(training=other, backend_factory=factory)
    restarted.preload_scenarios(("agent",))
    scenario = _scenario(restarted)
    assert _wait_for(lambda: len(other.submitted) == 1)
    assert _job_events(scenario) == ["submitted"]
    state = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state)
    assert state.job is not None and state.job.job_id == other.submitted[0].job_id


@pytest.mark.unit
def test_a_weights_start_cancels_the_running_job_and_moves_the_next_jobs_start(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A new start imports the release into the training component and cancels the running job. The next job
    starts from the imported release, not from the last checkpoint, and serving does not change."""
    training = HeldJobs(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    training.finish(_submitted(training, 1)[0])
    _settle_weights_job(scenario)
    _append_records(scenario, 2)
    assert _commit_step(scenario, WEIGHTS) == [("i2", "r2")]
    running = _submitted(training, 2)[-1]
    assert running.start == str(tmp_path / "checkpoints" / "1")

    served = scenario.current_artifact_ref()
    scenario.train_from(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
    # The training worker stops the job on the component once it sees the cancelled item.
    assert _wait_for(lambda: training.statuses[running.job_id].error == "cancelled")
    [imported] = training.imported
    assert scenario.current_artifact_ref() == served
    assert component_files(scenario, served)[WEIGHTS] == "weights job 1"
    # The harness trainer's start is its own: it does not move.
    assert scenario.store.history()[-1].component == WEIGHTS
    state = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state)
    assert (state.job, state.start) == (None, str(imported))
    _append_records(scenario, 3)
    assert _commit_step(scenario, WEIGHTS) == [("i3", "r3")]
    assert _submitted(training, 3)[-1].start == str(imported)


@pytest.mark.unit
def test_a_restart_before_the_trainers_next_commit_imports_its_start_again(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The process stops after the ``train_from`` record and before the trainer's next commit records the start.
    The trainer's state recovers from its earlier commit, so the restarted scenario imports the release again."""
    training = HeldJobs(tmp_path / "checkpoints")
    factory = _seeded_repository(tmp_path)
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True, backend_factory=factory)
    scenario = _loaded(dispatcher, 1)
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    training.finish(_submitted(training, 1)[0])
    _settle_weights_job(scenario)
    served = scenario.current_artifact_ref()
    scenario.train_from(f"{scenario.repository.backend.resolve_release('base').release_id}:{WEIGHTS}", WEIGHTS)
    assert JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state).start == str(training.imported[0])
    dispatcher.close()

    restarted, _ = dispatchers.open(training=training, hold_local_cycles=True, backend_factory=factory)
    scenario = _scenario(restarted, 2)
    assert len(training.imported) == 2
    assert scenario.current_artifact_ref() == served
    state = JobLoopState.from_dict(scenario.trainer_for(WEIGHTS).state)
    assert state.start == str(training.imported[-1])
    assert _commit_step(scenario, WEIGHTS) == [("i2", "r2")]
    assert _submitted(training, 2)[-1].start == state.start


@pytest.mark.unit
def test_deleting_a_scenario_cancels_its_running_training_job(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """Nothing would read the job's checkpoints once its scenario is gone, so the delete cancels the job."""
    training = HeldJobs(tmp_path / "checkpoints")
    dispatcher, _ = dispatchers.open(training=training, hold_local_cycles=True)
    assert _commit_step(_loaded(dispatcher, 1), WEIGHTS) == [("i1", "r1")]
    [job] = _submitted(training, 1)
    dispatcher.delete_scenario("agent")
    # The training worker stops the job on the component once it sees the cancelled item.
    assert _wait_for(lambda: training.statuses[job.job_id].error == "cancelled")
    assert not dispatcher.registry.has("agent")
