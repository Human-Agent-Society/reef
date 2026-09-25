"""One trainer per release component: independent workers that meet at the commit boundary."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
import uuid
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestServer

from reef.artifact import Artifact, ArtifactNotFound, ArtifactRef, InMemoryRepositoryBackend
from reef.artifact.release_chain import ArtifactReleaseChain
from reef.core import AgentRecord, RequestType
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError, ScenarioBusy
from reef.core.requirements import required_by
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.recipe import Recipe
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.runtime.interfaces import RuntimeContractError
from reef.scenario import Scenario, StaleTrainingResultError
from reef.scenario.scenario import validate_component_trainers
from reef.service.app import create_app
from reef.service.request_service import RequestService
from reef.storage.commits import (
    SCENARIO_METADATA_KEY,
    CommitLogError,
    CommitRecord,
    parse_scenario_metadata,
    scenario_metadata_for,
)
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.train import CandidateBackend, ComponentTrainer, PreparedStep, Trainer, TrainStepResult
from reef.train.evaluation import EvaluationResult, UpdateCandidate
from reef.train.processors.base import DataProcessor
from reef.train.processors.computed import ComputedFeedbackProcessor, SupportsReceipt
from reef.train.processors.reported import GroupDecision, ReportContext
from reef.train.types import TrainingBatch

from ._threshold_processor import ThresholdProcessor
from ._trajectories import policy_trajectory
from .runtime_stubs import StubTrainingRuntime

WEIGHTS = "weights"
HARNESS = "harness"


class _ComponentBackend(CandidateBackend):
    """A local candidate cycle publishing one file for its component per step."""

    def __init__(self, component: str, artifact_dir: Path, *, stale_policy: str = "refuse") -> None:
        self.component = component
        self.artifact_dir = artifact_dir
        self.stale_policy = stale_policy
        self.prepared = 0
        self.evaluated = 0
        self.reevaluations = 0
        self.reject_next = False
        self.hold_next = False
        self.result: TrainStepResult | None = None

    @property
    def stale_result_policy(self) -> str:
        return self.stale_policy

    def unblock(self) -> None:
        """Let go of any step the test holds, so the dispatcher can close."""

    def initial_state(self) -> Mapping[str, Any]:
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        step = int(state["steps"]) + 1
        self.prepared += 1
        if self.reject_next:
            # A rejected step publishes nothing: the row it leaves carries the head's own release.
            self.reject_next = False
            self.result = TrainStepResult({"steps": step}, metrics={"prepared": self.prepared, "selected": False})
            return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})
        path = self.artifact_dir / self.component / uuid.uuid4().hex
        path.mkdir(parents=True)
        (path / f"{self.component}.txt").write_text(f"{self.component} step {step}", encoding="utf-8")
        self.result = TrainStepResult(
            {"steps": step}, metrics={"prepared": self.prepared}, artifact=Artifact.local(path), pending=self.hold_next
        )
        self.hold_next = False
        return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})

    def prepare_reevaluation(self, prepared):
        self.reevaluations += 1
        return prepared

    def evaluate(self, candidate):
        self.evaluated += 1
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision):
        assert self.result is not None
        return self.result

    def abort_step(self, prepared):
        pass


class _SlowBackend(_ComponentBackend):
    """The same cycle whose evaluation waits for the test to let it go, as a minutes long episode run would."""

    def __init__(self, component: str, artifact_dir: Path, *, stale_policy: str = "merge") -> None:
        super().__init__(component, artifact_dir, stale_policy=stale_policy)
        self.evaluating = threading.Event()
        self.release = threading.Event()

    def unblock(self) -> None:
        self.release.set()

    def evaluate(self, candidate):
        self.evaluating.set()
        assert self.release.wait(30)
        return super().evaluate(candidate)


class _AwayBackend(_ComponentBackend):
    """A local cycle whose backend is away: every preparation fails."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.attempts = 0

    def prepare_step(self, batch, state, scenario_step):
        self.attempts += 1
        raise RuntimeError("proposer away")


class _FlakyBackend(_ComponentBackend):
    """A local cycle whose backend fails once, then answers."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.failures_left = 1

    def prepare_step(self, batch, state, scenario_step):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("proposer away once")
        return super().prepare_step(batch, state, scenario_step)


class _DispatchedBackend(_ComponentBackend):
    """The same cycle run as a dispatched job: its result carries a job identity the backend must finish."""

    def __init__(self, component: str, artifact_dir: Path, job_id: str) -> None:
        super().__init__(component, artifact_dir)
        self.job_id = job_id

    @property
    def dispatched(self) -> bool:
        return True

    def settle_step(self, prepared, decision):
        result = super().settle_step(prepared, decision)
        assert result.artifact is not None and result.artifact.local_path is not None
        load = f"inc:{self.prepared}"
        return TrainStepResult(
            result.state,
            metrics=result.metrics,
            artifact=Artifact.local(result.artifact.local_path, metadata={"runtime_load_id": load}),
            runtime_load_id=load,
            training_job_id=self.job_id,
        )


class _ColocatedBackend(_DispatchedBackend):
    """A dispatched job that holds the served engine while it runs, as a colocated Slime stack does."""

    @property
    def colocated(self) -> bool:
        return True


class _HybridThresholdProcessor(ThresholdProcessor):
    """The same processor, taking instructions too."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})


@dataclass(frozen=True)
class _TwoTrainerRecipe(Recipe):
    """A scenario serving weights and a harness tree, each evolved by its own local backend."""

    backends: Mapping[str, CandidateBackend]
    hybrid_components: frozenset[str] = frozenset()

    def build_surface(self, scenario: str) -> Surface:
        return Surface(
            components={
                WEIGHTS: ComponentSurface(),
                HARNESS: ComponentSurface(files=TextFileTree()),
            }
        )

    def processor_for(self, component: str) -> type[DataProcessor]:
        return _HybridThresholdProcessor if component in self.hybrid_components else ThresholdProcessor

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            processor = self.processor_for(component)
            return lambda context: processor(context.with_config({"batch_size": 1}))

        return tuple(
            ComponentTrainer(
                component,
                Trainer.build(
                    scenario,
                    records,
                    processor_factory=factory_for(component),
                    candidate_backend=backend,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                ),
            )
            for component, backend in self.backends.items()
        )


class _RecordingTracker(ExperimentTracker):
    """Keeps every training event the dispatcher records."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def bind_scenario(self, **kwargs):
        return NullExperimentLogger()

    def correlation_metrics(self, context):
        return {}

    def record(self, event):
        self.events.append(event)

    def record_rollback(self, event):
        pass

    def close(self):
        pass


def _records(step: int, scenario: str = "agent") -> tuple[AgentRecord, AgentRecord]:
    """One inference and the report on it; rows of scenarios other than ``agent`` carry the scenario's name."""
    prefix = "" if scenario == "agent" else f"{scenario}-"
    inference = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
        agent_record_id=f"{prefix}i{step}",
    )
    report = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"{prefix}i{step}"]},
        agent_record_id=f"{prefix}r{step}",
        references=(f"{prefix}i{step}",),
    )
    return inference, report


def _report(record_id: str, *references: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": list(references)},
        agent_record_id=record_id,
        references=references,
    )


def _append_records(scenario: Scenario, *steps: int) -> None:
    """Append each step's rows to the store directly, so the test, not a worker, owns when a trainer prepares."""
    for step in steps:
        for record in _records(step, scenario.name):
            scenario.records.append(record)


def _scenario(dispatcher: Dispatcher, *steps: int, name: str = "agent") -> Scenario:
    """The scenario, created on first use, with each step's rows appended."""
    scenario = dispatcher.get_or_create_scenario(name)
    assert scenario is not None
    _append_records(scenario, *steps)
    return scenario


def _wait_for(condition: Callable[[], bool], timeout_seconds: float = 10) -> bool:
    """Whether ``condition`` held within the timeout, checked every few milliseconds."""
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _seeded_repository(tmp_path: Path, components: tuple[str, ...] = (WEIGHTS, HARNESS)):
    """A repository whose base seeds one file per component, unless the test wrote its own base first."""
    initial = tmp_path / "initial"
    if not initial.exists():
        for component in components:
            (initial / component).mkdir(parents=True)
            (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    return InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")


def _local_pair(tmp_path: Path) -> dict[str, _ComponentBackend]:
    return {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}


def _dispatched_pair(
    tmp_path: Path, job_id: str = "job-1", *, harness: _ComponentBackend | None = None
) -> dict[str, _ComponentBackend]:
    return {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", job_id),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates") if harness is None else harness,
    }


def _dispatcher(
    tmp_path: Path,
    *,
    backends: dict[str, _ComponentBackend] | None = None,
    recipe: Recipe | None = None,
    hybrid_components: frozenset[str] = frozenset(),
    training: StubTrainingRuntime | None = None,
    records_dir: Path | None = None,
    backend_factory: Any = None,
    experiment_tracker: ExperimentTracker | None = None,
    hold_local_cycles: bool = False,
) -> tuple[Dispatcher, dict[str, _ComponentBackend]]:
    """A dispatcher over the two trainer recipe of ``backends``, or over ``recipe`` when one is given."""
    repository = _seeded_repository(tmp_path) if backend_factory is None else backend_factory
    if recipe is not None:
        backends = {} if backends is None else backends
    elif training is None:
        backends = _local_pair(tmp_path) if backends is None else backends
        recipe = _TwoTrainerRecipe(backends=backends, hybrid_components=hybrid_components)
    else:
        # A training runtime binds the scenario to the training thread, which drives the dispatched job.
        backends = _local_pair(tmp_path) if backends is None else backends
        recipe = _TwoTrainerRecipe(
            backends=backends,
            hybrid_components=hybrid_components,
            runtime=training.inference,
            training_runtime=training,
        )
    records = tmp_path / "records" if records_dir is None else records_dir
    dispatcher = Dispatcher(
        recipe,
        repository,
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
        experiment_tracker=experiment_tracker,
        hold_local_cycles=hold_local_cycles,
    )
    return dispatcher, backends


class _Dispatchers:
    """Opens a test's dispatchers over its directory and closes each at teardown, letting held backends go first."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.opened: list[tuple[Dispatcher, Mapping[str, _ComponentBackend]]] = []

    def open(self, **options: Any) -> tuple[Dispatcher, dict[str, _ComponentBackend]]:
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


def _component_files(scenario: Scenario, ref: ArtifactRef) -> dict[str, str]:
    head = scenario.repository.materialize(ref)
    return {
        component: (TextFileTree().read_files(head.component(component)) or {})[f"{component}.txt"]
        for component in (WEIGHTS, HARNESS)
    }


def _training_components(scenario: Scenario) -> list[str]:
    return [row["component"] for row in scenario.releases() if row["operation"] == "training"]


def _commit_step(scenario: Scenario, component: str) -> list[tuple[str, ...]] | None:
    """Prepare and commit ``component``'s next step; the sources of its batch, or None when nothing was ready."""
    result = scenario.prepare_training_step(component)
    if result is None:
        return None
    batch = scenario.trainer_for(component).pending_batch
    assert batch is not None
    sources = [tuple(item.source_agent_record_ids) for item in batch.items]
    scenario.commit(result, component=component)
    return sources


@pytest.mark.unit
def test_component_trainers_meet_at_the_commit_boundary(dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    assert [bound.component for bound in scenario.component_trainers] == [WEIGHTS, HARNESS]
    assert scenario.trainer_for(HARNESS) is scenario.component_trainers[1].trainer
    base = scenario.current_artifact_ref().release_id

    weights = scenario.prepare_training_step(WEIGHTS)
    harness = scenario.prepare_training_step(HARNESS)
    assert weights is not None and harness is not None
    assert scenario.trainer_for(WEIGHTS).pending_base_release_id == base

    scenario.commit(harness, component=HARNESS)
    after_harness = scenario.current_artifact_ref()
    assert _component_files(scenario, after_harness) == {WEIGHTS: "weights seed", HARNESS: "harness step 1"}

    # The weights result was prepared against the base, which the harness commit replaced.
    with pytest.raises(StaleTrainingResultError, match=WEIGHTS):
        scenario.commit(weights, component=WEIGHTS)
    assert scenario.scenario_step == 1
    scenario.retry_pending(WEIGHTS)
    assert scenario.trainer_for(WEIGHTS).pending_batch is not None
    retried = scenario.prepare_training_step(WEIGHTS)
    assert retried is not None and retried is not weights
    assert backends[WEIGHTS].prepared == 2
    assert scenario.trainer_for(WEIGHTS).pending_base_release_id == after_harness.release_id
    scenario.commit(retried, component=WEIGHTS)
    assert _component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness step 1",
    }

    records = scenario.store.history()
    assert [(record.component, record.base_release_id) for record in records] == [
        (HARNESS, base),
        (WEIGHTS, after_harness.release_id),
    ]
    # A commit retires no row: each trainer consumed the same rows on its own record.
    assert scenario.records.count("agent") == 2
    assert records[0].consumed_ids == records[1].consumed_ids == frozenset({"i1", "r1"})

    # Each trainer recovers from its own commits.
    reloaded = dispatcher._registry.reload("agent")
    assert reloaded.trainer_for(WEIGHTS).state == {"steps": 1}
    assert reloaded.trainer_for(HARNESS).state == {"steps": 1}
    assert reloaded.scenario_step == 2
    _append_records(reloaded, 2)
    assert reloaded.prepare_training_step(HARNESS) is not None
    assert reloaded.prepare_training_step(WEIGHTS) is not None


@pytest.mark.unit
def test_a_local_step_of_one_trainer_does_not_hold_up_the_others_or_the_status(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """While the harness evaluates for minutes, the weights job still reserves and commits, and status still reads."""
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path, harness=slow))
    scenario = _scenario(dispatcher)
    base = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    outcome: dict[str, Any] = {}

    def harness_step() -> None:
        outcome["result"] = scenario.prepare_training_step(HARNESS)

    worker = threading.Thread(target=harness_step)
    worker.start()
    assert slow.evaluating.wait(10)
    # The harness step is mid evaluation: the dispatched trainer reserves, executes and commits meanwhile...
    assert scenario.reserve_training_batch(WEIGHTS) is not None
    execution = scenario.execute_reserved_training_step(WEIGHTS)
    assert execution.outcome == "commit" and execution.result is not None
    scenario.commit(execution.result, component=WEIGHTS)
    served = scenario.current_artifact_ref().release_id
    assert served != base
    # ...and the status reads the record it made.
    last = scenario.last_commit_for(WEIGHTS)
    assert last is not None and last.step == 1
    assert dispatcher.build_training_status()["scenarios"]["agent"]["scenario_step"] == 1
    slow.release.set()
    worker.join(10)
    assert not worker.is_alive()
    harness = outcome["result"]
    assert harness is not None
    # Its result was prepared against the base; it merges onto the release served now.
    scenario.commit(harness, component=HARNESS)
    assert scenario.releases()[0]["metrics"]["merged_onto"] == served
    assert _component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness step 1",
    }


@pytest.mark.unit
def test_a_colocated_weights_job_waits_for_the_local_cycle_that_needs_the_engine(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A colocated job holds the served engine, so it starts only after the harness cycle mid evaluation ends."""
    colocated = _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1")
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends={WEIGHTS: colocated, HARNESS: slow})
    scenario = _scenario(dispatcher, 1)
    worker = threading.Thread(target=lambda: dispatcher._process_local_backend_step("agent", HARNESS))
    worker.start()
    assert slow.evaluating.wait(10)
    assert scenario.reserve_training_batch(WEIGHTS) is not None
    executions: list[Any] = []

    def job() -> None:
        with dispatcher.dispatched_turn(scenario, WEIGHTS):
            executions.append(scenario.execute_reserved_training_step(WEIGHTS))

    trainer = threading.Thread(target=job)
    trainer.start()
    trainer.join(1)
    # The job has not started: the harness cycle still holds the engine it evaluates through.
    assert trainer.is_alive()
    assert colocated.prepared == 0
    slow.release.set()
    worker.join(10)
    trainer.join(10)
    assert not trainer.is_alive()
    assert colocated.prepared == 1
    assert executions[0].outcome == "commit"
    assert scenario.scenario_step == 1


@pytest.mark.unit
def test_a_colocated_weights_job_waits_for_every_scenario_that_shares_the_engine(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """Admission is engine wide, so another scenario's harness cycle mid evaluation finishes first too."""
    colocated = _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1")
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends={WEIGHTS: colocated, HARNESS: slow})
    agent = _scenario(dispatcher, 1)
    _scenario(dispatcher, 1, name="other")
    worker = threading.Thread(target=lambda: dispatcher._process_local_backend_step("other", HARNESS))
    worker.start()
    assert slow.evaluating.wait(10)
    assert agent.reserve_training_batch(WEIGHTS) is not None
    executions: list[Any] = []

    def job() -> None:
        with dispatcher.dispatched_turn(agent, WEIGHTS):
            executions.append(agent.execute_reserved_training_step(WEIGHTS))

    trainer = threading.Thread(target=job)
    trainer.start()
    trainer.join(1)
    assert trainer.is_alive()
    assert colocated.prepared == 0
    slow.release.set()
    worker.join(10)
    trainer.join(10)
    assert not trainer.is_alive()
    assert executions[0].outcome == "commit"


@pytest.mark.unit
def test_a_dispatched_turn_wakes_every_local_worker_and_yields_before_the_next_cycle(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """Workers of every scenario stand aside for a waiting job and are woken when its turn ends."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    for name in ("agent", "other"):
        _scenario(dispatcher, 1, name=name)
    # A job is waiting for its turn: a worker woken now runs no cycle.
    dispatcher._training.turn_waiting.set()
    dispatcher._start_local_backend_worker("other", HARNESS)
    time.sleep(0.5)
    assert backends[HARNESS].prepared == 0
    # The turn ends: every worker is woken and the cycle runs.
    dispatcher._training.turn_waiting.clear()
    dispatcher.wake_local_workers()
    _wait_for(lambda: backends[HARNESS].prepared > 0)
    assert backends[HARNESS].prepared == 1


@pytest.mark.unit
def test_a_reload_after_a_training_failure_wakes_the_local_workers(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """The rebuilt instance holds the local components' rows unread; their workers look again."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher)
    dispatcher._training.turn_waiting.set()
    dispatcher._start_local_backend_worker("agent", HARNESS)
    time.sleep(0.2)
    dispatcher._training.turn_waiting.clear()
    _append_records(scenario, 1)
    dispatcher._reload_after_training_failure("agent", RuntimeError("job away"))
    _wait_for(lambda: backends[HARNESS].prepared > 0)
    assert backends[HARNESS].prepared == 1


@pytest.mark.unit
def test_a_reload_after_a_weights_failure_leaves_the_harness_cycle_in_flight_its_instance(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """The training thread rebuilds the scenario at once; the instance a harness cycle evaluates on stays open until
    the cycle ends, and the cycle then looks again on the rebuilt instance instead of failing."""
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path, harness=slow))
    scenario = _scenario(dispatcher, 1)
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
            outcome.append(dispatcher._process_local_backend_step("agent", HARNESS))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=cycle)
    worker.start()
    assert slow.evaluating.wait(10)
    reloading = threading.Thread(
        target=lambda: dispatcher._reload_after_training_failure("agent", RuntimeError("weights commit refused"))
    )
    reloading.start()
    reloading.join(5)
    # The training thread does not wait for the cycle: its next turn needs the rebuilt instance.
    assert not reloading.is_alive()
    rebuilt = dispatcher._registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    slow.release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert errors == [] and outcome == [True]
    assert closed_under_the_cycle == [False]
    assert dispatcher.build_training_status()["error"] is None
    # The rows the cycle held train on the rebuilt instance.
    assert dispatcher._process_local_backend_step("agent", HARNESS) is True
    assert _training_components(rebuilt) == [HARNESS]


@pytest.mark.unit
def test_a_harness_backlog_is_kept_for_the_next_start_when_the_service_stops(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A stopping service takes its routes away first, so a harness cycle's model calls through them fail and would
    come back as a skip that consumes the batch: the cycle ending then commits nothing and no further cycle starts."""
    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path, harness=slow))
    _scenario(dispatcher, 1, 2)
    dispatcher._start_local_backend_worker("agent", HARNESS)
    assert slow.evaluating.wait(10)
    dispatcher.stop_local_cycles()
    closing = threading.Thread(target=dispatcher.close)
    closing.start()
    slow.release.set()
    closing.join(10)
    assert not closing.is_alive()
    assert slow.prepared == 1
    # The next start finds both batches unread and trains them, with no new record to wake the harness.
    restarted, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    # What a restart does for every registration a repository lists (this in-memory one lists none).
    restarted._preload_scenarios(("agent",))
    _wait_for(lambda: backends[HARNESS].prepared >= 2)
    scenario = _scenario(restarted)
    assert backends[HARNESS].prepared == 2 and _training_components(scenario) == [HARNESS, HARNESS]


@pytest.mark.unit
def test_held_local_cycles_run_nothing_until_the_service_answers(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A service whose recipe calls itself holds its local cycles while it starts: a harness step woken by the
    preload or by a record would fail its model calls into a skip that consumes its batch."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path), hold_local_cycles=True)
    scenario = _scenario(dispatcher, 1)
    dispatcher._preload_scenarios(("agent",))
    for record in _records(2):
        dispatcher.accept_record(record)
    time.sleep(0.5)
    assert backends[HARNESS].prepared == 0
    dispatcher.open_local_cycles()
    _wait_for(lambda: backends[HARNESS].prepared >= 2)
    assert backends[HARNESS].prepared == 2 and _training_components(scenario) == [HARNESS, HARNESS]


@pytest.mark.unit
def test_the_app_opens_held_local_cycles_once_it_listens(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path), hold_local_cycles=True)
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
def test_an_instance_parked_while_its_cycle_lock_is_held_closes_when_the_holder_lets_go(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A reload lands while a colocated job holds every cycle lock: the instance it replaced closes when the job's
    turn ends, not at shutdown."""
    backends = {
        WEIGHTS: _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = dispatchers.open(backends=backends)
    scenario = _scenario(dispatcher)
    closed: list[bool] = []
    close = scenario.close

    def observed_close() -> None:
        closed.append(True)
        close()

    scenario.close = observed_close  # type: ignore[method-assign]
    with dispatcher.dispatched_turn(scenario, WEIGHTS):
        dispatcher._registry.reload("agent")
        assert closed == []
    assert closed == [True]
    with dispatcher._training.lock:
        assert dispatcher._training.replaced.get("agent") in (None, [])


@pytest.mark.unit
def test_deleting_a_scenario_whose_training_job_is_out_waits_for_the_job(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The job could neither commit nor be acknowledged without its scenario, so the delete answers busy until it lands."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher, 1)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    with pytest.raises(ScenarioBusy, match="training job is out"):
        dispatcher.delete_scenario("agent")
    assert dispatcher._registry.has("agent")
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert dispatcher.delete_scenario("agent")["scenario"] == "agent"
    assert not dispatcher._registry.has("agent")


@pytest.mark.unit
def test_a_scenario_removed_under_its_training_job_is_an_error_not_a_silent_replay(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The registry path a delete no longer takes: a job without its scenario needs operator recovery."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    old = _scenario(dispatcher, 1)
    batch = old.reserve_training_batch(WEIGHTS)
    assert batch is not None
    dispatcher._registry.remove("agent")
    with pytest.raises(RuntimeContractError, match="deleted under its training job"):
        dispatcher.run_dispatched_turn(old, WEIGHTS, backends[WEIGHTS], batch)


@pytest.mark.unit
def test_the_training_thread_idles_once_the_last_training_scenario_is_gone(dispatchers: _Dispatchers) -> None:
    """A wake that finds no training scenario is not a failure: nothing would ever clear its error."""
    dispatcher, _ = dispatchers.open()
    assert dispatcher._process_training() is False
    dispatcher._drain_training()
    assert dispatcher.build_training_status()["error"] is None


@pytest.mark.unit
def test_a_failed_instruction_of_a_scenario_that_is_gone_starts_no_worker(
    monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """Workers started for a deleted name would record errors that nothing clears."""
    dispatcher, _ = dispatchers.open()
    old = _scenario(dispatcher)
    dispatcher.delete_scenario("agent")
    monkeypatch.setattr(old.trainer_for(HARNESS), "fail_pending_instruction", lambda error: True)
    dispatcher._fail_instruction(old, RuntimeError("episode failed"), HARNESS)
    with dispatcher._training.lock:
        assert dispatcher._training.local_workers == {}


@pytest.mark.unit
def test_a_failing_local_cycle_runs_once_per_wake_and_keeps_its_error_in_the_status(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A backend that is away costs one attempt per wake, never a loop of reloads, and the failure stays visible."""
    away = _AwayBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path, harness=away))
    # Both rows first, then one wake: an accept wakes the worker per row, which would be two wakes.
    _scenario(dispatcher, 1)
    dispatcher._start_local_backend_worker("agent", HARNESS)
    _wait_for(lambda: away.attempts > 0)
    time.sleep(0.5)
    assert away.attempts == 1
    assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"
    # A job's turn ended: the worker looks again, once.
    dispatcher.wake_local_workers()
    _wait_for(lambda: away.attempts > 1)
    time.sleep(0.5)
    assert away.attempts == 2
    assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"


@pytest.mark.unit
def test_a_local_failure_under_a_dispatched_job_reloads_once_the_job_has_landed(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A reload under the job would hand its result to an instance that never reserved it."""
    backends = _dispatched_pair(tmp_path, harness=_AwayBackend(HARNESS, tmp_path / "candidates"))
    dispatcher, _ = dispatchers.open(backends=backends)
    scenario = _scenario(dispatcher, 1)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    with pytest.raises(RuntimeError, match="proposer away"):
        dispatcher._process_local_backend_step("agent", HARNESS)
    assert dispatcher._registry.get_optional("agent") is scenario
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert _training_components(scenario) == [WEIGHTS]
    # The next local cycle rebuilds the scenario as its first act, then looks again on the new instance.
    assert dispatcher._process_local_backend_step("agent", HARNESS) is True
    rebuilt = dispatcher._registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert rebuilt.scenario_step == 1
    # With no job out, a failure reloads at once.
    with pytest.raises(RuntimeError, match="proposer away"):
        dispatcher._process_local_backend_step("agent", HARNESS)
    assert dispatcher._registry.get_optional("agent") is not rebuilt


@pytest.mark.unit
def test_a_deferred_reload_never_lands_under_a_cycle_that_runs_on_the_instance(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A cycle woken while the job is still out commits on the instance; the rebuild waits for the cycle after."""
    backends = _dispatched_pair(tmp_path, harness=_FlakyBackend(HARNESS, tmp_path / "candidates"))
    dispatcher, _ = dispatchers.open(backends=backends)
    scenario = _scenario(dispatcher, 1)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    with pytest.raises(RuntimeError, match="away once"):
        dispatcher._process_local_backend_step("agent", HARNESS)
    assert dispatcher._process_local_backend_step("agent", HARNESS) is True
    assert dispatcher._registry.get_optional("agent") is scenario
    assert _training_components(scenario) == [HARNESS]
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert dispatcher._process_local_backend_step("agent", HARNESS) is True
    rebuilt = dispatcher._registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert _training_components(rebuilt) == [WEIGHTS, HARNESS]


class _BlockingAwayBackend(_ComponentBackend):
    """A local cycle whose preparation waits for the test to let it go, then fails."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.entered = threading.Event()
        self.go = threading.Event()

    def unblock(self) -> None:
        self.go.set()

    def prepare_step(self, batch, state, scenario_step):
        self.entered.set()
        assert self.go.wait(30)
        raise RuntimeError("proposer away")


@pytest.mark.unit
def test_a_sibling_waiting_at_the_cycle_lock_looks_again_after_a_reload_under_it(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """The failing cycle rebuilds the scenario under the lock; the sibling that read the old instance looks again
    on the new one instead of sleeping with its rows unread."""
    away = _BlockingAwayBackend(WEIGHTS, tmp_path / "candidates")
    harness = _ComponentBackend(HARNESS, tmp_path / "candidates")
    dispatcher, _ = dispatchers.open(backends={WEIGHTS: away, HARNESS: harness})
    scenario = _scenario(dispatcher, 1)
    failures: list[BaseException] = []

    def fail() -> None:
        try:
            dispatcher._process_local_backend_step("agent", WEIGHTS)
        except RuntimeError as exc:
            failures.append(exc)

    failing = threading.Thread(target=fail)
    failing.start()
    assert away.entered.wait(10)
    # The sibling reads the instance, then reaches the cycle lock the failing cycle holds.
    reached = threading.Event()
    cycle_lock = dispatcher.local_cycle_lock
    outcome: list[bool] = []
    sibling = threading.Thread(target=lambda: outcome.append(dispatcher._process_local_backend_step("agent", HARNESS)))

    def observed(name: str):
        lock = cycle_lock(name)
        if threading.current_thread() is sibling:
            reached.set()
        return lock

    monkeypatch.setattr(dispatcher, "local_cycle_lock", observed)
    sibling.start()
    assert reached.wait(10)
    away.go.set()
    failing.join(10)
    sibling.join(10)
    assert [str(error) for error in failures] == ["proposer away"]
    rebuilt = dispatcher._registry.get_optional("agent")
    assert rebuilt is not None and rebuilt is not scenario
    assert outcome == [True] and harness.prepared == 0
    # Looking again on the rebuilt instance commits the rows that were there all along.
    assert dispatcher._process_local_backend_step("agent", HARNESS) is True
    assert _training_components(rebuilt) == [HARNESS]


@pytest.mark.unit
def test_deleting_a_scenario_is_refused_while_the_runtime_cannot_say_whether_its_job_is_out(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """A runtime that does not answer gets a busy answer, not a crash: a blind delete could orphan a job. A marker
    names the scenario that owns its job and refuses that scenario's delete alone; a marker written before markers
    named an owner refuses none, and the scenario holding the job's reservation is still refused."""
    training = StubTrainingRuntime()
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path), training=training)
    _scenario(dispatcher)

    def unhealthy() -> None:
        raise RuntimeError("train group is unhealthy")

    monkeypatch.setattr(training, "training_job_status", unhealthy)
    with pytest.raises(ScenarioBusy, match="does not say whether its job is out"):
        dispatcher.delete_scenario("agent")
    assert dispatcher._registry.get_optional("agent") is not None
    marker: dict[str, Any] = {"status": "RUNNING", "training_job_id": "job-1", "scenario": "agent"}
    monkeypatch.setattr(training, "training_job_status", lambda: marker)
    assert dispatcher.training_job_in_flight("agent") is True
    assert dispatcher.training_job_in_flight("old") is False
    marker = {"status": "COMPLETE", "training_job_id": "job-1", "commit_acknowledged": False, "scenario": "agent"}
    with pytest.raises(ScenarioBusy, match="training job is out"):
        dispatcher.delete_scenario("agent")
    marker = {"status": "COMPLETE", "training_job_id": "job-1", "commit_acknowledged": True, "scenario": "agent"}
    assert dispatcher.delete_scenario("agent")["scenario"] == "agent"


@pytest.mark.unit
def test_after_a_restart_binds_a_stray_registration_its_delete_lets_the_jobs_owner_bind_and_finish(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """b trains on a one scenario runtime; a create of a is refused but leaves a's registration. b's job is out when
    the process dies, and after the restart a binds first. The marker names b, so b cannot be deleted while its job
    is out, and a, the stray, can: its delete and a restart let b bind again, the only log that can finish the job."""
    factory = _seeded_repository(tmp_path)
    dispatcher, _ = dispatchers.open(
        backends=_dispatched_pair(tmp_path), training=StubTrainingRuntime(), backend_factory=factory
    )
    assert dispatcher.get_or_create_scenario("b") is not None
    with pytest.raises(ReefError, match="already bound"):
        dispatcher.get_or_create_scenario("a")
    dispatcher.close()

    training = StubTrainingRuntime()
    marker = {"status": "UPDATING_WEIGHTS", "training_job_id": "job-b", "scenario": "b"}
    monkeypatch.setattr(training, "training_job_status", lambda: marker)
    restarted, _ = dispatchers.open(backends=_dispatched_pair(tmp_path), training=training, backend_factory=factory)
    thread = restarted._lifecycle.preload_thread
    if thread is not None:
        thread.join(30)
    assert restarted._registry.training_scenario_name == "a"
    with pytest.raises(ScenarioBusy, match="training job is out"):
        restarted.delete_scenario("b")
    assert restarted.delete_scenario("a")["scenario"] == "a"


@pytest.mark.unit
def test_the_delete_guard_reads_the_runtime_once(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """Every read of an executor runtime is a health call that can fail on its own: the guard makes one, and an
    answer that goes bad on a second read cannot turn the busy answer into another error."""
    from reef.train.runtime import ExecutorTrainingRuntime

    class Handle:
        calls = 0

        def health(self) -> dict[str, Any]:
            self.calls += 1
            if self.calls > 1:
                return {
                    "ok": False,
                    "phase": "dead",
                    "training_job": {"deferred_weight_update": True, "status": "IDLE"},
                }
            return {
                "ok": True,
                "training_job": {
                    "deferred_weight_update": True,
                    "status": "RUNNING",
                    "training_job_id": "j",
                    "scenario": "x",
                },
            }

    handle = Handle()
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path), training=StubTrainingRuntime())
    object.__setattr__(dispatcher._recipe, "training_runtime", ExecutorTrainingRuntime(handle))  # type: ignore[arg-type]
    try:
        assert dispatcher.training_job_in_flight("x") is True
        assert handle.calls == 1
    finally:
        object.__setattr__(dispatcher._recipe, "training_runtime", None)


@pytest.mark.unit
def test_deleting_a_scenario_whose_job_marker_is_out_at_the_backend_waits(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """The backend's marker outlives a rebuilt or unloaded instance; the delete reads it, not the reserved batch."""
    training = StubTrainingRuntime()
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path), training=training)
    _scenario(dispatcher)
    marker: dict[str, Any] = {"status": "CHECKPOINT", "training_job_id": "job-1", "scenario": "agent"}
    monkeypatch.setattr(training, "training_job_status", lambda: marker)
    with pytest.raises(ScenarioBusy, match="training job is out"):
        dispatcher.delete_scenario("agent")
    marker["scenario"] = "other"
    assert dispatcher.delete_scenario("agent")["scenario"] == "agent"
    _scenario(dispatcher, name="again")
    marker = {"status": "COMPLETE", "training_job_id": "job-2", "commit_acknowledged": False, "scenario": "again"}
    with pytest.raises(ScenarioBusy, match="training job is out"):
        dispatcher.delete_scenario("again")
    marker["commit_acknowledged"] = True
    assert dispatcher.delete_scenario("again")["scenario"] == "again"


@pytest.mark.unit
def test_a_scenario_of_several_components_refuses_a_checkpoint_interval_above_one(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A step that published no checkpoint would see its component dropped from the next release."""
    recipe = _TwoTrainerRecipe(backends=_local_pair(tmp_path), checkpoint_strategy=EveryNVersions(2))
    dispatcher, _ = dispatchers.open(recipe=recipe)
    with pytest.raises(ReefError, match="checkpoint interval must be 1"):
        dispatcher.get_or_create_scenario("agent")


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

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        return tuple(
            ComponentTrainer(
                name,
                Trainer.build(
                    scenario,
                    records,
                    processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
                    candidate_backend=_ComponentBackend(name, self.artifact_dir),
                    algorithm_state=algorithm_states.get(name),
                    experiment_logger=experiment_logger,
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


class _SlowDispatchedBackend(_DispatchedBackend):
    """A dispatched job whose execution takes a moment, as a real job takes minutes."""

    def evaluate(self, candidate):
        time.sleep(0.05)
        return super().evaluate(candidate)


@pytest.mark.unit
def test_a_weights_job_commits_while_the_harness_backend_is_away(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """On the training thread: the harness cycle keeps failing, the weights job still lands and the failure shows."""
    away = _AwayBackend(HARNESS, tmp_path / "candidates")
    backends = {WEIGHTS: _SlowDispatchedBackend(WEIGHTS, tmp_path / "candidates", "job-1"), HARNESS: away}
    dispatcher, _ = dispatchers.open(backends=backends, training=StubTrainingRuntime())
    _scenario(dispatcher)
    for record in _records(1):
        dispatcher.accept_record(record)
    assert _wait_for(
        lambda: (WEIGHTS, 1) in [(row.component, row.step) for row in _scenario(dispatcher).store.history()],
        timeout_seconds=15,
    )
    time.sleep(0.5)
    assert 1 <= away.attempts <= 3
    assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"


@pytest.mark.unit
def test_a_second_report_on_a_trained_inference_is_settled_after_a_restart(dispatchers: _Dispatchers) -> None:
    """The harness trained on i1 through r1a while the weights trainer still holds i1; after a restart r1b is not resolved against it."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    scenario.records.append(_report("r1b", "i1"))
    result = scenario.prepare_training_step(HARNESS)
    assert result is not None
    scenario.commit(result, component=HARNESS)
    assert scenario.records.count("agent") == 3
    assert scenario.prepare_training_step(HARNESS) is None
    rebuilt = dispatcher._registry.reload("agent")
    assert rebuilt.prepare_training_step(HARNESS) is None
    # Its source trained, so the rebuilt harness skips it; the weights trainer still reads it.
    assert "r1b" in rebuilt.trainer_for(HARNESS).skipped_record_ids
    assert "r1b" not in rebuilt.trainer_for(WEIGHTS).consumed_record_ids


@pytest.mark.unit
def test_a_report_settled_between_two_commits_stays_settled_after_a_restart(dispatchers: _Dispatchers) -> None:
    """r1b arrived after commit A trained i1 and before commit B: the replay settles it at A's watermark, as it was."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None
    scenario.records.append(_report("r1b", "i1"))
    assert scenario.prepare_training_step(HARNESS) is None
    _append_records(scenario, 2)
    assert _commit_step(scenario, HARNESS) is not None
    rebuilt = dispatcher._registry.reload("agent")
    assert rebuilt.prepare_training_step(HARNESS) is None
    # Settled, not live: commit B names it as consumed, so the rebuilt harness skips it.
    assert "r1b" in rebuilt.trainer_for(HARNESS).consumed_record_ids


@pytest.mark.unit
def test_a_report_on_an_inference_the_weights_job_trained_is_skipped_not_raised(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The weights job holds i1 while the harness commits and r1b arrives; the job's commit consumes i1 with r1b
    unread by the weights trainer. Every later read skips r1b, on the live path and after a restart."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher, 1)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    assert _commit_step(scenario, HARNESS) is not None
    scenario.records.append(_report("r1b", "i1"))
    assert scenario.prepare_training_step(HARNESS) is None
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert scenario.records.get("agent", "i1") is not None
    assert scenario.reserve_training_batch(WEIGHTS) is None
    assert "r1b" in scenario.trainer_for(WEIGHTS).skipped_record_ids
    _append_records(scenario, 2)
    assert _commit_step(scenario, HARNESS) is not None
    rebuilt = dispatcher._registry.reload("agent")
    assert rebuilt.prepare_training_step(HARNESS) is None
    weights = rebuilt.reserve_training_batch(WEIGHTS)
    assert weights is not None
    assert [item.source_agent_record_ids for item in weights.items] == [("i2", "r2")]


class _SlottedProcessor(ThresholdProcessor):
    """Reports batch by the group and the retry slot named in their metadata; a group is ready at two slots."""

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload["metadata"]
        return metadata["group"], metadata["slot"]

    def decide_group(self, key: Hashable, items: tuple[Any, ...]) -> GroupDecision:
        return GroupDecision.READY if len(items) >= 2 else GroupDecision.INCOMPLETE


@dataclass(frozen=True)
class _SlottedRecipe(_TwoTrainerRecipe):
    """The two trainer recipe whose harness batches reports by group and retry slot."""

    def processor_for(self, component: str) -> type[DataProcessor]:
        return _SlottedProcessor if component == HARNESS else ThresholdProcessor


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
    dispatcher, _ = dispatchers.open(recipe=_SlottedRecipe(backends=_local_pair(tmp_path)))
    scenario = _scenario(dispatcher)
    for step in (1, 2, 3):
        scenario.records.append(_records(step)[0])
    scenario.records.append(_slotted_report("r1", "i1", "G", 0))
    scenario.records.append(_slotted_report("r1dup", "i3", "G", 0))
    scenario.records.append(_slotted_report("r2", "i2", "G", 1))
    assert _commit_step(scenario, WEIGHTS) == [("i1", "r1")]
    assert _commit_step(scenario, HARNESS) == [("i1", "r1"), ("i2", "r2")]
    assert scenario.records.get("agent", "r1dup") is not None
    rebuilt = dispatcher._registry.reload("agent")
    rebuilt.records.append(_records(4)[0])
    rebuilt.records.append(_slotted_report("r4", "i4", "G", 2))
    # Live, slot 0 was never r1dup's: G holds r4 alone and waits.
    assert rebuilt.prepare_training_step(HARNESS) is None
    record = rebuilt.last_commit_for(HARNESS)
    assert record is not None and "r1dup" in record.consumed_ids


class _DroppingBackend(_DispatchedBackend):
    """A dispatched backend whose runtime drops the next batch as stale when told to, each job its own id."""

    def __init__(self, component: str, artifact_dir: Path, job_id: str) -> None:
        super().__init__(component, artifact_dir, job_id)
        self.drop_next = False

    def prepare_step(self, batch, state, scenario_step):
        if self.drop_next:
            self.drop_next = False
            self.prepared += 1
            return PreparedStep.dropped(state=state, metrics={"stale": 1})
        return super().prepare_step(batch, state, scenario_step)

    def settle_step(self, prepared, decision):
        self.job_id = f"job-{self.prepared}"
        return super().settle_step(prepared, decision)


def _dropping_pair(tmp_path: Path, harness_policy: str = "refuse") -> dict[str, _ComponentBackend]:
    return {
        WEIGHTS: _DroppingBackend(WEIGHTS, tmp_path / "candidates", "job-0"),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates", stale_policy=harness_policy),
    }


@pytest.mark.unit
def test_a_batch_the_weights_backend_dropped_stays_consumed_after_a_restart(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A drop consumes its batch without a commit and its rows stay stored for the harness. The drop's consumption
    receipt names what it consumed and its component, so neither a restart right after it nor one after a later
    commit reserves the dropped batch again."""
    dispatcher, backends = dispatchers.open(backends=_dropping_pair(tmp_path))
    weights = backends[WEIGHTS]
    assert isinstance(weights, _DroppingBackend)

    def turn(scenario: Scenario) -> list[tuple[str, ...]]:
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        sources = [item.source_agent_record_ids for item in batch.items]
        assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, weights, batch) is True
        return sources

    scenario = _scenario(dispatcher, 1, 2, 3)
    assert turn(scenario) == [("i1", "r1")]
    weights.drop_next = True
    assert turn(scenario) == [("i2", "r2")]
    assert scenario.records.get("agent", "r2") is not None
    rebuilt = dispatcher._registry.reload("agent")
    assert turn(rebuilt) == [("i3", "r3")]
    _append_records(rebuilt, 4)
    rebuilt = dispatcher._registry.reload("agent")
    assert turn(rebuilt) == [("i4", "r4")]
    (receipt,) = rebuilt.records.consumption_receipts("agent")
    assert receipt["metadata"]["outcome"] == "stale" and receipt["metadata"]["component"] == WEIGHTS
    assert set(receipt["consumed_ids"]) == {"i2", "r2"}


@pytest.mark.unit
def test_two_stale_drops_across_a_reload_are_two_receipts_and_the_rows_behind_them_train(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A rebuilt processor numbers its batches from 1 again, so a drop after a reload carries the batch id of a drop
    before it, over other rows. A receipt is keyed by its id and what it consumed: the second is a receipt of its
    own, not a conflict with the first, and the weights trainer goes on to the rows behind both drops."""
    dispatcher, backends = dispatchers.open(backends=_dropping_pair(tmp_path))
    weights = backends[WEIGHTS]
    assert isinstance(weights, _DroppingBackend)

    def turn(scenario: Scenario) -> tuple[str, list[tuple[str, ...]]]:
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        sources = [item.source_agent_record_ids for item in batch.items]
        assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, weights, batch) is True
        return batch.batch_id, sources

    scenario = _scenario(dispatcher, 1, 2, 3)
    weights.drop_next = True
    first_id, first = turn(scenario)
    assert first == [("i1", "r1")]
    rebuilt = dispatcher._registry.reload("agent")
    weights.drop_next = True
    second_id, second = turn(rebuilt)
    assert (second_id, second) == (first_id, [("i2", "r2")])
    rebuilt = dispatcher._registry.reload("agent")
    assert turn(rebuilt)[1] == [("i3", "r3")]
    stale = rebuilt.records.consumption_receipts("agent")
    assert sorted(sorted(row["consumed_ids"]) for row in stale) == [["i1", "r1"], ["i2", "r2"]]
    assert [row["receipt_id"] for row in stale] == [f"{WEIGHTS}:{first_id}"] * 2


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
@pytest.mark.parametrize("colocated", [False, True])
def test_a_harness_commit_that_fails_under_a_weights_job_leaves_its_rows_to_the_harness(
    tmp_path: Path, colocated: bool, monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """The harness acknowledges its batch before its record is durable. When that commit fails while a weights job
    is out, the reload waits for the job, and the job's commit leaves the harness's rows stored: after the job, the
    rebuilt harness trains them once. A colocated job holds every cycle lock, so the harness cannot retry before the
    job commits."""
    backends = _dispatched_pair(tmp_path)
    if colocated:
        backends[WEIGHTS] = _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1")
    dispatcher, _ = dispatchers.open(backends=backends)
    scenario = _scenario(dispatcher, 1)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    _fail_first_record_of(monkeypatch, scenario, HARNESS)
    with pytest.raises(OSError, match="transient store write error"):
        dispatcher._process_local_backend_step("agent", HARNESS)
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
    for _ in range(3):
        dispatcher._process_local_backend_step("agent", HARNESS)
    current = dispatcher._registry.get_optional("agent")
    assert current is not None
    history = current.store.history()
    assert [sorted(record.consumed_ids) for record in history if record.component == HARNESS] == [["i1", "r1"]]
    assert [sorted(record.consumed_ids) for record in history if record.component == WEIGHTS] == [["i1", "r1"]]


@pytest.mark.unit
def test_a_weights_commit_that_fails_leaves_its_batch_to_the_weights_trainer_after_a_harness_commit(
    monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """The reverse: a weights commit that fails after its acknowledgment, then a harness commit before the weights
    trainer's reload. The harness commit leaves the rows stored, so the rebuilt weights trainer trains the batch its
    failed commit had acknowledged."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) == [("i1", "r1")]
    _append_records(scenario, 2)
    _fail_first_record_of(monkeypatch, scenario, WEIGHTS)
    with pytest.raises(OSError, match="transient store write error"):
        _commit_step(scenario, WEIGHTS)
    assert _commit_step(scenario, HARNESS) == [("i2", "r2")]
    assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
    rebuilt = dispatcher._registry.reload("agent")
    assert _commit_step(rebuilt, WEIGHTS) == [("i1", "r1")]


@pytest.mark.unit
def test_a_stale_drop_under_a_failed_harness_commit_leaves_the_harness_rows_to_the_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """A drop consumes its rows for the weights trainer alone: a harness batch acknowledged by a commit that failed
    stays stored under the weights drop, and the harness trains it."""
    dispatcher, backends = dispatchers.open(backends=_dropping_pair(tmp_path))
    weights = backends[WEIGHTS]
    assert isinstance(weights, _DroppingBackend)
    scenario = _scenario(dispatcher, 1)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    _fail_first_record_of(monkeypatch, scenario, HARNESS)
    with pytest.raises(OSError, match="transient store write error"):
        dispatcher._process_local_backend_step("agent", HARNESS)
    weights.drop_next = True
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, weights, batch) is True
    assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
    (receipt,) = scenario.records.consumption_receipts("agent")
    assert set(receipt["consumed_ids"]) == {"i1", "r1"} and receipt["metadata"]["component"] == WEIGHTS
    for _ in range(3):
        dispatcher._process_local_backend_step("agent", HARNESS)
    current = dispatcher._registry.get_optional("agent")
    assert current is not None
    assert [sorted(record.consumed_ids) for record in current.store.history() if record.component == HARNESS] == [
        ["i1", "r1"]
    ]


@dataclass(frozen=True)
class _Judgment(SupportsReceipt):
    receipt: str


class _JudgedAtOnce:
    """A judge worker whose every judgment is ready at the next poll."""

    def __init__(self) -> None:
        self._done: list[_Judgment] = []

    def submit(self, job: _Judgment) -> bool:
        self._done.append(job)
        return True

    def poll(self) -> list[_Judgment]:
        done, self._done = self._done, []
        return done

    def close(self) -> None:
        return None


class _RetiringRetriesProcessor(ComputedFeedbackProcessor):
    """A turn is judged once its successor arrives, and a retry of a turn already seen is retired, as OpenClaw-RL
    retires a client's duplicate turn."""

    def __init__(self, context: Any) -> None:
        super().__init__(context, worker=_JudgedAtOnce())
        self._seen: set[str] = set()

    def ingest(self, item: AgentRecord) -> None:
        self.catch_up(time.monotonic())
        if item.payload.get("completes"):
            self.dispatch(_Judgment(item.payload["completes"]))
        if item.payload["turn"] in self._seen:
            self.retire(item.agent_record_id)
            return
        self._seen.add(item.payload["turn"])
        self.track(item)

    async def judge(self, job: _Judgment) -> _Judgment:
        return job

    def make_sample(self, record: AgentRecord, judgment: _Judgment) -> Any:
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            runtime_load_id="v1",
        )

    def make_batch(self, samples: Any, batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:retiring:{batch_number}", tuple(samples))


@dataclass(frozen=True)
class _ComputedWeightsRecipe(_TwoTrainerRecipe):
    """The two trainer recipe whose weights trainer computes its own feedback."""

    def processor_for(self, component: str) -> type[DataProcessor]:
        return _RetiringRetriesProcessor if component == WEIGHTS else ThresholdProcessor


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
    dispatcher, _ = dispatchers.open(recipe=_ComputedWeightsRecipe(backends=_local_pair(tmp_path)))
    scenario = _scenario(dispatcher)
    scenario.records.append(_turn("A", "t1"))
    scenario.records.append(_turn("B", "t2", completes="A"))
    assert _commit_step(scenario, WEIGHTS) == [("A",)]
    scenario.records.append(_turn("A-retry", "t1"))
    scenario.records.append(_turn("C", "t3", completes="B"))
    assert _commit_step(scenario, WEIGHTS) == [("B",)]
    assert "A-retry" in scenario.store.history()[-1].consumed_ids
    if reload:
        scenario = dispatcher._registry.reload("agent")
    scenario.records.append(_turn("E", "t4", completes="A-retry"))
    assert _commit_step(scenario, WEIGHTS) is None


@pytest.mark.unit
def test_a_run_report_on_an_inference_both_trainers_trained_is_skipped_by_both(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """run references i1 and i2 and is stored before the weights job commits i1, which the harness trained already.
    Each trainer then skips run, since one of its sources trained, instead of raising on it."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher, 1)
    scenario.records.append(_records(2)[0])
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    assert _commit_step(scenario, HARNESS) is not None
    scenario.records.append(_report("run", "i1", "i2"))
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert scenario.prepare_training_step(HARNESS) is None
    assert scenario.reserve_training_batch(WEIGHTS) is None
    for component in (WEIGHTS, HARNESS):
        assert "run" in scenario.trainer_for(component).skipped_record_ids, component


@pytest.mark.unit
def test_a_rollback_to_the_creation_after_a_rejected_first_step(dispatchers: _Dispatchers) -> None:
    """A rejected step records the creation without a checkpoint; the creation still has its own bytes."""
    dispatcher, backends = dispatchers.open()
    scenario = _scenario(dispatcher)
    creation = scenario.current_artifact_ref().release_id
    backends[HARNESS].reject_next = True
    for step in (1, 2):
        _append_records(scenario, step)
        assert _commit_step(scenario, HARNESS) is not None
    assert scenario.current_artifact_ref().release_id != creation
    assert next(row for row in scenario.releases() if row["operation"] == "creation")["restorable"] is True
    scenario.rollback(creation)
    assert _component_files(scenario, scenario.current_artifact_ref())[HARNESS] == "harness seed"


@pytest.mark.unit
def test_a_job_whose_scenario_was_reloaded_under_it_is_not_committed_on_the_new_instance(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The rebuilt trainer reserves the same rows and the backend replays the job under its marker."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    old = _scenario(dispatcher, 1)
    batch = old.reserve_training_batch(WEIGHTS)
    assert batch is not None
    new = dispatcher._registry.reload("agent")
    assert new is not old
    assert dispatcher.run_dispatched_turn(old, WEIGHTS, backends[WEIGHTS], batch) is True
    assert _training_components(new) == []
    assert new.reserve_training_batch(WEIGHTS) is not None


def _commit_weights_job(scenario: Scenario) -> None:
    """Reserve, run and commit the dispatched weights step, as the training thread does."""
    assert scenario.reserve_training_batch(WEIGHTS) is not None
    execution = scenario.execute_reserved_training_step(WEIGHTS)
    assert execution.outcome == "commit" and execution.result is not None
    scenario.commit(execution.result, component=WEIGHTS)


@pytest.mark.unit
def test_a_harness_only_rollback_keeps_the_proof_that_the_weights_job_was_committed(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The backend must still finish a job whose weights a later rollback carried forward unchanged."""
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher, 1)
    _commit_weights_job(scenario)
    assert scenario.committed_training_job_id == "job-1"
    for step in (2, 3):
        _append_records(scenario, step)
        assert _commit_step(scenario, HARNESS) is not None
    first_harness = scenario.releases()[1]["release_id"]
    assert scenario.committed_training_job_id == "job-1"
    scenario.rollback(first_harness)
    # Only the harness changed: the weights job's commit still stands.
    assert scenario.committed_training_job_id == "job-1"
    assert not scenario.committed_training_without_job_id


@pytest.mark.unit
def test_training_mode_switches_every_trainer_or_none(dispatchers: _Dispatchers) -> None:
    """A mode one component cannot run is refused before any component switches."""
    dispatcher, _ = dispatchers.open(hybrid_components=frozenset({WEIGHTS}))
    scenario = _scenario(dispatcher)
    with pytest.raises(NotImplementedError, match=r"harness \(ThresholdProcessor\)"):
        scenario.set_training_mode("hybrid")
    assert [bound.trainer.training_mode for bound in scenario.component_trainers] == ["auto", "auto"]
    with pytest.raises(ValueError, match="training_mode must be"):
        scenario.set_training_mode("bogus")


@pytest.mark.unit
def test_a_rejected_batch_is_consumed_for_its_own_trainer_alone(dispatchers: _Dispatchers) -> None:
    """A drop consumes its rows for the trainer that dropped them: they stay stored, and its receipt names it."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    assert scenario.prepare_training_step(WEIGHTS) is not None
    scenario.reject_pending(component=WEIGHTS)
    assert scenario.records.count("agent") == 2
    (receipt,) = scenario.records.consumption_receipts("agent")
    assert receipt["metadata"]["component"] == WEIGHTS and set(receipt["consumed_ids"]) == {"i1", "r1"}
    assert _commit_step(scenario, HARNESS) is not None
    assert scenario.store.history()[-1].consumed_ids == frozenset({"i1", "r1"})


@pytest.mark.unit
def test_composite_registration_refuses_a_flat_base(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """Files at the root of the base belong to no component and would be carried forward by none."""
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "harness.txt").write_text("harness seed", encoding="utf-8")
    dispatcher, _ = dispatchers.open()
    with pytest.raises(ReefError, match=r"keeps \['harness.txt'\] outside its components"):
        dispatcher.get_or_create_scenario("agent")


@pytest.mark.unit
def test_composite_registration_refuses_a_file_named_like_a_component(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A component keeps a directory; a file of that name would fail on every component view."""
    initial = tmp_path / "initial"
    (initial / HARNESS).mkdir(parents=True)
    (initial / HARNESS / f"{HARNESS}.txt").write_text("harness seed", encoding="utf-8")
    (initial / WEIGHTS).write_text("not a directory", encoding="utf-8")
    dispatcher, _ = dispatchers.open()
    with pytest.raises(ReefError, match=r"keeps \['weights'\] outside its components"):
        dispatcher.get_or_create_scenario("agent")


@pytest.mark.unit
def test_composite_registration_starts_an_unseeded_component_empty(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """A fresh repository seeds only the harness; the weights component starts empty and fills at its first step."""
    _seeded_repository(tmp_path, (HARNESS,))
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher)
    base = scenario.repository.materialize(scenario.current_artifact_ref())
    assert base.components is not None and base.components.names == (WEIGHTS, HARNESS)
    assert TextFileTree().read_files(base.component(WEIGHTS)) is None
    _append_records(scenario, 1)
    assert _commit_step(scenario, WEIGHTS) is not None
    assert _component_files(scenario, scenario.current_artifact_ref()) == {
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
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    assert _commit_step(scenario, HARNESS) is not None

    reloaded = dispatcher._registry.reload("agent")
    assert reloaded.prepare_training_step(HARNESS) is None
    assert _commit_step(reloaded, WEIGHTS) is not None
    assert reloaded.store.history()[-1].consumed_ids == frozenset({"i1", "r1"})
    assert reloaded.records.count("agent") == 2


@pytest.mark.unit
def test_dispatched_result_overtaken_by_another_trainer_is_merged(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """The remote job has published its weights; the step lands on the release served now instead of being refused."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher)
    base = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    assert scenario.reserve_training_batch(WEIGHTS) is not None
    execution = scenario.execute_reserved_training_step(WEIGHTS)
    assert execution.outcome == "commit" and execution.result is not None

    # The harness worker commits while the weights job is out.
    assert _commit_step(scenario, HARNESS) is not None
    after_harness = scenario.current_artifact_ref().release_id
    assert after_harness != base

    scenario.commit(execution.result, component=WEIGHTS)
    assert backends[WEIGHTS].prepared == 1
    assert scenario.scenario_step == 2
    assert _component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness step 1",
    }
    records = scenario.store.history()
    assert (records[-1].component, records[-1].base_release_id) == (WEIGHTS, base)
    assert scenario.committed_training_job_id == "job-1"


def _both_prepared(dispatchers: _Dispatchers, harness_policy: str):
    """Both trainers prepared against the base, then the weights step moves the head."""
    backends = {
        WEIGHTS: _ComponentBackend(WEIGHTS, dispatchers.tmp_path / "candidates"),
        HARNESS: _ComponentBackend(HARNESS, dispatchers.tmp_path / "candidates", stale_policy=harness_policy),
    }
    dispatcher, _ = dispatchers.open(backends=backends)
    scenario = _scenario(dispatcher)
    base = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    harness = scenario.prepare_training_step(HARNESS)
    weights = scenario.prepare_training_step(WEIGHTS)
    assert harness is not None and weights is not None
    scenario.commit(weights, component=WEIGHTS)
    return backends, scenario, harness, base


@pytest.mark.unit
def test_a_local_result_its_backend_calls_mergeable_lands_on_the_release_served_now(
    dispatchers: _Dispatchers,
) -> None:
    backends, scenario, harness, base = _both_prepared(dispatchers, "merge")
    scenario.commit(harness, component=HARNESS)
    assert _component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness step 1",
    }
    record = scenario.store.history()[-1]
    assert (record.component, record.base_release_id) == (HARNESS, base)
    assert backends[HARNESS].prepared == 1 and backends[HARNESS].evaluated == 1


@pytest.mark.unit
def test_a_local_result_its_backend_reevaluates_keeps_its_candidate(dispatchers: _Dispatchers) -> None:
    """The proposer is not asked again; the kept candidate is evaluated against the release served now."""
    backends, scenario, harness, base = _both_prepared(dispatchers, "reevaluate")
    with pytest.raises(StaleTrainingResultError) as refused:
        scenario.commit(harness, component=HARNESS)
    assert refused.value.policy == "reevaluate"
    scenario.retry_pending(HARNESS, keep_candidate=True)
    again = scenario.prepare_training_step(HARNESS)
    assert again is not None
    assert backends[HARNESS].prepared == 1 and backends[HARNESS].evaluated == 2
    assert backends[HARNESS].reevaluations == 1
    scenario.commit(again, component=HARNESS)
    assert _component_files(scenario, scenario.current_artifact_ref()) == {
        WEIGHTS: "weights step 1",
        HARNESS: "harness step 1",
    }
    assert scenario.store.history()[-1].base_release_id != base


@pytest.mark.unit
def test_a_weights_step_is_no_new_harness_head(dispatchers: _Dispatchers) -> None:
    """Clients pull the tree: a step that carried it forward unchanged is not announced or listed as a release."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher, 1)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    assert _commit_step(scenario, HARNESS) is not None
    harness_release = scenario.current_artifact_ref().release_id
    assert _commit_step(scenario, WEIGHTS) is not None
    assert scenario.current_artifact_ref().release_id != harness_release
    assert service.harness_head(headers) == harness_release
    manifest = service.harness_manifest(headers)
    assert manifest["release_id"] == harness_release
    assert manifest["files"] == {"harness.txt": "harness step 1"}
    catalog = service.harness_releases(headers)["releases"]
    assert [row.get("component") for row in catalog] == [None, HARNESS]
    assert dispatcher._experiment_context(scenario, WEIGHTS).component == WEIGHTS


@pytest.mark.unit
def test_each_components_step_reaches_the_tracker_under_its_name(dispatchers: _Dispatchers) -> None:
    """The event the dispatcher records for a step names the trainer that made it."""
    tracker = _RecordingTracker()
    dispatcher, _ = dispatchers.open(experiment_tracker=tracker)
    scenario = _scenario(dispatcher, 1)
    harness = scenario.prepare_training_step(HARNESS)
    assert harness is not None
    dispatcher._commit_result("agent", harness, HARNESS)
    weights = scenario.prepare_training_step(WEIGHTS)
    assert weights is not None
    dispatcher._commit_result("agent", weights, WEIGHTS)
    assert [event.context.component for event in tracker.events] == [HARNESS, WEIGHTS]
    assert [event.context.backend for event in tracker.events] == ["_ComponentBackend", "_ComponentBackend"]
    assert [event.context.step for event in tracker.events] == [1, 2]
    assert [row.get("component") for row in scenario.releases()] == [WEIGHTS, HARNESS, None]


@pytest.mark.unit
def test_the_harness_catalog_is_one_lineage_of_trees(dispatchers: _Dispatchers) -> None:
    """A rejected step, an unlisted weights step and a promote at step 1 never move the head or break the chain."""
    dispatcher, backends = dispatchers.open()
    scenario = _scenario(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    creation = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    harness = scenario.prepare_training_step(HARNESS)
    assert harness is not None
    requires = {"training_request": {"id": "r1", "requires": [{"name": "TOKEN", "kind": "env"}]}}
    scenario.commit(scenario.trainer_for(HARNESS).add_commit_metrics(harness, requires), component=HARNESS)
    h1 = scenario.current_artifact_ref().release_id
    assert _commit_step(scenario, WEIGHTS) is not None
    w1 = scenario.current_artifact_ref().release_id

    # A rejected harness step is listed for its page, but it published nothing and names no head.
    _append_records(scenario, 2)
    backends[HARNESS].reject_next = True
    rejected = scenario.prepare_training_step(HARNESS)
    assert rejected is not None and rejected.artifact is None
    scenario.commit(rejected, component=HARNESS)
    assert scenario.current_artifact_ref().release_id == w1
    assert service.harness_head(headers) == h1
    assert service.harness_manifest(headers)["release_id"] == h1
    catalog = service.harness_releases(headers)["releases"]
    # The rejected row is named by the listed release it ran on, so a client's poll agrees with the head;
    # like a flat scenario's rejected row, it lists the head's parent, not itself.
    assert [row["release_id"] for row in catalog] == [creation, h1, h1]
    assert catalog[-1]["metrics"]["selected"] is False
    assert catalog[-1]["composed_release_id"] == w1
    assert catalog[-1]["parent_release_id"] == catalog[1]["parent_release_id"] == creation
    assert "ran on" in service.harness_release_page(headers, 2).lower()
    assert f'"release_id": "{h1}"' in service.harness_install_script(headers, adapter="pi")

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
    assert service.harness_head(headers) == h2
    page = service.harness_release_page(headers, 3)
    assert h2 in page

    # A rollback to the unlisted weights release restores h1's tree: the row names the listed release
    # that target carried, so the requires chain walks on through it. The rejected step recorded w1's
    # reference without a checkpoint, which must not hide the checkpoint w1 has.
    scenario.rollback(w1)
    restored = scenario.current_artifact_ref().release_id
    catalog = service.harness_releases(headers)["releases"]
    assert [row["release_id"] for row in catalog] == [creation, h1, h1, h2, restored]
    assert catalog[-1]["rollback_target_release_id"] == h1
    assert catalog[-1]["composed_rollback_target_release_id"] == w1
    assert required_by(catalog, restored) == [{"name": "TOKEN", "kind": "env"}]
    assert service.harness_head(headers) == restored

    # A rollback to weights held for review, which carried the restored tree, names the listed release
    # that carried it; and the install script's fallback names a listed release.
    _append_records(scenario, 4)
    backends[WEIGHTS].hold_next = True
    weights = scenario.prepare_training_step(WEIGHTS)
    assert weights is not None and weights.pending
    scenario.commit(weights, component=WEIGHTS)
    held = next(row["release_id"] for row in scenario.releases() if row.get("pending"))
    _append_records(scenario, 5)
    assert _commit_step(scenario, HARNESS) is not None
    scenario.rollback(held)
    catalog = service.harness_releases(headers)["releases"]
    assert held not in {row["release_id"] for row in catalog}
    assert catalog[-1]["operation"] == "rollback"
    assert catalog[-1]["rollback_target_release_id"] == restored
    assert catalog[-1]["composed_rollback_target_release_id"] == held
    assert required_by(catalog, catalog[-1]["release_id"]) == [{"name": "TOKEN", "kind": "env"}]
    script = service.harness_install_script(headers, adapter="pi")
    assert w1 not in script and w2 not in script and held not in script


@pytest.mark.unit
def test_a_merged_result_names_the_release_it_landed_on_in_its_event(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """The tracker event carries what the commit record carries, merged_onto included."""
    tracker = _RecordingTracker()
    backends = {
        WEIGHTS: _ComponentBackend(WEIGHTS, tmp_path / "candidates"),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates", stale_policy="merge"),
    }
    dispatcher, _ = dispatchers.open(backends=backends, experiment_tracker=tracker)
    scenario = _scenario(dispatcher, 1)
    harness = scenario.prepare_training_step(HARNESS)
    weights = scenario.prepare_training_step(WEIGHTS)
    assert harness is not None and weights is not None
    dispatcher._commit_result("agent", weights, WEIGHTS)
    served = scenario.current_artifact_ref().release_id
    dispatcher._commit_result("agent", harness, HARNESS)
    assert [event.context.component for event in tracker.events] == [WEIGHTS, HARNESS]
    assert tracker.events[-1].metrics["merged_onto"] == served
    assert scenario.releases()[0]["metrics"]["merged_onto"] == served
    assert "merged_onto" not in tracker.events[0].metrics


@pytest.mark.unit
def test_a_creation_manifest_read_that_failed_is_asked_again(monkeypatch: Any, dispatchers: _Dispatchers) -> None:
    """One failed read of the creation artifact must not refuse every later promote for the process lifetime."""
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
    assert scenario.creation_components(0) is None
    # Not asked again at the same step (a page polls every few seconds; a remote read is a fetch)...
    assert scenario.creation_components(0) is None
    assert calls["failed"] == 1
    # ...but asked again after the next commit.
    components = scenario.creation_components(1)
    assert components is not None and set(components) == {WEIGHTS, HARNESS}
    assert calls["failed"] == 1
    assert scenario.creation_components(0) is not None


@pytest.mark.unit
def test_a_promote_of_weights_held_at_step_one_is_no_harness_release(dispatchers: _Dispatchers) -> None:
    """The creation artifact has no record; its manifest still says the promoted tree did not change."""
    dispatcher, backends = dispatchers.open()
    backends[WEIGHTS].hold_next = True
    scenario = _scenario(dispatcher)
    service = RequestService(dispatcher)
    headers = {"x-reef-scenario": "agent"}
    creation = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    weights = scenario.prepare_training_step(WEIGHTS)
    assert weights is not None and weights.pending
    scenario.commit(weights, component=WEIGHTS)
    held = next(row["release_id"] for row in scenario.releases() if row.get("pending"))
    scenario.rollback(held, operation="promote")
    assert _component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 1"
    assert service.harness_head(headers) == creation
    assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [creation]


@pytest.mark.unit
def test_a_rollback_under_the_same_tree_is_no_new_harness_head(dispatchers: _Dispatchers) -> None:
    """A rollback or promote that changed only the weights is not announced or listed as a harness release."""
    dispatcher, _ = dispatchers.open()
    scenario = _scenario(dispatcher)
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
    scenario.rollback(first_weights)
    assert _component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 1"
    assert service.harness_head(headers) == harness_release
    assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [base, harness_release]

    # Back to the base: the seed tree is served again, and that is a harness release.
    scenario.rollback(base)
    restored = scenario.current_artifact_ref().release_id
    assert service.harness_head(headers) == restored
    assert service.harness_manifest(headers)["release_id"] == restored
    assert service.harness_manifest(headers)["files"] == {"harness.txt": "harness seed"}
    assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [
        base,
        harness_release,
        restored,
    ]


@pytest.mark.unit
def test_committed_job_id_outlives_another_trainers_commit(tmp_path: Path, dispatchers: _Dispatchers) -> None:
    """The backend must finish the weights job even after the harness moved the scenario step."""
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher)
    assert scenario.dispatched_component == WEIGHTS
    base = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    _commit_weights_job(scenario)
    assert scenario.committed_training_job_id == "job-1"

    assert _commit_step(scenario, HARNESS) is not None
    assert scenario.scenario_step == 2
    assert scenario.committed_training_job_id == "job-1"
    assert scenario.committed_training_without_job_id is False

    # A rollback after the weights commit is the one thing that unmakes the proof.
    scenario.rollback(base)
    assert scenario.committed_training_job_id is None


@pytest.mark.unit
def test_adopted_checkpoint_is_attributed_to_the_trainer_that_made_it(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A checkpoint read back from the artifact head names its component, so a lost commit log resets no trainer."""
    backend_factory = _seeded_repository(tmp_path)
    dispatcher, _ = dispatchers.open(records_dir=tmp_path / "records-1", backend_factory=backend_factory)
    scenario = _scenario(dispatcher)
    base = scenario.current_artifact_ref().release_id
    _append_records(scenario, 1)
    assert _commit_step(scenario, HARNESS) is not None
    head = scenario.repository.materialize(scenario.current_artifact_ref())
    checkpoint = head.metadata[SCENARIO_METADATA_KEY]
    assert isinstance(checkpoint, Mapping)
    assert checkpoint["component"] == HARNESS
    assert checkpoint["base_release_id"] == base
    _, _, adopted = parse_scenario_metadata(checkpoint, checkpoint_head=head.ref)
    assert adopted is not None and adopted.component == HARNESS and adopted.base_release_id == base
    dispatcher.close()

    # The record store is gone; the checkpoint is adopted from the head and belongs to the harness trainer.
    recovered, _ = dispatchers.open(records_dir=tmp_path / "records-2", backend_factory=backend_factory)
    scenario = _scenario(recovered)
    assert scenario.scenario_step == 1
    assert scenario.trainer_for(HARNESS).state == {"steps": 1}
    assert scenario.trainer_for(WEIGHTS).state == {"steps": 0}
    last = scenario.last_commit_for(HARNESS)
    assert last is not None and last.step == 1 and last.component == HARNESS
    assert scenario.last_commit_for(WEIGHTS) is None


@pytest.mark.unit
def test_dispatcher_runs_one_worker_per_component(dispatchers: _Dispatchers) -> None:
    dispatcher, _ = dispatchers.open()
    _scenario(dispatcher)
    for step in (1, 2):
        for record in _records(step):
            dispatcher.accept_record(record)

    def both_committed_twice() -> bool:
        current = _scenario(dispatcher)
        committed = {record.component for record in current.store.history() if record.step <= 4}
        return current.scenario_step >= 4 and committed == {WEIGHTS, HARNESS}

    assert _wait_for(both_committed_twice, timeout_seconds=20), "both component workers did not commit twice"
    current = _scenario(dispatcher)
    assert _component_files(current, current.current_artifact_ref()) == {
        WEIGHTS: "weights step 2",
        HARNESS: "harness step 2",
    }
    assert all(record.base_release_id is not None for record in current.store.history())
    status = dispatcher.build_training_status()
    assert status["error"] is None
    # Local workers take turns for a whole cycle, so neither is ever refused as stale.
    components = status["scenarios"]["agent"]["components"]
    assert [components[name]["stale_refusals_total"] for name in (WEIGHTS, HARNESS)] == [0, 0]


@pytest.mark.unit
def test_stale_refusals_are_counted_and_bounded(dispatchers: _Dispatchers) -> None:
    """A local result overtaken again and again is reported and parked, not prepared and discarded forever."""
    dispatcher, backends = dispatchers.open()
    scenario = _scenario(dispatcher)
    outcomes = []
    for step in (1, 2, 3):
        _append_records(scenario, step)
        # The weights result is prepared against the served release; the harness cycle then replaces it.
        assert scenario.prepare_training_step(WEIGHTS) is not None
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        outcomes.append(dispatcher._process_local_backend_step("agent", WEIGHTS))
    assert outcomes == [True, True, False]
    assert backends[WEIGHTS].prepared == 3
    assert scenario.trainer_for(WEIGHTS).pending_batch is not None
    status = dispatcher.build_training_status()
    assert status["scenarios"]["agent"]["components"][WEIGHTS]["stale_refusals_total"] == 3
    assert "refused 3 times in a row" in status["error"]

    # The kept batch is prepared again on the next wake and commits.
    assert dispatcher._process_local_backend_step("agent", WEIGHTS) is True
    assert scenario.trainer_for(WEIGHTS).state == {"steps": 1}
    status = dispatcher.build_training_status()
    assert status["error"] is None
    assert status["scenarios"]["agent"]["components"][WEIGHTS]["stale_refusals_total"] == 3


@pytest.mark.unit
def test_commit_record_carries_component_and_base_release() -> None:
    record = CommitRecord(
        scenario="agent",
        step=3,
        artifact_ref=ArtifactRef("c", "r3", "r2"),
        checkpoint=True,
        algorithm_state={"steps": 1},
        high_water_sequence=2,
        high_water_offset=2,
        component=HARNESS,
        base_release_id="r2",
    )
    encoded = record.to_dict()
    assert encoded["component"] == HARNESS
    assert encoded["base_release_id"] == "r2"
    assert CommitRecord.from_dict(encoded) == record
    assert "component" not in CommitRecord.from_dict({**encoded, "component": None}).to_dict()
    with pytest.raises(CommitLogError, match="only training commits may carry component"):
        CommitRecord(
            scenario="agent",
            step=4,
            artifact_ref=ArtifactRef("c", "r4", "r3"),
            checkpoint=True,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
            operation="rollback",
            rollback_target_release_id="r1",
            component=HARNESS,
        )


@pytest.mark.unit
def test_component_trainers_must_match_the_surface() -> None:
    def trainer() -> Trainer:
        return Trainer.build("agent", SQLiteRecordStore(), processor_factory=DataProcessor)

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
@pytest.mark.parametrize("policy", ["refuse", "reevaluate"])
def test_a_stale_retry_of_a_batch_an_earlier_attempt_took_back_is_prepared_again_and_never_acknowledged_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str, dispatchers: _Dispatchers
) -> None:
    """The harness acknowledged its batch and its record write failed while a weights job was out, so its reload
    waits. The job lands and the next one goes out at once: the retry on the same instance is refused as stale. The
    batch stays taken, so the next cycle prepares or evaluates it again and records what the first attempt consumed,
    instead of asking the processor for a batch it gave back."""
    dispatcher, backends = dispatchers.open(backends=_dropping_pair(tmp_path, harness_policy=policy))
    scenario = _scenario(dispatcher, 1)
    first = scenario.reserve_training_batch(WEIGHTS)
    assert first is not None
    _fail_first_record_of(monkeypatch, scenario, HARNESS)
    with pytest.raises(OSError, match="transient store write error"):
        dispatcher._process_local_backend_step("agent", HARNESS)
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], first) is True
    _append_records(scenario, 2)
    second = scenario.reserve_training_batch(WEIGHTS)
    assert second is not None
    outcomes = [dispatcher._process_local_backend_step("agent", HARNESS) for _ in range(3)]
    assert dispatcher._registry.get_optional("agent") is scenario
    harness_rows = [row for row in scenario.store.history() if row.component == HARNESS]
    assert harness_rows and sorted(harness_rows[0].consumed_ids) == ["i1", "r1"]
    consumed = [row.consumed_ids for row in harness_rows]
    assert all(not (a & b) for index, a in enumerate(consumed) for b in consumed[index + 1 :])
    assert True in outcomes
    if policy == "reevaluate":
        assert backends[HARNESS].reevaluations >= 1


class _SkippingBackend(_ComponentBackend):
    """A local cycle whose proposer produced nothing: a step with no candidate."""

    def prepare_step(self, batch, state, scenario_step):
        self.prepared += 1
        return PreparedStep.skipped(state={"steps": int(state["steps"]) + 1}, metrics={"skipped": "no proposal"})


@pytest.mark.unit
def test_a_skip_meets_a_moved_head_and_commits_since_it_was_evaluated_against_no_release(
    tmp_path: Path, dispatchers: _Dispatchers
) -> None:
    """A step with no candidate is not stale whatever another trainer published meanwhile: refusing it would run
    the backend again (a failed instruction's skip would run the failed instruction) for nothing."""
    skipping = _SkippingBackend(HARNESS, tmp_path / "candidates", stale_policy="refuse")
    dispatcher, _ = dispatchers.open(
        backends={WEIGHTS: _ComponentBackend(WEIGHTS, tmp_path / "candidates"), HARNESS: skipping}
    )
    scenario = _scenario(dispatcher, 1)
    harness = scenario.prepare_training_step(HARNESS)
    weights = scenario.prepare_training_step(WEIGHTS)
    assert harness is not None and weights is not None
    scenario.commit(weights, component=WEIGHTS)
    scenario.commit(harness, component=HARNESS)
    assert [row.component for row in scenario.store.history()] == [WEIGHTS, HARNESS]
    assert skipping.prepared == 1


def _harness_record_durable_then_failed(
    dispatcher: Dispatcher, scenario: Any, monkeypatch: pytest.MonkeyPatch, fault: str = "install"
) -> None:
    """The harness commit reaches the log and then fails (a lost acknowledgment or a failed ref install)."""
    committer = scenario._committer
    name = "_append_commit_record" if fault == "append" else "_install_committed_checkpoint"
    original = committer._append_commit_record if fault == "append" else committer._install_committed_checkpoint
    failed: list[bool] = []

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        outcome = original(*args, **kwargs)
        if not failed:
            failed.append(True)
            raise OSError("acknowledgment lost")
        return outcome

    monkeypatch.setattr(committer, name, fail_once)
    with pytest.raises(OSError, match="acknowledgment lost"):
        dispatcher._process_local_backend_step("agent", HARNESS)


@pytest.mark.unit
@pytest.mark.parametrize("fault", ["append", "install"])
def test_a_sibling_commit_durable_but_not_settled_is_settled_before_the_next_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, dispatchers: _Dispatchers
) -> None:
    """The harness record reached the log and then its commit failed (a lost fsync acknowledgment, a failed ref
    install) while a weights job was out, so its reload waits. The weights commit settles that record first, as
    the harness's own retry would, and takes the next step: no conflict and no extra reload of the job."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher, 1, 2)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    _harness_record_durable_then_failed(dispatcher, scenario, monkeypatch, fault)
    assert [row.component for row in scenario.store.history()] == [HARNESS]
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    assert [row.component for row in scenario.store.history()] == [HARNESS, WEIGHTS]
    assert scenario.trainer_for(HARNESS).pending_batch is None
    assert [row.step for row in scenario.store.history()] == [1, 2]


@pytest.mark.unit
@pytest.mark.parametrize("fault", ["append", "install"])
def test_a_harness_cycle_in_flight_never_commits_a_result_the_weights_commit_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, dispatchers: _Dispatchers
) -> None:
    """The harness worker took its cached result, then waited for the registry lock the weights commit held. That
    commit settled the harness record, so the worker's result is on record: it looks again and publishes nothing,
    where it used to land a third step with nothing consumed and republish the harness artifact."""
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path))
    scenario = _scenario(dispatcher, 1, 2)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    _harness_record_durable_then_failed(dispatcher, scenario, monkeypatch, fault)
    prepared = threading.Event()
    go = threading.Event()
    real_prepare = scenario.prepare_training_step

    def paused_prepare(component: str | None = None) -> Any:
        result = real_prepare(component)
        prepared.set()
        assert go.wait(10)
        return result

    monkeypatch.setattr(scenario, "prepare_training_step", paused_prepare)
    outcome: dict[str, Any] = {}

    def harness_cycle() -> None:
        try:
            outcome["progressed"] = dispatcher._process_local_backend_step("agent", HARNESS)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=harness_cycle)
    worker.start()
    assert prepared.wait(10)
    releases_before = len(scenario.releases())
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    go.set()
    worker.join(10)
    assert outcome == {"progressed": True}
    assert [(row.step, row.component) for row in scenario.store.history()] == [(1, HARNESS), (2, WEIGHTS)]
    assert len(scenario.releases()) == releases_before + 1


@pytest.mark.unit
def test_a_commit_that_settles_a_sibling_record_gives_each_step_its_own_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dispatchers: _Dispatchers
) -> None:
    """The weights commit settled the harness record first: the harness step gets its event at step 1, from its
    record, and the weights commit gets its event at step 2, the step it took."""
    tracker = _RecordingTracker()
    dispatcher, backends = dispatchers.open(backends=_dispatched_pair(tmp_path), experiment_tracker=tracker)
    scenario = _scenario(dispatcher, 1, 2)
    batch = scenario.reserve_training_batch(WEIGHTS)
    assert batch is not None
    _harness_record_durable_then_failed(dispatcher, scenario, monkeypatch)
    assert dispatcher.run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
    history = scenario.store.history()
    assert [(row.step, row.component) for row in history] == [(1, HARNESS), (2, WEIGHTS)]
    events = [(event.context.step, event.context.component, event.training_job_id) for event in tracker.events]
    assert events == [(1, HARNESS, None), (2, WEIGHTS, "job-1")]
    weights_event = tracker.events[1]
    assert weights_event.context.source_artifact_ref.release_id == history[0].artifact_ref.release_id
    assert tracker.events[0].produced_artifact_ref.release_id == history[0].artifact_ref.release_id


@pytest.mark.unit
def test_the_status_names_the_job_out_and_its_owner_or_that_it_names_none(
    tmp_path: Path, monkeypatch: Any, dispatchers: _Dispatchers
) -> None:
    """An operator reads whose job the runtime holds out: a marker from before markers named their owner shows no
    owner, the window in which no delete on the runtime is safe."""
    training = StubTrainingRuntime()
    dispatcher, _ = dispatchers.open(backends=_dispatched_pair(tmp_path), training=training)
    _scenario(dispatcher)
    marker: dict[str, Any] | None = None
    monkeypatch.setattr(training, "training_job_status", lambda: marker)
    assert dispatcher.build_training_status()["training_job"] is None
    marker = {"status": "READY_TO_COMMIT", "training_job_id": "job-1", "scenario": "agent"}
    assert dispatcher.build_training_status()["training_job"] == {
        "status": "READY_TO_COMMIT",
        "training_job_id": "job-1",
        "owner": "agent",
    }
    marker = {"status": "CHECKPOINT", "training_job_id": "job-0"}
    assert dispatcher.build_training_status()["training_job"]["owner"] is None
    marker = {"status": "COMPLETE", "training_job_id": "job-1", "commit_acknowledged": True, "scenario": "agent"}
    assert dispatcher.build_training_status()["training_job"] is None


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
