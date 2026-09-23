"""One trainer per release component: independent workers that meet at the commit boundary."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

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
from reef.train.processors.base import DataProcessor, RetentionDecision
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


class _HybridThresholdProcessor(ThresholdProcessor):
    """The same processor, taking instructions too."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})


class _InferenceOnlyProcessor(DataProcessor):
    """A processor that never needs a report row."""

    required_request_types = frozenset({RequestType.INFERENCE})


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

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            processor = _HybridThresholdProcessor if component in self.hybrid_components else ThresholdProcessor
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


def _records(step: int) -> tuple[AgentRecord, AgentRecord]:
    inference = AgentRecord.create(
        scenario="agent",
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
        agent_record_id=f"i{step}",
    )
    report = AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"i{step}"]},
        agent_record_id=f"r{step}",
        references=(f"i{step}",),
    )
    return inference, report


def _report(record_id: str, reference: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": [reference]},
        agent_record_id=record_id,
        references=(reference,),
    )


def _dispatcher(
    tmp_path: Path,
    *,
    records_dir: Path | None = None,
    backend_factory: Any = None,
    backends: dict[str, _ComponentBackend] | None = None,
    hybrid_components: frozenset[str] = frozenset(),
    experiment_tracker: ExperimentTracker | None = None,
    training: StubTrainingRuntime | None = None,
) -> tuple[Dispatcher, dict[str, _ComponentBackend]]:
    initial = tmp_path / "initial"
    if not initial.exists():
        for component in (WEIGHTS, HARNESS):
            (initial / component).mkdir(parents=True)
            (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    if backends is None:
        backends = {
            component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)
        }
    records = tmp_path / "records" if records_dir is None else records_dir
    recipe = _TwoTrainerRecipe(backends=backends, hybrid_components=hybrid_components)
    if training is not None:
        # A training runtime binds the scenario to the training thread, which drives the dispatched job.
        recipe = _TwoTrainerRecipe(
            backends=backends,
            hybrid_components=hybrid_components,
            runtime=training.inference,
            training_runtime=training,
        )
    dispatcher = Dispatcher(
        recipe,
        backend_factory or InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
        experiment_tracker=experiment_tracker,
    )
    return dispatcher, backends


def _component_files(scenario: Scenario, ref: ArtifactRef) -> dict[str, str]:
    head = scenario.repository.materialize(ref)
    return {
        component: (TextFileTree().read_files(head.component(component)) or {})[f"{component}.txt"]
        for component in (WEIGHTS, HARNESS)
    }


@pytest.mark.unit
def test_component_trainers_meet_at_the_commit_boundary(tmp_path: Path) -> None:
    dispatcher, backends = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        assert [bound.component for bound in scenario.component_trainers] == [WEIGHTS, HARNESS]
        assert scenario.trainer_for(HARNESS) is scenario.component_trainers[1].trainer
        base = scenario.current_artifact_ref().release_id
        # Appended directly so the test, not a worker thread, owns when each trainer prepares.
        for record in _records(1):
            scenario.records.append(record)

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
        # Rows both trainers consumed are retired only once both have committed.
        assert records[0].compacted_ids == frozenset()
        assert records[1].compacted_ids == frozenset({"i1", "r1"})
        assert records[0].consumed_ids == records[1].consumed_ids == frozenset({"i1", "r1"})

        # Each trainer recovers from its own commits.
        reloaded = dispatcher._registry.reload("agent")
        assert reloaded.trainer_for(WEIGHTS).state == {"steps": 1}
        assert reloaded.trainer_for(HARNESS).state == {"steps": 1}
        assert reloaded.scenario_step == 2
        for record in _records(2):
            reloaded.records.append(record)
        assert reloaded.prepare_training_step(HARNESS) is not None
        assert reloaded.prepare_training_step(WEIGHTS) is not None
    finally:
        dispatcher.close()


class _ColocatedBackend(_DispatchedBackend):
    """A dispatched job that holds the served engine while it runs, as a colocated Slime stack does."""

    @property
    def colocated(self) -> bool:
        return True


def _dispatched_pair(tmp_path: Path, job_id: str = "job-1") -> dict[str, _ComponentBackend]:
    return {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", job_id),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates"),
    }


@pytest.mark.unit
def test_a_local_step_of_one_trainer_does_not_hold_up_the_others_or_the_status(tmp_path: Path) -> None:
    """While the harness evaluates for minutes, the weights job still reserves and commits, and status still reads."""
    backends = {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _SlowBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    slow = backends[HARNESS]
    assert isinstance(slow, _SlowBackend)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        base = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
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
    finally:
        slow.release.set()
        dispatcher.close()


@pytest.mark.unit
def test_a_colocated_weights_job_waits_for_the_local_cycle_that_needs_the_engine(tmp_path: Path) -> None:
    """A colocated job holds the served engine, so it starts only after the harness cycle mid evaluation ends."""
    backends = {
        WEIGHTS: _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _SlowBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    slow = backends[HARNESS]
    assert isinstance(slow, _SlowBackend)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
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
        assert backends[WEIGHTS].prepared == 0
        slow.release.set()
        worker.join(10)
        trainer.join(10)
        assert not trainer.is_alive()
        assert backends[WEIGHTS].prepared == 1
        assert executions[0].outcome == "commit"
        assert scenario.scenario_step == 1
    finally:
        slow.release.set()
        dispatcher.close()


def _records_for(scenario: str, step: int) -> tuple[AgentRecord, AgentRecord]:
    inference = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
        agent_record_id=f"{scenario}-i{step}",
    )
    report = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"{scenario}-i{step}"]},
        agent_record_id=f"{scenario}-r{step}",
        references=(f"{scenario}-i{step}",),
    )
    return inference, report


@pytest.mark.unit
def test_a_colocated_weights_job_waits_for_every_scenario_that_shares_the_engine(tmp_path: Path) -> None:
    """Admission is engine wide, so another scenario's harness cycle mid evaluation finishes first too."""
    backends = {
        WEIGHTS: _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _SlowBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    slow = backends[HARNESS]
    assert isinstance(slow, _SlowBackend)
    try:
        agent = dispatcher.get_or_create_scenario("agent")
        other = dispatcher.get_or_create_scenario("other")
        assert agent is not None and other is not None
        for record in _records_for("agent", 1):
            agent.records.append(record)
        for record in _records_for("other", 1):
            other.records.append(record)
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
        assert backends[WEIGHTS].prepared == 0
        slow.release.set()
        worker.join(10)
        trainer.join(10)
        assert not trainer.is_alive()
        assert executions[0].outcome == "commit"
    finally:
        slow.release.set()
        dispatcher.close()


@pytest.mark.unit
def test_a_dispatched_turn_wakes_every_local_worker_and_yields_before_the_next_cycle(tmp_path: Path) -> None:
    """Workers of every scenario stand aside for a waiting job and are woken when its turn ends."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        for name in ("agent", "other"):
            scenario = dispatcher.get_or_create_scenario(name)
            assert scenario is not None
            for record in _records_for(name, 1):
                scenario.records.append(record)
        # A job is waiting for its turn: a worker woken now runs no cycle.
        dispatcher._training.turn_waiting.set()
        dispatcher._start_local_backend_worker("other", HARNESS)
        time.sleep(0.5)
        assert backends[HARNESS].prepared == 0
        # The turn ends: every worker is woken and the cycle runs.
        dispatcher._training.turn_waiting.clear()
        dispatcher.wake_local_workers()
        deadline = time.monotonic() + 10
        while backends[HARNESS].prepared == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert backends[HARNESS].prepared == 1
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_reload_after_a_training_failure_wakes_the_local_workers(tmp_path: Path) -> None:
    """The rebuilt instance holds the local components' rows unread; their workers look again."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        dispatcher._training.turn_waiting.set()
        dispatcher._start_local_backend_worker("agent", HARNESS)
        time.sleep(0.2)
        dispatcher._training.turn_waiting.clear()
        for record in _records(1):
            scenario.records.append(record)
        dispatcher._reload_after_training_failure("agent", RuntimeError("job away"))
        deadline = time.monotonic() + 10
        while backends[HARNESS].prepared == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert backends[HARNESS].prepared == 1
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_deleting_a_scenario_whose_training_job_is_out_waits_for_the_job(tmp_path: Path) -> None:
    """The job could neither commit nor be acknowledged without its scenario, so the delete answers busy until it lands."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        with pytest.raises(ScenarioBusy, match="training job is out"):
            dispatcher.delete_scenario("agent")
        assert dispatcher._registry.has("agent")
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
        assert dispatcher.delete_scenario("agent")["scenario"] == "agent"
        assert not dispatcher._registry.has("agent")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_scenario_removed_under_its_training_job_is_an_error_not_a_silent_replay(tmp_path: Path) -> None:
    """The registry path a delete no longer takes: a job without its scenario needs operator recovery."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        old = dispatcher.get_or_create_scenario("agent")
        assert old is not None
        for record in _records(1):
            old.records.append(record)
        batch = old.reserve_training_batch(WEIGHTS)
        assert batch is not None
        dispatcher._registry.remove("agent")
        with pytest.raises(RuntimeContractError, match="deleted under its training job"):
            dispatcher._run_dispatched_turn(old, WEIGHTS, backends[WEIGHTS], batch)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_training_thread_idles_once_the_last_training_scenario_is_gone(tmp_path: Path) -> None:
    """A wake that finds no training scenario is not a failure: nothing would ever clear its error."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        assert dispatcher._process_training() is False
        dispatcher._drain_training()
        assert dispatcher.build_training_status()["error"] is None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_failed_instruction_of_a_scenario_that_is_gone_starts_no_worker(tmp_path: Path, monkeypatch: Any) -> None:
    """Workers started for a deleted name would record errors that nothing clears."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        old = dispatcher.get_or_create_scenario("agent")
        assert old is not None
        dispatcher.delete_scenario("agent")
        monkeypatch.setattr(old.trainer_for(HARNESS), "fail_pending_instruction", lambda error: True)
        dispatcher._fail_instruction(old, RuntimeError("episode failed"), HARNESS)
        with dispatcher._training.lock:
            assert dispatcher._training.local_workers == {}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_failing_local_cycle_runs_once_per_wake_and_keeps_its_error_in_the_status(tmp_path: Path) -> None:
    """A backend that is away costs one attempt per wake, never a loop of reloads, and the failure stays visible."""
    backends = {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _AwayBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    away = backends[HARNESS]
    assert isinstance(away, _AwayBackend)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        # Both rows first, then one wake: an accept wakes the worker per row, which would be two wakes.
        for record in _records(1):
            scenario.records.append(record)
        dispatcher._start_local_backend_worker("agent", HARNESS)
        deadline = time.monotonic() + 10
        while away.attempts == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        assert away.attempts == 1
        assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"
        # A job's turn ended: the worker looks again, once.
        dispatcher.wake_local_workers()
        deadline = time.monotonic() + 10
        while away.attempts == 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        assert away.attempts == 2
        assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_local_failure_under_a_dispatched_job_reloads_once_the_job_has_landed(tmp_path: Path) -> None:
    """A reload under the job would hand its result to an instance that never reserved it."""
    backends = {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _AwayBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        with pytest.raises(RuntimeError, match="proposer away"):
            dispatcher._process_local_backend_step("agent", HARNESS)
        assert dispatcher._registry.get_optional("agent") is scenario
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
        assert [row["component"] for row in scenario.releases() if row["operation"] == "training"] == [WEIGHTS]
        # The next local cycle rebuilds the scenario as its first act, then looks again on the new instance.
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        rebuilt = dispatcher._registry.get_optional("agent")
        assert rebuilt is not None and rebuilt is not scenario
        assert rebuilt.scenario_step == 1
        # With no job out, a failure reloads at once.
        with pytest.raises(RuntimeError, match="proposer away"):
            dispatcher._process_local_backend_step("agent", HARNESS)
        assert dispatcher._registry.get_optional("agent") is not rebuilt
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_deferred_reload_never_lands_under_a_cycle_that_runs_on_the_instance(tmp_path: Path) -> None:
    """A cycle woken while the job is still out commits on the instance; the rebuild waits for the cycle after."""
    backends = {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _FlakyBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        with pytest.raises(RuntimeError, match="away once"):
            dispatcher._process_local_backend_step("agent", HARNESS)
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        assert dispatcher._registry.get_optional("agent") is scenario
        assert [row["component"] for row in scenario.releases() if row["operation"] == "training"] == [HARNESS]
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        rebuilt = dispatcher._registry.get_optional("agent")
        assert rebuilt is not None and rebuilt is not scenario
        assert [row["component"] for row in rebuilt.releases() if row["operation"] == "training"] == [WEIGHTS, HARNESS]
    finally:
        dispatcher.close()


class _BlockingAwayBackend(_ComponentBackend):
    """A local cycle whose preparation waits for the test to let it go, then fails."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(component, artifact_dir)
        self.entered = threading.Event()
        self.go = threading.Event()

    def prepare_step(self, batch, state, scenario_step):
        self.entered.set()
        assert self.go.wait(30)
        raise RuntimeError("proposer away")


@pytest.mark.unit
def test_a_sibling_waiting_at_the_cycle_lock_looks_again_after_a_reload_under_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The failing cycle rebuilds the scenario under the lock; the sibling that read the old instance looks again
    on the new one instead of sleeping with its rows unread."""
    backends = {
        WEIGHTS: _BlockingAwayBackend(WEIGHTS, tmp_path / "candidates"),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    away = backends[WEIGHTS]
    assert isinstance(away, _BlockingAwayBackend)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
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
        cycle_lock = dispatcher._local_cycle_lock
        outcome: list[bool] = []
        sibling = threading.Thread(
            target=lambda: outcome.append(dispatcher._process_local_backend_step("agent", HARNESS))
        )

        def observed(name: str):
            lock = cycle_lock(name)
            if threading.current_thread() is sibling:
                reached.set()
            return lock

        monkeypatch.setattr(dispatcher, "_local_cycle_lock", observed)
        sibling.start()
        assert reached.wait(10)
        away.go.set()
        failing.join(10)
        sibling.join(10)
        assert [str(error) for error in failures] == ["proposer away"]
        rebuilt = dispatcher._registry.get_optional("agent")
        assert rebuilt is not None and rebuilt is not scenario
        assert outcome == [True] and backends[HARNESS].prepared == 0
        # Looking again on the rebuilt instance commits the rows that were there all along.
        assert dispatcher._process_local_backend_step("agent", HARNESS) is True
        assert [row["component"] for row in rebuilt.releases() if row["operation"] == "training"] == [HARNESS]
    finally:
        away.go.set()
        dispatcher.close()


@pytest.mark.unit
def test_deleting_a_scenario_is_refused_while_the_runtime_cannot_say_whether_its_job_is_out(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A runtime that does not answer gets a busy answer, not a crash: a blind delete could orphan a job. A marker
    names the scenario that owns its job and refuses that scenario's delete alone; a marker written before markers
    named an owner refuses none, and the scenario holding the job's reservation is still refused."""
    training = StubTrainingRuntime()
    dispatcher, _ = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"), training=training)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None

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
        # An earlier build's marker names no owner: it says nothing about whose job is out.
        marker = {"status": "RUNNING", "training_job_id": "job-1"}
        assert dispatcher.training_job_in_flight("agent") is False
        assert dispatcher.training_job_in_flight("old") is False
        marker = {"status": "COMPLETE", "training_job_id": "job-1", "commit_acknowledged": True, "scenario": "agent"}
        assert dispatcher.delete_scenario("agent")["scenario"] == "agent"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_after_a_restart_binds_a_stray_registration_its_delete_lets_the_jobs_owner_bind_and_finish(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """b trains on a one scenario runtime; a create of a is refused but leaves a's registration. b's job is out when
    the process dies, and after the restart a binds first. The marker names b, so b cannot be deleted while its job
    is out, and a, the stray, can: its delete and a restart let b bind again, the only log that can finish the job.
    A marker an earlier build wrote names no owner and refuses neither delete, so the same way out stays open."""
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher, _ = _dispatcher(
        tmp_path, backends=_dispatched_pair(tmp_path), training=StubTrainingRuntime(), backend_factory=factory
    )
    try:
        assert dispatcher.get_or_create_scenario("b") is not None
        with pytest.raises(ReefError, match="already bound"):
            dispatcher.get_or_create_scenario("a")
    finally:
        dispatcher.close()

    def restarted(marker: dict[str, Any]) -> Dispatcher:
        training = StubTrainingRuntime()
        monkeypatch.setattr(training, "training_job_status", lambda: marker)
        restarted_dispatcher, _ = _dispatcher(
            tmp_path, backends=_dispatched_pair(tmp_path), training=training, backend_factory=factory
        )
        thread = restarted_dispatcher._lifecycle.preload_thread
        if thread is not None:
            thread.join(30)
        return restarted_dispatcher

    dispatcher = restarted({"status": "UPDATING_WEIGHTS", "training_job_id": "job-b", "scenario": "b"})
    try:
        assert dispatcher._registry.training_scenario_name == "a"
        with pytest.raises(ScenarioBusy, match="training job is out"):
            dispatcher.delete_scenario("b")
        assert dispatcher.delete_scenario("a")["scenario"] == "a"
    finally:
        dispatcher.close()
    dispatcher = restarted({"status": "UPDATING_WEIGHTS", "training_job_id": "job-b"})
    try:
        assert dispatcher._registry.training_scenario_name == "b"
        assert dispatcher.training_job_in_flight("b") is False
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_delete_guard_reads_the_runtime_once(tmp_path: Path) -> None:
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
    dispatcher, _ = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path), training=StubTrainingRuntime())
    try:
        object.__setattr__(dispatcher._recipe, "training_runtime", ExecutorTrainingRuntime(handle))  # type: ignore[arg-type]
        assert dispatcher.training_job_in_flight("x") is True
        assert handle.calls == 1
    finally:
        object.__setattr__(dispatcher._recipe, "training_runtime", None)
        dispatcher.close()


@pytest.mark.unit
def test_deleting_a_scenario_whose_job_marker_is_out_at_the_backend_waits(tmp_path: Path, monkeypatch: Any) -> None:
    """The backend's marker outlives a rebuilt or unloaded instance; the delete reads it, not the reserved batch."""
    training = StubTrainingRuntime()
    dispatcher, _ = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"), training=training)
    try:
        assert dispatcher.get_or_create_scenario("agent") is not None
        marker: dict[str, Any] = {"status": "CHECKPOINT", "training_job_id": "job-1", "scenario": "agent"}
        monkeypatch.setattr(training, "training_job_status", lambda: marker)
        with pytest.raises(ScenarioBusy, match="training job is out"):
            dispatcher.delete_scenario("agent")
        marker["scenario"] = "other"
        assert dispatcher.delete_scenario("agent")["scenario"] == "agent"
        assert dispatcher.get_or_create_scenario("again") is not None
        marker = {"status": "COMPLETE", "training_job_id": "job-2", "commit_acknowledged": False, "scenario": "again"}
        with pytest.raises(ScenarioBusy, match="training job is out"):
            dispatcher.delete_scenario("again")
        marker["commit_acknowledged"] = True
        assert dispatcher.delete_scenario("again")["scenario"] == "again"
    finally:
        dispatcher.close()


class _GroupedProcessor(ThresholdProcessor):
    """Reports batch by the group named in their metadata; a group is ready at two members."""

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return context.report.payload["metadata"]["group"], None

    def decide_group(self, key: Hashable, items: tuple[Any, ...]) -> GroupDecision:
        return GroupDecision.READY if len(items) >= 2 else GroupDecision.INCOMPLETE


@dataclass(frozen=True)
class _GroupedRecipe(_TwoTrainerRecipe):
    """The two trainer recipe whose harness batches reports by group."""

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            processor = _GroupedProcessor if component == HARNESS else ThresholdProcessor
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


def _grouped_report(record_id: str, reference: str, group: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [reference], "metadata": {"group": group}},
        agent_record_id=record_id,
        references=(reference,),
    )


@pytest.mark.unit
def test_a_report_live_in_an_incomplete_group_before_a_restart_still_trains_after_it(tmp_path: Path) -> None:
    """The replay rebuilds the live state as it was: a report ingested before its source trained stays live."""
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    records = tmp_path / "records"
    dispatcher = Dispatcher(
        _GroupedRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for step in (1, 2, 3):
            scenario.records.append(_records(step)[0])
        scenario.records.append(_grouped_report("r1c", "i1", "B"))
        scenario.records.append(_grouped_report("r1a", "i1", "A"))
        scenario.records.append(_grouped_report("r2", "i2", "A"))
        result = scenario.prepare_training_step(HARNESS)
        assert result is not None
        scenario.commit(result, component=HARNESS)
        assert scenario.prepare_training_step(HARNESS) is None
        rebuilt = dispatcher._registry.reload("agent")
        rebuilt.records.append(_grouped_report("r3", "i3", "B"))
        assert rebuilt.prepare_training_step(HARNESS) is not None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_scenario_of_several_components_refuses_a_checkpoint_interval_above_one(tmp_path: Path) -> None:
    """A step that published no checkpoint would see its component dropped from the next release."""
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    records = tmp_path / "records"
    dispatcher = Dispatcher(
        _TwoTrainerRecipe(backends=backends, checkpoint_strategy=EveryNVersions(2)),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
    )
    try:
        with pytest.raises(ReefError, match="checkpoint interval must be 1"):
            dispatcher.get_or_create_scenario("agent")
    finally:
        dispatcher.close()


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
def test_a_recipe_serving_fewer_components_than_registered_is_refused(tmp_path: Path) -> None:
    """A step carries forward only the components the recipe serves; one it does not serve would leave every release."""
    names = ("weights", "harness", "config")
    initial = tmp_path / "initial"
    for name in names:
        (initial / name).mkdir(parents=True)
        (initial / name / f"{name}.txt").write_text(f"{name} seed", encoding="utf-8")
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    records = tmp_path / "records"

    def serve(served: tuple[str, ...]) -> Dispatcher:
        return Dispatcher(
            _NamedComponentsRecipe(names=served, artifact_dir=tmp_path / "candidates"),
            factory,
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=records,
            scenario_storage=SQLiteScenarioStorage(records),
        )

    first = serve(names)
    try:
        assert first.get_or_create_scenario("agent") is not None
    finally:
        first.close()
    second = serve(("weights", "harness"))
    try:
        with pytest.raises(ReefError, match=r"does not serve \['config'\]"):
            second.get_or_create_scenario("agent")
    finally:
        second.close()


class _SlowDispatchedBackend(_DispatchedBackend):
    """A dispatched job whose execution takes a moment, as a real job takes minutes."""

    def evaluate(self, candidate):
        time.sleep(0.05)
        return super().evaluate(candidate)


@pytest.mark.unit
def test_a_weights_job_commits_while_the_harness_backend_is_away(tmp_path: Path) -> None:
    """On the training thread: the harness cycle keeps failing, the weights job still lands and the failure shows."""
    training = StubTrainingRuntime()
    backends = {
        WEIGHTS: _SlowDispatchedBackend(WEIGHTS, tmp_path / "candidates", "job-1"),
        HARNESS: _AwayBackend(HARNESS, tmp_path / "candidates"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends, training=training)
    away = backends[HARNESS]
    assert isinstance(away, _AwayBackend)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            dispatcher.accept_record(record)
        deadline = time.monotonic() + 15
        history: list[tuple[str | None, int]] = []
        while time.monotonic() < deadline:
            current = dispatcher.get_or_create_scenario("agent")
            assert current is not None
            history = [(row.component, row.step) for row in current.store.history()]
            if (WEIGHTS, 1) in history:
                break
            time.sleep(0.05)
        assert (WEIGHTS, 1) in history
        time.sleep(0.5)
        assert 1 <= away.attempts <= 3
        assert dispatcher.build_training_status()["error"] == "agent: RuntimeError: proposer away"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_second_report_on_a_trained_inference_is_settled_after_a_restart(tmp_path: Path) -> None:
    """The harness trained on i1 through r1a while the weights trainer still holds i1; after a restart r1b is not resolved against it."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        scenario.records.append(
            AgentRecord.create(
                scenario="agent",
                request_type=RequestType.REPORT,
                payload={"score": 0.5, "references": ["i1"]},
                agent_record_id="r1b",
                references=("i1",),
            )
        )
        result = scenario.prepare_training_step(HARNESS)
        assert result is not None
        scenario.commit(result, component=HARNESS)
        assert scenario.records.count("agent") == 3
        assert scenario.prepare_training_step(HARNESS) is None
        rebuilt = dispatcher._registry.reload("agent")
        assert rebuilt.prepare_training_step(HARNESS) is None
        assert "r1b" in rebuilt.trainer_for(HARNESS).releasable_agent_record_ids()
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_report_settled_between_two_commits_stays_settled_after_a_restart(tmp_path: Path) -> None:
    """r1b arrived after commit A trained i1 and before commit B: the replay settles it at A's watermark, as it was."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        scenario.records.append(_report("r1b", "i1"))
        assert scenario.prepare_training_step(HARNESS) is None
        for record in _records(2):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        rebuilt = dispatcher._registry.reload("agent")
        assert rebuilt.prepare_training_step(HARNESS) is None
        assert "r1b" in rebuilt.trainer_for(HARNESS).releasable_agent_record_ids()
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_report_on_an_inference_the_sibling_commit_retired_is_released_not_raised(tmp_path: Path) -> None:
    """The weights job holds i1 while the harness commits and r1b arrives; the job's commit retires i1 with r1b
    still stored, unread by the weights trainer. Every later read releases r1b, on the live path and after a restart.
    """
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        scenario.records.append(_report("r1b", "i1"))
        assert scenario.prepare_training_step(HARNESS) is None
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
        assert scenario.records.get("agent", "i1") is None
        assert scenario.records.get("agent", "r1b") is not None
        assert scenario.reserve_training_batch(WEIGHTS) is None
        assert "r1b" in scenario.trainer_for(WEIGHTS).releasable_agent_record_ids()
        for record in _records(2):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        assert scenario.records.get("agent", "r1b") is None
        rebuilt = dispatcher._registry.reload("agent")
        assert rebuilt.prepare_training_step(HARNESS) is None
        weights = rebuilt.reserve_training_batch(WEIGHTS)
        assert weights is not None
        assert [item.source_agent_record_ids for item in weights.items] == [("i2", "r2")]
    finally:
        dispatcher.close()


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

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            processor = _SlottedProcessor if component == HARNESS else ThresholdProcessor
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


def _slotted_report(record_id: str, reference: str, group: str, slot: int) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [reference], "metadata": {"group": group, "slot": slot}},
        agent_record_id=record_id,
        references=(reference,),
    )


@pytest.mark.unit
def test_a_retry_settled_for_a_taken_slot_stays_settled_after_a_restart(tmp_path: Path) -> None:
    """r1dup retried slot 0 while r1 held it, so the harness settled it; the harness commit consumed r1, which both
    trainers had trained, and retired it. The weights trainer has not read r1dup, so it stays stored. The replay no
    longer sees r1 hold the slot: the commit record names r1dup as settled, and the rebuilt harness keeps it out."""
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    records = tmp_path / "records"
    dispatcher = Dispatcher(
        _SlottedRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for step in (1, 2, 3):
            scenario.records.append(_records(step)[0])
        scenario.records.append(_slotted_report("r1", "i1", "G", 0))
        scenario.records.append(_slotted_report("r1dup", "i3", "G", 0))
        scenario.records.append(_slotted_report("r2", "i2", "G", 1))
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        assert [item.source_agent_record_ids for item in scenario.trainer_for(WEIGHTS).pending_batch.items] == [
            ("i1", "r1")
        ]
        scenario.commit(weights, component=WEIGHTS)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        assert [item.source_agent_record_ids for item in scenario.trainer_for(HARNESS).pending_batch.items] == [
            ("i1", "r1"),
            ("i2", "r2"),
        ]
        scenario.commit(harness, component=HARNESS)
        assert scenario.records.get("agent", "r1") is None
        assert scenario.records.get("agent", "r1dup") is not None
        rebuilt = dispatcher._registry.reload("agent")
        rebuilt.records.append(_records(4)[0])
        rebuilt.records.append(_slotted_report("r4", "i4", "G", 2))
        # Live, slot 0 was never r1dup's: G holds r4 alone and waits.
        assert rebuilt.prepare_training_step(HARNESS) is None
        assert "r1dup" in rebuilt.trainer_for(HARNESS).releasable_agent_record_ids()
        record = rebuilt.last_commit_for(HARNESS)
        assert record is not None and "r1dup" in record.settled_ids
    finally:
        dispatcher.close()


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


@pytest.mark.unit
def test_a_batch_the_weights_backend_dropped_stays_consumed_after_a_restart(tmp_path: Path) -> None:
    """A drop consumes its batch without a commit; the harness still holds its rows, so they stay stored. The drop's
    receipt names what it consumed and how far the trainer read, so neither a restart right after it nor one after a
    later commit reserves the dropped batch again."""
    weights = _DroppingBackend(WEIGHTS, tmp_path / "candidates", "job-0")
    dispatcher, _ = _dispatcher(
        tmp_path, backends={WEIGHTS: weights, HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates")}
    )

    def turn(scenario: Scenario) -> list[tuple[str, ...]]:
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        sources = [item.source_agent_record_ids for item in batch.items]
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, weights, batch) is True
        return sources

    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for step in (1, 2, 3):
            for record in _records(step):
                scenario.records.append(record)
        assert turn(scenario) == [("i1", "r1")]
        weights.drop_next = True
        assert turn(scenario) == [("i2", "r2")]
        assert scenario.records.get("agent", "r2") is not None
        rebuilt = dispatcher._registry.reload("agent")
        assert turn(rebuilt) == [("i3", "r3")]
        for record in _records(4):
            rebuilt.records.append(record)
        rebuilt = dispatcher._registry.reload("agent")
        assert turn(rebuilt) == [("i4", "r4")]
        (receipt,) = [
            row for row in rebuilt.records.compaction_receipts("agent") if row["metadata"]["outcome"] == "stale"
        ]
        assert receipt["metadata"]["component"] == WEIGHTS
        assert set(receipt["metadata"]["consumed_ids"]) == {"i2", "r2"}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_two_stale_drops_across_a_reload_are_two_receipts_and_the_rows_behind_them_train(tmp_path: Path) -> None:
    """A rebuilt processor numbers its batches from 1 again, so a drop after a reload carries the batch id of a drop
    before it, over other rows. Each drop's receipt is keyed by what it consumed: the second is a receipt of its
    own, not a conflict with the first, and the weights trainer goes on to the rows behind both drops."""
    weights = _DroppingBackend(WEIGHTS, tmp_path / "candidates", "job-0")
    dispatcher, _ = _dispatcher(
        tmp_path, backends={WEIGHTS: weights, HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates")}
    )

    def turn(scenario: Scenario) -> tuple[str, list[tuple[str, ...]]]:
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        sources = [item.source_agent_record_ids for item in batch.items]
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, weights, batch) is True
        return batch.batch_id, sources

    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for step in (1, 2, 3):
            for record in _records(step):
                scenario.records.append(record)
        weights.drop_next = True
        first_id, first = turn(scenario)
        assert first == [("i1", "r1")]
        rebuilt = dispatcher._registry.reload("agent")
        weights.drop_next = True
        second_id, second = turn(rebuilt)
        assert (second_id, second) == (first_id, [("i2", "r2")])
        rebuilt = dispatcher._registry.reload("agent")
        assert turn(rebuilt)[1] == [("i3", "r3")]
        stale = [row for row in rebuilt.records.compaction_receipts("agent") if row["metadata"]["outcome"] == "stale"]
        assert sorted(sorted(row["metadata"]["consumed_ids"]) for row in stale) == [["i1", "r1"], ["i2", "r2"]]
        assert len({row["receipt_id"] for row in stale}) == 2
        assert all(str(row["receipt_id"]).startswith(f"{first_id}:") for row in stale)
    finally:
        dispatcher.close()


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
    tmp_path: Path, colocated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness acknowledges its batch before its record is durable. When that commit fails while a weights job
    is out, the reload waits for the job, and the job's commit must not retire the harness's rows on the strength of
    an acknowledgment no record carries: after the job, the rebuilt harness trains them. A colocated job holds every
    cycle lock, so the harness cannot retry before the job commits."""
    backends = _dispatched_pair(tmp_path, "job-1")
    if colocated:
        backends[WEIGHTS] = _ColocatedBackend(WEIGHTS, tmp_path / "candidates", "job-1")
    dispatcher, _ = _dispatcher(tmp_path, backends=backends)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        _fail_first_record_of(monkeypatch, scenario, HARNESS)
        with pytest.raises(OSError, match="transient store write error"):
            dispatcher._process_local_backend_step("agent", HARNESS)
        # The acknowledgment is in the harness processor's memory only: siblings do not read it.
        assert not {"i1", "r1"} & scenario.trainer_for(HARNESS).releasable_agent_record_ids()
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
        assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
        for _ in range(3):
            dispatcher._process_local_backend_step("agent", HARNESS)
        current = dispatcher._registry.get_optional("agent")
        assert current is not None
        history = current.store.history()
        assert [sorted(record.consumed_ids) for record in history if record.component == HARNESS] == [["i1", "r1"]]
        assert [sorted(record.compacted_ids) for record in history if record.component == WEIGHTS] == [[]]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_weights_commit_that_fails_leaves_its_batch_to_the_weights_trainer_after_a_harness_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse: a weights commit that fails after its acknowledgment, then a harness commit before the weights
    trainer's reload. The harness commit retires only what the weights trainer's durable state releases, so the
    rebuilt weights trainer trains the batch its failed commit had acknowledged."""
    dispatcher, _ = _dispatcher(tmp_path)

    def step(scenario: Scenario, component: str) -> list[tuple[str, ...]] | None:
        result = scenario.prepare_training_step(component)
        if result is None:
            return None
        batch = scenario.trainer_for(component).pending_batch
        assert batch is not None
        sources = [item.source_agent_record_ids for item in batch.items]
        scenario.commit(result, component=component)
        return sources

    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        assert step(scenario, HARNESS) == [("i1", "r1")]
        for record in _records(2):
            scenario.records.append(record)
        _fail_first_record_of(monkeypatch, scenario, WEIGHTS)
        with pytest.raises(OSError, match="transient store write error"):
            step(scenario, WEIGHTS)
        assert not {"i1", "r1"} & scenario.trainer_for(WEIGHTS).releasable_agent_record_ids()
        assert step(scenario, HARNESS) == [("i2", "r2")]
        assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
        rebuilt = dispatcher._registry.reload("agent")
        assert step(rebuilt, WEIGHTS) == [("i1", "r1")]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_stale_drop_under_a_failed_harness_commit_retires_none_of_the_harness_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drop retires only rows every other trainer releases, like a commit: a harness batch acknowledged by a commit
    that failed is not released yet, so the weights drop leaves it stored and the harness trains it."""
    weights = _DroppingBackend(WEIGHTS, tmp_path / "candidates", "job-0")
    dispatcher, _ = _dispatcher(
        tmp_path, backends={WEIGHTS: weights, HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates")}
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        _fail_first_record_of(monkeypatch, scenario, HARNESS)
        with pytest.raises(OSError, match="transient store write error"):
            dispatcher._process_local_backend_step("agent", HARNESS)
        weights.drop_next = True
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, weights, batch) is True
        assert scenario.records.get("agent", "i1") is not None and scenario.records.get("agent", "r1") is not None
        (receipt,) = [
            row for row in scenario.records.compaction_receipts("agent") if row["metadata"]["outcome"] == "stale"
        ]
        assert set(receipt["metadata"]["consumed_ids"]) == {"i1", "r1"} and receipt["compacted_ids"] == ()
        for _ in range(3):
            dispatcher._process_local_backend_step("agent", HARNESS)
        current = dispatcher._registry.get_optional("agent")
        assert current is not None
        assert [sorted(record.consumed_ids) for record in current.store.history() if record.component == HARNESS] == [
            ["i1", "r1"]
        ]
    finally:
        dispatcher.close()


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

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            if component == WEIGHTS:
                return lambda context: _RetiringRetriesProcessor(context.with_config({"batch_size": 1}))
            return lambda context: ThresholdProcessor(context.with_config({"batch_size": 1}))

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


def _turn(record_id: str, turn: str, completes: str | None = None) -> AgentRecord:
    payload: dict[str, Any] = {"turn": turn, "tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    if completes:
        payload["completes"] = completes
    return AgentRecord.create(
        scenario="agent", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


@pytest.mark.unit
@pytest.mark.parametrize("reload", [False, True])
def test_a_retry_a_computed_processor_retired_never_trains_after_a_reload(tmp_path: Path, reload: bool) -> None:
    """A retries turn t1 after A trained; the weights processor retires the retry, and its next commit names the
    retry as settled because the harness still holds it. Settled means released without training: a reload keeps it
    released, so a later turn that completes the retry finds nothing to judge, as on the live path."""
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    dispatcher = Dispatcher(
        _ComputedWeightsRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )

    def step(scenario: Scenario) -> list[tuple[str, ...]] | None:
        result = scenario.prepare_training_step(WEIGHTS)
        if result is None:
            return None
        batch = scenario.trainer_for(WEIGHTS).pending_batch
        assert batch is not None
        sources = [tuple(item.source_agent_record_ids) for item in batch.items]
        scenario.commit(result, component=WEIGHTS)
        return sources

    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        scenario.records.append(_turn("A", "t1"))
        scenario.records.append(_turn("B", "t2", completes="A"))
        assert step(scenario) == [("A",)]
        scenario.records.append(_turn("A-retry", "t1"))
        scenario.records.append(_turn("C", "t3", completes="B"))
        assert step(scenario) == [("B",)]
        assert "A-retry" in scenario.store.history()[-1].settled_ids
        if reload:
            scenario = dispatcher._registry.reload("agent")
        scenario.records.append(_turn("E", "t4", completes="A-retry"))
        assert step(scenario) is None
    finally:
        dispatcher.close()


def _run_report(record_id: str, references: tuple[str, ...]) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload={"score": 0.5, "references": list(references)},
        agent_record_id=record_id,
        references=references,
    )


@pytest.mark.unit
def test_a_run_report_settled_for_a_retired_inference_releases_its_other_inference(tmp_path: Path) -> None:
    """run references i1 and i2 and is stored before the weights job retires i1. Each trainer then settles run for
    the retired row, and the ownership rule releases i2 with it, as when run is read before the retirement."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in (*_records(1), _records(2)[0]):
            scenario.records.append(record)
        batch = scenario.reserve_training_batch(WEIGHTS)
        assert batch is not None
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        scenario.records.append(_run_report("run", ("i1", "i2")))
        assert dispatcher._run_dispatched_turn(scenario, WEIGHTS, backends[WEIGHTS], batch) is True
        assert scenario.records.get("agent", "i1") is None
        assert scenario.prepare_training_step(HARNESS) is None
        assert scenario.reserve_training_batch(WEIGHTS) is None
        for component in (WEIGHTS, HARNESS):
            decision = scenario.trainer_for(component).processor.retention_decision()
            assert "i2" not in decision.protected_agent_record_ids, component
            assert {"i2", "run"} <= scenario.trainer_for(component).releasable_agent_record_ids(), component
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_rollback_to_the_creation_after_a_rejected_first_step(tmp_path: Path) -> None:
    """A rejected step records the creation without a checkpoint; the creation still has its own bytes."""
    dispatcher, backends = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        creation = scenario.current_artifact_ref().release_id
        backends[HARNESS].reject_next = True
        for step in (1, 2):
            for record in _records(step):
                scenario.records.append(record)
            result = scenario.prepare_training_step(HARNESS)
            assert result is not None
            scenario.commit(result, component=HARNESS)
        assert scenario.current_artifact_ref().release_id != creation
        assert next(row for row in scenario.releases() if row["operation"] == "creation")["restorable"] is True
        scenario.rollback(creation)
        assert _component_files(scenario, scenario.current_artifact_ref())[HARNESS] == "harness seed"
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_job_whose_scenario_was_reloaded_under_it_is_not_committed_on_the_new_instance(tmp_path: Path) -> None:
    """The rebuilt trainer reserves the same rows and the backend replays the job under its marker."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        old = dispatcher.get_or_create_scenario("agent")
        assert old is not None
        for record in _records(1):
            old.records.append(record)
        batch = old.reserve_training_batch(WEIGHTS)
        assert batch is not None
        backend = backends[WEIGHTS]
        new = dispatcher._registry.reload("agent")
        assert new is not old
        assert dispatcher._run_dispatched_turn(old, WEIGHTS, backend, batch) is True
        assert [row["component"] for row in new.releases() if row["operation"] == "training"] == []
        assert new.reserve_training_batch(WEIGHTS) is not None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_harness_only_rollback_keeps_the_proof_that_the_weights_job_was_committed(tmp_path: Path) -> None:
    """The backend must still finish a job whose weights a later rollback carried forward unchanged."""
    dispatcher, _ = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path, "job-1"))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        assert scenario.reserve_training_batch(WEIGHTS) is not None
        execution = scenario.execute_reserved_training_step(WEIGHTS)
        assert execution.outcome == "commit" and execution.result is not None
        scenario.commit(execution.result, component=WEIGHTS)
        assert scenario.committed_training_job_id == "job-1"
        for step in (2, 3):
            for record in _records(step):
                scenario.records.append(record)
            harness = scenario.prepare_training_step(HARNESS)
            assert harness is not None
            scenario.commit(harness, component=HARNESS)
        first_harness = scenario.releases()[1]["release_id"]
        assert scenario.committed_training_job_id == "job-1"
        scenario.rollback(first_harness)
        # Only the harness changed: the weights job's commit still stands.
        assert scenario.committed_training_job_id == "job-1"
        assert not scenario.committed_training_without_job_id
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_training_mode_switches_every_trainer_or_none(tmp_path: Path) -> None:
    """A mode one component cannot run is refused before any component switches."""
    dispatcher, _ = _dispatcher(tmp_path, hybrid_components=frozenset({WEIGHTS}))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        with pytest.raises(NotImplementedError, match=r"harness \(ThresholdProcessor\)"):
            scenario.set_training_mode("hybrid")
        assert [bound.trainer.training_mode for bound in scenario.component_trainers] == ["auto", "auto"]
        with pytest.raises(ValueError, match="training_mode must be"):
            scenario.set_training_mode("bogus")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_rejected_batch_retires_rows_from_every_trainer(tmp_path: Path) -> None:
    """Rows a rejection retires leave the other trainers' memory too, as after a commit."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        assert scenario.trainer_for(HARNESS).releasable_agent_record_ids() == frozenset({"i1", "r1"})
        assert scenario.prepare_training_step(WEIGHTS) is not None
        scenario.reject_pending(component=WEIGHTS)
        assert scenario.records.count("agent") == 0
        assert scenario.trainer_for(HARNESS).releasable_agent_record_ids() == frozenset()
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_rows_a_trainer_never_ingests_are_released_by_it() -> None:
    """A row outside the processor's request types is nobody's to protect, so the trainer releases it at once."""
    store = SQLiteRecordStore()
    trainer = Trainer.build("agent", store, processor_factory=_InferenceOnlyProcessor)
    for record in _records(1):
        store.append(record)
    assert trainer.run_once(0) is None
    assert "r1" in trainer.releasable_agent_record_ids()
    assert "i1" not in trainer.releasable_agent_record_ids()
    trainer.compaction_applied(frozenset({"r1"}))
    assert "r1" not in trainer.releasable_agent_record_ids()


class _ReportOnlyProcessor(DataProcessor):
    """A weights processor that trains on reports alone: the inference rows beside them are nobody's to protect."""

    required_request_types = frozenset({RequestType.REPORT})

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self.items: list[str] = []
        self.done: set[str] = set()
        self.batches = 0

    def ingest(self, item: AgentRecord) -> None:
        self.items.append(item.agent_record_id)

    def ready(self) -> bool:
        return bool(self.items)

    def build_batch(self) -> TrainingBatch:
        self.batches += 1
        return TrainingBatch(f"b{self.batches}", ())

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        consumed = frozenset(self.items)
        self.items.clear()
        self.done |= consumed
        return consumed

    def retention_decision(self) -> RetentionDecision:
        return RetentionDecision(
            protected_agent_record_ids=frozenset(self.items), releasable_agent_record_ids=frozenset(self.done)
        )

    def compaction_applied(self, compacted_ids: frozenset[str]) -> None:
        self.done -= compacted_ids


@dataclass(frozen=True)
class _ReportOnlyWeightsRecipe(_TwoTrainerRecipe):
    """Weights train on reports alone; the harness holds every inference it reads, its batch never full."""

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            if component == HARNESS:
                return lambda context: ThresholdProcessor(context.with_config({"batch_size": 10**6}))
            return _ReportOnlyProcessor

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


@pytest.mark.unit
def test_a_settled_row_of_a_type_the_processor_does_not_take_is_named_once_across_reloads(tmp_path: Path) -> None:
    """The weights trainer releases each inference beside its report at once, and its record names the row once
    while the harness keeps it stored; a reload reads that record back, so the next record does not name every such
    row again."""
    backends: dict[str, CandidateBackend] = {
        component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)
    }
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    dispatcher = Dispatcher(
        _ReportOnlyWeightsRecipe(backends=backends),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for step in range(1, 13):
            for record in _records(step):
                scenario.records.append(record)
            assert scenario.prepare_training_step(HARNESS) is None
            result = scenario.prepare_training_step(WEIGHTS)
            assert result is not None
            scenario.commit(result, component=WEIGHTS)
            if step % 4 == 0:
                scenario = dispatcher._registry.reload("agent")
        records = [record for record in scenario.store.history() if record.component == WEIGHTS]
        assert [sorted(record.settled_ids) for record in records] == [[f"i{step}"] for step in range(1, 13)]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_composite_registration_refuses_a_flat_base(tmp_path: Path) -> None:
    """Files at the root of the base belong to no component and would be carried forward by none."""
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "harness.txt").write_text("harness seed", encoding="utf-8")
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        with pytest.raises(ReefError, match=r"keeps \['harness.txt'\] outside its components"):
            dispatcher.get_or_create_scenario("agent")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_composite_registration_refuses_a_file_named_like_a_component(tmp_path: Path) -> None:
    """A component keeps a directory; a file of that name would fail on every component view."""
    initial = tmp_path / "initial"
    (initial / HARNESS).mkdir(parents=True)
    (initial / HARNESS / f"{HARNESS}.txt").write_text("harness seed", encoding="utf-8")
    (initial / WEIGHTS).write_text("not a directory", encoding="utf-8")
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        with pytest.raises(ReefError, match=r"keeps \['weights'\] outside its components"):
            dispatcher.get_or_create_scenario("agent")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_composite_registration_starts_an_unseeded_component_empty(tmp_path: Path) -> None:
    """A fresh repository seeds only the harness; the weights component starts empty and fills at its first step."""
    initial = tmp_path / "initial"
    (initial / HARNESS).mkdir(parents=True)
    (initial / HARNESS / f"{HARNESS}.txt").write_text("harness seed", encoding="utf-8")
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        base = scenario.repository.materialize(scenario.current_artifact_ref())
        assert base.components is not None and base.components.names == (WEIGHTS, HARNESS)
        assert TextFileTree().read_files(base.component(WEIGHTS)) is None
        for record in _records(1):
            scenario.records.append(record)
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        scenario.commit(weights, component=WEIGHTS)
        assert _component_files(scenario, scenario.current_artifact_ref()) == {
            WEIGHTS: "weights step 1",
            HARNESS: "harness seed",
        }
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_composite_recovery_refuses_a_registration_without_a_component_manifest(tmp_path: Path) -> None:
    """A scenario registered by a recipe serving one component is not read as a composed one."""
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    backend = backend_factory("agent")
    selected = backend.resolve_release(None)
    # Registration as written before releases named their components.
    backend.fork(
        selected.release_id,
        metadata={SCENARIO_METADATA_KEY: scenario_metadata_for(name="agent", base_artifact=selected)},
    )
    dispatcher, _ = _dispatcher(tmp_path, backend_factory=backend_factory)
    try:
        with pytest.raises(ReefError, match="registered without a component manifest"):
            dispatcher.get_or_create_scenario("agent")
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_rows_consumed_before_a_restart_are_still_retired(tmp_path: Path) -> None:
    """A trainer rebuilt after a restart keeps releasing the rows its committed steps consumed."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        assert scenario.store.history()[-1].compacted_ids == frozenset()

        # Restart between the two trainers' commits of the same rows.
        reloaded = dispatcher._registry.reload("agent")
        weights = reloaded.prepare_training_step(WEIGHTS)
        assert weights is not None
        reloaded.commit(weights, component=WEIGHTS)
        assert reloaded.store.history()[-1].compacted_ids == frozenset({"i1", "r1"})
        assert reloaded.records.count("agent") == 0
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_dispatched_result_overtaken_by_another_trainer_is_merged(tmp_path: Path) -> None:
    """The remote job has published its weights; the step lands on the release served now instead of being refused."""
    dispatcher, backends = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        base = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
        assert scenario.reserve_training_batch(WEIGHTS) is not None
        execution = scenario.execute_reserved_training_step(WEIGHTS)
        assert execution.outcome == "commit" and execution.result is not None

        # The harness worker commits while the weights job is out.
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
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
    finally:
        dispatcher.close()


def _both_prepared(tmp_path: Path, harness_policy: str):
    """Both trainers prepared against the base, then the weights step moves the head."""
    backends = {
        WEIGHTS: _ComponentBackend(WEIGHTS, tmp_path / "candidates"),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates", stale_policy=harness_policy),
    }
    dispatcher, backends = _dispatcher(tmp_path, backends=backends)
    scenario = dispatcher.get_or_create_scenario("agent")
    assert scenario is not None
    base = scenario.current_artifact_ref().release_id
    for record in _records(1):
        scenario.records.append(record)
    harness = scenario.prepare_training_step(HARNESS)
    weights = scenario.prepare_training_step(WEIGHTS)
    assert harness is not None and weights is not None
    scenario.commit(weights, component=WEIGHTS)
    return dispatcher, backends, scenario, harness, base


@pytest.mark.unit
def test_a_local_result_its_backend_calls_mergeable_lands_on_the_release_served_now(tmp_path: Path) -> None:
    dispatcher, backends, scenario, harness, base = _both_prepared(tmp_path, "merge")
    try:
        scenario.commit(harness, component=HARNESS)
        assert _component_files(scenario, scenario.current_artifact_ref()) == {
            WEIGHTS: "weights step 1",
            HARNESS: "harness step 1",
        }
        record = scenario.store.history()[-1]
        assert (record.component, record.base_release_id) == (HARNESS, base)
        assert backends[HARNESS].prepared == 1 and backends[HARNESS].evaluated == 1
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_local_result_its_backend_reevaluates_keeps_its_candidate(tmp_path: Path) -> None:
    """The proposer is not asked again; the kept candidate is evaluated against the release served now."""
    dispatcher, backends, scenario, harness, base = _both_prepared(tmp_path, "reevaluate")
    try:
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
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_weights_step_is_no_new_harness_head(tmp_path: Path) -> None:
    """Clients pull the tree: a step that carried it forward unchanged is not announced or listed as a release."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        harness_release = scenario.current_artifact_ref().release_id
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        scenario.commit(weights, component=WEIGHTS)
        assert scenario.current_artifact_ref().release_id != harness_release
        assert service.harness_head(headers) == harness_release
        manifest = service.harness_manifest(headers)
        assert manifest["release_id"] == harness_release
        assert manifest["files"] == {"harness.txt": "harness step 1"}
        catalog = service.harness_releases(headers)["releases"]
        assert [row.get("component") for row in catalog] == [None, HARNESS]
        assert dispatcher._experiment_context(scenario, WEIGHTS).component == WEIGHTS
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_each_components_step_reaches_the_tracker_under_its_name(tmp_path: Path) -> None:
    """The event the dispatcher records for a step names the trainer that made it."""
    tracker = _RecordingTracker()
    dispatcher, _ = _dispatcher(tmp_path, experiment_tracker=tracker)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
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
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_the_harness_catalog_is_one_lineage_of_trees(tmp_path: Path) -> None:
    """A rejected step, an unlisted weights step and a promote at step 1 never move the head or break the chain."""
    dispatcher, backends = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        creation = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        requires = {"training_request": {"id": "r1", "requires": [{"name": "TOKEN", "kind": "env"}]}}
        scenario.commit(scenario.trainer_for(HARNESS).add_commit_metrics(harness, requires), component=HARNESS)
        h1 = scenario.current_artifact_ref().release_id
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        scenario.commit(weights, component=WEIGHTS)
        w1 = scenario.current_artifact_ref().release_id

        # A rejected harness step is listed for its page, but it published nothing and names no head.
        for record in _records(2):
            scenario.records.append(record)
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
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        scenario.commit(weights, component=WEIGHTS)
        w2 = scenario.current_artifact_ref().release_id
        for record in _records(3):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
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
        for record in _records(4):
            scenario.records.append(record)
        backends[WEIGHTS].hold_next = True
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None and weights.pending
        scenario.commit(weights, component=WEIGHTS)
        held = next(row["release_id"] for row in scenario.releases() if row.get("pending"))
        for record in _records(5):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        scenario.rollback(held)
        catalog = service.harness_releases(headers)["releases"]
        assert held not in {row["release_id"] for row in catalog}
        assert catalog[-1]["operation"] == "rollback"
        assert catalog[-1]["rollback_target_release_id"] == restored
        assert catalog[-1]["composed_rollback_target_release_id"] == held
        assert required_by(catalog, catalog[-1]["release_id"]) == [{"name": "TOKEN", "kind": "env"}]
        script = service.harness_install_script(headers, adapter="pi")
        assert w1 not in script and w2 not in script and held not in script
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_merged_result_names_the_release_it_landed_on_in_its_event(tmp_path: Path) -> None:
    """The tracker event carries what the commit record carries, merged_onto included."""
    tracker = _RecordingTracker()
    backends = {
        WEIGHTS: _ComponentBackend(WEIGHTS, tmp_path / "candidates"),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates", stale_policy="merge"),
    }
    dispatcher, _ = _dispatcher(tmp_path, backends=backends, experiment_tracker=tracker)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for record in _records(1):
            scenario.records.append(record)
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
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_creation_manifest_read_that_failed_is_asked_again(tmp_path: Path, monkeypatch: Any) -> None:
    """One failed read of the creation artifact must not refuse every later promote for the process lifetime."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
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
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_promote_of_weights_held_at_step_one_is_no_harness_release(tmp_path: Path) -> None:
    """The creation artifact has no record; its manifest still says the promoted tree did not change."""
    dispatcher, backends = _dispatcher(tmp_path)
    backends[WEIGHTS].hold_next = True
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        creation = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None and weights.pending
        scenario.commit(weights, component=WEIGHTS)
        held = next(row["release_id"] for row in scenario.releases() if row.get("pending"))
        scenario.rollback(held, operation="promote")
        assert _component_files(scenario, scenario.current_artifact_ref())[WEIGHTS] == "weights step 1"
        assert service.harness_head(headers) == creation
        assert [row["release_id"] for row in service.harness_releases(headers)["releases"]] == [creation]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_rollback_under_the_same_tree_is_no_new_harness_head(tmp_path: Path) -> None:
    """A rollback or promote that changed only the weights is not announced or listed as a harness release."""
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        base = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        harness_release = scenario.current_artifact_ref().release_id
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        scenario.commit(weights, component=WEIGHTS)
        first_weights = scenario.current_artifact_ref().release_id
        for record in _records(2):
            scenario.records.append(record)
        weights = scenario.prepare_training_step(WEIGHTS)
        assert weights is not None
        scenario.commit(weights, component=WEIGHTS)

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
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_committed_job_id_outlives_another_trainers_commit(tmp_path: Path) -> None:
    """The backend must finish the weights job even after the harness moved the scenario step."""
    dispatcher, _ = _dispatcher(tmp_path, backends=_dispatched_pair(tmp_path))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        assert scenario.dispatched_component == WEIGHTS
        base = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
        assert scenario.reserve_training_batch(WEIGHTS) is not None
        execution = scenario.execute_reserved_training_step(WEIGHTS)
        assert execution.outcome == "commit" and execution.result is not None
        scenario.commit(execution.result, component=WEIGHTS)
        assert scenario.committed_training_job_id == "job-1"

        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        assert scenario.scenario_step == 2
        assert scenario.committed_training_job_id == "job-1"
        assert scenario.committed_training_without_job_id is False

        # A rollback after the weights commit is the one thing that unmakes the proof.
        scenario.rollback(base)
        assert scenario.committed_training_job_id is None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_adopted_checkpoint_is_attributed_to_the_trainer_that_made_it(tmp_path: Path) -> None:
    """A checkpoint read back from the artifact head names its component, so a lost commit log resets no trainer."""
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher, _ = _dispatcher(tmp_path, records_dir=tmp_path / "records-1", backend_factory=backend_factory)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        base = scenario.current_artifact_ref().release_id
        for record in _records(1):
            scenario.records.append(record)
        harness = scenario.prepare_training_step(HARNESS)
        assert harness is not None
        scenario.commit(harness, component=HARNESS)
        head = scenario.repository.materialize(scenario.current_artifact_ref())
        checkpoint = head.metadata[SCENARIO_METADATA_KEY]
        assert isinstance(checkpoint, Mapping)
        assert checkpoint["component"] == HARNESS
        assert checkpoint["base_release_id"] == base
        _, _, adopted = parse_scenario_metadata(checkpoint, checkpoint_head=head.ref)
        assert adopted is not None and adopted.component == HARNESS and adopted.base_release_id == base
    finally:
        dispatcher.close()

    # The record store is gone; the checkpoint is adopted from the head and belongs to the harness trainer.
    recovered, _ = _dispatcher(tmp_path, records_dir=tmp_path / "records-2", backend_factory=backend_factory)
    try:
        scenario = recovered.get_or_create_scenario("agent")
        assert scenario is not None
        assert scenario.scenario_step == 1
        assert scenario.trainer_for(HARNESS).state == {"steps": 1}
        assert scenario.trainer_for(WEIGHTS).state == {"steps": 0}
        last = scenario.last_commit_for(HARNESS)
        assert last is not None and last.step == 1 and last.component == HARNESS
        assert scenario.last_commit_for(WEIGHTS) is None
    finally:
        recovered.close()


@pytest.mark.unit
def test_dispatcher_runs_one_worker_per_component(tmp_path: Path) -> None:
    dispatcher, _ = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        for step in (1, 2):
            for record in _records(step):
                dispatcher.accept_record(record)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = dispatcher.get_or_create_scenario("agent")
            assert current is not None
            committed = {record.component for record in current.store.history() if record.step <= 4}
            if current.scenario_step >= 4 and committed == {WEIGHTS, HARNESS}:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("both component workers did not commit twice")
        current = dispatcher.get_or_create_scenario("agent")
        assert current is not None
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
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_stale_refusals_are_counted_and_bounded(tmp_path: Path) -> None:
    """A local result overtaken again and again is reported and parked, not prepared and discarded forever."""
    dispatcher, backends = _dispatcher(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        outcomes = []
        for step in (1, 2, 3):
            for record in _records(step):
                scenario.records.append(record)
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
    finally:
        dispatcher.close()


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
