"""One trainer per release component: independent workers that meet at the commit boundary."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reef.artifact import Artifact, ArtifactRef, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.components import RECORDS_COMPONENT
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.scenario import Scenario, StaleTrainingResultError
from reef.scenario.scenario import validate_component_trainers
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

from ._threshold_processor import ThresholdProcessor

WEIGHTS = "weights"
HARNESS = "harness"


class _ComponentBackend(CandidateBackend):
    """A local candidate cycle publishing one file for its component per step."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        self.component = component
        self.artifact_dir = artifact_dir
        self.prepared = 0
        self.result: TrainStepResult | None = None

    def initial_state(self) -> Mapping[str, Any]:
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        step = int(state["steps"]) + 1
        path = self.artifact_dir / self.component / uuid.uuid4().hex
        path.mkdir(parents=True)
        (path / f"{self.component}.txt").write_text(f"{self.component} step {step}", encoding="utf-8")
        self.prepared += 1
        self.result = TrainStepResult(
            {"steps": step}, metrics={"prepared": self.prepared}, artifact=Artifact.local(path)
        )
        return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})

    def evaluate(self, candidate):
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision):
        assert self.result is not None
        return self.result

    def abort_step(self, prepared):
        pass


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


def _dispatcher(
    tmp_path: Path,
    *,
    records_dir: Path | None = None,
    backend_factory: Any = None,
    backends: dict[str, _ComponentBackend] | None = None,
    hybrid_components: frozenset[str] = frozenset(),
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
    dispatcher = Dispatcher(
        _TwoTrainerRecipe(backends=backends, hybrid_components=hybrid_components),
        backend_factory or InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=records,
        scenario_storage=SQLiteScenarioStorage(records),
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


def _dispatched_pair(tmp_path: Path, job_id: str = "job-1") -> dict[str, _ComponentBackend]:
    return {
        WEIGHTS: _DispatchedBackend(WEIGHTS, tmp_path / "candidates", job_id),
        HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates"),
    }


@pytest.mark.unit
def test_training_mode_switches_every_trainer_or_none(tmp_path: Path) -> None:
    """A mode one component cannot run is refused before any component switches."""
    dispatcher, _ = _dispatcher(tmp_path, hybrid_components=frozenset({WEIGHTS}))
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        with pytest.raises(NotImplementedError, match=r"\['harness'\]"):
            scenario.set_training_mode("hybrid")
        assert [bound.trainer.training_mode for bound in scenario.component_trainers] == ["auto", "auto"]
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
