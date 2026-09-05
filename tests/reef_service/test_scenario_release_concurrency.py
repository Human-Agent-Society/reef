"""Release readers stay live during evolution without observing partial publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.scenario import Scenario
from reef.train import PreparedStep, Trainer, TrainingBackend, TrainStepResult
from reef.train.evaluation import EvaluationResult, UpdateCandidate

from ._threshold_processor import ThresholdProcessor


class _LocalBackend(TrainingBackend):
    """An ordinary local candidate cycle with controllable preparation/evaluation."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.block_phase: str | None = None
        self.started = Event()
        self.resume = Event()
        self.result: TrainStepResult | None = None

    def initial_state(self):
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        step = scenario_step + 1
        artifact_path = self.artifact_dir / str(step)
        artifact_path.mkdir(parents=True)
        (artifact_path / "model.txt").write_text(f"step {step}", encoding="utf-8")
        self.result = TrainStepResult(
            {"steps": step},
            metrics={"score": float(step)},
            artifact=Artifact.local(artifact_path),
        )
        self._wait_if_blocked("prepare")
        return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})

    def evaluate(self, candidate):
        self._wait_if_blocked("evaluate")
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision):
        assert self.result is not None
        return self.result

    def abort_step(self, prepared):
        pass

    def _wait_if_blocked(self, phase: str) -> None:
        if self.block_phase == phase:
            self.started.set()
            assert self.resume.wait(10), f"test did not release {phase}"


@dataclass(frozen=True)
class _LocalRecipe(Recipe):
    backend: TrainingBackend

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
            training_backend=self.backend,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )


@pytest.fixture
def local_scenario(tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("initial", encoding="utf-8")
    backend = _LocalBackend(tmp_path / "candidates")
    dispatcher = Dispatcher(
        _LocalRecipe(backend=backend),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
    )
    scenario = dispatcher.get_or_create_scenario("math")
    assert scenario is not None
    try:
        yield scenario, backend
    finally:
        backend.resume.set()
        dispatcher.close()


def _queue_batch(scenario: Scenario, step: int) -> None:
    # Append directly so the test, rather than the dispatcher's worker, owns
    # exactly when the real local training cycle starts and commits.
    scenario.records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
            agent_record_id=f"i{step}",
        )
    )
    scenario.records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0, "references": [f"i{step}"]},
            agent_record_id=f"r{step}",
            references=(f"i{step}",),
        )
    )


def _prepare(scenario: Scenario) -> TrainStepResult:
    _queue_batch(scenario, scenario.scenario_step + 1)
    result = scenario.prepare_training_step()
    assert result is not None
    return result


def _model_text(artifact: Artifact) -> str:
    local_path = artifact.materialize().local_path
    assert local_path is not None
    return local_path.joinpath("model.txt").read_text(encoding="utf-8")


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["prepare", "evaluate"])
@pytest.mark.parametrize("operation", ["commit", "rollback"])
def test_release_reads_remain_live_during_local_training(local_scenario, phase, operation) -> None:
    scenario, backend = local_scenario
    initial_ref = scenario.current_artifact_ref()
    scenario.commit(_prepare(scenario))
    current_ref = scenario.current_artifact_ref()
    expected_releases = scenario.releases()
    backend.block_phase = phase
    _queue_batch(scenario, 2)
    writer_started = Event()

    def write():
        writer_started.set()
        if operation == "commit":
            assert backend.result is not None
            return scenario.commit(backend.result)
        return scenario.rollback(initial_ref.release_id)

    with ThreadPoolExecutor(max_workers=7) as executor:
        training = executor.submit(scenario.prepare_training_step)
        try:
            assert backend.started.wait(2), "local backend did not start"
            writer = executor.submit(write)
            assert writer_started.wait(2), "writer did not start"
            # A queued writer must wait on the operation lock without first
            # taking the publication lock and starving the read-side callers.
            catalog = executor.submit(scenario.releases)
            latest = executor.submit(scenario.artifact_snapshot)
            historical = executor.submit(scenario.artifact_snapshot, initial_ref.release_id)
            version = executor.submit(scenario.artifact_for_version, current_ref.release_id)
            old_version = executor.submit(scenario.artifact_for_version, initial_ref.release_id)

            assert catalog.result(timeout=2) == expected_releases
            artifact, metrics = latest.result(timeout=2)
            assert artifact.ref == current_ref
            assert _model_text(artifact) == "step 1"
            assert metrics == {"score": 1.0}
            artifact, metrics = historical.result(timeout=2)
            assert artifact.ref == initial_ref
            assert _model_text(artifact) == "initial"
            assert metrics is None
            assert version.result(timeout=2).ref == current_ref
            assert old_version.result(timeout=2).ref == initial_ref
            with pytest.raises(TimeoutError):
                writer.result(timeout=0.1)
            assert not training.done()
        finally:
            backend.resume.set()

        result = training.result(timeout=2)
        assert result is backend.result
        if operation == "commit":
            assert writer.result(timeout=2) == {"steps": 2}
            assert scenario.scenario_step == 2
            assert scenario.artifact_snapshot()[1] == {"score": 2.0}
        else:
            with pytest.raises(ReefError, match="pending commit"):
                writer.result(timeout=2)
            assert scenario.current_artifact_ref() == current_ref
            assert scenario.scenario_step == 1


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["commit", "rollback"])
@pytest.mark.parametrize("phase", ["journal", "settlement"])
def test_release_reads_wait_for_complete_publication(local_scenario, monkeypatch, operation, phase) -> None:
    scenario, _ = local_scenario
    initial_ref = scenario.current_artifact_ref()
    scenario.commit(_prepare(scenario))
    scenario.commit(_prepare(scenario))
    previous_ref = scenario.current_artifact_ref()
    result = _prepare(scenario) if operation == "commit" else None
    publication_started = Event()
    resume_publication = Event()

    assert scenario.commit_log is not None
    target = scenario.commit_log if phase == "journal" else scenario.trainer
    method = "append" if phase == "journal" else "commit_applied"
    original = target.append if phase == "journal" else target.commit_applied

    def block_publication(value):
        publication_started.set()
        assert resume_publication.wait(10), "test did not release publication"
        return original(value)

    monkeypatch.setattr(target, method, block_publication)
    readers_started = [Event() for _ in range(4)]

    def read_catalog():
        readers_started[0].set()
        return scenario.releases()

    def read_latest():
        readers_started[1].set()
        return scenario.artifact_snapshot()

    def read_historical():
        readers_started[2].set()
        return scenario.artifact_snapshot(previous_ref.release_id)

    def read_version():
        readers_started[3].set()
        return scenario.artifact_for_version(previous_ref.release_id)

    with ThreadPoolExecutor(max_workers=5) as executor:
        writer = (
            executor.submit(scenario.commit, result)
            if operation == "commit"
            else executor.submit(scenario.rollback, initial_ref.release_id)
        )
        try:
            assert publication_started.wait(2), "publication did not start"
            # The repository has moved, but the journal or scenario state has
            # not caught up. This is the precise window readers must exclude.
            assert scenario.current_artifact_ref() != previous_ref
            assert scenario.scenario_step == 2
            readers = [executor.submit(read) for read in (read_catalog, read_latest, read_historical, read_version)]
            for started in readers_started:
                assert started.wait(2), "reader did not start"
            for reader in readers:
                with pytest.raises(TimeoutError):
                    reader.result(timeout=0.1)
        finally:
            resume_publication.set()

        writer.result(timeout=2)
        catalog, latest, historical, version = [reader.result(timeout=2) for reader in readers]

    current_ref = scenario.current_artifact_ref()
    assert scenario.scenario_step == 3
    assert catalog[0]["release_id"] == current_ref.release_id
    assert catalog[0]["current"] is True
    assert catalog[0]["operation"] == ("training" if operation == "commit" else "rollback")
    assert sum(row["current"] for row in catalog) == 1
    artifact, metrics = latest
    assert artifact.ref == current_ref
    assert _model_text(artifact) == ("step 3" if operation == "commit" else "initial")
    assert metrics == ({"score": 3.0} if operation == "commit" else None)
    assert historical[0].ref == previous_ref
    assert historical[1] == {"score": 2.0}
    assert version.ref == previous_ref
