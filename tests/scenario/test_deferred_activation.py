"""Control commits artifact selections while inference prepares their serving routes."""

from contextlib import closing
from pathlib import Path

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.component_trainers import _records
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent
from tests.support.samples_service import make_samples

from reef.artifact import InMemoryRepositoryBackend
from reef.core.served_models import ScenarioModel, ServedArtifact
from reef.core.training_request import normalize_training_request_payload
from reef.observability import NullExperimentTracker
from reef.scenario.factory import ScenarioFactory
from reef.storage.job_data import read_job_data
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.training_requests import TrainingRequestQueue


@pytest.fixture
def deferred_factory(tmp_path: Path):
    runtime = StubInferenceRuntime()
    training = StubTrainingComponent(tmp_path / "checkpoints")
    with (
        closing(SQLiteScenarioStorage(tmp_path / "records")) as storage,
        closing(memory_coordination()) as coordination,
    ):
        factory = ScenarioFactory(
            TestPolicyRecipe(training, runtime=runtime),
            InMemoryRepositoryBackend.factory(directory=tmp_path / "releases"),
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
            served_models=coordination.served_models,
            training_requests=TrainingRequestQueue(coordination.work_queue),
        )
        yield factory, runtime, coordination


def test_load_training_activation_and_recovery_do_not_load_an_engine(deferred_factory) -> None:
    factory, runtime, coordination = deferred_factory
    lease = coordination.control_lease.acquire("test-control")
    with closing(factory.load_or_create("math")) as scenario:
        root = scenario.current_snapshot()
        assert runtime.activated == []
        for record in _records(1, scenario.serving_artifact_id(), "math"):
            scenario.records.append(record)
        make_samples(scenario)
        # Submission, checkpoint, and job completion each have their own commit.
        for _ in range(3):
            coordination.served_models.write(
                ScenarioModel("math"),
                ServedArtifact(scenario.serving_artifact_id(), "http://engine", "selected-model"),
                control_lease_number=lease.number,
            )
            result = scenario.next_commit()
            assert result is not None
            scenario.commit(result)
        assert scenario.current_snapshot() != root
        scenario.activate(scenario.repository.resolve(root).artifact_id)
        assert scenario.current_snapshot() == root
        assert runtime.activated == []

    with closing(factory.load_or_create("math")) as recovered:
        assert recovered.current_snapshot() == root
        assert runtime.activated == []


def test_training_waits_for_the_selected_artifact_then_excludes_old_samples(deferred_factory) -> None:
    factory, runtime, coordination = deferred_factory
    with closing(factory.load_or_create("math")) as scenario:
        selected = scenario.serving_artifact_id()
        assert scenario.served_artifact_id() is None
        lease = coordination.control_lease.acquire("test-control")
        assert lease is not None
        coordination.served_models.write(
            ScenarioModel("math"),
            ServedArtifact("previous-artifact", "http://engine", "previous-model"),
            control_lease_number=lease.number,
        )
        for record in _records(1, "previous-artifact", "math"):
            scenario.records.append(record)
        make_samples(scenario)

        before = dict(scenario.trainer.state)
        assert scenario.next_commit() is None
        assert scenario.trainer.state == before
        assert scenario.served_artifact_id() == "previous-artifact"
        assert scenario.serving_artifact_id() == selected
        assert runtime.activated == []

        coordination.served_models.write(
            ScenarioModel("math"),
            ServedArtifact(selected, "http://engine", "selected-model"),
            control_lease_number=lease.number,
        )
        for record in _records(2, selected, "math"):
            scenario.records.append(record)
        make_samples(scenario)
        result = scenario.next_commit()
        assert result is not None
        scenario.commit(result)
        (job,) = scenario.training_component.submitted
        assert [sample.source_agent_record_ids for sample in read_job_data(job.data_dir).samples] == [
            ("math-i2", "math-r2")
        ]


def test_a_request_trains_while_the_selected_artifact_does_not_serve(deferred_factory) -> None:
    """A request names its samples, so it does not wait for inference, also when no artifact serves."""
    factory, runtime, _ = deferred_factory
    with closing(factory.load_or_create("math")) as scenario:
        scenario.set_training_mode("manual")
        for record in _records(1, scenario.serving_artifact_id(), "math"):
            scenario.records.append(record)
        make_samples(scenario)
        assert scenario.served_artifact_id() is None
        assert scenario.training_requests is not None
        samples = [stored.sample_id for stored in scenario.trainer.pipeline.store.page()]
        payload = {"text": "train", "session": "session-1", "artifact_id": "release-1", "samples": samples}
        scenario.training_requests.file("t1", normalize_training_request_payload(payload))

        result = scenario.next_commit()
        assert result is not None
        scenario.commit(result)
        (job,) = scenario.training_component.submitted
        assert job.request_id == "t1"
        assert runtime.activated == []
