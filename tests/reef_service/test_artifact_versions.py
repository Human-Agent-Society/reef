"""Release listing and rollback tests (issue #78, phase 4)."""

from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.observability import NullExperimentLogger
from reef.runtime import ActivatedModel, ModelCandidate, PreparedTrainingStep, TrainingRuntime
from reef.runtime.inference import InferenceBackend
from reef.scenario import ReleaseNotRestorable
from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.service.app import RequestService, create_app

from ._policy_recipe import TestPolicyRecipe


class RollbackRuntime(TrainingRuntime):
    def __init__(self, checkpoint_dir) -> None:
        super().__init__(base_url="http://trainer")
        self.checkpoint_dir = checkpoint_dir
        self.trained = 0
        self.restored: list[str] = []
        self.candidate_versions: dict[str, str] = {}

    @property
    def inference_backend(self):
        return None

    def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
        del batch, step_preparer
        return PreparedTrainingStep(
            action="train",
            payload={"rollout_id": scenario_step},
            next_algorithm_state={"steps": int(algorithm_state.get("steps", 0)) + 1},
            metrics={},
        )

    def train_candidate(self, payload):
        self.trained += 1
        version = f"w{self.trained}"
        checkpoint = self.checkpoint_dir / str(payload["rollout_id"])
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.txt").write_text(version)
        job_id = f"job-{payload['rollout_id']}"
        self.candidate_versions[job_id] = version
        return ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=str(checkpoint),
            current_runtime_load_id=None,
        )

    def activate_candidate(self, candidate):
        return ActivatedModel(candidate.candidate_id, self.candidate_versions[candidate.candidate_id])

    def reject_candidate(self, candidate, decision):
        del decision
        self.candidate_versions.pop(candidate.candidate_id, None)

    def restore_checkpoint(self, artifact):
        version = artifact.local_path.joinpath("model.txt").read_text()
        self.restored.append(version)
        return f"restored:{version}"


class BlockingBackend(InferenceBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.release_id: str | None = None

    async def inference(self, artifact, path, payload):
        del path, payload
        self.release_id = artifact.ref.release_id
        self.started.set()
        await self.finish.wait()
        return {"choices": [{"message": {"content": "ok"}}], "metadata": {"runtime_load_id": "w1"}}


class RecordingExperimentTracker:
    def __init__(self) -> None:
        self.events = []
        self.rollbacks = []

    def bind_scenario(self, **kwargs):
        del kwargs
        return NullExperimentLogger()

    def correlation_metrics(self, context):
        del context
        return {}

    def record(self, event):
        self.events.append(event)

    def record_rollback(self, event):
        self.rollbacks.append(event)

    def close(self):
        pass


def inference(index: int) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
        agent_record_id=f"i{index}",
    )


def report(index: int) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"i{index}"]},
        agent_record_id=f"r{index}",
        references=(f"i{index}",),
    )


def dispatcher(tmp_path, *, checkpoint_every: int = 1, experiment_tracker=None):
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    (initial / "model.txt").write_text("w0")
    runtime = RollbackRuntime(tmp_path / "exported")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    value = Dispatcher(
        TestPolicyRecipe(
            runtime,
            batch_size=1,
            checkpoint_strategy=EveryNVersions(checkpoint_every),
        ),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "agent-record",
        experiment_tracker=experiment_tracker,
    )
    return value, runtime, backend_factory


def train(dispatcher: Dispatcher, index: int) -> None:
    dispatcher.accept_record(inference(index))
    dispatcher.accept_record(report(index))
    for _ in range(1000):
        if dispatcher.get_or_create_scenario("math").scenario_step >= index:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not commit")


@pytest.mark.unit
def test_versions_are_wal_backed_and_rollback_appends_a_new_commit(tmp_path) -> None:
    value, runtime, _ = dispatcher(tmp_path)
    created = value.get_or_create_scenario("math").releases()[0]["release_id"]
    train(value, 1)
    train(value, 2)
    train(value, 3)

    before = value.list_releases("math")
    assert len(before) == 4
    assert all(version["restorable"] for version in before)
    assert before[-1]["release_id"] == created
    target_version = before[-2]["release_id"]
    last_training_step = value.build_training_status()["scenarios"]["math"]["last_committed_step"]

    published = value.rollback("math", target_version)

    assert published.release_id != target_version
    assert (
        value.get_or_create_scenario("math").repository.resolve(published).local_path.joinpath("model.txt").read_text()
        == "w1"
    )
    assert runtime.restored == ["w1"]
    record = value.get_or_create_scenario("math").commit_log.records()[-1]
    assert record.operation == "rollback"
    assert record.rollback_target_release_id == target_version
    assert record.artifact_ref == published
    status = value.build_training_status()["scenarios"]["math"]
    assert status["scenario_step"] == 4
    assert status["last_committed_step"] == last_training_step


@pytest.mark.unit
def test_rollback_retry_settles_the_recorded_release_without_republishing(tmp_path, monkeypatch) -> None:
    value, _, _ = dispatcher(tmp_path)
    created = value.get_or_create_scenario("math").releases()[0]["release_id"]
    train(value, 1)
    scenario = value.get_or_create_scenario("math")
    original = scenario.trainer.commit_applied
    should_fail = True

    def fail_after_effect(state):
        nonlocal should_fail
        original(state)
        if should_fail:
            should_fail = False
            raise RuntimeError("injected settlement failure")

    monkeypatch.setattr(scenario.trainer, "commit_applied", fail_after_effect)

    with pytest.raises(RuntimeError, match="injected settlement failure"):
        value.rollback("math", created)

    assert scenario.scenario_step == 1
    assert scenario.commit_log is not None
    recorded = scenario.commit_log.records()[-1]
    assert recorded.step == 2
    assert recorded.operation == "rollback"

    published = value.rollback("math", created)

    assert published == recorded.artifact_ref
    assert scenario.scenario_step == 2
    assert scenario.commit_log.records()[-1] == recorded
    assert len(scenario.commit_log.records()) == 2


@pytest.mark.unit
def test_rollback_starts_a_new_experiment_run_segment(tmp_path) -> None:
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    train(value, 1)
    train(value, 2)
    target_version = value.list_releases("math")[-2]["release_id"]

    value.rollback("math", target_version)
    train(value, 4)

    assert [(event.context.run_segment, event.context.run_step) for event in tracker.events] == [
        (0, 0),
        (0, 1),
        (3, 0),
    ]
    assert len(tracker.rollbacks) == 1
    rollback = tracker.rollbacks[0]
    assert rollback.step == 3
    assert rollback.run_segment == 0
    assert rollback.target_release_id == target_version


@pytest.mark.unit
def test_recovery_adopts_a_lost_rollback_record_without_treating_it_as_training(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    train(value, 1)
    train(value, 2)
    target_version = value.list_releases("math")[-2]["release_id"]
    value.rollback("math", target_version)

    scenario = value.get_or_create_scenario("math")
    assert scenario is not None
    path = scenario.commit_log.path
    records = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")

    restarted = Dispatcher(
        TestPolicyRecipe(
            RollbackRuntime(tmp_path / "restarted-export"),
            checkpoint_strategy=EveryNVersions(1),
        ),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        agent_record_dir=tmp_path / "agent-record",
    )
    recovered = restarted.get_or_create_scenario("math")
    assert recovered is not None
    adopted = recovered.commit_log.records()[-1]

    assert adopted.operation == "rollback"
    assert adopted.operation_verified is True
    assert adopted.rollback_target_release_id == target_version
    assert recovered.committed_training_without_job_id is False


@pytest.mark.unit
def test_older_versions_and_version_catalog_survive_restart(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    for index in (1, 2, 3):
        train(value, index)
    scenario = value.get_or_create_scenario("math")

    version_one = scenario.releases()[-2]
    version_one_ref = scenario.repository.backend.resolve_release(version_one["release_id"])
    assert scenario.repository.resolve(version_one_ref).local_path.joinpath("model.txt").read_text() == "w1"

    restarted_runtime = RollbackRuntime(tmp_path / "restarted-export")
    restarted = Dispatcher(
        TestPolicyRecipe(restarted_runtime, checkpoint_strategy=EveryNVersions(1)),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        agent_record_dir=tmp_path / "agent-record",
    )
    recovered = restarted.get_or_create_scenario("math")
    assert [version["operation"] for version in recovered.releases()] == [
        "training",
        "training",
        "training",
        "creation",
    ]


@pytest.mark.unit
def test_live_version_is_listed_but_cannot_be_a_rollback_target(tmp_path) -> None:
    value, _, _ = dispatcher(tmp_path, checkpoint_every=2)
    train(value, 1)
    train(value, 2)

    versions = value.list_releases("math")
    live = next(version for version in versions if not version["checkpoint"])
    assert not live["restorable"]
    with pytest.raises(ReleaseNotRestorable, match="no durable checkpoint bytes"):
        value.rollback("math", live["release_id"])


@pytest.mark.unit
def test_request_records_the_artifact_ref_captured_before_inference(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        train(value, 1)
        selected_version = value.get_or_create_scenario("math").current_artifact_ref().release_id
        backend = BlockingBackend()
        service = RequestService(value)
        request = asyncio.create_task(
            service.infer_with_data(
                {"x-reef-scenario": "math"},
                {"model": "reef", "messages": [{"role": "user", "content": "hi"}]},
                "/v1/chat/completions",
                backend,
            )
        )
        await backend.started.wait()

        train(value, 2)

        backend.finish.set()
        _, item = await request
        assert backend.release_id == selected_version
        assert item.artifact_ref is not None
        assert item.artifact_ref.release_id == selected_version

    asyncio.run(run())


@pytest.mark.unit
def test_version_and_rollback_http_api(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        train(value, 1)
        train(value, 2)
        client = TestClient(TestServer(create_app(value)))
        await client.start_server()
        try:
            listed = await client.get("/reef/scenarios/math/releases")
            assert listed.status == 200
            listed_body = await listed.json()
            assert len(listed_body["releases"]) == 3
            target_version = listed_body["releases"][-2]["release_id"]

            rolled_back = await client.post(
                "/reef/scenarios/math/rollback",
                json={"release_id": target_version},
            )
            assert rolled_back.status == 200
            assert (await rolled_back.json())["release_id"] != target_version

            missing = await client.post(
                "/reef/scenarios/math/rollback",
                json={"release_id": "missing"},
            )
            assert missing.status == 404
        finally:
            await client.close()

    asyncio.run(run())
