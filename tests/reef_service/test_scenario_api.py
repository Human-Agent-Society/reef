from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from unittest.mock import ANY

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.records import report, token_inference
from tests.support.runtime_stubs import StubFacts, StubInferenceRuntime, StubTrainingComponent
from tests.support.trajectory_uploads import dispatcher_for

from reef.artifact import (
    ArtifactNotFoundError,
    ArtifactPublicationError,
    GitLFSRepositoryBackend,
    GitLFSRepositoryBackendFactory,
    InMemoryRepositoryBackend,
)
from reef.core import ScenarioNotFoundError
from reef.core.errors import ScenarioDeletedError
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.dispatcher.scenario_deletes import DELETES_QUEUE, request_cleanup
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.recipe import Recipe
from reef.record2sample.service import SAMPLES_QUEUE
from reef.scenario.admission import ScenarioLimits
from reef.scenario.live_objects import LiveObjects, RecipeChecks
from reef.service.app import create_app
from reef.storage.sql_coordination import memory_coordination, open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.work_queue import NewWork
from reef.train.job_queue import (
    CHECKS_QUEUE,
    JOBS_QUEUE,
    SCENARIO_DELETED_ERROR,
    TrainingJobWorker,
    archive_item_id,
    job_records_content,
)
from reef.train.pipeline import TrainingPipeline


def _dispatcher(tmp_path, *, allow_implicit_creation: bool = True) -> Dispatcher:
    return Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        allow_implicit_creation=allow_implicit_creation,
        scenario_storage=SQLiteScenarioStorage(None),
        coordination=memory_coordination(),
    )


def test_get_or_create_scenario_returns_same_instance(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    first = dispatcher.get_or_create_scenario("code-repair")
    assert first is not None
    again = dispatcher.get_or_create_scenario("code-repair")
    assert again is first


def test_implicit_creation_off_returns_none_for_unknown_scenarios(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path, allow_implicit_creation=False)
    assert dispatcher.get_or_create_scenario("typo-name") is None


def test_implicit_creation_off_still_resolves_created_scenarios(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path, allow_implicit_creation=False)
    dispatcher.get_or_create_scenario("code-repair", allow_implicit_creation=True)
    assert dispatcher.get_or_create_scenario("code-repair").name == "code-repair"


def test_implicit_creation_on_keeps_current_behavior(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    assert dispatcher.get_or_create_scenario("fresh").name == "fresh"


def test_list_scenarios_shows_loaded_bindings(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("a")
    dispatcher.get_or_create_scenario("b")
    rows = dispatcher.list_scenarios()
    names = [row["scenario"] for row in rows]
    assert names == ["a", "b"]
    assert all(row["loaded"] and row["components"] for row in rows)
    assert all("release_id" not in row and "content_id" not in row for row in rows)
    assert all("recipe" not in row for row in rows)


def test_list_scenarios_empty_when_nothing_exists(tmp_path) -> None:
    assert _dispatcher(tmp_path).list_scenarios() == ()


def test_scenario_contract_returns_processor_and_request_types(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path)
    dispatcher.get_or_create_scenario("math")
    contract = dispatcher.scenario_contract("math")
    assert contract["scenario"] == "math"
    assert "recipe" not in contract
    assert contract["processor"] == "NoJobData"
    assert contract["required_request_types"] == ["inference", "report"]
    assert contract["parameters_schema"] == {}, "a recipe without a training component declares no schema"


def test_scenario_contract_serves_the_training_components_schema_by_sample_kind(tmp_path) -> None:
    class Declaring(StubTrainingComponent):
        def parameters_schema(self, kind):
            return {"title": kind} if kind == "task" else None

    dispatcher = _dispatcher(tmp_path)
    scenario = dispatcher.get_or_create_scenario("math")
    scenario._binding = replace(scenario._binding, training_component=Declaring())
    assert dispatcher.scenario_contract("math")["parameters_schema"] == {"task": {"title": "task"}}


def test_scenario_contract_raises_for_unknown_scenario(tmp_path) -> None:
    with pytest.raises(ScenarioNotFoundError):
        _dispatcher(tmp_path).scenario_contract("nope")


def test_a_deleted_scenario_leaves_the_registry_and_its_id_is_never_created_again(tmp_path) -> None:
    """A deleted scenario leaves the registry and the listing, and its lock slot goes. Its ID is not used again: an
    explicit creation and an implicit one, as an incoming record makes, are refused."""
    value = _dispatcher(tmp_path)
    assert value.get_or_create_scenario("delivery") is not None

    result = value.delete_scenario("delivery")

    assert result["scenario"] == "delivery"
    assert not value.has_scenario("delivery")
    assert "delivery" not in {row["scenario"] for row in value.list_scenarios()}
    assert value.get_or_create_scenario("delivery", allow_implicit_creation=False) is None
    with pytest.raises(ScenarioNotFoundError):
        value.delete_scenario("delivery")
    with pytest.raises(ScenarioDeletedError, match="create a scenario with a new ID"):
        value.get_or_create_scenario("delivery", allow_implicit_creation=True)
    with pytest.raises(ScenarioDeletedError):
        value.create_scenario("delivery")
    assert not value.has_scenario("delivery")
    assert value.create_scenario("delivery-2")[1]


class FailingDiscard(InMemoryRepositoryBackend):
    """An artifact store that does not answer the first discard of all content."""

    failures = 1

    def discard_all_content(self) -> None:
        if type(self).failures:
            type(self).failures -= 1
            raise ArtifactPublicationError("the artifact store did not answer")
        super().discard_all_content()


def cleanup_items(dispatcher: Dispatcher, scenario: str) -> list[tuple[str, object]]:
    """The state and the result of each cleanup item of ``scenario``."""
    return [(item.state, item.result) for item in dispatcher.queue.items(DELETES_QUEUE, scenario=scenario)]


def test_a_delete_refuses_before_it_changes_anything_and_a_failed_cleanup_is_finished_by_the_next_sweep(
    tmp_path, monkeypatch, swept_by_the_test
) -> None:
    factory = FailingDiscard.factory(directory=tmp_path / "artifacts")
    value = Dispatcher(
        Recipe(),
        factory,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    created = value.get_or_create_scenario("delivery")
    assert created is not None
    factory("delivery").store_content("job:records", tmp_path / "artifacts")
    # A backend that cannot archive a registration, such as a remote Git repository, refuses first.
    monkeypatch.setattr(type(factory), "archives_registrations", False)
    with pytest.raises(NotImplementedError, match="cannot archive a scenario"):
        value.delete_scenario("delivery")
    assert value.get_or_create_scenario("delivery", allow_implicit_creation=False) is created
    assert cleanup_items(value, "delivery") == []
    monkeypatch.undo()
    assert not GitLFSRepositoryBackendFactory(
        GitLFSRepositoryBackend,
        "https://example.invalid/artifacts.git",
        work_dir=tmp_path / "w",
        cache_dir=tmp_path / "c",
    ).archives_registrations

    assert value.delete_scenario("delivery")["scenario"] == "delivery"
    with pytest.raises(ScenarioNotFoundError):
        value.delete_scenario("delivery")
    assert [state for state, _ in cleanup_items(value, "delivery")] == ["queued"]
    # The store does not answer the discard: the scenario stays deleted, and its item waits for the next sweep.
    value.sweeper.sweep()
    assert [state for state, _ in cleanup_items(value, "delivery")] == ["failed"]
    assert not value.has_scenario("delivery")
    with pytest.raises(ScenarioDeletedError):
        value.create_scenario("delivery")
    assert factory("delivery").materialize_content("job:records").is_dir()
    # The next sweep finishes the cleanup, and no content stays.
    value.sweeper.sweep()
    ((state, result),) = cleanup_items(value, "delivery")
    assert state == "finished" and result["deleted"] is True
    with pytest.raises(ArtifactNotFoundError):
        factory("delivery").materialize_content("job:records")
    assert not [path for path in (tmp_path / "records").iterdir() if path.is_file()]
    value.close()


@pytest.fixture
def swept_by_the_test(monkeypatch) -> None:
    """Control's sweep runs only when the test calls it: its thread does not start."""
    monkeypatch.setattr(EndedWorkSweeper, "start", lambda self: None)


def durable_dispatcher(tmp_path, factory) -> Dispatcher:
    """A dispatcher over the store of the directory ``tmp_path``, as a process of a deployment builds it."""
    return Dispatcher(
        Recipe(),
        factory,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=open_coordination("sqlite", agent_record_dir=str(tmp_path / "records"), record_database_url=None),
    )


def test_a_delete_that_stopped_after_its_mark_is_finished_by_the_next_sweep_and_the_scenario_is_gone_meanwhile(
    tmp_path, swept_by_the_test
) -> None:
    """A process stops right after the delete's mark: nothing of the cleanup ran, and nothing is in memory. For the
    next process, the scenario does not exist, and its sweep removes the rest."""
    factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts")
    first = durable_dispatcher(tmp_path, factory)
    current = first.get_or_create_scenario("delivery")
    assert current is not None
    root = current.serving_artifact_id()
    factory("delivery").store_content("job:records", tmp_path / "artifacts")
    request_cleanup(first.queue, "delivery")
    first.registry.archive_registration("delivery")
    first.close()

    value = durable_dispatcher(tmp_path, factory)
    assert not value.has_scenario("delivery")
    assert "delivery" not in {row["scenario"] for row in value.list_scenarios()}
    with pytest.raises(ScenarioNotFoundError):
        value.scenario_contract("delivery")
    with pytest.raises(ScenarioNotFoundError):
        value.delete_scenario("delivery")
    with pytest.raises(ScenarioDeletedError):
        value.create_scenario("delivery")
    with pytest.raises(ScenarioDeletedError):
        value.accept_record(token_inference("i1", scenario="delivery", artifact_id=root))
    with pytest.raises(ScenarioDeletedError):
        value.registry.reload("delivery")
    assert [state for state, _ in cleanup_items(value, "delivery")] == ["queued"]

    value.sweeper.sweep()
    ((state, result),) = cleanup_items(value, "delivery")
    assert state == "finished" and result["deleted"] is True
    with pytest.raises(ArtifactNotFoundError):
        factory("delivery").materialize_content("job:records")
    assert not [path for path in (tmp_path / "records").iterdir() if path.suffix == ".sqlite3"]
    assert list((tmp_path / "records" / "archived").iterdir())
    value.close()


def test_a_delete_that_stopped_before_its_mark_leaves_the_scenario_whole_and_a_repeated_delete_finishes(
    tmp_path, swept_by_the_test
) -> None:
    factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts")
    value = durable_dispatcher(tmp_path, factory)
    assert value.get_or_create_scenario("delivery") is not None
    request_cleanup(value.queue, "delivery")
    value.sweeper.sweep()
    assert cleanup_items(value, "delivery") == [("finished", {"deleted": False})]
    assert value.get_or_create_scenario("delivery", allow_implicit_creation=False) is not None
    assert any(path.suffix == ".sqlite3" for path in (tmp_path / "records").iterdir())

    value.delete_scenario("delivery")
    value.sweeper.sweep()
    assert cleanup_items(value, "delivery")[-1] == ("finished", {"deleted": True, "archived": ANY})
    assert not value.has_scenario("delivery")
    value.close()


class StoringOrphanJob(InMemoryRepositoryBackend):
    """The artifact store of a scenario that a job of it stores into while the scenario's content is discarded.

    The job is one that the cancel of the scenario's jobs did not see, because a trainer's cycle added it after the
    look. Right after the discard of all content, its training worker stores the job's record files and checks the
    scenario, as ``TrainingJobWorker.store_job_records`` does."""

    worker: TrainingJobWorker | None = None

    def discard_all_content(self) -> None:
        super().discard_all_content()
        worker = type(self).worker
        if worker is None or "orphan" in [item.id for item in worker.queue.items(JOBS_QUEUE)]:
            return
        worker.queue.append(NewWork(JOBS_QUEUE, "orphan", "delivery", "training_job:stub", {}))
        item = worker.queue.take(JOBS_QUEUE, worker.worker_id, ["training_job:stub"])
        assert item is not None
        files = self.root.parent / "orphan-records"
        files.mkdir(exist_ok=True)
        self.store_content(job_records_content(item.id), files)
        worker.refuses_deleted_scenario(item)


def test_the_records_that_an_orphan_job_stores_while_its_scenario_is_deleted_do_not_stay(
    tmp_path, swept_by_the_test
) -> None:
    """The window of a delete that discards before it marks: a job that the cancel missed stores its records after
    the discard, and the worker's check still finds the scenario. With the mark first, the check finds none."""
    factory = StoringOrphanJob.factory(directory=tmp_path / "artifacts")
    value = durable_dispatcher(tmp_path, factory)
    assert value.get_or_create_scenario("delivery") is not None
    backend = factory("delivery")
    StoringOrphanJob.worker = TrainingJobWorker(
        StubTrainingComponent(),
        value.queue,
        value.job_directory_store,
        factory,
        checks=RecipeChecks(Recipe()),
        worker_id="training-1",
    )
    try:
        value.delete_scenario("delivery")
        value.sweeper.sweep()
    finally:
        StoringOrphanJob.worker = None
    with pytest.raises(ArtifactNotFoundError):
        backend.materialize_content(job_records_content("orphan"))
    assert value.queue.read(JOBS_QUEUE, "orphan").error == SCENARIO_DELETED_ERROR
    value.close()


def test_delete_scenario_refuses_names_that_are_paths() -> None:
    value = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    for name in ("", ".", "..", "a/b"):
        with pytest.raises(ScenarioNotFoundError):
            value.delete_scenario(name)


def test_delete_scenario_archives_its_records_and_commit_log(tmp_path) -> None:
    """With a record directory, the scenario's SQLite store and commit history move under ``archived``; the
    directory keeps nothing of the name at the top level, so a restart does not find it."""
    record_dir = tmp_path / "records"
    value = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(record_dir),
        coordination=memory_coordination(),
    )
    assert value.get_or_create_scenario("delivery") is not None
    before = {path.name for path in record_dir.iterdir() if path.is_file()}
    assert any(name.endswith(".sqlite3") for name in before)

    value.delete_scenario("delivery")
    value.sweeper.sweep()

    assert not [path for path in record_dir.iterdir() if path.is_file()]
    archived = list((record_dir / "archived").iterdir())
    assert len(archived) == 1 and any(path.name.endswith(".sqlite3") for path in archived[0].iterdir())
    ((state, result),) = cleanup_items(value, "delivery")
    assert state == "finished" and all(item.startswith(str(record_dir / "archived")) for item in result["archived"])
    value.close()


@pytest.mark.unit
def test_delete_scenario_over_http() -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(
                create_app(
                    build_default_dispatcher(
                        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
                    )
                )
            )
        )
        await client.start_server()
        try:
            created = await client.post("/reef/scenarios", json={"name": "delivery"})
            assert created.status == 201
            deleted = await client.delete("/reef/scenarios/delivery")
            assert deleted.status == 200
            assert (await deleted.json())["scenario"] == "delivery"
            assert await (await client.get("/reef/scenarios")).json() == {"scenarios": []}
            assert (await client.delete("/reef/scenarios/delivery")).status == 404
            assert (await client.get("/reef/scenarios/delivery/artifacts")).status == 404
            again = await client.post("/reef/scenarios", json={"name": "delivery"})
            assert again.status == 409 and "was deleted" in await again.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_scenario_limits_answer_409_and_503_over_http(tmp_path) -> None:
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        scenario_admission=ScenarioLimits(max_loaded_scenarios=1, max_scenarios=2),
        coordination=memory_coordination(),
    )

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            assert (await client.post("/reef/scenarios", json={"name": "a"})).status == 201
            busy = dispatcher.get_or_create_scenario("a")
            assert busy is not None
            # A running record write keeps the loaded scenario busy, so it cannot be unloaded.
            measurement = busy.operations.start("ingest/write")
            full = await client.post("/reef/scenarios", json={"name": "b"})
            assert full.status == 503
            assert full.headers["Retry-After"] == "30"
            measurement.finish(succeeded=True)
            assert (await client.post("/reef/scenarios", json={"name": "b"})).status == 201
            status = await (await client.get("/reef/status")).json()
            assert status["loaded_scenarios"] == 1
            refused = await client.post("/reef/scenarios", json={"name": "c"})
            assert refused.status == 409
            assert "reef.max_scenarios" in await refused.text()
        finally:
            await client.close()

    asyncio.run(run())


def test_a_selection_between_its_read_and_its_write_when_the_mark_lands_stores_no_sample(
    tmp_path, monkeypatch, swept_by_the_test
) -> None:
    """The samples service read the scenario's records and is about to write their samples when a delete marks the
    scenario. The write checks the mark and stores nothing. Control's sweep, which runs meanwhile, archives the
    storage only after the selection ended, so a write that passed its check before the mark would be archived too."""
    component = StubTrainingComponent()
    dispatcher = dispatcher_for(
        tmp_path,
        TestPolicyRecipe(),
        live=LiveObjects(runtime=StubInferenceRuntime(), training_facts=StubFacts(component)),
    )
    # No training service runs: the sweep stops waiting for its archive soon, and the test answers it below.
    dispatcher.sweeper.deletions.answer_seconds = 2.0
    sweeping: list[threading.Thread] = []
    original = TrainingPipeline.write_record_samples

    def marked_then_written(pipeline, group, record_ids, samples):
        if not sweeping:
            request_cleanup(dispatcher.queue, "math")
            dispatcher.registry.archive_registration("math")
            sweeping.append(threading.Thread(target=dispatcher.sweeper.sweep))
            sweeping[0].start()
            # The sweep holds the cleanup item and waits for the selections, the delete selection among them.
            assert became_true(
                lambda: any(
                    "trigger:delete" in item.metadata
                    for item in dispatcher.queue.items(SAMPLES_QUEUE, scenario="math")
                )
            )
            assert [item.state for item in dispatcher.queue.items(DELETES_QUEUE)] in (["assigned"], ["running"])
        return original(pipeline, group, record_ids, samples)

    monkeypatch.setattr(TrainingPipeline, "write_record_samples", marked_then_written)
    try:
        current = dispatcher.get_or_create_scenario("math")
        assert current is not None
        root = current.serving_artifact_id()
        dispatcher.accept_record(token_inference("i1", artifact_id=root))
        dispatcher.accept_record(report("r1", "i1"))
        assert became_true(lambda: bool(sweeping))
        sweeping[0].join(30)
        assert not sweeping[0].is_alive()
        assert dispatcher.data_store.query("math", kind="trajectory") == ()
        assert dispatcher.data_store.query("math", kind="task") == ()
        assert dispatcher.data_store.query("math", scenario="math") == ()
        selections = dispatcher.queue.items(SAMPLES_QUEUE, scenario="math")
        assert SCENARIO_DELETED_ERROR in [item.error for item in selections]
        # The storage was archived after the selection ended; the cleanup waits for the training service only.
        assert not list((tmp_path / "records").glob("*.sqlite3"))
        (archive,) = dispatcher.queue.items(CHECKS_QUEUE, scenario="math")
        taken = dispatcher.queue.take(CHECKS_QUEUE, "training", [archive.kind])
        dispatcher.queue.finish(CHECKS_QUEUE, taken.id, "training", {"refusal": None})
        dispatcher.sweeper.sweep()
        ((state, result),) = cleanup_items(dispatcher, "math")
        assert state == "finished" and result["deleted"] is True
    finally:
        dispatcher.close()


def became_true(condition, seconds: float = 20.0) -> bool:
    """Whether ``condition`` holds within ``seconds``."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def test_a_process_forgets_only_a_deleted_scenario(tmp_path) -> None:
    """A service may forget a scenario on any hint of a delete: a scenario that was not deleted keeps its backend."""
    factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts")
    live = factory("live")
    factory.forget_scenario("live")
    assert factory("live") is live
    gone = factory("gone")
    factory.archive_registration("gone")
    factory.forget_scenario("gone")
    assert factory("gone") is not gone


def test_a_mark_that_lands_after_another_instances_sweep_ended_the_first_item_still_gets_cleaned_up(
    tmp_path, monkeypatch, swept_by_the_test
) -> None:
    """At a lease takeover, the new control instance sweeps between the old one's first cleanup item and its mark:
    it finds no mark and ends the item. The delete adds the item again after the mark, so the next sweep cleans."""
    import reef.dispatcher.dispatcher as dispatcher_module

    value = durable_dispatcher(tmp_path, InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts"))
    assert value.get_or_create_scenario("delivery") is not None
    calls: list[str] = []

    def then_swept(queue, scenario):
        request_cleanup(queue, scenario)
        if not calls:
            calls.append(scenario)
            value.sweeper.sweep()

    monkeypatch.setattr(dispatcher_module, "request_cleanup", then_swept)
    value.delete_scenario("delivery")
    assert cleanup_items(value, "delivery") == [("finished", {"deleted": False}), ("queued", None)]
    value.sweeper.sweep()
    assert cleanup_items(value, "delivery")[-1] == ("finished", {"deleted": True, "archived": ANY})
    assert not any(path.suffix == ".sqlite3" for path in (tmp_path / "records").iterdir())
    value.close()


def test_the_cleanup_ends_only_after_the_training_service_archived_and_a_failed_archive_is_queued_again(
    tmp_path, swept_by_the_test
) -> None:
    component = StubTrainingComponent()
    value = dispatcher_for(
        tmp_path,
        TestPolicyRecipe(),
        live=LiveObjects(runtime=StubInferenceRuntime(), training_facts=StubFacts(component)),
    )
    value.sweeper.deletions.answer_seconds = 0.2
    assert value.get_or_create_scenario("math") is not None
    value.delete_scenario("math")
    value.sweeper.sweep()
    # No training service answers: the cleanup waits, and the next sweep tries again.
    assert [state for state, _ in cleanup_items(value, "math")] == ["failed"]
    (archive,) = value.queue.items(CHECKS_QUEUE, scenario="math")
    assert archive.id == archive_item_id("math", component.name)
    # The training service fails the archive for good.
    taken = value.queue.take(CHECKS_QUEUE, "training", [archive.kind])
    value.queue.fail(CHECKS_QUEUE, taken.id, "training", "the directory did not move", retryable=False)
    value.sweeper.sweep()
    assert value.queue.read(CHECKS_QUEUE, archive.id).state == "queued"
    assert [state for state, _ in cleanup_items(value, "math")] == ["failed"]
    # It archives on the next attempt, and the next sweep ends the cleanup.
    taken = value.queue.take(CHECKS_QUEUE, "training", [archive.kind])
    value.queue.finish(CHECKS_QUEUE, taken.id, "training", {"refusal": None})
    value.sweeper.sweep()
    assert cleanup_items(value, "math")[-1][0] == "finished"
    value.close()


def test_a_control_instance_without_the_lease_fails_the_withdraw_step(tmp_path, swept_by_the_test) -> None:
    value = durable_dispatcher(tmp_path, InMemoryRepositoryBackend.factory(directory=tmp_path / "artifacts"))
    assert value.get_or_create_scenario("delivery") is not None
    value.publisher.held = False
    value.delete_scenario("delivery")
    value.sweeper.sweep()
    ((state, _),) = cleanup_items(value, "delivery")
    assert state == "failed" and "control lease" in value.queue.items(DELETES_QUEUE)[0].error
    value.publisher.held = True
    value.sweeper.sweep()
    assert cleanup_items(value, "delivery")[-1][0] == "finished"
    value.close()
