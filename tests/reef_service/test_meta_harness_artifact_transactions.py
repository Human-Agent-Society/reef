"""Published harnesses remain staged until their scenario journal commits."""

import copy
import dataclasses

import pytest

from reef.artifact import GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactPublicationError
from reef.dispatcher import Dispatcher
from reef.storage.commit_log import CommitLogScenarioStore
from reef.storage.scenario import SQLiteScenarioStorage

from .test_meta_harness import (
    IMPROVED,
    SEED,
    QueueChat,
    _report_once,
    build_recipe,
    content_id,
    make_binary,
    reply,
    runtime,
    sections,
)


@pytest.fixture(params=["memory", "git"])
def publication_case(tmp_path, request):
    make_binary(tmp_path)
    config = sections(tmp_path)
    chat = QueueChat(reply(content_id(SEED), IMPROVED))
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=runtime())
    recipe = dataclasses.replace(recipe, models={"proposer": chat})
    initial = tmp_path / "initial"
    initial.mkdir()
    memory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "memory")

    def factory():
        if request.param == "memory":
            return memory
        return GitLFSRepositoryBackend.factory(
            tmp_path / "artifacts.git",
            work_dir=tmp_path / "work",
            cache_dir=tmp_path / "cache",
        )

    return recipe, factory, chat, tmp_path / "records"


@pytest.mark.parametrize("recovery", ["retry", "restart"])
def test_failed_selected_commit_never_serves_or_recovers_staged_artifact(publication_case, monkeypatch, recovery):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        scenario = dispatcher.get_or_create_scenario("selected-commit")
        previous = copy.deepcopy(scenario.trainer.state)
        head = scenario.current_artifact_ref()
        _report_once(scenario, "selected-commit", "1")
        result = scenario.prepare_training_step()
        assert result.artifact is not None and len(chat.prompts) == 1
        assert isinstance(scenario.store, CommitLogScenarioStore)
        journal = scenario.store.commit_log
        assert journal is not None
        append = journal.append

        def fail(record):
            assert scenario.current_artifact_ref() == head
            assert scenario.repository.backend.current() == head
            raise RuntimeError("journal unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(journal, "append", fail)
            with pytest.raises(RuntimeError, match="journal unavailable"):
                scenario.commit(result)
        assert scenario.trainer.state == previous
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() == head
        assert scenario.store.history() == ()
        if recovery == "retry":

            def durable_append(record):
                assert scenario.current_artifact_ref() == scenario.repository.backend.current() == head
                return append(record)

            monkeypatch.setattr(journal, "append", durable_append)
            scenario.commit(result)
            assert scenario.scenario_step == 1
            assert scenario.trainer.state == result.state
            assert scenario.current_artifact_ref() == scenario.repository.backend.current() != head
            assert len(scenario.store.history()) == 1
    finally:
        dispatcher.close()
    restarted = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        scenario = restarted.get_or_create_scenario("selected-commit")
        assert scenario.trainer.state == (result.state if recovery == "retry" else previous)
        assert scenario.scenario_step == (1 if recovery == "retry" else 0)
        assert scenario.current_artifact_ref() == scenario.repository.backend.current()
        assert len(chat.prompts) == 1
    finally:
        restarted.close()


def test_restart_repairs_artifact_head_from_successful_journal_commit(publication_case, monkeypatch):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        scenario = dispatcher.get_or_create_scenario("stale-artifact-head")
        old_head = scenario.current_artifact_ref()
        _report_once(scenario, "stale-artifact-head", "1")
        result = scenario.prepare_training_step()

        def unavailable(*args, **kwargs):
            raise ArtifactPublicationError("backend pointer unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(scenario.repository.backend, "commit_release", unavailable)
            scenario.commit(result)
        assert scenario.scenario_step == 1 and scenario.trainer.state == result.state
        committed_head = scenario.current_artifact_ref()
        assert committed_head != old_head
        assert scenario.repository.backend.current() == old_head
        sync = scenario.commit_status["artifact_head_sync"]
        assert sync == {
            "state": "pending",
            "release_id": committed_head.release_id,
            "error": "backend pointer unavailable",
        }
        assert dispatcher.build_training_status()["scenarios"]["stale-artifact-head"]["artifact_head_sync"] == sync
        assert scenario.store.history()[-1].artifact_ref == committed_head
    finally:
        dispatcher.close()
    restarted = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        scenario = restarted.get_or_create_scenario("stale-artifact-head")
        assert scenario.scenario_step == 1 and scenario.trainer.state == result.state
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() == committed_head
        assert len(scenario.store.history()) == 1 and len(chat.prompts) == 1
    finally:
        restarted.close()


def test_without_a_journal_the_backend_publication_remains_the_commit(publication_case, monkeypatch):
    recipe, factory, _, _ = publication_case
    dispatcher = Dispatcher(recipe, factory(), scenario_storage=SQLiteScenarioStorage())
    try:
        scenario = dispatcher.get_or_create_scenario("no-journal")
        head = scenario.current_artifact_ref()

        def unexpected(*args, **kwargs):
            pytest.fail("a non-journaled publication must commit its backend head directly")

        monkeypatch.setattr(scenario.repository.backend, "commit_release", unexpected)
        _report_once(scenario, "no-journal", "1")
        result = scenario.prepare_training_step()
        scenario.commit(result)

        assert not scenario.store.durable
        assert scenario.trainer.state == result.state
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() != head
        assert scenario.commit_status["artifact_head_sync"] == {
            "state": "synchronized",
            "release_id": scenario.current_artifact_ref().release_id,
            "error": None,
        }
    finally:
        dispatcher.close()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_lost_journal_ack_retries_the_committed_release_without_evaluating_again(
    publication_case, monkeypatch, checkpoint
):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        scenario = dispatcher.get_or_create_scenario("lost-journal-ack")
        monkeypatch.setattr(scenario._committer, "_should_checkpoint", lambda result: checkpoint)
        _report_once(scenario, "lost-journal-ack", "1")
        result = scenario.prepare_training_step()
        assert isinstance(scenario.store, CommitLogScenarioStore)
        journal = scenario.store.commit_log
        assert journal is not None
        append = journal.append

        def lose_ack(record):
            append(record)
            raise OSError("journal acknowledgment lost")

        with monkeypatch.context() as patch:
            patch.setattr(journal, "append", lose_ack)
            with pytest.raises(OSError, match="acknowledgment lost"):
                scenario.commit(result)
        committed = scenario.store.history()[-1]
        scenario.commit(result)

        assert scenario.scenario_step == 1 and len(chat.prompts) == 1
        assert scenario.current_artifact_ref() == committed.artifact_ref
        assert scenario.repository.resolve(committed.artifact_ref).local_path.is_dir()
        assert len(scenario.store.history()) == 1
    finally:
        dispatcher.close()


def test_postcommit_conflict_keeps_durable_step_and_rejects_unrelated_head(publication_case, monkeypatch):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        scenario = dispatcher.get_or_create_scenario("postcommit-head-conflict")
        backend = scenario.repository.backend
        old_head = backend.current()
        competing_path = records.parent / "competing-artifact"
        competing_path.mkdir()
        (competing_path / "foreign.txt").write_text("another writer's artifact")
        competing = Artifact.local(competing_path, metadata=backend.metadata())
        commit_release = backend.commit_release
        unrelated = None

        def conflict_after_journal(ref, *, expected_parent):
            nonlocal unrelated
            assert scenario.store.history()[-1].artifact_ref == ref
            unrelated = backend.publish(competing, expected_parent=expected_parent)
            return commit_release(ref, expected_parent=expected_parent)

        _report_once(scenario, "postcommit-head-conflict", "1")
        result = scenario.prepare_training_step()
        with monkeypatch.context() as patch:
            patch.setattr(backend, "commit_release", conflict_after_journal)
            scenario.commit(result)

        committed = scenario.store.history()[-1]
        assert scenario.scenario_step == committed.step == 1
        assert scenario.trainer.state == committed.algorithm_state == result.state
        assert scenario.current_artifact_ref() == committed.artifact_ref
        assert backend.current() == unrelated
        assert scenario.commit_status["artifact_head_sync"]["state"] == "conflict"
        assert scenario.commit_status["artifact_head_sync"]["release_id"] == committed.artifact_ref.release_id
        assert unrelated not in (old_head, committed.artifact_ref)
        with pytest.raises(ArtifactConflict):
            scenario.repository.synchronize_checkpoint()
        assert backend.current() == unrelated and len(scenario.store.history()) == 1
    finally:
        dispatcher.close()

    restarted = Dispatcher(
        recipe, factory(), agent_record_dir=records, scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        with pytest.raises(ArtifactConflict):
            restarted.get_or_create_scenario("postcommit-head-conflict")
        assert len(chat.prompts) == 1
    finally:
        restarted.close()
