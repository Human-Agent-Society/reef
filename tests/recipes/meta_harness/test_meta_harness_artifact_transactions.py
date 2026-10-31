"""Published harnesses remain staged until their scenario journal commits."""

import copy
import dataclasses

import pytest
from tests.support.git_lfs import fake_git_lfs
from tests.support.harness_methods import IMPROVED, SEED, QueueChat, meta_harness_recipe, proposal_reply
from tests.support.harness_service import checkpoint_result, report_once

from reef.artifact import GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactPublicationError
from reef.dispatcher import Dispatcher
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

__all__ = ["fake_git_lfs"]


@pytest.fixture(params=["memory", "git"])
def publication_case(tmp_path, request):
    chat = QueueChat(proposal_reply(SEED, IMPROVED))
    recipe = meta_harness_recipe(tmp_path, chat)
    # The base release holds the seed, as a deployment's does.
    initial = tmp_path / "initial"
    recipe.seed.write(initial)
    memory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "memory")

    if request.param == "git":
        # The commit protocol, not Git LFS, is under test here; tests/artifact covers real LFS.
        request.getfixturevalue("fake_git_lfs")

    def factory():
        if request.param == "memory":
            return memory
        return GitLFSRepositoryBackend.factory(
            tmp_path / "artifacts.git",
            work_dir=tmp_path / "work",
            cache_dir=tmp_path / "cache",
            base=recipe.base_artifact_source(),
        )

    return recipe, factory, chat, tmp_path / "records"


@pytest.mark.parametrize("recovery", ["retry", "restart"])
def test_failed_selected_commit_never_serves_or_recovers_staged_artifact(publication_case, monkeypatch, recovery):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe,
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("selected-commit")
        head = scenario.current_artifact_ref()
        report_once(scenario, "1")
        result = checkpoint_result(scenario)
        # The state after the job's submission, which the failed checkpoint commit keeps.
        previous = copy.deepcopy(scenario.trainer.state)
        assert result.artifact is not None and len(chat.prompts) == 1
        assert isinstance(scenario.store, CommitHistoryScenarioStore)
        journal = scenario.store.commit_history
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
        assert len(scenario.store.history()) == 1
        if recovery == "retry":

            def durable_append(record):
                assert scenario.current_artifact_ref() == scenario.repository.backend.current() == head
                return append(record)

            monkeypatch.setattr(journal, "append", durable_append)
            scenario.commit(result)
            assert scenario.scenario_step == 2
            assert scenario.trainer.state == result.state
            assert scenario.current_artifact_ref() == scenario.repository.backend.current() != head
            assert len(scenario.store.history()) == 2
    finally:
        dispatcher.close()
    restarted = Dispatcher(
        # Each process makes its own harness component, as each service process builds its recipe.
        dataclasses.replace(recipe, harness_component=None),
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        scenario = restarted.get_or_create_scenario("selected-commit")
        assert scenario.trainer.state == (result.state if recovery == "retry" else previous)
        assert scenario.scenario_step == (2 if recovery == "retry" else 1)
        assert scenario.current_artifact_ref() == scenario.repository.backend.current()
        assert len(chat.prompts) == 1
    finally:
        restarted.close()


def test_restart_repairs_artifact_head_from_successful_journal_commit(publication_case, monkeypatch):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe,
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("stale-artifact-head")
        old_head = scenario.current_artifact_ref()
        report_once(scenario, "1")
        result = checkpoint_result(scenario)

        def unavailable(*args, **kwargs):
            raise ArtifactPublicationError("backend pointer unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(scenario.repository.backend, "commit_release", unavailable)
            scenario.commit(result)
        assert scenario.scenario_step == 2 and scenario.trainer.state == result.state
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
        # Each process makes its own harness component, as each service process builds its recipe.
        dataclasses.replace(recipe, harness_component=None),
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        scenario = restarted.get_or_create_scenario("stale-artifact-head")
        assert scenario.scenario_step == 2 and scenario.trainer.state == result.state
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() == committed_head
        assert len(scenario.store.history()) == 2 and len(chat.prompts) == 1
    finally:
        restarted.close()


def test_without_a_journal_the_backend_publication_remains_the_commit(publication_case, monkeypatch):
    recipe, factory, _, _ = publication_case
    dispatcher = Dispatcher(
        recipe,
        factory(),
        scenario_storage=SQLiteScenarioStorage(),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("no-journal")
        head = scenario.current_artifact_ref()

        def unexpected(*args, **kwargs):
            pytest.fail("a non-journaled publication must commit its backend head directly")

        monkeypatch.setattr(scenario.repository.backend, "commit_release", unexpected)
        report_once(scenario, "1")
        result = checkpoint_result(scenario)
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


def test_lost_journal_ack_retries_the_committed_release_without_evaluating_again(publication_case, monkeypatch):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe,
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("lost-journal-ack")
        report_once(scenario, "1")
        result = checkpoint_result(scenario)
        commit_step = scenario.store.commit_step

        def lose_ack(**keywords):
            commit_step(**keywords)
            raise OSError("journal acknowledgment lost")

        with monkeypatch.context() as patch:
            patch.setattr(scenario.store, "commit_step", lose_ack)
            with pytest.raises(OSError, match="acknowledgment lost"):
                scenario.commit(result)
        committed = scenario.store.history()[-1]
        scenario.commit(result)

        assert scenario.scenario_step == 2 and len(chat.prompts) == 1
        assert scenario.current_artifact_ref() == committed.artifact_ref
        assert scenario.repository.resolve(committed.artifact_ref).local_path.is_dir()
        assert len(scenario.store.history()) == 2
    finally:
        dispatcher.close()


def test_postcommit_conflict_keeps_durable_step_and_rejects_unrelated_head(publication_case, monkeypatch):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(
        recipe,
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
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

        report_once(scenario, "1")
        result = checkpoint_result(scenario)
        with monkeypatch.context() as patch:
            patch.setattr(backend, "commit_release", conflict_after_journal)
            scenario.commit(result)

        committed = scenario.store.history()[-1]
        assert scenario.scenario_step == committed.step == 2
        assert scenario.trainer.state == committed.algorithm_state == result.state
        assert scenario.current_artifact_ref() == committed.artifact_ref
        assert backend.current() == unrelated
        assert scenario.commit_status["artifact_head_sync"]["state"] == "conflict"
        assert scenario.commit_status["artifact_head_sync"]["release_id"] == committed.artifact_ref.release_id
        assert unrelated not in (old_head, committed.artifact_ref)
        with pytest.raises(ArtifactConflict):
            scenario.repository.synchronize_checkpoint()
        assert backend.current() == unrelated and len(scenario.store.history()) == 2
    finally:
        dispatcher.close()

    restarted = Dispatcher(
        # Each process makes its own harness component, as each service process builds its recipe.
        dataclasses.replace(recipe, harness_component=None),
        factory(),
        scenario_storage=SQLiteScenarioStorage(records),
        hold_local_cycles=True,
        coordination=memory_coordination(),
    )
    try:
        with pytest.raises(ArtifactConflict):
            restarted.get_or_create_scenario("postcommit-head-conflict")
        assert len(chat.prompts) == 1
    finally:
        restarted.close()
