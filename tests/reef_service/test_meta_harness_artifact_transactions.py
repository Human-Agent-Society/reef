"""Published harnesses remain staged until their scenario journal commits."""

import copy
import dataclasses

import pytest

from reef.artifact import GitLFSRepositoryBackend, InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact, ArtifactConflict, ArtifactPublicationError
from reef.dispatcher import Dispatcher

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
    dispatcher = Dispatcher(recipe, factory(), agent_record_dir=records)
    try:
        scenario = dispatcher.get_or_create_scenario("selected-commit")
        previous = copy.deepcopy(scenario.trainer.state)
        head = scenario.current_artifact_ref()
        _report_once(scenario, "selected-commit", "1")
        result = scenario.prepare_training_step()
        assert result.artifact is not None and len(chat.prompts) == 1
        append = scenario.commit_log.append

        def fail(record):
            assert scenario.current_artifact_ref() == head
            assert scenario.repository.backend.current() == head
            raise RuntimeError("journal unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(scenario.commit_log, "append", fail)
            with pytest.raises(RuntimeError, match="journal unavailable"):
                scenario.commit(result)
        assert scenario.trainer.state == previous
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() == head
        assert scenario.commit_log.records() == ()
        if recovery == "retry":

            def durable_append(record):
                assert scenario.current_artifact_ref() == scenario.repository.backend.current() == head
                return append(record)

            monkeypatch.setattr(scenario.commit_log, "append", durable_append)
            scenario.commit(result)
            assert scenario.scenario_step == 1
            assert scenario.trainer.state == result.state
            assert scenario.current_artifact_ref() == scenario.repository.backend.current() != head
            assert len(scenario.commit_log.records()) == 1
    finally:
        dispatcher.close()
    restarted = Dispatcher(recipe, factory(), agent_record_dir=records)
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
    dispatcher = Dispatcher(recipe, factory(), agent_record_dir=records)
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
        assert scenario.commit_log.records()[-1].artifact_ref == committed_head
    finally:
        dispatcher.close()
    restarted = Dispatcher(recipe, factory(), agent_record_dir=records)
    try:
        scenario = restarted.get_or_create_scenario("stale-artifact-head")
        assert scenario.scenario_step == 1 and scenario.trainer.state == result.state
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() == committed_head
        assert len(scenario.commit_log.records()) == 1 and len(chat.prompts) == 1
    finally:
        restarted.close()


def test_postcommit_conflict_keeps_durable_step_and_rejects_unrelated_head(publication_case, monkeypatch):
    recipe, factory, chat, records = publication_case
    dispatcher = Dispatcher(recipe, factory(), agent_record_dir=records)
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
            assert scenario.commit_log.records()[-1].artifact_ref == ref
            unrelated = backend.publish(competing, expected_parent=expected_parent)
            return commit_release(ref, expected_parent=expected_parent)

        _report_once(scenario, "postcommit-head-conflict", "1")
        result = scenario.prepare_training_step()
        with monkeypatch.context() as patch:
            patch.setattr(backend, "commit_release", conflict_after_journal)
            scenario.commit(result)

        committed = scenario.commit_log.records()[-1]
        assert scenario.scenario_step == committed.step == 1
        assert scenario.trainer.state == committed.algorithm_state == result.state
        assert scenario.current_artifact_ref() == committed.artifact_ref
        assert backend.current() == unrelated
        assert unrelated not in (old_head, committed.artifact_ref)
        with pytest.raises(ArtifactConflict):
            scenario.repository.synchronize_checkpoint()
        assert backend.current() == unrelated and len(scenario.commit_log.records()) == 1
    finally:
        dispatcher.close()

    restarted = Dispatcher(recipe, factory(), agent_record_dir=records)
    try:
        with pytest.raises(ArtifactConflict):
            restarted.get_or_create_scenario("postcommit-head-conflict")
        assert len(chat.prompts) == 1
    finally:
        restarted.close()
