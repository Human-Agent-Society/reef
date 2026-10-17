"""Generated tasks persist before rollout and recover after interrupted publication."""

import dataclasses
import json
from pathlib import Path

import pytest
from tests.support.task_service_stubs import task_for

from reef.core.tasks import GeneratedTask, PlayArm, manifest_task_paths, read_harbor_task
from reef.core.tasks.split import TaskSplitConflict
from reef.record2task.store import FileTaskStore, PlayedResult, TaskConflict, TaskStoreError

TASK_NAME = "stored-task"
SOURCE_ID = "source-session"
PORT = 8471
SEED = 7


def candidate(name: str = TASK_NAME, *, source: str = SOURCE_ID, port: int = PORT) -> GeneratedTask:
    """Return a task with a measurement arm and explicit original sources."""
    return GeneratedTask(
        task=task_for(name=name, record_id=source, port=port), measurement_arms=(PlayArm(name="hint", count=1),)
    )


@pytest.mark.unit
def test_saved_task_reopens_with_its_files_sources_split_and_measurement_arms(tmp_path: Path) -> None:
    generated = candidate()
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        store.save_candidates(generation=0, candidates=(generated,))
        first = store.publish_task(generation=0, index=0)
        assert store.publish_task(generation=0, index=0) == first
        store.complete_generation(generation=0)
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        (saved,) = reopened.tasks(split="train")
        assert saved.generated == generated
        assert saved.source_record_ids == (SOURCE_ID,)
        assert read_harbor_task(path=saved.path) == generated.task
        assert reopened.generation(generation=0).is_complete


@pytest.mark.unit
def test_unfinished_validation_recovers_candidates_without_generating_again(tmp_path: Path) -> None:
    generated = (candidate(), candidate(name="second-task", source="second-source"))
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        store.save_candidates(generation=0, candidates=generated)
        store.publish_task(generation=0, index=0)
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        generation = reopened.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        assert generation.candidates == generated
        assert generation.pending_indices == (1,)
        with pytest.raises(TaskStoreError, match="unfinished"):
            reopened.complete_generation(generation=0)
        reopened.reject_task(generation=0, index=1, reason="oracle failed")
        reopened.complete_generation(generation=0)
        assert len(reopened.tasks()) == 1


@pytest.mark.unit
def test_eval_tasks_have_a_readable_manifest_without_rollout(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=1.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        store.save_candidates(generation=0, candidates=(candidate(),))
        saved = store.publish_task(generation=0, index=0)
        store.complete_generation(generation=0)
        assert saved.split == "eval"
        assert store.tasks(split="train") == ()
        assert manifest_task_paths(manifest_path=store.manifest_path, root=store.task_root, split="eval") == (
            saved.path,
        )


@pytest.mark.unit
def test_a_closed_worker_cannot_publish_after_another_writer_opens_the_store(tmp_path: Path) -> None:
    store = FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED)
    store.begin_generation(generation=0, source_record_ids=())
    store.save_candidates(generation=0, candidates=(candidate(),))
    store.close()
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        with pytest.raises(TaskStoreError, match="closed"):
            store.publish_task(generation=0, index=0)
        assert reopened.tasks() == ()
        assert not (reopened.task_root / TASK_NAME).exists()


@pytest.mark.unit
def test_name_conflicts_leave_the_existing_task_unchanged(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=())
        store.save_candidates(generation=0, candidates=(candidate(),))
        first = store.publish_task(generation=0, index=0)
        store.begin_generation(generation=1, source_record_ids=())
        store.save_candidates(generation=1, candidates=(candidate(port=PORT + 1),))
        with pytest.raises(TaskConflict, match="different digest"):
            store.publish_task(generation=1, index=0)
        assert read_harbor_task(path=first.path).digest == first.generated.task.digest
        assert len(store.tasks()) == 1


@pytest.mark.unit
def test_a_second_writer_is_refused_and_close_releases_the_lock(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED), pytest.raises(TaskStoreError, match="writer"):
        FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED)
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED):
        pass


@pytest.mark.unit
def test_split_configuration_cannot_change_on_reopen(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED):
        pass
    with pytest.raises(TaskStoreError, match="split configuration"):
        FileTaskStore(root=tmp_path, eval_fraction=0.5, seed=SEED)


@pytest.mark.unit
def test_an_unknown_store_version_is_refused(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED):
        pass
    settings_path = tmp_path / "store.json"
    settings = json.loads(settings_path.read_text())
    settings["version"] += 1
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(TaskStoreError, match="unsupported task store version"):
        FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED)


@pytest.mark.unit
def test_reopening_clears_interrupted_json_and_harbor_staging(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED):
        for staging in (tmp_path / ".staging", tmp_path / "tasks" / ".staging"):
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "unfinished").write_text("interrupted write")
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED):
        assert not (tmp_path / ".staging" / "unfinished").exists()
        assert not (tmp_path / "tasks" / ".staging").exists()


@pytest.mark.unit
def test_missing_published_files_fail_on_reopen(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=())
        store.save_candidates(generation=0, candidates=(candidate(),))
        saved = store.publish_task(generation=0, index=0)
    (saved.path / "instruction.md").unlink()
    with pytest.raises(TaskStoreError, match="stored-task"):
        FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED)


@pytest.mark.unit
def test_authoring_receipts_are_separate_from_split_sources(tmp_path: Path) -> None:
    generated = candidate()
    generated = dataclasses.replace(
        generated,
        task=dataclasses.replace(
            generated.task,
            source_agent_record_ids=("designer-receipt", SOURCE_ID),
            metadata={**generated.task.metadata, "designer_record_id": "designer-receipt"},
        ),
    )
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        store.save_candidates(generation=0, candidates=(generated,))
        saved = store.publish_task(generation=0, index=0)
        assert saved.source_record_ids == (SOURCE_ID,)
        assert saved.designer_record_ids == ("designer-receipt",)


@pytest.mark.unit
def test_a_conflicting_source_bridge_stays_out_of_both_splits_after_reopen(tmp_path: Path) -> None:
    train = candidate(name="train-task", source="train-source")
    evaluation = candidate(name="eval-task", source="eval-source")
    bridge = candidate(name="bridge-task")
    bridge = dataclasses.replace(
        bridge, task=dataclasses.replace(bridge.task, source_agent_record_ids=("train-source", "eval-source"))
    )
    with FileTaskStore(root=tmp_path, eval_fraction=0.5, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=())
        store.save_candidates(generation=0, candidates=(train, evaluation, bridge))
        assert store.publish_task(generation=0, index=0).split == "train"
        assert store.publish_task(generation=0, index=1).split == "eval"
        with pytest.raises(TaskSplitConflict, match="split conflict"):
            store.publish_task(generation=0, index=2)
        store.reject_task(generation=0, index=2, reason="split conflict")
        store.complete_generation(generation=0)
    with FileTaskStore(root=tmp_path, eval_fraction=0.5, seed=SEED) as store:
        assert store.split().train == ("train-task",)
        assert store.split().eval == ("eval-task",)
        assert store.counts()["rejected"] == 1
        assert not (store.task_root / "bridge-task").exists()


@pytest.mark.unit
@pytest.mark.parametrize("boundary", ["sync_task", "save_generation", "write_manifest"])
def test_interrupted_publication_recovers_without_losing_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    def interrupt(**fields: object) -> None:
        raise OSError("simulated publication interruption")

    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        store.save_candidates(generation=0, candidates=(candidate(),))
        with monkeypatch.context() as interruption:
            interruption.setattr(store, boundary, interrupt)
            with pytest.raises(OSError, match="publication interruption"):
                store.publish_task(generation=0, index=0)
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        state = reopened.generation(generation=0)
        assert state.candidates == (candidate(),)
        reopened.publish_task(generation=0, index=0)
        reopened.complete_generation(generation=0)
        (saved,) = reopened.tasks()
        assert read_harbor_task(path=saved.path) == candidate().task
        assert reopened.generation(generation=0).is_complete


@pytest.mark.unit
def test_a_failed_candidate_checkpoint_does_not_create_a_completed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(**fields: object) -> None:
        raise OSError("simulated checkpoint interruption")

    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        with monkeypatch.context() as interruption:
            interruption.setattr(store, "write_atomic", interrupt)
            with pytest.raises(OSError, match="checkpoint interruption"):
                store.save_candidates(generation=0, candidates=(candidate(),))
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        state = reopened.generation(generation=0)
        assert state.candidates is None and not state.is_complete
        assert reopened.tasks() == ()


@pytest.mark.unit
def test_interrupted_source_completion_keeps_the_published_task_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(**fields: object) -> None:
        raise OSError("simulated completion interruption")

    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        store.begin_generation(generation=0, source_record_ids=(SOURCE_ID,))
        store.save_candidates(generation=0, candidates=(candidate(),))
        store.publish_task(generation=0, index=0)
        with monkeypatch.context() as interruption:
            interruption.setattr(store, "write_atomic", interrupt)
            with pytest.raises(OSError, match="completion interruption"):
                store.complete_generation(generation=0)
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        assert len(reopened.tasks()) == 1
        assert not reopened.generation(generation=0).is_complete
        reopened.complete_generation(generation=0)
        assert reopened.generation(generation=0).is_complete


def completed(store: FileTaskStore, generation: int, *names: str, sources: tuple[str, ...] = ()) -> None:
    """Save and publish one complete generation of train tasks."""
    store.begin_generation(generation=generation, source_record_ids=sources)
    store.save_candidates(
        generation=generation,
        candidates=tuple(
            candidate(name, source=f"source-{name}", port=PORT + index) for index, name in enumerate(names)
        ),
    )
    for index in range(len(names)):
        store.publish_task(generation=generation, index=index)
    store.complete_generation(generation=generation)


@pytest.mark.unit
def test_play_results_make_tasks_unavailable_and_complete_a_generations_feedback(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        completed(store, 0, "first-task", "second-task")
        assert [saved.generated.task.name for saved in store.available_tasks()] == ["first-task", "second-task"]
        store.record_result(generation=0, result=PlayedResult("first-task", (1.0, 0.0), reported_plays=2))
        assert [saved.generated.task.name for saved in store.available_tasks()] == ["second-task"]
        assert store.deliverable_generations() == ()
        with pytest.raises(TaskStoreError, match="already has a result"):
            store.record_result(generation=0, result=PlayedResult("first-task", (1.0,)))
        store.record_result(
            generation=0, result=PlayedResult("second-task", error="sandbox gone", infrastructure_failure=True)
        )
        (state,) = store.deliverable_generations()
        first, second = state.task_results()
        assert first.training_rewards == (1.0, 0.0) and first.is_played
        assert second.infrastructure_failure and second.error == "sandbox gone"
        assert store.counts()["played/train"] == 2
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as reopened:
        assert reopened.available_tasks() == () and len(reopened.deliverable_generations()) == 1
        reopened.mark_delivered(generation=0)
        assert reopened.deliverable_generations() == ()


@pytest.mark.unit
def test_an_incomplete_generation_offers_no_task_and_is_found_by_its_sources(tmp_path: Path) -> None:
    with FileTaskStore(root=tmp_path, eval_fraction=0.0, seed=SEED) as store:
        completed(store, 0, "done-task", sources=("done-source",))
        store.begin_generation(generation=1, source_record_ids=("open-source",))
        store.save_candidates(generation=1, candidates=(candidate("open-task", source="open-source"),))
        store.publish_task(generation=1, index=0)
        assert [saved.generated.task.name for saved in store.available_tasks()] == ["done-task"]
        assert store.completed_source_record_ids() == {"done-source"}
        found = store.find_generation(source_record_ids=("open-source",))
        assert found is not None and found.generation == 1 and not found.is_complete
        assert store.find_generation(source_record_ids=()) is None
        assert store.next_generation() == 2
