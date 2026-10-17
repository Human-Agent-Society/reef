"""The episode runner stages a saved task again after a stop or a redeployment, and keeps each episode's outcome."""

import asyncio
from pathlib import Path

import pytest
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.record2task.generator.client import TaskNameConflict
from reef.record2task.generator.runner import GeneratorEpisodeRunner


@pytest.mark.unit
def test_a_saved_task_stages_again_on_an_empty_root_and_plays_with_its_labels(tmp_path: Path) -> None:
    task = task_for("saved-task", "designer-0", 9000)
    first = StandInGenerator(tmp_path / "first", rewards={"plain": 0.5})
    runner = GeneratorEpisodeRunner(first)
    path = asyncio.run(runner.stage(task))
    assert asyncio.run(runner.stage(task)) == path, "a stop between staging and play stages again"
    redeployed = GeneratorEpisodeRunner(StandInGenerator(tmp_path / "redeployed", rewards={"plain": 0.5}))
    episodes = asyncio.run(
        redeployed.play(
            asyncio.run(redeployed.stage(task)),
            scenario="tasks",
            arm="plain",
            plays=2,
            extra_instruction_files=(),
            labels={"batch": "tasks:generated-tasks:1", "task_use": "use-saved-task-0"},
            model="Qwen/Qwen3-8B",
        )
    )
    assert [episode.reward for episode in episodes] == [0.5, 0.5] and all(episode.is_reported for episode in episodes)
    assert not any(episode.is_infrastructure_failure for episode in episodes)
    service = redeployed.service
    assert isinstance(service, StandInGenerator)
    assert service.plays[0]["tags"] == {"batch": "tasks:generated-tasks:1", "task_use": "use-saved-task-0"}


@pytest.mark.unit
def test_an_episode_that_could_not_run_is_an_infrastructure_failure(tmp_path: Path) -> None:
    runner = GeneratorEpisodeRunner(StandInGenerator(tmp_path / "tasks", play_error="Failed to start tmux session."))
    task = task_for("saved-task", "designer-0", 9000)
    (episode,) = asyncio.run(
        runner.play(
            asyncio.run(runner.stage(task)),
            scenario="tasks",
            arm="plain",
            plays=1,
            extra_instruction_files=(),
            labels={},
            model=None,
        )
    )
    assert episode.is_infrastructure_failure and episode.error == "Failed to start tmux session."
    assert not episode.is_reported


@pytest.mark.unit
def test_a_different_task_under_a_staged_name_is_refused(tmp_path: Path) -> None:
    runner = GeneratorEpisodeRunner(StandInGenerator(tmp_path / "tasks"))
    asyncio.run(runner.stage(task_for("saved-task", "designer-0", 9000)))
    with pytest.raises(TaskNameConflict):
        asyncio.run(runner.stage(task_for("saved-task", "designer-1", 9001)))


@pytest.mark.unit
def test_unstage_removes_only_a_copy_this_runner_created_and_no_other_stage_reuses(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    task = task_for("saved-task", "designer-0", 9000)
    owner = GeneratorEpisodeRunner(service)
    path = asyncio.run(owner.stage(task))
    asyncio.run(owner.stage(task))
    borrower = GeneratorEpisodeRunner(service)
    asyncio.run(borrower.stage(task))
    asyncio.run(borrower.unstage(task.name))
    assert path.is_dir(), "a runner that found the copy staged does not remove it"
    asyncio.run(owner.unstage(task.name))
    assert path.is_dir(), "the owner's second stage still uses the copy"
    asyncio.run(owner.unstage(task.name))
    assert not path.exists() and service.deleted == [task.name]
