"""The shipped scorer for task directory runs: the Harbor verifier's reward from the verifier row."""

from __future__ import annotations


import pytest

from reef.core.tasks import primary_reward
from reef.evaluation.harness import EvaluationTask, TaskScorer, VerifierRewardScorer, verifier_reward
from reef.harness.tasks.run import TaskResult

TASK = "/tasks/openenv-00012-003-deduction"


def task_run(*events: dict[str, object], exit_code: int = 0) -> TaskResult:
    return TaskResult(exit_code=exit_code, stdout="", stderr="", trajectory=tuple(events), residue=())


def verifier(
    rewards: dict[str, object], *, task: str = TASK, failed: bool = False, error: str = ""
) -> dict[str, object]:
    return {
        "type": "verifier",
        "task": task,
        "rewards": rewards,
        "reward": primary_reward(rewards),
        "failed": failed,
        "error": error,
    }


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (task_run(verifier({"reward": 1.0}), {"type": "step"}), 1.0),
        (task_run(verifier({"reward": 0.25})), 0.25),
        (task_run(verifier({"reward": 0})), 0.0),
        (task_run(verifier({"tests_total": 10, "tests_passed": 7, "reward": 0.7})), 0.7),
        (task_run(verifier({"accuracy": 1})), 1.0),
        (task_run(verifier({}, failed=True)), 0.0),
        (task_run(verifier({}, failed=True), exit_code=1), 0.0),
        (task_run(exit_code=1), 0.0),
    ],
)
def test_the_verifier_reward_is_the_score(result: TaskResult, expected: float) -> None:
    assert verifier_reward(EvaluationTask.of_directory(TASK), result) == expected


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (task_run({"type": "step"}), "expected one verifier record"),
        (task_run(verifier({"reward": 1.0}), verifier({"reward": 0.0})), "expected one verifier record"),
        (task_run(verifier({"reward": 1.0}, task="/tasks/other")), "names '/tasks/other'"),
        (
            task_run(verifier({"tests_total": 10, "tests_passed": 7})),
            "wrote \\['tests_passed', 'tests_total'\\] and no",
        ),
        (task_run(verifier({"reward": float("nan")})), "must be a finite number"),
        (task_run(verifier({"reward": True})), "must be a finite number"),
        (task_run(verifier({"reward": "1"})), "must be a finite number"),
    ],
)
def test_a_record_that_cannot_be_scored_is_an_error(result: TaskResult, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        verifier_reward(EvaluationTask.of_directory(TASK), result)


def test_an_older_row_without_the_rewards_mapping_still_scores() -> None:
    row = {"type": "verifier", "task": TASK, "reward": 0.5, "failed": False}
    assert verifier_reward(EvaluationTask.of_directory(TASK), task_run(row)) == 0.5


def test_the_scorer_resolves_from_its_class_reference() -> None:
    scorer = TaskScorer.from_setting(
        "reef.evaluation.harness:VerifierRewardScorer", setting="evolution.evaluation.scorer"
    )
    assert isinstance(scorer, VerifierRewardScorer)
    assert scorer.score(EvaluationTask.of_directory(TASK), task_run(verifier({"reward": 0.5}))) == 0.5
