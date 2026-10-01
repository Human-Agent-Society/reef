"""The shipped gate scorer for task directory episodes: the Harbor verifier's reward from the terminus row."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.executor import LocalExecutor
from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import primary_reward
from reef.harness.runners.terminus.runner import trial_record
from reef.train.cordis_backend import ScoreUnavailable
from reef.train.cordis_backend.backend import EpisodeEvaluationWorker
from reef.train.cordis_backend.strategies import required_verifier_reward, resolve_episode_scorer, verifier_reward

TASK = "/tasks/openenv-00012-003-deduction"


def episode(*events: dict[str, object], exit_code: int = 0) -> EpisodeResult:
    return EpisodeResult(exit_code=exit_code, stdout="", stderr="", trajectory=tuple(events), residue=())


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
        (episode(verifier({"reward": 1.0}), {"type": "step"}), 1.0),
        (episode(verifier({"reward": 0.25})), 0.25),
        (episode(verifier({"reward": 0})), 0.0),
        (episode(verifier({"tests_total": 10, "tests_passed": 7, "reward": 0.7})), 0.7),
        (episode(verifier({"accuracy": 1})), 1.0),
        (episode(verifier({}, failed=True)), 0.0),
        (episode(verifier({}, failed=True), exit_code=1), 0.0),
        (episode(exit_code=1), 0.0),
    ],
)
def test_the_verifier_reward_is_the_score(result: EpisodeResult, expected: float) -> None:
    assert verifier_reward(TASK, result) == expected


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (episode({"type": "step"}), "expected one verifier record"),
        (episode(verifier({"reward": 1.0}), verifier({"reward": 0.0})), "expected one verifier record"),
        (episode(verifier({"reward": 1.0}, task="/tasks/other")), "names '/tasks/other'"),
        (
            episode(verifier({"tests_total": 10, "tests_passed": 7})),
            "wrote \\['tests_passed', 'tests_total'\\] and no",
        ),
        (episode(verifier({"reward": float("nan")})), "must be a finite number"),
        (episode(verifier({"reward": True})), "must be a finite number"),
        (episode(verifier({"reward": "1"})), "must be a finite number"),
    ],
)
def test_a_record_that_cannot_be_scored_is_an_error(result: EpisodeResult, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        verifier_reward(TASK, result)


def test_an_older_row_without_the_rewards_mapping_still_scores() -> None:
    row = {"type": "verifier", "task": TASK, "reward": 0.5, "failed": False}
    assert verifier_reward(TASK, episode(row)) == 0.5


def test_the_scorer_resolves_from_its_dotted_reference() -> None:
    scorer = resolve_episode_scorer("reef.train.cordis_backend.strategies:verifier_reward")
    assert scorer(TASK, episode(verifier({"reward": 0.5}))) == 0.5


def test_the_terminus_row_carries_harbor_primary_reward(tmp_path: Path) -> None:
    record = trial_record(TASK, {"tests_total": 10, "tests_passed": 7, "reward": 0.7}, tmp_path)
    assert record["reward"] == 0.7 and not record["failed"]
    assert trial_record(TASK, {"accuracy": 1.0}, tmp_path)["reward"] == 1.0
    assert trial_record(TASK, {"a": 1, "b": 2}, tmp_path)["reward"] is None
    failed = trial_record(TASK, {}, tmp_path, "docker compose build failed")
    assert failed["failed"] and failed["reward"] is None and failed["error"] == "docker compose build failed"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (episode(verifier({}, failed=True)), "the verifier for '/tasks/openenv-00012-003-deduction' failed"),
        (episode(verifier({}, failed=True), exit_code=1), "failed"),
        (episode(exit_code=1), "exited 1 without a verifier record"),
        (episode(verifier({})), "wrote no reward"),
    ],
)
def test_the_required_reward_has_no_score_where_the_plain_reward_writes_zero(
    result: EpisodeResult, message: str
) -> None:
    assert verifier_reward(TASK, result) == 0.0
    with pytest.raises(ScoreUnavailable, match=message):
        required_verifier_reward(TASK, result)


def test_the_required_reward_scores_and_refuses_as_the_plain_reward_does() -> None:
    assert required_verifier_reward(TASK, episode(verifier({"reward": 0}))) == 0.0
    assert required_verifier_reward(TASK, episode(verifier({"tests_passed": 7, "reward": 0.7}))) == 0.7
    assert required_verifier_reward(TASK, episode(verifier({"accuracy": 1}))) == 1.0
    with pytest.raises(ValueError, match="names '/tasks/other'"):
        required_verifier_reward(TASK, episode(verifier({"reward": 1.0}, task="/tasks/other")))
    with pytest.raises(ValueError, match="must be a finite number"):
        required_verifier_reward(TASK, episode(verifier({"reward": float("nan")})))


def test_an_episode_without_a_reward_is_invalid_and_the_infrastructures_fault() -> None:
    """The worker catches the scorer's ``ScoreUnavailable``: the episode ran, so it is invalid, not an error."""
    worker = EpisodeEvaluationWorker(
        descriptor=get_adapter("terminus"),
        scorer=resolve_episode_scorer(required_verifier_reward),
        binary=None,
        timeout=10,
        executor=LocalExecutor(),
        forbid_residue=False,
    )
    scored = worker._score_result(episode(verifier({}), exit_code=1), TASK)
    assert scored.score is None and scored.label == "invalid" and scored.fault == "infrastructure"
    assert scored.failure is not None and scored.failure.stage == "score"
    assert scored.failure.cause == f"the verifier for {TASK!r} wrote no reward"
    rewarded = worker._score_result(episode(verifier({"reward": 1.0})), TASK)
    assert rewarded.score == 1.0 and rewarded.label == "valid" and rewarded.fault is None


def test_a_runner_that_exited_before_any_trial_is_the_harnesss_failure() -> None:
    """No verifier row and a nonzero exit: the terminus runner stopped before its trial, as when the tree cannot
    load. The current tree's own crash is then a candidate win, not a void pair."""
    worker = EpisodeEvaluationWorker(
        descriptor=get_adapter("terminus"),
        scorer=resolve_episode_scorer(required_verifier_reward),
        binary=None,
        timeout=10,
        executor=LocalExecutor(),
        forbid_residue=False,
    )
    crashed = worker._score_result(
        EpisodeResult(exit_code=1, stdout="", stderr="TerminusTreeError: no model", trajectory=(), residue=()), TASK
    )
    assert crashed.score is None and crashed.label == "execution_error" and crashed.fault == "harness"
    assert crashed.failure is not None and crashed.failure.stage == "exit"
    assert crashed.failure.cause == "exit 1: TerminusTreeError: no model"
