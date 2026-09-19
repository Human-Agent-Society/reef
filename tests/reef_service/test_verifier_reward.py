"""The shipped gate scorer for task directory episodes: the Harbor verifier's reward from the terminus row."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import primary_reward
from reef.harness.runners.terminus.runner import trial_record
from reef.train.cordis_backend.strategies import resolve_episode_scorer, verifier_reward

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
