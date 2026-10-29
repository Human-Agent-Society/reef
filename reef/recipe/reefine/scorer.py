"""Reefine's rollout scorer: the profile's health task, graded by its expected answer.

The health task checks that a tree still works (the model binding, the tools,
the extensions load), not that a requested change does. A prompt task scores
1.0 when the last line of the final reply is the expected answer for the
task's prefix; the health task directory, which an adapter that runs a Harbor
task directory runs in the prompt's place, scores its verifier's reward.
"""

from __future__ import annotations

from pathlib import Path

from reef.evaluation.harness import RolloutScorer, verifier_reward
from reef.harness.rollouts.run import RolloutResult
from reef.harness.rollouts.trajectory import final_assistant_text

#: Expected final answers, keyed by the stable prefix each task starts with (the reefine profile's tasks).
ANSWERS = {
    "[health]": "reef-ok",
}

#: The health task as a Harbor task directory, for an adapter that runs a task directory rather than a prompt
#: (terminus): the same check, scored by its verifier's reward instead of the reply's last line.
HEALTH_TASK_DIRECTORY = str(Path(__file__).with_name("health"))


def grade_text(task: str, text: str | None) -> float:
    """1.0 when the last non-empty line of ``text`` is the expected answer for ``task``'s prefix, else 0.0."""
    expected = next((answer for prefix, answer in ANSWERS.items() if task.startswith(prefix)), None)
    if expected is None or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == expected else 0.0


class AnswerScorer(RolloutScorer):
    """Score a rollout of the health task: its verifier's reward for the task directory, else ``grade_text``."""

    def score(self, task: str, result: RolloutResult) -> float:
        if task == HEALTH_TASK_DIRECTORY:
            return verifier_reward(task, result)
        return grade_text(task, final_assistant_text(result.trajectory))


__all__ = ["ANSWERS", "HEALTH_TASK_DIRECTORY", "AnswerScorer", "grade_text"]
