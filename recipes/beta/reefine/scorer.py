"""Reefine's run scorer: the serve config's health task, graded by its answer.

The health task checks that a tree still works (the model binding, the tools,
the extensions load), not that a requested change does. An instruction task
scores 1.0 when the last line of the final reply is the task's answer; the
health task directory, which an adapter that runs Harbor task directories
runs in the instruction's place, scores its verifier's reward.
"""

from __future__ import annotations

from pathlib import Path

from reef.evaluation.harness import EvaluationTask, TaskScorer, verifier_reward
from reef.harness.tasks.run import TaskResult
from reef.harness.tasks.trajectory import final_assistant_text

#: The id of the example serve config's health task.
HEALTH_TASK_ID = "health"

#: The health task as a Harbor task directory, for an adapter that runs a task directory rather than an
#: instruction (terminus): the same check, scored by its verifier's reward instead of the reply's last line.
HEALTH_TASK_DIRECTORY = str(Path(__file__).with_name("health"))


def grade_text(answer: str | None, text: str | None) -> float:
    """1.0 when the last non-empty line of ``text`` is ``answer``, else 0.0; a task without an answer scores 0."""
    if answer is None or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == answer else 0.0


class AnswerScorer(TaskScorer):
    """Score a run: a task directory's verifier reward, else ``grade_text`` of the task's answer."""

    def score(self, task: EvaluationTask, result: TaskResult) -> float:
        if task.directory is not None:
            return verifier_reward(task, result)
        return grade_text(task.answer, final_assistant_text(result.trajectory))


__all__ = ["HEALTH_TASK_DIRECTORY", "HEALTH_TASK_ID", "AnswerScorer", "grade_text"]
