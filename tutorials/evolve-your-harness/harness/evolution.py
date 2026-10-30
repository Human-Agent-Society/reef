"""The tutorial's grader, shared by both variants and by ``run.py``.

Each task in the serve files carries its own answer, which the step's
evaluation passes to the scorer with the task and ``run.py`` reads from
``work/tasks.json``. The grader stays dependency-free so ``run.py`` still
needs only reef-client.
"""


def grade_text(answer: str | None, text: str | None) -> float:
    """The shared grader: 1.0 when the last non-empty line is ``answer``, else
    0.0. ``run.py`` scores the recorded traffic with exactly this function."""
    if answer is None or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == answer else 0.0


def evaluate(task, result) -> float:
    """Grade the run's final assistant text against the task's answer."""
    from reef.harness.tasks.trajectory import final_assistant_text

    return grade_text(task.answer, final_assistant_text(result.trajectory))
