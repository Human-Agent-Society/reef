"""The tutorial's grader, shared by both variants and by ``run.py``.

The grader and its three arithmetic answers are this tutorial's own, since
its deployments keep those tasks. The grader stays dependency-free so
``run.py`` still needs only reef-client.
"""

ANSWERS = {
    "[sieve]": "9592",
    "[fib]": "2880067194370816120",
    "[csv]": "30",
}


def grade_text(task: str, text: str | None) -> float:
    """The shared grader: 1.0 when the last non-empty line is the expected
    answer for the task's prefix, else 0.0. ``run.py`` scores the recorded
    traffic with exactly this function."""
    expected = next((answer for prefix, answer in ANSWERS.items() if task.startswith(prefix)), None)
    if expected is None or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == expected else 0.0


def evaluate(task: str, result) -> float:
    """Grade the run's final assistant text against this tutorial's own answers."""
    from reef.harness.tasks.trajectory import final_assistant_text

    return grade_text(task, final_assistant_text(result.trajectory))
