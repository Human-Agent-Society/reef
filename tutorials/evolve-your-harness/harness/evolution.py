"""The tutorial's grader, shared by both variants and by ``run.py``.

Proposals on pi come from the built-in Reefine text proposer, which the serve
file names; the grader and its three arithmetic answers are this tutorial's
own, since its deployments keep those tasks. The grader stays dependency-free
so ``run.py`` still needs only reef-client.
"""

import json

ANSWERS = {
    "[sieve]": "9592",
    "[fib]": "2880067194370816120",
    "[csv]": "30",
}


def failures_text(samples):
    """The failing samples as the proposer reads them: one object per sample with the request as served, the
    score its report gave and the report's feedback verbatim (``null`` when the report carried none)."""
    from reef.core.trajectories import recorded_payload

    views = [
        {
            "request": recorded_payload(sample),
            "score": sample.metadata.get("reward"),
            "feedback": sample.metadata.get("feedback"),
        }
        for sample in samples
    ]
    return json.dumps(views, indent=2, default=str)


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
    """Grade the episode's final assistant text against this tutorial's own answers."""
    from reef.harness.episodes.trajectory import final_assistant_text

    return grade_text(task, final_assistant_text(result.trajectory))
