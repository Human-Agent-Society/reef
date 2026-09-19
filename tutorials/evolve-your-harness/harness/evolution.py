"""Compatibility entrypoints for the evolve-your-harness tutorial.

Proposals and the reading of an episode's final answer live in the built-in
Reefine recipe; the grader and its three arithmetic answers are this
tutorial's own, since its deployments keep those tasks. The small client
grader stays dependency-free so run.py still needs only reef-client.
"""

import json
import re

ANSWERS = {
    "[sieve]": "9592",
    "[fib]": "2880067194370816120",
    "[csv]": "30",
}


_ENTRY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


API_SKILL_NAME = "reef-pi-extension-api"


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


def propose(nodes, samples, models, *, requests=(), entries=()):
    """Delegate service-side proposals to the built-in recipe, the tree's entries included."""
    from reef.recipe.reefine.evolution import propose as reefine_propose

    return reefine_propose(nodes, samples, models, requests=requests, entries=entries)


def evaluate(task: str, result) -> float:
    """Grade the episode's final assistant text, read as the built-in recipe reads it, against this tutorial's
    own answers."""
    from reef.recipe.reefine.evolution import final_assistant_text

    return grade_text(task, final_assistant_text(result.trajectory))
