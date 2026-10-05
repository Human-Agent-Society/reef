"""The Reefine profile's health scorer; this alone does not verify a request."""

from pathlib import Path
from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import final_assistant_text
from reef.train.cordis_backend.strategies import verifier_reward

HEALTH_TASK_DIRECTORY = str(Path(__file__).parents[2] / "recipe" / "reefine" / "health")


def evaluate(task: str, result: EpisodeResult) -> float:
    if task == HEALTH_TASK_DIRECTORY:
        return verifier_reward(task, result)
    return grade_text(task, final_assistant_text(result.trajectory))


def grade_text(task: str, text: str | None) -> float:
    if not task.startswith("[health]") or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == "reef-ok" else 0.0
