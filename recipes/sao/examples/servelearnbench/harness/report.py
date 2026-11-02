"""The score report of one serving episode, against the episode's last inference receipt."""

import uuid

from servelearnbench import protocol

from .agent import ReefLLM


def episode_end_reason(trace: list[dict]) -> str:
    """Return the official terminal event, independently of per-turn finish_reason."""
    if not trace:
        raise ValueError("An episode must contain a trace event")
    last = trace[-1]
    if last["kind"] == "finish":
        return "finish"
    if last["name"] in ("(step_cap)", "(token_cap)", "(context_overflow)"):
        return last["name"][1:-1]
    raise ValueError(f"Unrecognized official episode ending: {last['kind']}/{last['name']}")


def episode_artifact_id(model: ReefLLM) -> str:
    """Require one served artifact across all recorded calls of a task."""
    artifacts = set(model.artifact_ids)
    if len(artifacts) != 1:
        raise RuntimeError("An episode must use exactly one served artifact")
    return artifacts.pop()


def report_id(run_id: str, task_id: str) -> str:
    """The report ID of a task in a run; the same task reports the same ID again on a retry."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"servelearnbench:{run_id}:{task_id}").hex


def report_episode(model: ReefLLM, run_id: str, task_id: str, reward: float, end_reason: str) -> dict:
    """Submit one scalar outcome against the episode's last inference receipt.

    Reef links the episode's calls by their shared conversation prefix, so the
    report names only the last call and trains every reply of the episode.
    This function accepts serving episodes only through the campaign driver.
    The report excludes hidden rules, verifier details, and category labels.
    """
    if not model.receipts or not 0.0 <= reward <= 1.0:
        raise ValueError("A report requires inference receipts and a reward in [0, 1]")
    return model.reef.report(
        model.scenario,
        {
            "agent_record_id": report_id(run_id, task_id),
            "score": reward,
            "feedback": protocol.score_line(reward),
            "references": [model.receipts[-1]],
            "metadata": {"task": {"name": task_id}, "end_reason": end_reason, "calls": len(model.receipts)},
        },
    )
