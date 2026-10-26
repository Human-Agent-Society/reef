"""The score report of one serving episode, against its ordered inference receipts."""

import uuid
from urllib.parse import quote

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


def episode_runtime_load_id(model: ReefLLM) -> str:
    """Require one engine weight version across all recorded turns of a task."""
    versions: set[str] = set()
    for receipt in model.receipts:
        record = model.reef.get(f"/reef/scenarios/{quote(model.scenario, safe='')}/records/{quote(receipt, safe='')}")
        version = record["payload"].get("runtime_load_id")
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"Receipt {receipt} has no single producing weight version")
        versions.add(version)
    if len(versions) != 1:
        raise RuntimeError("An episode must use exactly one producing weight version")
    return versions.pop()


def report_episode(model: ReefLLM, run_id: str, task_id: str, reward: float, end_reason: str) -> dict:
    """Submit one scalar outcome against all ordered inference receipts.

    This function accepts serving episodes only through the campaign driver.
    The report excludes hidden rules, verifier details, and category labels.
    """
    if not model.receipts or not 0.0 <= reward <= 1.0:
        raise ValueError("A report requires inference receipts and a reward in [0, 1]")
    report_id = uuid.uuid5(uuid.NAMESPACE_URL, f"servelearnbench:{run_id}:{task_id}").hex
    return model.reef.report(
        model.scenario,
        {
            "agent_record_id": report_id,
            "score": reward,
            "feedback": protocol.score_line(reward),
            "references": list(model.receipts),
            "metadata": {"task": {"name": task_id}, "end_reason": end_reason},
        },
    )
