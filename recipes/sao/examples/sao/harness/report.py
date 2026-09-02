"""Attach a Harbor trial's verifier reward to the Reef inferences that earned it.

The agent (``harness.agent``) records its rollouts through Reef and leaves the
receipts in the Harbor agent context. Harbor then runs the verifier and ends the
trial by writing ``result.json``; :func:`post_report` turns that trial result
into one Reef report — the verifier reward as its score, the trial's inference
receipts as its references.
"""

import uuid

from reef_client import ReefClient


def post_report(result: dict, *, client: ReefClient, scenario: str) -> dict:
    trial_id = result["id"]
    task_name = result["task_name"]
    score = result["verifier_result"]["rewards"]["reward"]
    payload = {
        # A report id derived from the trial makes a duplicate post a no-op on
        # Reef's side, not a second report about the same trial.
        "agent_record_id": uuid.uuid5(uuid.NAMESPACE_URL, f"reef:harbor:{trial_id}").hex,
        "score": score,
        "feedback": f"harbor verifier reward for {task_name}: {score}",
        "references": result["agent_result"]["metadata"]["reef"]["agent_record_ids"],
        "metadata": {"harbor": {"trial_id": trial_id, "task_name": task_name}},
    }
    return client.report(scenario, payload)
