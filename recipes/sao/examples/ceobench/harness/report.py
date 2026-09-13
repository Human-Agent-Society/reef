"""Attach a CEO-Bench trial's verifier reward to the Reef inferences that earned it.

The agent (``harness.agent``) routes every model call of one episode through
Reef and leaves the receipts, in order, in the Harbor agent context. Harbor
then runs the verifier (``harbor/tests/score.py``) and ends the trial by
writing ``result.json``; :func:`post_reports` turns that trial result into one
Reef report per receipt: the episode score as the report's score, that turn's
receipt as its only reference. One reference per report is what the ``sao``
recipe trains on; the episode-level score applied to every turn is the
reward-shaping choice the README documents.

A turn longer than the trainer's window (``max_tokens``, prompt and completion
together) is skipped: the engine served it, Reef recorded it, but the trainer
could not hold it, so it stays evaluation-only.
"""

import uuid

from reef_client import ReefClient


def post_reports(result: dict, *, client: ReefClient, scenario: str, max_tokens: int = 0) -> list[dict]:
    trial_id = result["id"]
    task_name = result["task_name"]
    rewards = result["verifier_result"]["rewards"]
    score = float(rewards["reward"])
    reef = result["agent_result"]["metadata"]["reef"]
    receipts = list(reef["agent_record_ids"])
    tokens = list(reef.get("agent_record_tokens") or [0] * len(receipts))
    feedback = (
        f"ceobench final cash {rewards.get('final_cash')} after {rewards.get('survival_days')} days"
        f" (bankrupt={rewards.get('bankrupt')}); score {score} over {len(receipts)} turns"
    )
    posted = []
    for turn, (receipt, turn_length) in enumerate(zip(receipts, tokens, strict=True)):
        if max_tokens and turn_length > max_tokens:
            continue
        payload = {
            # A report id derived from the trial and the receipt makes a
            # duplicate post a no-op on Reef's side, not a second report.
            "agent_record_id": uuid.uuid5(uuid.NAMESPACE_URL, f"reef:harbor:{trial_id}:{receipt}").hex,
            "score": score,
            "feedback": feedback,
            "references": [receipt],
            "metadata": {
                "harbor": {"trial_id": trial_id, "task_name": task_name},
                "ceobench": {"turn": turn, "turns": len(receipts), **rewards},
            },
        }
        posted.append(client.report(scenario, payload))
    return posted
