"""Post one week of a CEO-Bench episode to Reef, one report per turn.

The agent (``harness.agent``) routes every model call of an episode through
Reef and knows, from the dashboard each request carries, which simulated week
a call belongs to and the cash the week started with. When the next week's
dashboard appears the week is over, and :func:`post_week_reports` turns it
into one Reef report per turn: the week's cash change over the starting
balance as the score, that turn's receipt as the only reference. One
reference per report is what the ``sao`` recipe trains on. The last week of
an episode ends with the verifier's final cash instead of a next dashboard.

A turn longer than the trainer's window (``max_tokens``, prompt and
completion together) is skipped: the engine served it and Reef recorded it,
but the trainer could not hold it, so it stays evaluation-only.
"""

import uuid
from collections.abc import Sequence

from reef_client import ReefClient

#: The benchmark's starting balance (the runner's ``--cash`` default).
INITIAL_CASH = 1_000_000.0


def week_score(cash_start: float, cash_end: float, initial_cash: float = INITIAL_CASH) -> float:
    """A week's cash change in units of the starting balance."""
    return (cash_end - cash_start) / initial_cash


def post_week_reports(
    client: ReefClient,
    scenario: str,
    *,
    week: int,
    day: int,
    cash_start: float,
    cash_end: float,
    turns: Sequence[tuple[str, int]],
    max_tokens: int = 0,
) -> list[dict]:
    """Report one finished week against each of its turns' receipts.

    ``turns`` are ``(receipt, tokens)`` pairs in call order.
    """
    score = week_score(cash_start, cash_end)
    feedback = (
        f"ceobench week {week} (from day {day}): cash {cash_start:.0f} -> {cash_end:.0f},"
        f" score {score:.4f} over {len(turns)} turns"
    )
    posted = []
    for index, (receipt, tokens) in enumerate(turns):
        if max_tokens and tokens > max_tokens:
            continue
        payload = {
            # A report id derived from the receipt makes a duplicate post a
            # no-op on Reef's side, not a second report about the same turn.
            "agent_record_id": uuid.uuid5(uuid.NAMESPACE_URL, f"reef:ceobench:{receipt}").hex,
            "score": score,
            "feedback": feedback,
            "references": [receipt],
            "metadata": {
                "ceobench": {
                    "week": week,
                    "day": day,
                    "cash_start": cash_start,
                    "cash_end": cash_end,
                    "turn": index,
                    "turns": len(turns),
                }
            },
        }
        posted.append(client.report(scenario, payload))
    return posted
