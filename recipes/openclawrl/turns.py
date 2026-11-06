"""What one turn's judgment gets and answers.

:class:`TurnJob` is everything one turn's judgment needs;
:class:`TurnJudgment` is what the judge answers. The worker that runs
judgments lives on the shared engine
(:class:`reef.execution.jobs.BackgroundJobWorker`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.openclawrl.sessions import Binding
from reef.core.records_types import AgentRecord
from reef.core.trajectories import exchange_messages
from reef.execution.jobs import SupportsReceipt


@dataclass(frozen=True)
class TurnJob(SupportsReceipt):
    """Everything one turn's judgment needs, lifted off its record."""

    receipt: str
    request_messages: tuple[Any, ...]
    request_tools: Any
    response_message: Mapping[str, Any]
    next_state_text: str
    next_state_role: str


@dataclass(frozen=True)
class TurnJudgment(SupportsReceipt):
    """One judged turn: a trainable score with teacher candidates, or terminal.

    Each teacher candidate is ``{"hint", "messages"}``: an accepted hint and the hint-enhanced request that the
    teacher reads, or ``{"hint": "", "messages": None}``, the un-enhanced anchor, whose teacher reads the student's
    own sequence. The training component renders each with the model's chat template.
    """

    receipt: str
    #: ``None`` is a declined judgment: the turn never trains.
    score: float | None = None
    teacher_cands: tuple[Mapping[str, Any], ...] = ()


def _declared_turn_type(payload: Any) -> str | None:
    """The harness-declared ``x-reef-tag-turn-type``, if any.

    This is upstream's own split: it stamps ``turn_type`` from the agent's
    trigger, and its extension marks only ``{heartbeat, memory, cron}`` as
    side — so every iteration of a tool loop is main and is its own training
    datum. A harness that can declare per request should, and this honours it.
    """
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    value = tags.get("turn-type") if isinstance(tags, dict) else None
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def main_turn_message(payload: Any) -> Any:
    """The assistant message of a main turn, else ``None``.

    Without a declaration, the presence of tool schemas stands in: an agent
    loop ships its toolset, housekeeping calls (titles, compaction summaries)
    do not. That is reef's heuristic for harnesses that cannot declare, NOT
    upstream's rule — upstream would train a user-triggered call that shipped
    no toolset, and would skip a housekeeping call that shipped one.
    """
    declared = _declared_turn_type(payload)
    if declared == "side":
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        return None
    if declared != "main" and not payload.get("tools"):
        return None
    _, replies = exchange_messages(payload)
    message = replies[-1] if replies else None
    return dict(message) if isinstance(message, dict) else None


def turn_request_messages(payload: Any) -> list[Any]:
    """The request messages of a turn, as the provider received them."""
    return list(exchange_messages(payload)[0]) if isinstance(payload, dict) else []


def turn_job(record: AgentRecord, binding: Binding) -> TurnJob:
    payload = record.payload
    return TurnJob(
        receipt=binding.receipt,
        request_messages=tuple(turn_request_messages(payload)),
        request_tools=payload.get("tools"),
        response_message=main_turn_message(payload) or {},
        next_state_text=binding.next_state_text,
        next_state_role=binding.next_state_role,
    )
