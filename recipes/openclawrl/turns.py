"""What one turn's judgment gets and answers, and its teacher-row fixup.

:class:`TurnJob` is everything one turn's judgment needs;
:class:`TurnJudgment` is what the judge answers. The worker that runs
judgments lives on the shared engine
(:class:`reef.train.processors.computed.JudgingWorker`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.openclawrl.sessions import Binding
from reef.core.records_types import AgentRecord
from reef.train.types import PolicySample


@dataclass(frozen=True)
class TurnJob:
    """Everything one turn's judgment needs, lifted off its record."""

    receipt: str
    request_messages: tuple[Any, ...]
    request_tools: Any
    response_message: Mapping[str, Any]
    next_state_text: str
    next_state_role: str
    topk_indices: list[list[int]] | None = None
    native_tokens: list[int] | None = None
    response_token_count: int | None = None


@dataclass(frozen=True)
class TurnJudgment:
    """One judged turn: a trainable score with teacher rows, or terminal."""

    receipt: str
    #: ``None`` is a declined judgment: the turn never trains.
    score: float | None = None
    teacher_cands: tuple[Mapping[str, Any], ...] = ()


def validate_teacher_cands(cands: tuple[Any, ...], sample: PolicySample) -> tuple[dict[str, Any], ...] | None:
    """Normalize candidate token sequences for the Megatron teacher pass.

    Alignment is exact by construction — every candidate ends with the
    native response ids verbatim, so the teacher's response positions are
    the sample's own (upstream materializes ``prompt_ids + response_ids``
    the same way). What is checked is that construction: a candidate whose
    tail is not the native response, or that carries no prompt at all,
    would silently gather the wrong positions, and the sample must not
    train on it.
    """
    response_length = len(sample.loss_mask)
    native_tail = [int(v) for v in sample.tokens[-response_length:]]
    validated: list[dict[str, Any]] = []
    for cand in cands:
        tokens = cand.get("teacher_tokens")
        if not isinstance(tokens, list | tuple) or len(tokens) <= response_length:
            return None
        ids = [int(v) for v in tokens]
        if ids[-response_length:] != native_tail:
            return None
        validated.append({"hint": str(cand.get("hint", "")), "teacher_tokens": ids})
    return tuple(validated)


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
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    training = response.get("training")
    if isinstance(training, dict) and training.get("request_tools"):
        message = training.get("response_message")
        return message if isinstance(message, dict) else None
    if declared != "main" and not payload.get("tools"):
        return None
    choices = response.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else None
    return message if isinstance(message, dict) else None


def turn_request_messages(payload: Any) -> list[Any]:
    """Provider-neutral request messages retained by the rollout backend."""
    response = payload.get("response") if isinstance(payload, dict) else None
    training = response.get("training") if isinstance(response, dict) else None
    messages = training.get("request_messages") if isinstance(training, dict) else None
    return list(messages) if isinstance(messages, list) else list(payload.get("messages") or [])


def turn_job(record: AgentRecord, binding: Binding) -> TurnJob:
    payload = record.payload
    response = payload.get("response") or {}
    training = response.get("training")
    training = training if isinstance(training, dict) else {}
    request_messages = turn_request_messages(payload)
    request_tools = training.get("request_tools", payload.get("tools"))
    return TurnJob(
        receipt=binding.receipt,
        request_messages=tuple(request_messages),
        request_tools=request_tools,
        response_message=main_turn_message(payload) or {},
        next_state_text=binding.next_state_text,
        next_state_role=binding.next_state_role,
        topk_indices=training.get("topk_indices"),
        native_tokens=training.get("tokens"),
        response_token_count=training.get("response_length"),
    )
