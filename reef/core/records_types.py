"""Core record types: ``AgentRecord`` and ``RequestType``.

The stored entity is a record; its id keeps the wire spelling
(``agent_record_id``, header ``x-reef-agent-record-id``) — the "receipt" a
client quotes back.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reef.core.artifact_ref import ArtifactRef


class RequestType(str, Enum):
    INFERENCE = "inference"
    REPORT = "report"
    TRAIN = "train"


#: Delivery diagnostic retained by older records that stripped the abort finish reason.
INFERENCE_ABORT_ERROR = "upstream engine aborted the generation"


def inference_aborted(response: Mapping[str, object]) -> bool:
    """Whether a native, provider, or stored response reports an engine abort.

    Read finish reasons without requiring training tensors. Older stored
    Anthropic aborts kept only the stream-delivery error after stripping tensors.
    """
    delivery = response.get("stream_delivery")
    if isinstance(delivery, Mapping) and delivery.get("error") == INFERENCE_ABORT_ERROR:
        return True

    def aborted(value: object) -> bool:
        return value == "abort" or (isinstance(value, Mapping) and value.get("type") == "abort")

    meta = response.get("meta_info")
    if aborted(response.get("finish_reason")) or (isinstance(meta, Mapping) and aborted(meta.get("finish_reason"))):
        return True
    training = response.get("training")
    if isinstance(training, Mapping) and aborted(training.get("finish_reason")):
        return True
    choices = response.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, Mapping)
        and (
            aborted(choice.get("finish_reason"))
            or (isinstance(choice.get("meta_info"), Mapping) and aborted(choice["meta_info"].get("finish_reason")))
        )
        for choice in choices
    )


def parse_references(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("references", ())
    if not isinstance(value, (list, tuple)) or any(not isinstance(reference, str) for reference in value):
        raise ValueError("references must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class AgentRecord:
    agent_record_id: str
    scenario: str
    request_type: RequestType
    payload: Mapping[str, Any]
    created_at: float
    references: tuple[str, ...] = ()
    artifact_ref: ArtifactRef | None = None

    @classmethod
    def create(
        cls,
        *,
        scenario: str,
        request_type: RequestType,
        payload: Mapping[str, Any],
        agent_record_id: str | None = None,
        created_at: float | None = None,
        references: tuple[str, ...] | None = None,
        artifact_ref: ArtifactRef | None = None,
    ) -> AgentRecord:
        return cls(
            agent_record_id=agent_record_id or uuid.uuid4().hex,
            scenario=scenario,
            request_type=request_type,
            payload=dict(payload),
            created_at=time.time() if created_at is None else created_at,
            references=parse_references(payload) if references is None else references,
            artifact_ref=artifact_ref,
        )
