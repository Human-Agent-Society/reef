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
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class RecordLoss:
    """Durable capacity-eviction totals for one scenario's stored records or record items."""

    record_count: int = 0
    body_bytes: int = 0
    first_sequence: int = 0
    last_sequence: int = 0


class RequestType(StrEnum):
    INFERENCE = "inference"
    REPORT = "report"


#: The request header that marks an inference call of a rollout that Reef runs itself. The service
#: serves the call from the scenario's snapshot as it serves any other, keeps no record of it, and returns the
#: response with its training block. Such a rollout is not user data, so it never becomes a record.
ROLLOUT_HEADER = "x-reef-rollout"


def parse_references(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("references", ())
    if not isinstance(value, (list, tuple)) or any(not isinstance(reference, str) for reference in value):
        raise ValueError("references must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class AgentRecord:
    """One inference or report, with its receipt, payload, time and referenced receipts.

    ``artifact_id`` is the nonempty opaque ID of the artifact that answered,
    or None when no artifact is known. Stores persist this value; training and
    record exports read it without resolving a storage snapshot.
    """

    agent_record_id: str
    scenario: str
    request_type: RequestType
    payload: Mapping[str, Any]
    created_at: float
    references: tuple[str, ...] = ()
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if self.artifact_id is not None and (not isinstance(self.artifact_id, str) or not self.artifact_id):
            raise ValueError("a record artifact_id must be a nonempty string or null")

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
        artifact_id: str | None = None,
    ) -> AgentRecord:
        return cls(
            agent_record_id=agent_record_id or uuid.uuid4().hex,
            scenario=scenario,
            request_type=request_type,
            payload=dict(payload),
            created_at=time.time() if created_at is None else created_at,
            references=parse_references(payload) if references is None else references,
            artifact_id=artifact_id,
        )


def decode_record_artifact_id(value: object, payload: Mapping[str, object]) -> str | None:
    """Read a record's artifact ID: ``value``, or the ``artifact_id`` of ``payload`` when ``value`` is None."""
    identity = payload.get("artifact_id") if value is None else value
    if identity is not None and (not isinstance(identity, str) or not identity):
        raise ValueError("a record artifact_id must be a nonempty string or null")
    return identity


def stored_record_artifact_id(record: Mapping[str, object], payload: Mapping[str, object]) -> str | None:
    """The artifact ID of a stored or uploaded record mapping ``record`` with ``payload``."""
    return decode_record_artifact_id(record.get("artifact_id"), payload)
