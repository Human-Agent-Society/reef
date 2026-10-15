"""Storage contract for service-local scheduler control operations and completed IDs."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.records_types import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest


@dataclass(frozen=True)
class PendingTrainingRequest:
    """An accepted instruction retained until its scenario's records own it."""

    item: AgentRecord
    generation: int

    @classmethod
    def from_value(cls, value: object, scenario_name: str) -> PendingTrainingRequest | None:
        if value is None:
            return None
        if not isinstance(value, dict) or not isinstance(value.get("item"), dict):
            raise ValueError("pending training request must contain a record")
        item = value["item"]
        payload = item.get("payload")
        record_id = item.get("agent_record_id")
        created_at = item.get("created_at")
        generation = value.get("generation")
        if (
            item.get("scenario") != scenario_name
            or item.get("request_type") != "train"
            or not isinstance(record_id, str)
            or not record_id.strip()
            or not isinstance(payload, dict)
            or any(not isinstance(key, str) for key in payload)
            or isinstance(created_at, bool)
            or not isinstance(created_at, int | float)
            or not math.isfinite(created_at)
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or not 1 <= generation <= 2**63 - 1
            or item.get("references") not in ([], ())
            or item.get("artifact_ref") is not None
        ):
            raise ValueError("invalid pending training request")
        TrainingRequest.from_dict(payload)
        return cls(
            AgentRecord.create(
                scenario=scenario_name,
                request_type=RequestType.TRAIN,
                payload=payload,
                agent_record_id=record_id,
                created_at=created_at,
                references=(),
            ),
            generation,
        )


class ScenarioAdmissionStore(ABC):
    """Persist control operations and retained instructions without scenario execution."""

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether saved identities survive service restart."""

    @abstractmethod
    def read(self, scenario_name: str) -> object:
        """Return a control record, or None when this name has never been assigned."""

    @abstractmethod
    def write(self, scenario_name: str, value: Mapping[str, object]) -> None:
        """Durably replace a control record before execution or release acknowledgement."""

    @abstractmethod
    def scenario_names(self) -> tuple[str, ...]:
        """Enumerate stored identities to recover pending instructions after restart."""
