"""Native request-driven batching, independent of a recipe's automatic gates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.types import ProcessorContext, TrainingBatch


class ManualTrainingProcessor(DataProcessor, ABC):
    """One durable TRAIN record authorizes one batch containing its instruction.

    Implements the ``*_manual`` hooks on the same processor instance; automatic
    engines can be combined with this engine through ordinary inheritance.
    Ordinary traffic remains audit data. It never enters automatic batching
    or model-based feedback derivation. Only the instruction receipt is
    consumed by a committed manual step.
    """

    required_request_types = frozenset(RequestType)
    supported_training_modes = frozenset({"manual"})

    def __init__(self, context: ProcessorContext) -> None:
        if not context.config.get("manual_enabled", True):
            self.supported_training_modes = self.supported_training_modes - {"manual"}
        super().__init__(context)
        self._manual_audit_ids: set[str] = set()
        self._manual_requests: dict[str, AgentRecord] = {}
        self._manual_released: set[str] = set()

    @abstractmethod
    def make_request_batch(self, request: AgentRecord) -> TrainingBatch:
        """Shape one explicit instruction into the method's batch schema, without model calls."""

    def ingest_manual(self, item: AgentRecord) -> None:
        if item.scenario != self.scenario:
            raise ValueError("manual training records must belong to the processor's scenario")
        if item.request_type is RequestType.TRAIN:
            TrainingRequest.from_dict(item.payload)
            if item.agent_record_id not in self._manual_released:
                self._manual_requests.setdefault(item.agent_record_id, item)
        else:
            self._manual_audit_ids.add(item.agent_record_id)

    def ready_manual(self) -> bool:
        return bool(self._manual_requests)

    def build_batch_manual(self, batch_number: int) -> TrainingBatch:
        record = next(iter(self._manual_requests.values()))
        batch = self.make_request_batch(record)
        return replace(
            batch,
            batch_id=f"{self.scenario}:manual:{record.agent_record_id}",
            request=replace(TrainingRequest.from_dict(record.payload), id=record.agent_record_id),
        )

    def acknowledge_manual(self) -> frozenset[str]:
        receipt = next(iter(self._manual_requests))
        self._manual_requests.pop(receipt)
        self._manual_released.add(receipt)
        return frozenset({receipt})

    def retention_decision_manual(self) -> RetentionDecision:
        return RetentionDecision(
            protected_agent_record_ids=frozenset(self._manual_audit_ids) | frozenset(self._manual_requests),
            releasable_agent_record_ids=frozenset(self._manual_released),
        )

    def compaction_applied_manual(self, agent_record_ids: frozenset[str]) -> None:
        self._manual_released -= agent_record_ids

    def status(self) -> Mapping[str, Any]:
        if self.training_mode != "manual":
            return super().status()
        return {
            "training_mode": "manual",
            "buffered_requests": len(self._manual_requests),
            "retained_records": len(self._manual_audit_ids),
        }
