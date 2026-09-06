"""Optional composition for processors with separate implementations per mode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from reef.core import AgentRecord
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.types import ProcessorContext, TrainingBatch


class ModeDataProcessor(DataProcessor):
    """Delegate the complete lifecycle to the implementation selected by the processor.

    A recipe processor declares ``mode_processors``; an absent mode raises
    ``NotImplementedError`` at construction. This helper is optional: a
    processor can instead implement multiple modes in its own methods.
    Trainer sees the same DataProcessor interface in either case.
    """

    mode_processors: ClassVar[Mapping[str, type[DataProcessor]]] = {}

    def __init__(self, context: ProcessorContext) -> None:
        self.supported_training_modes = frozenset(self.mode_processors)
        super().__init__(context)
        self._implementation = self.mode_processors[context.training_mode](context)
        if self._implementation.training_mode != context.training_mode:
            self._implementation.close()
            raise ValueError("mode implementation must preserve the requested training_mode")
        self.output_schema = self._implementation.output_schema
        self.required_request_types = self._implementation.required_request_types

    def ingest(self, item: AgentRecord) -> None:
        self._implementation.ingest(item)

    def ready(self) -> bool:
        return self._implementation.ready()

    def build_batch(self) -> TrainingBatch:
        return self._implementation.build_batch()

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        return self._implementation.acknowledge(batch_id)

    def retention_decision(self) -> RetentionDecision:
        return self._implementation.retention_decision()

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        self._implementation.compaction_applied(agent_record_ids)

    def derivation_pending(self) -> bool:
        return self._implementation.derivation_pending()

    def status(self) -> Mapping[str, Any]:
        return self._implementation.status()

    def close(self) -> None:
        self._implementation.close()

    def prepare_reconfiguration(self, context: ProcessorContext) -> DataProcessor:
        return type(self)(context)

    def bind_config_revision(self, revision: int) -> None:
        super().bind_config_revision(revision)
        self._implementation.bind_config_revision(revision)
