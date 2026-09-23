"""Bounded, repeatable consumption of self-contained records from storage."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace

from reef.core.records_types import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.storage.commits import CommitRecord
from reef.storage.records import RecordStore
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch


class DatasetProcessor(DataProcessor, ABC):
    """Train each storage snapshot for K passes without retaining the dataset.

    Recipes assemble one self-contained INFERENCE record with ``make_sample``.
    The snapshot is the storage tail when consumption starts; later appends
    form the next snapshot. Only the current batch is assembled in memory.
    Report correlation and cross-record grouping belong to other processors.
    """

    required_request_types = frozenset({RequestType.INFERENCE, RequestType.TRAIN})
    supported_training_modes = frozenset({"auto", "hybrid"})

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        epochs = context.config.get("dataset_epochs", 1)
        batch_bytes = context.config.get("dataset_batch_bytes", 64 * 1024**2)
        for name, value in (("dataset_epochs", epochs), ("dataset_batch_bytes", batch_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.epochs: int = epochs
        self.batch_bytes_limit: int = batch_bytes
        self.start_sequence = 0
        self.end_sequence = 0
        self.epoch = 1
        self.cursor = 0
        self.batch_end_sequence = 0
        self.samples: list[TrainDataItem] = []
        self.batch_bytes = 0
        self.batch_complete = False
        self.released_records: set[str] = set()
        self.committed_epoch = 1

    @abstractmethod
    def make_sample(self, record: AgentRecord) -> TrainDataItem:
        """Assemble one persisted record without caching it or its sample."""

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        """Shape the current batch; recipes may override its concrete output type."""
        return TrainingBatch(f"{self.scenario}:dataset:{batch_number}", items)

    def set_training_mode(self, training_mode: str) -> None:
        if self.end_sequence and training_mode != self.training_mode:
            raise ValueError("select dataset training_mode before consumption begins")
        super().set_training_mode(training_mode)

    def consume(self, records: RecordStore, *, after_sequence: int, offset: int) -> tuple[int, int]:
        if self.ready():
            return after_sequence, offset
        if self.epoch > self.epochs or self.end_sequence == 0:
            end_sequence = records.latest_sequence(self.scenario)
            if end_sequence <= self.end_sequence:
                return after_sequence, offset
            self.start_sequence = self.end_sequence
            self.end_sequence = end_sequence
            self.cursor = self.start_sequence
            self.batch_end_sequence = self.cursor
            self.epoch = 1

        while True:
            while len(self.samples) < self._batch_size:
                # One raw record at a time: a page of 256 large trajectories can
                # itself exhaust memory before the processor applies its budget.
                page = records.replay_page(self.scenario, after_sequence=self.batch_end_sequence, limit=1)
                if not page or page[0][0] > self.end_sequence:
                    self.batch_end_sequence = self.end_sequence
                    break
                sequence, record = page[0]
                if record.request_type is RequestType.INFERENCE:
                    payload_bytes = len(json.dumps(record.payload, ensure_ascii=False).encode("utf-8"))
                    if self.samples and self.batch_bytes + payload_bytes > self.batch_bytes_limit:
                        break
                    sample = self.make_sample(record)
                    if sample.source_agent_record_ids not in ((), (record.agent_record_id,)):
                        raise ValueError("dataset samples must be self-contained within their input record")
                    self.samples.append(replace(sample, source_agent_record_ids=(record.agent_record_id,)))
                    self.batch_bytes += payload_bytes
                elif record.request_type is RequestType.TRAIN and self.training_mode == "hybrid":
                    if self._training_requests:
                        break
                    self.ingest(record)
                self.batch_end_sequence = sequence
                if sequence > after_sequence:
                    after_sequence = sequence
                    offset += 1
                if self.batch_bytes >= self.batch_bytes_limit or sequence == self.end_sequence:
                    break

            self.batch_complete = True
            if self.samples or self._training_requests:
                return after_sequence, offset
            self.advance_cursor()
            if self.epoch > self.epochs:
                return after_sequence, offset

    def ready(self) -> bool:
        return self._pending is not None or (self.batch_complete and bool(self.samples or self._training_requests))

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        batch = self.make_batch(tuple(self.samples), batch_number)
        return replace(batch, batch_id=f"{self.scenario}:dataset:{self.end_sequence}:{self.epoch}:{self.cursor}")

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        consumed = super().acknowledge(batch_id)
        record_ids = frozenset(record_id for sample in self.samples for record_id in sample.source_agent_record_ids)
        self.committed_epoch = self.epoch
        if self.epoch == self.epochs:
            self.released_records.update(record_ids)
        self.advance_cursor()
        self.samples.clear()
        self.batch_bytes = 0
        self.batch_complete = False
        return consumed | record_ids

    def advance_cursor(self) -> None:
        self.cursor = self.batch_end_sequence
        if self.cursor == self.end_sequence:
            self.epoch += 1
            if self.epoch <= self.epochs:
                self.cursor = self.start_sequence
        self.batch_end_sequence = self.cursor

    def discard_request(self, request_id: str) -> frozenset[str]:
        self.committed_epoch = self.epoch
        # Keep the read cursor unchanged: only the failed instruction is
        # consumed. Its durable compaction makes replay skip it after restart.
        return super().discard_request(request_id)

    def dropped(self, batch_id: str) -> None:
        raise ValueError("dataset consumption requires a commit; a stale batch cannot be discarded")

    def retention_decision(self) -> RetentionDecision:
        # Every other input stays in storage by default: only final-pass
        # records and completed instructions are explicitly releasable.
        return RetentionDecision(
            protected_agent_record_ids=frozenset(self._training_requests),
            releasable_agent_record_ids=frozenset(self.released_records | self._consumed_requests),
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        super().compaction_applied(agent_record_ids)
        self.released_records -= agent_record_ids

    def consumption_metrics(self) -> Mapping[str, object]:
        return {
            "dataset_epoch": self.committed_epoch,
            "dataset_state": {
                "epochs": self.epochs,
                "start_sequence": self.start_sequence,
                "end_sequence": self.end_sequence,
                "epoch": self.epoch,
                "cursor": self.cursor,
            },
        }

    def restore_consumption(self, commits: Sequence[CommitRecord]) -> bool:
        for commit in reversed(commits):
            metrics = commit.metrics or {}
            if "dataset_state" not in metrics:
                if commit.consumed_ids:
                    raise ValueError("committed training is missing dataset consumption state; use a new scenario")
                continue
            state = metrics["dataset_state"]
            names = ("epochs", "start_sequence", "end_sequence", "epoch", "cursor")
            if not isinstance(state, Mapping) or any(
                isinstance(state.get(name), bool) or not isinstance(state.get(name), int) for name in names
            ):
                raise ValueError("invalid dataset consumption state")
            if state["epochs"] != self.epochs:
                raise ValueError("dataset_epochs must match the committed configuration")
            if not (
                0 <= state["start_sequence"] <= state["cursor"] <= state["end_sequence"]
                and 1 <= state["epoch"] <= self.epochs + 1
            ):
                raise ValueError("invalid dataset consumption cursor")
            self.start_sequence = state["start_sequence"]
            self.end_sequence = state["end_sequence"]
            self.epoch = state["epoch"]
            self.cursor = state["cursor"]
            self.batch_end_sequence = self.cursor
            self.committed_epoch = metrics["dataset_epoch"]
            break
        return True
