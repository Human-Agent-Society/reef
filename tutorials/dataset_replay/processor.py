"""A recipe-owned dataset cursor: N passes over a prefix, then live records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from reef.core.records_types import RequestType
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import make_trajectory
from reef.storage.records import RecordStore
from reef.train.processors import DataProcessor
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem


@dataclass(frozen=True)
class ReplayProgress:
    """Next read position; epoch == epochs + 1 means the continual stream."""

    dataset_last_sequence: int
    epochs: int
    epoch: int = 1
    after_sequence: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"replay {name} must be a nonnegative integer")
        if self.epochs < 1 or not 1 <= self.epoch <= self.epochs + 1:
            raise ValueError("replay epoch must be between 1 and epochs + 1")
        if self.epoch <= self.epochs and self.after_sequence > self.dataset_last_sequence:
            raise ValueError("dataset cursor exceeds the dataset boundary")
        if self.epoch > self.epochs and self.after_sequence < self.dataset_last_sequence:
            raise ValueError("stream cursor precedes the dataset boundary")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ReplayProgress:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid replay progress")
        return cls(**value)


@dataclass(frozen=True, kw_only=True)
class ReplayBatch(TrainingBatch):
    epoch: int
    phase: str
    next_progress: ReplayProgress


class ReplayProcessor(DataProcessor):
    """Pull at most one batch from storage instead of buffering the dataset.

    The recipe injects storage and the last committed cursor. Normal trainer
    ingestion is used only for optional TRAIN instructions. Record selection
    and repeated reads belong to this processor, not the trainer's forward cursor.
    """

    required_request_types = frozenset({RequestType.TRAIN})
    supported_training_modes = frozenset({"auto", "hybrid", "manual"})
    output_schema = ReplayBatch

    def __init__(self, context: ProcessorContext, *, records: RecordStore, progress: ReplayProgress) -> None:
        super().__init__(context)
        self.records = records
        self.progress = progress
        self.available: ReplayBatch | None = None

    def fill_batch(self) -> None:
        if self.available is not None:
            return
        progress = self.progress
        while True:
            is_dataset = progress.epoch <= progress.epochs
            sequence = progress.after_sequence
            items: list[TrajectoryItem] = []
            exhausted = False
            while len(items) < self._batch_size:
                page = self.records.replay_page(
                    self.scenario, after_sequence=sequence, limit=self._batch_size - len(items)
                )
                if not page:
                    exhausted = True
                    break
                for position, record in page:
                    if is_dataset and position > progress.dataset_last_sequence:
                        exhausted = True
                        break
                    sequence = position
                    if record.request_type is RequestType.INFERENCE:
                        items.append(make_trajectory((record,)))
                if exhausted or (is_dataset and sequence >= progress.dataset_last_sequence):
                    exhausted = True
                    break
            next_progress = replace(progress, after_sequence=sequence)
            if is_dataset and exhausted:
                next_epoch = progress.epoch + 1
                next_progress = replace(
                    progress,
                    epoch=next_epoch,
                    after_sequence=0 if next_epoch <= progress.epochs else progress.dataset_last_sequence,
                )
            if items:
                self.available = ReplayBatch(
                    batch_id=f"{self.scenario}:epoch:{progress.epoch}:through:{sequence}",
                    items=tuple(items),
                    epoch=progress.epoch,
                    phase="dataset" if is_dataset else "stream",
                    next_progress=next_progress,
                )
                return
            # Empty/evicted ranges need no training commit. A later batch saves
            # this cursor; after a crash scanning those ranges again is harmless.
            progress = next_progress
            self.progress = progress
            if not is_dataset:
                return

    def ready(self) -> bool:
        if self._pending is not None:
            return True
        if self.training_mode == "manual":
            return bool(self._training_requests)
        self.fill_batch()
        return self.available is not None or (self.training_mode == "hybrid" and bool(self._training_requests))

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> ReplayBatch:
        self.fill_batch()
        if self.available is not None:
            return self.available
        if request is None:
            raise RuntimeError("no records available")
        return ReplayBatch(
            batch_id=f"{self.scenario}:empty:{batch_number}",
            epoch=self.progress.epoch,
            phase="instruction",
            next_progress=self.progress,
        )

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        if not isinstance(self._pending, ReplayBatch) or self._pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        next_progress = self._pending.next_progress
        consumed = super().acknowledge(batch_id)
        self.progress = next_progress
        self.available = None
        return consumed

    def status(self) -> Mapping[str, object]:
        return {
            **self.progress.to_dict(),
            "buffered_records": 0 if self.available is None else len(self.available.items),
        }
