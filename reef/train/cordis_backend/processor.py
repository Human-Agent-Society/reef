"""Harness-evolution processor: pair recorded requests with reported scores, unmodified."""

from __future__ import annotations

from reef.core import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import make_trajectory
from reef.train.experience import ArrivalOrder, ExperienceBuffer, ExperienceUnit, SelectionPolicy
from reef.train.processors.base import DataProcessor
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, reported_task
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class CordisProcessor(ReportedFeedbackProcessor):
    """Pair recorded requests with reported scores and batch them unmodified.

    Requests are recorded post-transform, so a trace shows exactly what the
    backend served. Every valid report contributes a trace. A report may reference
    one request or a whole run's worth; several references become one
    trajectory sample. The backends consume
    the resulting trace batches without adding processor logic. In ``hybrid``
    a queued instruction batches with the reported traces an automatic batch
    would take next, up to ``batch_size``, so the proposer reads the request
    beside them; in ``manual`` it runs alone.
    """

    output_schema = TrainingBatch
    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        if request is not None and self.training_mode == "manual":
            self.experience_buffer.reserve(0)
            return TrainingBatch(request.id, ())
        # In hybrid an instruction takes the units an automatic batch would, none included; the base attaches it.
        return self._make_pending(batch_number)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = make_trajectory(context.inferences, context.require_score(), context.report.payload.get("feedback"))
        task = reported_task(context.report.payload.get("metadata"))
        return sample if task is None else sample.with_metadata(task=task)

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:harness_evolve:{batch_number}", items)


class RecordDrivenTraceProcessor(DataProcessor):
    """Batch recorded inference traffic every ``batch_size`` requests, unscored.

    The report-free half of harness evolution: a deployment that only serves
    still evolves. Each recorded inference is one unit, in arrival order;
    when ``batch_size`` have accumulated they batch as trace samples with
    ``score=None``, and the proposer contract requires handling unscored
    samples. Reports that arrive under this policy are released untouched;
    a deployment with real outcome signal selects the reported policy
    instead, because a measured result beats model self judgment. In ``hybrid``
    a queued instruction batches with the oldest held records, up to
    ``batch_size``, as an automatic batch would; in ``manual`` it runs alone.
    """

    output_schema = TrainingBatch
    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        if request is not None and self.training_mode == "manual":
            self.experience_buffer.reserve(0)
            return TrainingBatch(request.id, ())
        # In hybrid an instruction takes the records an automatic batch would, none included; the base attaches it.
        return self._make_pending(batch_number)

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        # Each inference record is one unit.
        self.experience_buffer: ExperienceBuffer[str, AgentRecord] = ExperienceBuffer(self.selection_policy())
        self._released: set[str] = set()

    def selection_policy(self) -> SelectionPolicy:
        """Return the policy that puts records in batch order. The constructor calls this method one time."""
        return ArrivalOrder()

    def ingest(self, item: AgentRecord) -> None:
        if item.request_type is RequestType.TRAIN:
            super().ingest(item)
        elif item.request_type is RequestType.INFERENCE:
            index = self.experience_buffer.next_arrival_index()
            self.experience_buffer.put(ExperienceUnit(item.agent_record_id, (item,), index))
        else:
            self._released.add(item.agent_record_id)

    def _ready_count(self) -> int:
        return len(self.experience_buffer)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        selected = self.experience_buffer.reserve(self._batch_size)
        return TrainingBatch(
            f"{self.scenario}:harness_evolve:{batch_number}",
            tuple(make_trajectory(unit.members) for unit in selected),
        )

    def _consume_pending(self) -> frozenset[str]:
        if self.experience_buffer.reserved is None:
            raise RuntimeError("no pending trace batch to consume")
        consumed = frozenset(unit.unit_id for unit in self.experience_buffer.consume_reserved())
        self._released |= consumed
        return consumed

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset(self._released | self._consumed_requests)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        self._released -= agent_record_ids
