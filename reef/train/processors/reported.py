"""Reported feedback: assemble existing records, group samples, and reserve batches.

Ingress validates references against storage before accepting reports. The
processor consumes records in append order, so references are already present.
Deduplication, consumed-source tracking, and group slots preserve retry behavior;
retention protects every live report and the inference records it references.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, cast

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError, validate_report_payload
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.processors.common import (
    make_multi_turn_policy_trajectory,
    make_policy_trajectory,
    report_score,
    sample_assembly_config_fields,
)
from reef.train.types import ProcessorContext, TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

__all__ = [
    "GroupDecision",
    "ReportContext",
    "ReportedFeedbackProcessor",
    "SampleAssembly",
]


# ---------------------------------------------------------------- value types


class GroupDecision(Enum):
    """The decision for a report group: ready, incomplete, or invalid."""

    READY = "ready"
    INCOMPLETE = "incomplete"
    DISCARD = "discard"


@dataclass(frozen=True)
class ReportContext:
    """A valid report and its inference records, in reference order."""

    report: AgentRecord
    score: float | None
    inferences: tuple[AgentRecord, ...]
    parsed_report: ReportBase | None = None

    @property
    def references(self) -> tuple[str, ...]:
        return self.report.references

    def require_score(self) -> float:
        """Read a finite reward for a training method that needs one."""
        if self.score is None or not math.isfinite(self.score):
            raise ValueError("training data requires a finite report score")
        return self.score


@dataclass(frozen=True)
class _PendingReport:
    """Processor-owned assembly and consumption state; never passed to recipes."""

    order: int
    item: TrainDataItem
    report: AgentRecord
    group_key: Hashable | None
    slot: Hashable


# ------------------------------------------------- reported-feedback processor


class ReportedFeedbackProcessor(DataProcessor, ABC):
    """Assemble valid reports and their existing inference records into batches.

    Recipes implement ``make_sample`` and ``make_batch``, plus ``grouping``
    and ``decide_group`` for grouped methods. The engine owns deduplication, group slots, reservations,
    consumption, and retention. Invalid references raise immediately; training
    data failures propagate instead of silently dropping reports.
    """

    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        # The sample-assembly settings belong to every reported-feedback processor's config
        # surface: validate them here so a bad deployment fails at construction
        # even for recipes that never assemble a policy sample.
        sample_assembly_config_fields(context.config)

        # --- inference store ---
        self._inferences: dict[str, AgentRecord] = {}

        # --- report lifecycle ---
        self._reports: dict[str, AgentRecord] = {}  # live: assembled or failed, not consumed
        self._seen_reports: set[str] = set()  # dedup
        self._consumed: set[str] = set()  # acknowledged batch members
        self._terminal: set[str] = set()  # terminal reports

        # --- source ownership ---
        self._trained_sources: set[str] = set()  # consumed by an acknowledged batch
        self._terminal_owned_sources: set[str] = set()  # owned by terminal reports

        # --- buffered reports ---
        self.singletons: dict[str, _PendingReport] = {}  # report id → buffered report
        self._groups: dict[Hashable, dict[Hashable, _PendingReport]] = {}  # group key → slot → buffered report
        self._ready_groups: set[Hashable] = set()
        self._discarded_groups: set[Hashable] = set()

        # --- pending batch ---
        self._pending_reports: tuple[_PendingReport, ...] | None = None
        self._next_order = 0
        self._manual_limit_warned = False

    # ------------------------------------------------------- the recipe hooks

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Unconsumed reports, excluding the reserved batch; readiness is recipe-owned."""
        reserved = {pending.report.agent_record_id for pending in self._pending_reports or ()}
        waiting = [report for record_id, report in self._reports.items() if record_id not in reserved]
        return {
            **super().operational_metrics(),
            "unreserved_reports": len(waiting),
            "reserved_reports": len(reserved),
            "oldest_report_wait_seconds": (
                max(0.0, time.time() - min(report.created_at for report in waiting)) if waiting else 0.0
            ),
        }

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: Whether a terminal report owns its referenced sources outright.
    exclusive_sources: bool = False
    #: Batch ready groups in group-key order instead of arrival order.
    ordered_groups: bool = False
    #: Units held in manual mode beyond this many batches are released, oldest first.
    manual_unit_cap_batches: int = 4

    @abstractmethod
    def make_sample(self, context: ReportContext) -> TrainDataItem:
        """Assemble valid feedback into data; raise on a broken training contract.

        Runs on the trainer thread without network or model calls. Every
        valid report produces a sample; this hook does not accept/reject reports.
        """

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        """Return the batching group and retry slot; defaults to an independent report.

        These coordinates belong to ingestion, separately from an item's
        comparison group. For example, TTTD batches a whole step containing
        several comparison groups. A None slot uses the report id.
        """
        return None, None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        """Decide whether the group's assembled items are ready, incomplete, or invalid."""
        raise NotImplementedError(f"{type(self).__name__} produced a group without a decide_group override")

    @abstractmethod
    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        """Build a batch from selected items, in group and arrival order.

        The processor tracks consumption independently, including any selected
        items the recipe omits from the resulting batch.
        """

    def ingest(self, item: AgentRecord) -> None:
        if item.scenario != self.scenario:
            raise ReportValidationError("records must belong to the processor's scenario")
        if item.request_type is RequestType.TRAIN:
            super().ingest(item)
            return
        if item.request_type is RequestType.INFERENCE:
            self._inferences[item.agent_record_id] = item
            return
        if item.request_type is not RequestType.REPORT or item.agent_record_id in self._seen_reports:
            return
        validate_report_payload(item.payload)
        if not item.references or len(set(item.references)) != len(item.references):
            raise ReportValidationError("report references must be non-empty and unique")
        if any(ref in self._trained_sources for ref in item.references):
            self._seen_reports.add(item.agent_record_id)
            self._terminate(item)
            return
        context = self._report_context(item)
        # Retain the report before assembly: a contract failure must not let
        # compaction delete its inputs or turn a retry into a successful no-op.
        self._reports[item.agent_record_id] = item
        sample = self.make_sample(context)
        if not isinstance(sample, (TrajectoryItem, TaskItem)):
            raise TypeError(f"{type(self).__name__}.make_sample must return a TrainDataItem")
        key, slot = self.grouping(context)
        slot = item.agent_record_id if slot is None else slot
        hash(key)
        hash(slot)
        self._seen_reports.add(item.agent_record_id)
        if key is not None and (key in self._discarded_groups or slot in self._groups.get(key, {})):
            self._terminate(item)
            return
        self._next_order += 1
        pending = _PendingReport(
            order=self._next_order,
            item=replace(sample, source_agent_record_ids=(*item.references, item.agent_record_id)),
            report=item,
            group_key=key,
            slot=slot,
        )
        if key is None:
            self.singletons[item.agent_record_id] = pending
        else:
            self._groups.setdefault(key, {})[slot] = pending
            self._refresh_group(key)
        self.cap_manual_units()

    def set_training_mode(self, training_mode: str) -> None:
        super().set_training_mode(training_mode)
        self.cap_manual_units()

    def cap_manual_units(self) -> None:
        """Manual mode batches on instructions, not units, so the pile is bounded here instead."""
        limit = self._batch_size * self.manual_unit_cap_batches
        if self.training_mode != "manual" or self._ready_count() <= limit:
            return
        # The reserved batch is handed out until acknowledged, so its units stay put.
        reserved = {pending.report.agent_record_id for pending in self._pending_reports or ()}
        for unit in self._ordered_units():
            if self._ready_count() <= limit:
                return
            if unit[0].report.agent_record_id in reserved:
                continue
            self._release_unit(unit)
            if not self._manual_limit_warned:
                logger.warning(
                    "%s scenario %r released report %s beyond the manual limit %d (further releases are not logged)",
                    type(self).__name__,
                    self.scenario,
                    unit[0].report.agent_record_id,
                    limit,
                )
                self._manual_limit_warned = True

    def _release_unit(self, unit: tuple[_PendingReport, ...]) -> None:
        """Release a whole buffered group or singleton and its sources."""
        key = unit[0].group_key
        if key is not None:
            self._discard_group(key)
        for pending in unit:
            report_id = pending.report.agent_record_id
            report = self._reports.pop(report_id, None)
            self.singletons.pop(report_id, None)
            if report is not None:
                self._terminate(report)
            self._terminal_owned_sources.update(pending.report.references)

    def _terminate(self, report: AgentRecord) -> None:
        """Drop a report from the live set and mark it terminal.

        Terminal reports release their own record, and the sources they own
        outright: a report that claims more than one inference,
        or any report when the recipe declares
        ``exclusive_sources``.
        """
        report_id = report.agent_record_id
        self._reports.pop(report_id, None)
        self._terminal.add(report_id)
        if not report.references:
            return
        if self.exclusive_sources or len(report.references) > 1:
            self._terminal_owned_sources.update(report.references)

    def _report_context(self, report: AgentRecord) -> ReportContext:
        missing = [ref for ref in report.references if ref not in self._inferences]
        if missing:
            raise ReportValidationError(f"report references unavailable inference records: {missing!r}")
        inferences = tuple(self._inferences[ref] for ref in report.references)
        report_type = self.context.report_type
        parsed_report = None if report_type is None else report_type.from_dict(report.payload)
        return ReportContext(report, report_score(report), inferences, parsed_report)

    # ---------------------------------------------------------------- groups

    def _group_reports(self, key: Hashable) -> tuple[_PendingReport, ...]:
        return tuple(sorted(self._groups[key].values(), key=lambda pending: pending.order))

    def _refresh_group(self, key: Hashable) -> None:
        decision = self.decide_group(key, tuple(pending.item for pending in self._group_reports(key)))
        if decision is GroupDecision.READY:
            self._ready_groups.add(key)
        elif decision is GroupDecision.INCOMPLETE:
            self._ready_groups.discard(key)
        elif decision is GroupDecision.DISCARD:
            self._discard_group(key)
        else:
            raise TypeError("decide_group must return GroupDecision")

    def _discard_group(self, key: Hashable) -> None:
        self._ready_groups.discard(key)
        self._discarded_groups.add(key)
        group = self._groups.pop(key)
        # Remove every member from the live set first, so the wholesale
        # release is not blocked by siblings of the same discarded group.
        members = [
            report
            for pending in group.values()
            if (report := self._reports.pop(pending.report.agent_record_id, None)) is not None
        ]
        for report in members:
            self._terminate(report)

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one accepted singleton report or one ready group; the
    # engine's half of the shared cycle in base.py is the three methods below.

    def _ready_count(self) -> int:
        return len(self.singletons) + len(self._ready_groups)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        units = self._ordered_units()[: self._batch_size]
        self._pending_reports = tuple(pending for unit in units for pending in unit)
        return self.make_batch(tuple(pending.item for pending in self._pending_reports), batch_number)

    def _ordered_units(self) -> list[tuple[_PendingReport, ...]]:
        """Ready groups and singleton reports, preserving the configured priority."""
        singletons = [(pending,) for pending in self.singletons.values()]
        if self.ordered_groups:
            # Ordered groups require sortable keys, such as step indices.
            ordered = sorted(cast("set[Any]", self._ready_groups))
            return singletons + [self._group_reports(key) for key in ordered]
        units = singletons + [self._group_reports(key) for key in self._ready_groups]
        units.sort(key=lambda unit: unit[0].order)
        return units

    def _consume_pending(self) -> frozenset[str]:
        if self._pending_reports is None:
            raise RuntimeError("cannot consume a batch before reports are pending")
        consumed_reports: set[str] = set()
        trained_sources: set[str] = set()
        changed_groups: set[Hashable] = set()
        for pending in self._pending_reports:
            report_id = pending.report.agent_record_id
            self._consumed.add(report_id)
            consumed_reports.add(report_id)
            self._reports.pop(report_id, None)
            self.singletons.pop(report_id, None)
            trained_sources.update(pending.report.references)
            if pending.group_key is not None:
                group = self._groups.get(pending.group_key)
                if group is not None:
                    group.pop(pending.slot, None)
                    changed_groups.add(pending.group_key)
        for key in changed_groups:
            if self._groups[key]:
                self._refresh_group(key)
            else:
                self._groups.pop(key)
                self._ready_groups.discard(key)
        self._trained_sources.update(trained_sources)
        self._pending_reports = None
        return frozenset(consumed_reports | trained_sources)

    # -------------------------------------------------------------- retention

    def _live_references(self) -> set[str]:
        return {ref for report in self._reports.values() for ref in report.references}

    def retention_decision(self) -> RetentionDecision:
        """Derive retention from live state — a pure read, nothing mutates.

        The releasable-source set is recomputed here every time: a source is
        releasable while a terminal report owns it (or a batch consumed it)
        and no live report references it.  Live claims and terminal ownership
        both move between reads, so the answer is never latched at event time.
        """
        live_references = self._live_references()
        releasable_sources = (self._terminal_owned_sources | self._trained_sources) - live_references
        releasable = self._consumed | self._terminal | releasable_sources
        protected = set(self._reports) | live_references
        protected.update(inference_id for inference_id in self._inferences if inference_id not in releasable_sources)
        return RetentionDecision(
            protected_agent_record_ids=frozenset(protected | self._training_requests.keys()),
            releasable_agent_record_ids=frozenset(releasable | self._consumed_requests),
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        super().compaction_applied(agent_record_ids)
        # --- scalar id sets ---
        self._consumed -= agent_record_ids
        self._terminal -= agent_record_ids
        self._trained_sources -= agent_record_ids
        self._terminal_owned_sources -= agent_record_ids
        self._seen_reports -= agent_record_ids

        # Stored records: only inferences can be here. The trainer compacts
        # ``releasable - protected``, and every live report, every reference
        # a live report holds, and every buffered report are protected —
        # so a compacted id is never in _reports, singletons, or a
        # group.
        for agent_record_id in agent_record_ids:
            self._inferences.pop(agent_record_id, None)


# --------------------------------------------------------- shared helpers


def reported_task(metadata: object) -> dict[str, Any] | None:
    """The task a report names, so a recipe can group samples by task; None when the report names none.

    ``metadata.task`` is a mapping with at least a ``name`` (the task player also sends ``path`` and
    ``digest``); a Harbor report from the shipped harness names the task as ``metadata.harbor.task_name``.
    """
    if not isinstance(metadata, Mapping):
        return None
    task = metadata.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("name"), str) and task["name"]:
        return dict(task)
    harbor = metadata.get("harbor")
    if isinstance(harbor, Mapping) and isinstance(harbor.get("task_name"), str) and harbor["task_name"]:
        return {"name": harbor["task_name"]}
    return None


@dataclass(frozen=True)
class SampleAssembly:
    """Shape a resolved report into a policy sample, without recipe policy.

    One model call becomes one sample via ``make_sample`` (default: the shared
    tensor reader); an ordered multi-reference report becomes one assembled
    multi-turn sample. ``accept_multi_turn`` gates consumption only —
    assembly always runs first. Unsupported or unassemblable trajectories
    raise a training data error and retain their source records.
    """

    accept_multi_turn: bool = False
    realign_threshold: int = 1024
    scaffold_tolerance: int = 0
    make_sample: Callable[[AgentRecord, float], TrajectoryItem] | None = None

    @classmethod
    def from_config(
        cls,
        context: ProcessorContext,
        make_sample: Callable[[AgentRecord, float], TrajectoryItem] | None = None,
    ) -> SampleAssembly:
        accept_multi_turn, realign_threshold, scaffold_tolerance = sample_assembly_config_fields(context.config)
        return cls(accept_multi_turn, realign_threshold, scaffold_tolerance, make_sample)

    def build(self, context: ReportContext, score: float) -> TrajectoryItem:
        """Build training data, raising if the recorded trajectory cannot be used."""
        inferences = context.inferences
        if inferences is None:
            raise RuntimeError("sample assembly requires resolved inferences")
        if len(inferences) == 1:
            sample: TrajectoryItem | None = (
                make_policy_trajectory(inferences[0], score)
                if self.make_sample is None
                else self.make_sample(inferences[0], score)
            )
        else:
            sample = make_multi_turn_policy_trajectory(
                inferences,
                score,
                source_agent_record_id=context.report.agent_record_id,
                realign_threshold=self.realign_threshold,
                scaffold_tolerance=self.scaffold_tolerance,
            )
        if sample is None:
            raise ValueError("training data cannot assemble the recorded multi-turn trajectory")
        if (sample.training.get("turn_count", 1) > 1) and not self.accept_multi_turn:
            raise ValueError("training data requires accept_multi_turn_policy_samples for this trajectory")
        fields: dict[str, Any] = {
            "feedback": context.report.payload.get("feedback"),
            "report_agent_record_id": context.report.agent_record_id,
            "references": list(context.references),
        }
        task = reported_task(context.report.payload.get("metadata"))
        if task is not None:
            fields["task"] = task
        return sample.with_metadata(**fields)
