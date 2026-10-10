"""Reported feedback: assemble existing records, group samples, and reserve batches.

Ingress validates references against storage before accepting reports. The
processor consumes records in append order, so references are already present.
Deduplication, consumed-source tracking, and group slots preserve retry behavior;
buffer release preserves every live report and the inference records it references.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError, validate_report_payload
from reef.train.experience import ArrivalOrder, ExperienceBuffer, ExperienceUnit, GroupKeyOrder, SelectionPolicy
from reef.train.processors.base import DataProcessor
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


@dataclass(frozen=True, kw_only=True)
class ReportUnit(ExperienceUnit):
    """A singleton report or a ready group in the experience buffer, with its reports in arrival order."""

    reports: tuple[_PendingReport, ...]


def report_units(units: Sequence[ExperienceUnit]) -> tuple[ReportUnit, ...]:
    """Return buffer units as report units. The reported-feedback engine puts only report units in its buffer."""
    selected = tuple(unit for unit in units if isinstance(unit, ReportUnit))
    if len(selected) != len(units):
        raise TypeError("the reported-feedback experience buffer must hold only report units")
    return selected


# ------------------------------------------------- reported-feedback processor


def accepted_by(report_type: type[ReportBase], payload: Mapping[str, Any]) -> bool:
    """Whether ``report_type`` parses ``payload``."""
    try:
        report_type.from_dict(payload)
    except ReportValidationError:
        return False
    return True


class ReportedFeedbackProcessor(DataProcessor, ABC):
    """Assemble valid reports and their existing inference records into batches.

    Recipes implement ``make_sample`` and ``make_batch``, plus ``grouping``
    and ``decide_group`` for grouped methods. The engine owns deduplication, group slots, reservations,
    consumption, and buffer release. Invalid references raise immediately; training
    data failures propagate instead of silently dropping reports.

    The units for a batch are accepted singleton reports and ready groups. They
    wait in an :class:`~reef.train.experience.ExperienceBuffer`.
    ``selection_policy`` sets their batch order.
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
        self._groups: dict[Hashable, dict[Hashable, _PendingReport]] = {}  # group key → slot → buffered report
        self._discarded_groups: set[Hashable] = set()
        # Singleton reports and ready groups, plus the reserved batch.
        self.experience_buffer = ExperienceBuffer(self.selection_policy())
        self._manual_limit_warned = False

    # ------------------------------------------------------- the recipe hooks

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Unconsumed reports, excluding the reserved batch; readiness is recipe-owned."""
        reserved = self.reserved_report_ids()
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
    #: Batch ready groups in group-key order instead of arrival order; read by the default ``selection_policy``.
    ordered_groups: bool = False
    #: Units held in manual mode beyond this many batches are released, oldest first.
    manual_unit_cap_batches: int = 4

    def selection_policy(self) -> SelectionPolicy:
        """Return the policy that puts singleton reports and ready groups in batch order.

        The constructor calls this method one time. The default policy takes
        units in arrival order. If ``ordered_groups`` is true, it takes groups
        in key order.
        """
        return GroupKeyOrder() if self.ordered_groups else ArrivalOrder()

    @abstractmethod
    def make_sample(self, context: ReportContext) -> TrainDataItem:
        """Assemble valid feedback into data; raise on a broken training contract.

        Runs on the trainer thread without network or model calls. Every
        valid report produces a sample; this hook does not accept/reject reports.
        """

    def is_training_report(self, report: AgentRecord) -> bool:
        """Whether a valid report is this method's training data.

        A report that is not, such as another role's signal sharing the
        scenario, is released and never assembled; nothing about it can fail
        ingestion. Its sources are released with it only under ``exclusive_sources``
        or when it references more than one inference; a single referenced
        inference stays retained for a report that trains on it. The default
        takes every report.
        """
        return True

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
        if any(ref in self._trained_sources for ref in item.references) or not self.is_training_report(item):
            self._seen_reports.add(item.agent_record_id)
            self._terminate(item)
            return
        report_type = self.context.report_type
        parsed_report: ReportBase | None = None
        if report_type is not None:
            try:
                parsed_report = report_type.from_dict(item.payload)
            except ReportValidationError as refusal:
                # A scenario of several components admits what any of them accepts: a report the ingress
                # contract takes and this one refuses is another component's, not this method's training
                # data, and this trainer releases it. A report every component refuses still raises.
                admitted = self.context.admitted_report_type
                if admitted is None or admitted is report_type or not accepted_by(admitted, item.payload):
                    raise
                # Named in the log: a report meant for this trainer with a broken field also lands here.
                logger.warning(
                    "scenario %r releases report %s to the other components: %s refused it: %s",
                    self.scenario,
                    item.agent_record_id,
                    report_type.__name__,
                    refusal,
                )
                self._seen_reports.add(item.agent_record_id)
                self._terminate(item)
                return
        context = self._report_context(item, parsed_report)
        # Retain the report before assembly: a contract failure must not let
        # buffer release drop its inputs or turn a retry into a successful no-op.
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
        pending = _PendingReport(
            order=self.experience_buffer.next_arrival_index(),
            item=replace(sample, source_agent_record_ids=(*item.references, item.agent_record_id)),
            report=item,
            group_key=key,
            slot=slot,
        )
        if key is None:
            self.experience_buffer.put(
                ReportUnit(unit_id=("report", item.agent_record_id), arrival_index=pending.order, reports=(pending,))
            )
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
        reserved = self.reserved_report_ids()
        for unit in report_units(self.experience_buffer.ordered_units()):
            if self._ready_count() <= limit:
                return
            if unit.reports[0].report.agent_record_id in reserved:
                continue
            self._release_unit(unit)
            if not self._manual_limit_warned:
                logger.warning(
                    "%s scenario %r released report %s beyond the manual limit %d (further releases are not logged)",
                    type(self).__name__,
                    self.scenario,
                    unit.reports[0].report.agent_record_id,
                    limit,
                )
                self._manual_limit_warned = True

    def _release_unit(self, unit: ReportUnit) -> None:
        """Release a whole buffered group or singleton and its sources."""
        if unit.group_key is not None:
            self._discard_group(unit.group_key)
        else:
            self.experience_buffer.remove(unit.unit_id)
        for pending in unit.reports:
            report = self._reports.pop(pending.report.agent_record_id, None)
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

    def _report_context(self, report: AgentRecord, parsed_report: ReportBase | None = None) -> ReportContext:
        missing = [ref for ref in report.references if ref not in self._inferences]
        if missing:
            raise ReportValidationError(f"report references unavailable inference records: {missing!r}")
        inferences = tuple(self._inferences[ref] for ref in report.references)
        report_type = self.context.report_type
        if parsed_report is None and report_type is not None:
            parsed_report = report_type.from_dict(report.payload)
        return ReportContext(report, report_score(report), inferences, parsed_report)

    # ---------------------------------------------------------------- groups

    def _group_reports(self, key: Hashable) -> tuple[_PendingReport, ...]:
        return tuple(sorted(self._groups[key].values(), key=lambda pending: pending.order))

    def _refresh_group(self, key: Hashable) -> None:
        reports = self._group_reports(key)
        decision = self.decide_group(key, tuple(pending.item for pending in reports))
        if decision is GroupDecision.READY:
            self.experience_buffer.put(
                ReportUnit(unit_id=("group", key), arrival_index=reports[0].order, group_key=key, reports=reports)
            )
        elif decision is GroupDecision.INCOMPLETE:
            self.experience_buffer.remove(("group", key))
        elif decision is GroupDecision.DISCARD:
            self._discard_group(key)
        else:
            raise TypeError("decide_group must return GroupDecision")

    def _discard_group(self, key: Hashable) -> None:
        self.experience_buffer.remove(("group", key))
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
        return len(self.experience_buffer)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        units = report_units(self.experience_buffer.reserve(self._batch_size))
        return self.make_batch(tuple(pending.item for unit in units for pending in unit.reports), batch_number)

    def reserved_report_ids(self) -> set[str]:
        """Return the IDs of the reports in the reserved batch."""
        return {
            pending.report.agent_record_id
            for unit in report_units(self.experience_buffer.reserved_units())
            for pending in unit.reports
        }

    def _consume_pending(self) -> frozenset[str]:
        if self.experience_buffer.reserved is None:
            raise RuntimeError("cannot consume a batch before reports are pending")
        consumed_reports: set[str] = set()
        trained_sources: set[str] = set()
        changed_groups: set[Hashable] = set()
        consumed_units = report_units(self.experience_buffer.consume_reserved())
        for pending in (pending for unit in consumed_units for pending in unit.reports):
            report_id = pending.report.agent_record_id
            self._consumed.add(report_id)
            consumed_reports.add(report_id)
            self._reports.pop(report_id, None)
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
                self.experience_buffer.remove(("group", key))
        self._trained_sources.update(trained_sources)
        return frozenset(consumed_reports | trained_sources)

    # ---------------------------------------------------------- buffer release

    def _live_references(self) -> set[str]:
        return {ref for report in self._reports.values() for ref in report.references}

    def releasable_record_ids(self) -> frozenset[str]:
        """Find completed records with no remaining buffered dependents.

        The releasable-source set is recomputed here every time: a source is
        releasable while a terminal report owns it (or a batch consumed it)
        and no live report references it.  Live claims and terminal ownership
        both move between reads, so the answer is never latched at event time.
        """
        live_references = self._live_references()
        releasable_sources = (self._terminal_owned_sources | self._trained_sources) - live_references
        releasable = self._consumed | self._terminal | releasable_sources
        return frozenset(releasable | self._consumed_requests)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        # --- scalar id sets ---
        self._consumed -= agent_record_ids
        self._terminal -= agent_record_ids
        self._trained_sources -= agent_record_ids
        self._terminal_owned_sources -= agent_record_ids
        self._seen_reports -= agent_record_ids

        # Only completed inferences without live report references are released.
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
