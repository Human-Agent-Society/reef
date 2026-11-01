"""Reports as samples: one valid report and the calls of the trajectories that it scores become one sample.

``ReportSamples`` reads inference and report records. It links the inference records into trajectories
(``reef.core.trajectory_index``, RFC #1054). A report references the last call of each trajectory that it scores;
the stage expands each reference to the calls of its trajectory. It validates each report, filters the reports that
are not training data, and forms one ``ReportCandidate`` per report, which ``make_sample`` turns into one sample: by
default the trajectories as one ATIF trajectory sample with the report's score, which a recipe overrides. It then
places each report that has samples in its group (``ReportGrouping``) and makes singleton reports and ready groups
ready, as ``ReportSet`` record set units. It owns deduplication, group slots, consumption, and record release. No
inference is in two candidates: a report whose trajectories hold an inference of another candidate ends.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Self

from reef.core.job_data import Sample, TaskSample, TrajectorySample
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError, validate_report_payload
from reef.core.trajectories import make_trajectory
from reef.core.trajectory_index import TrajectoryIndex
from reef.pipeline_context import PipelineContext
from reef.record2sample.base import Record2Sample, Record2SampleSettings, SampleResult
from reef.record2sample.sets import RecordSet, RecordSetUnit

logger = logging.getLogger(__name__)


class GroupDecision(Enum):
    """The decision for a report group: ready, incomplete, or invalid."""

    READY = "ready"
    INCOMPLETE = "incomplete"
    DISCARD = "discard"


@dataclass(frozen=True)
class ReportContext:
    """A valid report and the calls of the trajectories that it scores.

    Values: ``inferences`` holds the calls of the trajectory of each of the report's references, oldest first,
    in reference order, each call once (``reef.core.trajectory_index``). ``score`` is the report's score
    (``report_score``) and ``parsed_report`` its payload parsed by the recipe's report type, when it has one.
    Purpose: ``ReportSamples`` makes it, and its ``make_sample`` reads it.
    """

    report: AgentRecord
    score: float | None
    inferences: tuple[AgentRecord, ...]
    parsed_report: ReportBase | None = None

    @property
    def references(self) -> tuple[str, ...]:
        """The IDs of the calls of the report's trajectories, as ``inferences`` holds them."""
        return tuple(record.agent_record_id for record in self.inferences)

    @property
    def ends(self) -> tuple[AgentRecord, ...]:
        """The last call of each trajectory, in reference order: the calls that the report references. A method
        that trains one call trains an end, whose request holds its whole trajectory."""
        by_id = {record.agent_record_id: record for record in self.inferences}
        return tuple(by_id[reference] for reference in self.report.references if reference in by_id)

    def require_score(self) -> float:
        """Read a finite reward for a training method that needs one."""
        if self.score is None or not math.isfinite(self.score):
            raise ValueError("training data requires a finite report score")
        return self.score


def report_score(report: AgentRecord) -> float | None:
    """Return the report's numeric score, or ``None`` when absent or non-numeric.

    Bool is not a score. Callers layer their own selection policy (windows,
    finiteness, minimums) on top of this one shared reading of the payload.
    """
    score = report.payload.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return float(score)


def accepted_by(report_type: type[ReportBase], payload: Mapping[str, Any]) -> bool:
    """Whether ``report_type`` parses ``payload``."""
    try:
        report_type.from_dict(payload)
    except ReportValidationError:
        return False
    return True


class ReportGrouping:
    """A recipe's report policy: which reports train, and how reports form groups.

    The default takes every report and trains each one by itself. A grouped
    method overrides ``placement`` and ``decide``. These methods run on the
    trainer thread and read only records.
    """

    def is_training_report(self, report: AgentRecord) -> bool:
        """Whether a valid report is this method's training data.

        A report that is not, such as another role's signal sharing the
        scenario, is released and never assembled; nothing about it can fail
        a push. Its sources are released with it only under ``exclusive_sources``
        or when it references more than one inference; a single referenced
        inference stays retained for a report that trains on it.
        """
        return True

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        """Return the training group and retry slot of a report that has samples. The default is an independent report.

        These coordinates belong to the stage, separately from an item's
        comparison group. For example, TTTD trains a whole step containing
        several comparison groups. A None slot uses the report ID.
        """
        return None, None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        """Decide whether the group's reports, in arrival order, are ready, incomplete, or invalid."""
        raise NotImplementedError(f"{type(self).__name__} placed a report in a group without a decide override")

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe group outcomes, such as groups that can never train."""
        return {}


@dataclass(frozen=True, kw_only=True)
class ReportCandidate(RecordSet):
    """One valid report and the inference records that it references.

    Values: a record set whose records are the inferences of ``context``, in
    reference order, and then the report. ``set_id`` is the report ID.
    Purpose: ``ReportSamples`` forms it and makes its samples (``make_samples``).
    """

    context: ReportContext

    def records(self) -> tuple[AgentRecord, ...]:
        return (*self.context.inferences, self.context.report)


@dataclass(frozen=True)
class ReportEntry:
    """One report candidate that has samples, with its place in a group slot or by itself."""

    candidate: ReportCandidate
    group_key: Hashable | None
    slot: Hashable

    @property
    def context(self) -> ReportContext:
        return self.candidate.context

    @property
    def report_id(self) -> str:
        return self.context.report.agent_record_id


@dataclass(frozen=True, kw_only=True)
class ReportSet(RecordSetUnit):
    """A singleton report or a ready group.

    Values: a record set unit whose record sets are the candidates of ``entries``,
    in arrival order. A singleton has one entry with no ``group_key``. A group
    holds every entry of its group key.
    Purpose: ``ReportSamples`` makes it ready; the pipeline stores its samples as one group.
    """

    entries: tuple[ReportEntry, ...]

    def record_sets(self) -> tuple[RecordSet, ...]:
        return tuple(entry.candidate for entry in self.entries)


def report_set(unit: RecordSetUnit) -> ReportSet:
    """Return ``unit`` as a report set. A report stage makes only report sets ready."""
    if not isinstance(unit, ReportSet):
        raise TypeError(f"the report stage makes report sets ready, not {type(unit).__name__}")
    return unit


class ReportStatus(Enum):
    """Where a report is in the stage, from its first read to its release."""

    #: A candidate whose samples the stage has not made yet. A failed assembly leaves it here for a retry.
    FORMED = "formed"
    #: The report has samples and waits for its group, or for the pipeline to store them.
    HELD = "held"
    #: The pipeline stored the report's samples. Its record is releasable.
    STORED = "stored"
    #: The report ended without training. Its record is releasable.
    ENDED = "ended"


@dataclass
class ReportState:
    """One report the stage read, where it is, and the calls that it claims.

    ``calls`` holds the calls of the report's trajectories once the stage expanded them, and the report's own
    references before.
    """

    record: AgentRecord
    status: ReportStatus
    calls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.calls:
            self.calls = self.record.references

    @property
    def live(self) -> bool:
        """Whether the report still claims its inference records."""
        return self.status in (ReportStatus.FORMED, ReportStatus.HELD)


@dataclass
class ReportGroup:
    """The reports placed under one group key, by slot. A discarded group ends every later report for its key."""

    entries: dict[Hashable, ReportEntry] = field(default_factory=dict)
    discarded: bool = False


@dataclass(frozen=True)
class SkippedReport:
    """``make_sample`` returns this for a valid report that cannot train. The stage releases the report."""

    reason: str


class ReportSamples(Record2Sample):
    """Turn each valid report and the calls of its trajectories into one sample, alone or in groups.

    A recipe overrides ``make_sample``; the default makes the trajectory sample. The stage attaches the calls and the
    report ID to the sample's ``source_agent_record_ids``. It forms report candidates, and makes singleton reports
    and ready groups ready as report sets. Invalid references raise immediately. Deduplication, consumed-source
    tracking, and group slots preserve retry behavior. Release keeps every live report and the inference records that
    it references.

    * ``grouping``: which reports train, and how they form groups; ``None`` trains every valid report by itself.
    * ``exclusive_sources``: a report that ends owns its referenced inference records outright. Use it when each
      report has its own inferences, as one scored rollout or one distillation sample does.
    * ``max_staleness``: the staleness limit of ``Record2Sample``.

    Config: ``Record2SampleSettings``.
    """

    def __init__(
        self,
        context: PipelineContext,
        grouping: ReportGrouping | None = None,
        *,
        exclusive_sources: bool = False,
        max_staleness: int = 0,
    ) -> None:
        super().__init__(max_staleness=max_staleness)
        self.scenario = context.scenario
        self.report_type = context.report_type
        self.admitted_report_type = context.admitted_report_type
        self.grouping = ReportGrouping() if grouping is None else grouping
        self.exclusive_sources = exclusive_sources
        #: The inference records, by ID, until their release.
        self.inferences: dict[str, AgentRecord] = {}
        #: The trajectories of the inference records that the stage read.
        self.trajectories = TrajectoryIndex()
        #: Every report the stage read, by ID, until its release.
        self.reports: dict[str, ReportState] = {}
        #: Inference records of the reports whose samples the pipeline stored, until their release.
        self.stored_sources: set[str] = set()
        #: Inference records that ended reports own. They are released once no live report references them.
        self.owned_sources: set[str] = set()
        self.groups: dict[Hashable, ReportGroup] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        """Every valid report trains by itself. A subclass with a grouping or settings overrides it."""
        settings = Record2SampleSettings.parse(config, plugin_class=cls)
        return cls(context, max_staleness=settings.max_staleness)

    def make_sample(self, context: ReportContext) -> Sample | SkippedReport:
        """Assemble valid feedback into data; raise on a broken training contract.

        The default is the report's trajectories as one ATIF trajectory sample (``make_trajectory``) with the
        report's score and its ``feedback``, when that is a string or an object. Runs on the trainer thread without
        network or model calls. Return ``SkippedReport`` for a valid report that cannot train.
        """
        feedback = context.report.payload.get("feedback")
        return make_trajectory(
            context.inferences, context.score, feedback if isinstance(feedback, (str, Mapping)) else None
        )

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        context = report_candidate(record_set).context
        sample = self.make_sample(context)
        if isinstance(sample, SkippedReport):
            return SampleResult.skipped()
        if not isinstance(sample, (TrajectorySample, TaskSample)):
            raise TypeError(f"{type(self).__name__}.make_sample must return a Sample")
        report_id = context.report.agent_record_id
        return SampleResult.ready((replace(sample, source_agent_record_ids=(*context.references, report_id)),))

    def context_changed(self, context: PipelineContext) -> None:
        self.admitted_report_type = context.admitted_report_type

    # ------------------------------------------------------------- formation

    def push(self, record: AgentRecord) -> None:
        if record.scenario != self.scenario:
            raise ReportValidationError("records must belong to the pipeline's scenario")
        if record.request_type is RequestType.INFERENCE:
            self.inferences[record.agent_record_id] = record
            self.trajectories.add(record)
            return
        if record.request_type is not RequestType.REPORT:
            return
        state = self.reports.get(record.agent_record_id)
        if state is not None and state.status is not ReportStatus.FORMED:
            # A retry of a report the stage already placed or ended. A formed report whose assembly failed is read again.
            return
        validate_report_payload(record.payload)
        if not record.references or len(set(record.references)) != len(record.references):
            raise ReportValidationError("report references must be non-empty and unique")
        if not self.grouping.is_training_report(record):
            self.handle_other_report(record)
            return
        parsed_report: ReportBase | None = None
        if self.report_type is not None:
            try:
                parsed_report = self.report_type.from_dict(record.payload)
            except ReportValidationError as refusal:
                # A scenario of several components admits what any of them accepts: a report the ingress
                # contract takes and this one refuses is another component's, not this method's training
                # data, and this trainer releases it. A report every component refuses still raises.
                admitted = self.admitted_report_type
                if admitted is None or admitted is self.report_type or not accepted_by(admitted, record.payload):
                    raise
                # Named in the log: a report meant for this trainer with a broken field also lands here.
                logger.warning(
                    "scenario %r releases report %s to the other components: %s refused it: %s",
                    self.scenario,
                    record.agent_record_id,
                    self.report_type.__name__,
                    refusal,
                )
                self.end_report(record)
                return
        calls = self.report_calls(record)
        if calls is None:
            self.end_report(record)
            return
        # No inference is in two candidates: a valid report on an inference of another candidate ends.
        report_id = record.agent_record_id
        if any(self.set_of_record.get(call, report_id) != report_id for call in calls):
            self.end_report(record, calls)
            return
        context = ReportContext(
            record, report_score(record), tuple(self.inferences[call] for call in calls), parsed_report
        )
        # The report is live before its samples are made: a contract failure must not let
        # buffer release drop its inputs or turn a retry into a successful no-op.
        self.reports[record.agent_record_id] = ReportState(record, ReportStatus.FORMED, calls)
        self.form(ReportCandidate(set_id=record.agent_record_id, context=context))

    def handle_other_report(self, report: AgentRecord) -> None:
        """End a valid report that this method does not train on. A subclass can keep it for another use."""
        self.end_report(report)

    def report_calls(self, report: AgentRecord) -> tuple[str, ...] | None:
        """The calls of the trajectories that ``report`` scores, in reference order, each once; ``None`` when the
        report cannot train.

        Raise ``ReportValidationError`` when a reference is not an inference that the stage read. A report that
        references a call inside the trajectory of another reference, as reports of earlier versions listed every
        call, cannot train, and neither can a report whose trajectory holds a call that the stage released.
        """
        missing = [ref for ref in report.references if ref not in self.inferences]
        if missing:
            raise ReportValidationError(f"report references unavailable inference records: {missing!r}")
        calls: dict[str, None] = {}
        for reference in report.references:
            trajectory = self.trajectories.trajectory(reference)
            inner = [other for other in report.references if other != reference and other in trajectory]
            if inner:
                logger.warning(
                    "scenario %r ends report %s: it references %r inside the trajectory of %r; a report references "
                    "only the last call of each trajectory",
                    self.scenario,
                    report.agent_record_id,
                    inner,
                    reference,
                )
                return None
            calls.update(dict.fromkeys(trajectory))
        released = [call for call in calls if call not in self.inferences]
        if released:
            logger.warning(
                "scenario %r ends report %s: its trajectories hold calls that already trained or ended: %r",
                self.scenario,
                report.agent_record_id,
                released,
            )
            return None
        return tuple(calls)

    def accept(self, record_set: RecordSet) -> None:
        candidate = report_candidate(record_set)
        context = candidate.context
        report = context.report
        key, slot = self.grouping.placement(context)
        slot = report.agent_record_id if slot is None else slot
        hash(key)
        hash(slot)
        group = None if key is None else self.groups.setdefault(key, ReportGroup())
        if group is not None and (group.discarded or slot in group.entries):
            self.end_report(report)
            return
        self.reports[report.agent_record_id].status = ReportStatus.HELD
        entry = ReportEntry(candidate=candidate, group_key=key, slot=slot)
        if group is None:
            unit_id = ("report", report.agent_record_id)
            self.ready_units[unit_id] = ReportSet(unit_id=unit_id, entries=(entry,))
        else:
            group.entries[slot] = entry
            self.refresh_group(key)

    def reject(self, record_set: RecordSet) -> None:
        self.end_report(report_candidate(record_set).context.report)

    def end_report(self, report: AgentRecord, calls: tuple[str, ...] = ()) -> None:
        """End a report without training. Its record is releasable, with the sources that it owns outright.

        A report's sources are the calls of its trajectories: ``calls``, those of its state, or else its
        references. A report owns its sources when it was a candidate, because no other
        candidate can take them. A report that ended before it was a candidate
        owns its sources when it claims more than one inference, or any number
        under ``exclusive_sources``. The pipeline forgets the samples of the report's record set.
        """
        state = self.reports.get(report.agent_record_id)
        was_candidate = state is not None and state.live
        sources = calls or (state.calls if state is not None else report.references)
        self.reports[report.agent_record_id] = ReportState(report, ReportStatus.ENDED, sources)
        self.end_set(report.agent_record_id)
        if was_candidate or self.exclusive_sources or len(sources) > 1:
            self.owned_sources.update(sources)

    # ---------------------------------------------------------------- groups

    def refresh_group(self, key: Hashable) -> None:
        """Decide the group again after an entry joined it: make it ready, wait, or discard it.

        A slot takes one entry and a filled slot refuses another, so the entries are in arrival order.
        """
        entries = tuple(self.groups[key].entries.values())
        decision = self.grouping.decide(key, tuple(entry.context for entry in entries))
        if decision is GroupDecision.READY:
            self.ready_units[("group", key)] = ReportSet(unit_id=("group", key), entries=entries)
        elif decision is GroupDecision.INCOMPLETE:
            self.ready_units.pop(("group", key), None)
        elif decision is GroupDecision.DISCARD:
            self.discard_group(key)
        else:
            raise TypeError("decide must return GroupDecision")

    def discard_group(self, key: Hashable) -> None:
        """End every report of the group, and refuse later reports for its key."""
        self.ready_units.pop(("group", key), None)
        group = self.groups[key]
        group.discarded = True
        for entry in group.entries.values():
            self.end_report(entry.context.report)
        group.entries = {}

    # ------------------------------------------------------------ consumption

    def consume(self, unit: RecordSetUnit) -> None:
        """End the reports of ``unit``, whose samples the pipeline stored; a stored group reopens for later reports.

        A ready group holds every entry of its key (``refresh_group``), so storing it empties the group.
        """
        for entry in report_set(unit).entries:
            self.reports[entry.report_id] = ReportState(
                entry.context.report, ReportStatus.STORED, entry.context.references
            )
            self.stored_sources.update(entry.context.references)
            self.end_set(entry.report_id)
            if entry.group_key is not None:
                self.groups.pop(entry.group_key, None)

    # ---------------------------------------------------------------- release

    def live_references(self) -> set[str]:
        return {call for state in self.reports.values() if state.live for call in state.calls}

    def releasable_record_ids(self) -> frozenset[str]:
        """Find completed records with no remaining live dependents.

        The releasable-source set is recomputed here every time: a source is
        releasable while an ended report owns it (or a stored report used it)
        and no live report references it. Live claims and ownership both
        move between reads, so the answer is never latched at event time.
        """
        ended_reports = {report_id for report_id, state in self.reports.items() if not state.live}
        releasable_sources = (self.owned_sources | self.stored_sources) - self.live_references()
        return frozenset(ended_reports | releasable_sources)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        for agent_record_id in agent_record_ids:
            state = self.reports.get(agent_record_id)
            if state is not None and not state.live:
                self.reports.pop(agent_record_id)
            # Only completed inferences without live report references are released.
            self.inferences.pop(agent_record_id, None)
        self.trajectories.forget(agent_record_ids)
        self.stored_sources -= agent_record_ids
        self.owned_sources -= agent_record_ids

    def status(self) -> Mapping[str, Any]:
        return self.grouping.status()

    def operational_metrics(self) -> Mapping[str, float | int]:
        """The live reports, which wait for their group or for the store."""
        waiting = [state.record for state in self.reports.values() if state.live]
        return {
            "waiting_reports": len(waiting),
            "oldest_report_wait_seconds": (
                max(0.0, time.time() - min(report.created_at for report in waiting)) if waiting else 0.0
            ),
        }


def report_candidate(record_set: RecordSet) -> ReportCandidate:
    """Return ``record_set`` as a report candidate. A report stage forms only report candidates."""
    if not isinstance(record_set, ReportCandidate):
        raise TypeError(f"the report stage forms report candidates, not {type(record_set).__name__}")
    return record_set
