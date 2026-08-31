"""The reported-feedback processor and shared building blocks.

:class:`ReportedFeedbackProcessor` owns the exactly-once state that pairs asynchronously
arriving inference and report records. Every structure it keeps answers one
hard requirement of continual serving — delete any one and a documented
failure returns:

* dedup (``_seen_reports``) — exactly-once under client retries and crash
  replay;
* the waiting index — reports legally arrive before the inference records
  they reference;
* trained-source refusal — a rollout an acknowledged batch consumed must
  never train twice, whatever the replay order;
* groups and slots — grouped recipes need a complete-set barrier and
  retry-safe coordinates;
* recomputed retention — the store deletes records, so what is safe to
  delete must be answerable at any moment, never latched at event time.

Retention is derived, never authored per recipe: live reports and their
references are protected, consumed and terminal records are releasable, and a
source is releasable when a terminal report owns it (or training consumed it)
and no live report still claims it.  That releasable-source set is recomputed
from live state on every read — never latched at event time — so a late retry
re-claims a not-yet-compacted source, and a claimant's later death frees the
source it blocked.

"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.processors.common import (
    make_multi_turn_policy_sample,
    make_policy_sample,
    report_is_trainable,
    report_score,
    sample_assembly_config_fields,
)
from reef.train.types import PolicySample, ProcessorContext, TrainingBatch

logger = logging.getLogger(__name__)

__all__ = [
    "NEVER",
    "WAIT",
    "BatchUnit",
    "Candidate",
    "GroupDecision",
    "Outcome",
    "ReportContext",
    "ReportDecision",
    "ReportedFeedbackProcessor",
    "SampleAssembly",
]


# ---------------------------------------------------------------- value types


class Outcome(Enum):
    """The one tri-state answer ``judge`` returns per report."""

    TRAIN = "train"
    WAIT = "wait"
    NEVER = "never"


@dataclass(frozen=True)
class ReportDecision:
    """The decision for one report, plus its candidate when accepted for training.

    ``group_key`` addresses the candidate into a group (``None`` means the
    candidate is its own singleton batch unit). ``slot`` is an optional
    idempotency key within the group: the first accepted report at a slot
    wins, and a retry at an occupied slot is terminal.

    ``reason`` names why a ``NEVER`` outcome was reached (a schema violation,
    a tensor-contract failure); the engine logs each distinct reason once and
    counts it, so malformed reports surface instead of vanishing silently.
    """

    outcome: Outcome
    value: Any = None
    group_key: Hashable | None = None
    slot: Hashable | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        # The engine branches on ``outcome is Outcome.X``; a non-Outcome (say,
        # a ReportDecision passed where its outcome was meant) would silently fall
        # through every branch and be treated as TRAIN. Fail at construction.
        if not isinstance(self.outcome, Outcome):
            raise TypeError(
                f"ReportDecision.outcome must be an Outcome, got {type(self.outcome).__name__}: {self.outcome!r}"
            )

    @classmethod
    def train(cls, value: Any, *, group_key: Hashable | None = None, slot: Hashable | None = None) -> ReportDecision:
        return cls(Outcome.TRAIN, value=value, group_key=group_key, slot=slot)

    @classmethod
    def never(cls, reason: str) -> ReportDecision:
        """A named terminal decision — use over bare :data:`NEVER` whenever the cause is known."""
        return cls(Outcome.NEVER, reason=reason)


#: Shared non-training report decisions.
WAIT = ReportDecision(Outcome.WAIT)
NEVER = ReportDecision(Outcome.NEVER)


class GroupDecision(Enum):
    """The decision for a candidate group: ready, incomplete, or invalid."""

    READY = "ready"
    INCOMPLETE = "incomplete"
    DISCARD = "discard"


@dataclass(frozen=True)
class ReportContext:
    """One report plus the score, eligibility, and references resolved for its decision.

    ``inferences`` is ``None`` while at least one referenced inference has not
    arrived (the only state where ``WAIT`` is a legal outcome) and the empty
    tuple when the reference list itself is malformed (duplicate references).
    ``parsed_report`` is the typed instance of the schema owned by the bound
    recipe; it is ``None`` when that recipe deliberately keeps ingress open.
    """

    report: AgentRecord
    score: float | None
    trainable: bool
    inferences: tuple[AgentRecord, ...] | None
    parsed_report: ReportBase | None = None

    @property
    def references(self) -> tuple[str, ...]:
        return self.report.references

    def eligibility(self) -> ReportDecision | None:
        """The one eligibility gate every score-based judge applies first.

        Returns the :data:`NEVER` decision for a report that can never train —
        no references, no finite numeric score (``NaN``/``inf`` never train),
        or an explicit ineligibility marker; the :data:`WAIT` decision while a
        referenced inference has not arrived; and ``None`` for a resolved,
        eligible report whose decision is now the recipe's own policy.
        """
        if not self.references or self.score is None or not math.isfinite(self.score) or not self.trainable:
            return NEVER
        if self.inferences is None:
            return WAIT
        return None


@dataclass(frozen=True)
class Candidate:
    """One accepted report's contribution to a batch, cached at decision time."""

    order: int
    value: Any
    report_agent_record_id: str
    source_agent_record_ids: frozenset[str]
    slot: Hashable


@dataclass(frozen=True)
class BatchUnit:
    """One batch unit: a singleton candidate or one ready group."""

    group_key: Hashable | None
    candidates: tuple[Candidate, ...]


# ------------------------------------------------- reported-feedback processor


class ReportedFeedbackProcessor(DataProcessor, ABC):
    """The report↔inference engine: reports judged, paired, and batched.

    A recipe subclasses it and writes:

    * ``judge`` — accept one resolved report (with the candidate it
      contributes), wait for its references, or reject it;
    * ``make_batch`` — shape the selected units into its batch type;
    * ``decide_group`` — grouped recipes only: is a group ready, incomplete,
      or invalid;
    * one-line class attributes: ``output_schema``,
      ``exclusive_sources``, ``ordered_groups``.

    Everything else — dedup, the waiting index, exactly-once consumption,
    retention, crash replay — is the engine's.

    The line, in reading order: ``ingest`` (arrival), ``_judge`` (the
    recipe's judge routed: WAIT / NEVER / TRAIN), ``build_batch`` (the
    recipe's ``make_batch`` over the selected units), ``acknowledge``
    (consumed; retention releases).
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
        self._reports: dict[str, AgentRecord] = {}  # live: waiting or accepted-but-unconsumed
        self._seen_reports: set[str] = set()  # dedup
        self._consumed: set[str] = set()  # acknowledged batch members
        self._terminal: set[str] = set()  # terminal reports
        #: Named NEVER reasons (``ReportDecision.never``), counted per reason so
        #: violations are observable instead of silent drops.
        self._never_reasons: Counter[str] = Counter()

        # --- source ownership ---
        self._trained_sources: set[str] = set()  # consumed by an acknowledged batch
        self._terminal_owned_sources: set[str] = set()  # owned by terminal reports

        # --- waiting index ---
        self._waiting: dict[str, set[str]] = {}  # inference id → waiting report ids

        # --- candidate cache ---
        self._singletons: dict[str, Candidate] = {}  # report id → singleton candidate
        self._groups: dict[Hashable, dict[Hashable, Candidate]] = {}  # group key → slot → candidate
        self._ready_groups: set[Hashable] = set()
        self._discarded_groups: set[Hashable] = set()

        # --- pending batch ---
        self._pending_units: tuple[BatchUnit, ...] | None = None
        self._next_order = 0

    # ------------------------------------------------------- the recipe hooks

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: Whether a terminal report owns its referenced sources outright.
    exclusive_sources: bool = False
    #: Batch ready groups in group-key order instead of arrival order.
    ordered_groups: bool = False

    @abstractmethod
    def judge(self, context: ReportContext) -> ReportDecision:
        """Accept, reject, or wait on one resolved report.

        Called inside ``ingest`` on the trainer's thread — a pure decision
        on data already in hand, never a model call (the computed-feedback processor's
        ``async def judge`` is the tier for that).
        """

    def decide_group(self, key: Hashable, candidates: tuple[Candidate, ...]) -> GroupDecision:
        """Decide whether an accepted candidate group is ready, incomplete, or invalid."""
        raise NotImplementedError(
            f"{type(self).__name__} produced a grouped candidate without a decide_group override"
        )

    @abstractmethod
    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> TrainingBatch:
        """Shape the selected units into the recipe's typed batch."""

    @property
    def never_reasons(self) -> Mapping[str, int]:
        """Counts of named terminal rejections (``ReportDecision.never`` reasons) since construction."""
        return dict(self._never_reasons)

    # ---------------------------------------------------------------- ingest

    def ingest(self, item: AgentRecord) -> None:
        # 1. An inference record: store it and re-judge the reports that
        #    were waiting on it, once their last reference has arrived.
        if item.request_type is RequestType.INFERENCE:
            self._inferences[item.agent_record_id] = item
            for report_id in tuple(self._waiting.pop(item.agent_record_id, ())):
                report = self._reports.get(report_id)
                if report is not None and all(ref in self._inferences for ref in report.references):
                    self._judge(report)
            return
        if item.request_type is not RequestType.REPORT:
            return
        # 2. A report, exactly once: a replayed or retried one changes
        #    nothing.
        if item.agent_record_id in self._seen_reports:
            return
        self._seen_reports.add(item.agent_record_id)
        # 3. A source an acknowledged batch consumed must never train twice,
        #    whatever the replay timing. (A source merely marked releasable
        #    does not block a late retry — while the record exists, the
        #    retry re-claims it.)
        if any(ref in self._trained_sources for ref in item.references):
            self._terminate(item)
            return
        # 4. The recipe's judge decides: TRAIN, WAIT, or NEVER.
        self._judge(item)

    # --------------------------------------------------------------- judging

    def _judge(self, report: AgentRecord) -> None:
        """Run the recipe's judge and route its report decision in one read."""
        try:
            context = self._report_context(report)
        except ReportValidationError as violation:
            # Ingress rejects malformed reports before persistence. This
            # fallback covers direct processor use and durable records replayed
            # after their recipe's schema has changed.
            decision = ReportDecision.never(str(violation))
        else:
            decision = self.judge(context)
        # 1. WAIT: park until every referenced inference has arrived; the
        #    waiting index re-judges the report when the last one lands.
        if decision.outcome is Outcome.WAIT:
            missing = [ref for ref in set(report.references) if ref not in self._inferences]
            if not missing:
                raise RuntimeError(f"{type(self).__name__} judged resolved report {report.agent_record_id!r} as WAIT")
            self._reports[report.agent_record_id] = report
            for ref in missing:
                self._waiting.setdefault(ref, set()).add(report.agent_record_id)
            return
        # 2. NEVER: count the named reason (warned once per distinct
        #    reason), release the report.
        if decision.outcome is Outcome.NEVER:
            if decision.reason is not None:
                if decision.reason not in self._never_reasons:
                    logger.warning(
                        "%s scenario %r rejected report %s: %s (further occurrences counted silently)",
                        type(self).__name__,
                        self.scenario,
                        report.agent_record_id,
                        decision.reason,
                    )
                self._never_reasons[decision.reason] += 1
            self._terminate(report)
            return
        # 3. TRAIN: a retry at an occupied slot (or a discarded group) is
        #    terminal; otherwise the candidate takes its singleton place or
        #    its group slot.
        key = decision.group_key
        slot = report.agent_record_id if decision.slot is None else decision.slot
        if key is not None and (key in self._discarded_groups or slot in self._groups.get(key, {})):
            self._terminate(report)
            return
        report_id = report.agent_record_id
        self._reports[report_id] = report
        self._next_order += 1
        candidate = Candidate(
            order=self._next_order,
            value=decision.value,
            report_agent_record_id=report_id,
            source_agent_record_ids=frozenset(report.references),
            slot=slot,
        )
        if key is None:
            self._singletons[report_id] = candidate
        else:
            self._groups.setdefault(key, {})[slot] = candidate
            self._refresh_group(key)

    def _terminate(self, report: AgentRecord) -> None:
        """Drop a report from the live set and mark it terminal.

        Terminal reports release their own record, and the sources they own
        outright: a report that claims more than one inference, an
        ineligible one, or any report at all when the recipe declares
        ``exclusive_sources``.
        """
        report_id = report.agent_record_id
        self._reports.pop(report_id, None)
        self._terminal.add(report_id)
        for ref in report.references:
            waiting = self._waiting.get(ref)
            if waiting is not None:
                waiting.discard(report_id)
                if not waiting:
                    self._waiting.pop(ref)
        if not report.references:
            return
        if self.exclusive_sources or len(report.references) > 1 or not report_is_trainable(report):
            self._terminal_owned_sources.update(report.references)

    def _report_context(self, report: AgentRecord) -> ReportContext:
        references = report.references
        inferences: tuple[AgentRecord, ...] | None
        if len(set(references)) != len(references):
            inferences = ()
        else:
            resolved = tuple(self._inferences[ref] for ref in references if ref in self._inferences)
            inferences = resolved if len(resolved) == len(references) else None
        report_type = self.context.report_type
        parsed_report = None if report_type is None else report_type.from_dict(report.payload)
        return ReportContext(
            report,
            report_score(report),
            report_is_trainable(report),
            inferences,
            parsed_report,
        )

    # ---------------------------------------------------------------- groups

    def _group_candidates(self, key: Hashable) -> tuple[Candidate, ...]:
        return tuple(sorted(self._groups[key].values(), key=lambda c: c.order))

    def _refresh_group(self, key: Hashable) -> None:
        decision = self.decide_group(key, self._group_candidates(key))
        if decision is GroupDecision.READY:
            self._ready_groups.add(key)
        elif decision is GroupDecision.INCOMPLETE:
            self._ready_groups.discard(key)
        else:
            self._discard_group(key)

    def _discard_group(self, key: Hashable) -> None:
        self._ready_groups.discard(key)
        self._discarded_groups.add(key)
        group = self._groups.pop(key)
        # Remove every member from the live set first, so the wholesale
        # release is not blocked by siblings of the same discarded group.
        members = [
            report
            for candidate in group.values()
            if (report := self._reports.pop(candidate.report_agent_record_id, None)) is not None
        ]
        for report in members:
            self._terminate(report)

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one accepted singleton candidate or one ready group; the
    # engine's half of the shared cycle in base.py is the three methods below.

    def _ready_count(self) -> int:
        return len(self._singletons) + len(self._ready_groups)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        units = self._select_units()
        self._pending_units = units
        return self.make_batch(units, batch_number)

    def _select_units(self) -> tuple[BatchUnit, ...]:
        """Select up to ``batch_size`` batch units in priority order.

        By default everything shares arrival order. With ``ordered_groups``,
        singletons batch first (arrival order), then groups in key order.
        """
        singletons = [BatchUnit(None, (candidate,)) for candidate in self._singletons.values()]
        if self.ordered_groups:
            # ordered_groups requires sortable group keys (step indices, say).
            ordered = sorted(cast("set[Any]", self._ready_groups))
            groups = [BatchUnit(key, self._group_candidates(key)) for key in ordered]
            units = singletons + groups
        else:
            units = singletons + [BatchUnit(key, self._group_candidates(key)) for key in self._ready_groups]
            units.sort(key=lambda unit: unit.candidates[0].order)
        return tuple(units[: self._batch_size])

    def _consume_pending(self) -> frozenset[str]:
        if self._pending_units is None:
            raise RuntimeError("cannot consume a batch before units are pending")
        consumed_reports: set[str] = set()
        trained_sources: set[str] = set()
        for unit in self._pending_units:
            group = self._groups.get(unit.group_key) if unit.group_key is not None else None
            for candidate in unit.candidates:
                report_id = candidate.report_agent_record_id
                self._consumed.add(report_id)
                consumed_reports.add(report_id)
                self._reports.pop(report_id, None)
                self._singletons.pop(report_id, None)
                trained_sources |= candidate.source_agent_record_ids
                if group is not None:
                    group.pop(candidate.slot, None)
            if unit.group_key is not None and group is not None:
                if group:
                    self._refresh_group(unit.group_key)
                else:
                    self._groups.pop(unit.group_key)
                    self._ready_groups.discard(unit.group_key)
        self._trained_sources.update(trained_sources)
        self._pending_units = None
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
            protected_agent_record_ids=frozenset(protected),
            releasable_agent_record_ids=frozenset(releasable),
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        # --- scalar id sets ---
        self._consumed -= agent_record_ids
        self._terminal -= agent_record_ids
        self._trained_sources -= agent_record_ids
        self._terminal_owned_sources -= agent_record_ids
        self._seen_reports -= agent_record_ids

        # Stored records: only inferences can be here. The trainer compacts
        # ``releasable - protected``, and every live report, every reference
        # a live report holds, and every candidate's report are protected —
        # so a compacted id is never in _reports, _singletons, _waiting, or a
        # group.
        for agent_record_id in agent_record_ids:
            self._inferences.pop(agent_record_id, None)


# --------------------------------------------------------- shared helpers


@dataclass(frozen=True)
class SampleAssembly:
    """Shape a resolved report into a policy sample, without recipe policy.

    One model call becomes one sample via ``make_sample`` (default: the shared
    tensor reader); an ordered multi-reference report becomes one assembled
    multi-turn sample. ``accept_multi_turn`` gates consumption only —
    assembly always runs first, so a rejected multi-turn report was still
    fully resolved before the recipe refused its representation.
    """

    accept_multi_turn: bool = False
    realign_threshold: int = 1024
    scaffold_tolerance: int = 0
    make_sample: Callable[[AgentRecord, float], PolicySample] | None = None

    @classmethod
    def from_config(
        cls,
        context: ProcessorContext,
        make_sample: Callable[[AgentRecord, float], PolicySample] | None = None,
    ) -> SampleAssembly:
        accept_multi_turn, realign_threshold, scaffold_tolerance = sample_assembly_config_fields(context.config)
        return cls(accept_multi_turn, realign_threshold, scaffold_tolerance, make_sample)

    def build(self, context: ReportContext, score: float) -> PolicySample | None:
        """Build the sample for a resolved report; ``None`` is unusable."""
        inferences = context.inferences
        if inferences is None:
            raise RuntimeError("sample assembly requires resolved inferences")
        if len(inferences) == 1:
            sample: PolicySample | None = (
                make_policy_sample(inferences[0], score)
                if self.make_sample is None
                else self.make_sample(inferences[0], score)
            )
        else:
            sample = make_multi_turn_policy_sample(
                inferences,
                score,
                source_agent_record_id=context.report.agent_record_id,
                realign_threshold=self.realign_threshold,
                scaffold_tolerance=self.scaffold_tolerance,
            )
        if sample is not None and sample.is_multi_turn and not self.accept_multi_turn:
            return None
        return sample
