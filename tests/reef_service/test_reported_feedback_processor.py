"""Focused unit tests for the reported-feedback processor's state machine.

The recipe adapters' suites (test_trainer, test_tttd, test_openclawrl, ...)
pin the per-recipe behavior; these tests pin the engine mechanics themselves
with a minimal processor, where recipe policy cannot blur the state
machine.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

import pytest

from reef.core import AgentRecord, RequestType
from reef.train.processors.reported import (
    NEVER,
    WAIT,
    BatchUnit,
    Candidate,
    GroupDecision,
    Outcome,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
)
from reef.train.types import PolicyBatch, ProcessorContext, TrainingBatch


def inference(agent_record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={},
        agent_record_id=agent_record_id,
    )


def report(agent_record_id: str, *references: str, score: float = 1.0, trainable: bool = True) -> AgentRecord:
    payload = {"score": score, "references": list(references)}
    if not trainable:
        payload["metadata"] = {"training": {"eligible": False}}
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload=payload,
        agent_record_id=agent_record_id,
        references=references,
    )


def emit_ids(units: tuple[BatchUnit, ...], batch_number: int) -> TrainingBatch:
    values = tuple(candidate.value for unit in units for candidate in unit.candidates)
    return PolicyBatch(f"batch:{batch_number}", values)


def simple_judge(context: ReportContext) -> ReportDecision:
    """Train any scored, resolved report; its value is the report id."""
    if not context.references or context.score is None:
        return NEVER
    if context.inferences is None:
        return WAIT
    return ReportDecision.train(context.report.agent_record_id)


def engine(
    judge_fn: Callable[[ReportContext], ReportDecision] = simple_judge,
    *,
    batch_size: int = 1,
    decide_group_fn: Callable[[Hashable, tuple[Candidate, ...]], GroupDecision] | None = None,
    ordered: bool = False,
    exclusive: bool = False,
) -> ReportedFeedbackProcessor:
    """Build a minimal processor with the given recipe hooks."""

    class _TestProcessor(ReportedFeedbackProcessor):
        output_schema = PolicyBatch
        exclusive_sources = exclusive
        ordered_groups = ordered

        def judge(self, context: ReportContext) -> ReportDecision:
            return judge_fn(context)

        def decide_group(self, key: Hashable, candidates: tuple[Candidate, ...]) -> GroupDecision:
            if decide_group_fn is not None:
                return decide_group_fn(key, candidates)
            raise NotImplementedError

        def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> TrainingBatch:
            return emit_ids(units, batch_number)

    return _TestProcessor(ProcessorContext("math", {"batch_size": batch_size}))


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_outcome",
    [ReportDecision(Outcome.TRAIN), "train", None],
    ids=["decision", "string", "none"],
)
def test_report_decision_rejects_a_non_outcome_outcome(bad_outcome: object) -> None:
    # Regression: the engine branches on ``outcome is Outcome.X``, so a
    # non-Outcome (say, a ReportDecision passed where its outcome was meant) used to
    # fall through every branch and silently be treated as TRAIN.
    with pytest.raises(TypeError, match=r"ReportDecision\.outcome must be an Outcome, got "):
        ReportDecision(bad_outcome)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), float("-inf")])
def test_eligibility_rejects_missing_and_non_finite_scores(bad_score: float | None) -> None:
    context = ReportContext(report("r1", "i1"), bad_score, True, (inference("i1"),))
    assert context.eligibility() is NEVER


@pytest.mark.unit
def test_eligibility_walks_never_wait_and_spec_policy() -> None:
    # The gate returns the terminal or waiting ReportDecision itself, so a judge can
    # hand it straight back without re-wrapping a bare outcome.
    resolved = (inference("i1"),)
    assert ReportContext(report("r1"), 1.0, True, None).eligibility() is NEVER
    assert ReportContext(report("r1", "i1"), 1.0, False, resolved).eligibility() is NEVER
    assert ReportContext(report("r1", "i1"), 1.0, True, None).eligibility() is WAIT
    assert ReportContext(report("r1", "i1"), 1.0, True, resolved).eligibility() is None


@pytest.mark.unit
def test_report_decisions_are_final_and_computed_once_per_resolution() -> None:
    judged: list[str] = []

    def counting_judge(context: ReportContext) -> ReportDecision:
        judged.append(context.report.agent_record_id)
        return simple_judge(context)

    processor = engine(counting_judge)
    processor.ingest(report("r1", "i1"))
    for _ in range(3):
        assert not processor.ready()
    processor.ingest(inference("i1"))
    for _ in range(3):
        assert processor.ready()

    # Judged once unresolved (WAIT), once at resolution (TRAIN) — polling is free.
    assert judged == ["r1", "r1"]


@pytest.mark.unit
def test_wait_decision_requires_an_unresolved_reference() -> None:
    processor = engine(lambda context: WAIT)
    processor.ingest(inference("i1"))
    with pytest.raises(RuntimeError, match="WAIT"):
        processor.ingest(report("r1", "i1"))


@pytest.mark.unit
def test_never_decision_releases_report_and_owned_sources() -> None:
    processor = engine(lambda context: NEVER, exclusive=True)
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))

    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "r1"})


@pytest.mark.unit
def test_never_decision_keeps_a_source_claimed_by_a_live_report() -> None:
    def judge(context: ReportContext) -> ReportDecision:
        if context.report.agent_record_id == "dead":
            return NEVER
        return simple_judge(context)

    processor = engine(judge, exclusive=True)
    processor.ingest(inference("i1"))
    processor.ingest(report("live", "i1"))
    processor.ingest(report("dead", "i1"))

    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset({"i1", "live"})
    assert decision.releasable_agent_record_ids == frozenset({"dead"})


@pytest.mark.unit
def test_reports_referencing_consumed_sources_are_terminal_at_ingest() -> None:
    processor = engine(exclusive=True)
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    batch = processor.build_batch()
    processor.acknowledge(batch.batch_id)
    assert "i1" in processor.retention_decision().releasable_agent_record_ids

    processor.ingest(report("late", "i1"))
    assert not processor.ready()
    assert "late" in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
@pytest.mark.parametrize("dead_report_first", [True, False])
def test_late_retry_reclaims_a_source_owned_by_a_dead_report(dead_report_first: bool) -> None:
    """A terminal report's released source is derived, not latched: while the
    source record exists, a late retry re-claims it and trains, in either
    arrival order."""

    def judge(context: ReportContext) -> ReportDecision:
        if context.report.agent_record_id == "dead":
            return NEVER
        return simple_judge(context)

    processor = engine(judge, exclusive=True)
    processor.ingest(inference("i1"))
    order = ("dead", "retry") if dead_report_first else ("retry", "dead")
    for report_id in order:
        processor.ingest(report(report_id, "i1"))
        processor.retention_decision()  # a read between arrivals must not poison the retry

    assert processor.ready()
    batch = processor.build_batch()
    assert batch.samples == ("retry",)
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset({"retry", "i1"})
    assert decision.releasable_agent_record_ids == frozenset({"dead"})

    processor.acknowledge(batch.batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"dead", "retry", "i1"})


@pytest.mark.unit
def test_blocked_terminal_source_is_releasable_once_its_claimant_dies() -> None:
    """A source owned by a terminal report but claimed by a live one is
    blocked, not forgotten: when the claimant later dies without owning it,
    the next retention read reports the source releasable."""

    def judge(context: ReportContext) -> ReportDecision:
        if not context.trainable:
            return NEVER
        if context.inferences is None:
            return WAIT
        return NEVER  # assembly failed: the claimant dies without owning i1

    processor = engine(judge, exclusive=False)
    processor.ingest(report("claimant", "i1"))  # live, waiting on i1
    processor.ingest(report("owner", "i1", trainable=False))  # terminal, owns i1, blocked by the claimant
    decision = processor.retention_decision()
    assert "i1" in decision.protected_agent_record_ids
    assert decision.releasable_agent_record_ids == frozenset({"owner"})

    processor.ingest(inference("i1"))  # resolves the claimant, which is judged NEVER
    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset({"owner", "claimant", "i1"})


@pytest.mark.unit
def test_retention_decision_is_a_pure_read() -> None:
    processor = engine()
    processor.ingest(report("r1", "i1"))
    processor.ingest(inference("i2"))

    first = processor.retention_decision()
    second = processor.retention_decision()
    assert first == second
    assert first.protected_agent_record_ids == frozenset({"r1", "i1", "i2"})


@pytest.mark.unit
def test_group_discard_releases_every_member_and_poisons_the_key() -> None:
    def grouped_judge(context: ReportContext) -> ReportDecision:
        decision = simple_judge(context)
        if decision.outcome is not Outcome.TRAIN:
            return decision
        return ReportDecision.train(decision.value, group_key="g")

    def barrier(key, candidates) -> GroupDecision:
        return GroupDecision.DISCARD if len(candidates) >= 2 else GroupDecision.INCOMPLETE

    processor = engine(grouped_judge, decide_group_fn=barrier, exclusive=True)
    for index in (1, 2):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))

    assert not processor.ready()
    decision = processor.retention_decision()
    assert decision.releasable_agent_record_ids == frozenset({"i1", "i2", "r1", "r2"})

    # The discarded key is poisoned: a later member is terminal on arrival.
    processor.ingest(inference("i3"))
    processor.ingest(report("r3", "i3"))
    assert "r3" in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
def test_slot_collision_keeps_the_first_durable_report() -> None:
    def slotted_judge(context: ReportContext) -> ReportDecision:
        decision = simple_judge(context)
        if decision.outcome is not Outcome.TRAIN:
            return decision
        return ReportDecision.train(decision.value, group_key="g", slot="only")

    def barrier(key, candidates) -> GroupDecision:
        return GroupDecision.READY if len(candidates) >= 1 else GroupDecision.INCOMPLETE

    processor = engine(slotted_judge, decide_group_fn=barrier, exclusive=True)
    processor.ingest(inference("i1"))
    processor.ingest(report("first", "i1"))
    processor.ingest(inference("i2"))
    processor.ingest(report("retry", "i2"))

    batch = processor.build_batch()
    assert batch.samples == ("first",)
    decision = processor.retention_decision()
    assert "retry" in decision.releasable_agent_record_ids
    assert "i2" in decision.releasable_agent_record_ids


@pytest.mark.unit
def test_ordered_groups_coexist_with_singleton_decisions() -> None:
    """Group-key order is only compared among groups, never against
    singleton arrival order: singletons batch first, sorted groups follow."""

    def judge(context: ReportContext) -> ReportDecision:
        decision = simple_judge(context)
        if decision.outcome is not Outcome.TRAIN or not context.report.agent_record_id.startswith("g"):
            return decision
        return ReportDecision.train(decision.value, group_key=f"key-{context.report.agent_record_id}")

    def barrier(key, candidates) -> GroupDecision:
        return GroupDecision.READY if len(candidates) >= 1 else GroupDecision.INCOMPLETE

    processor = engine(judge, batch_size=3, decide_group_fn=barrier, ordered=True)
    processor.ingest(inference("i1"))
    processor.ingest(report("g-z", "i1"))  # grouped candidates arrive first...
    processor.ingest(inference("i2"))
    processor.ingest(report("g-a", "i2"))
    processor.ingest(inference("i3"))
    processor.ingest(report("s1", "i3"))  # ...the singleton last

    batch = processor.build_batch()
    assert batch.samples == ("s1", "g-a", "g-z")


@pytest.mark.unit
def test_units_emit_in_report_arrival_order_across_late_resolution() -> None:
    processor = engine(batch_size=2)
    processor.ingest(report("r1", "i1"))
    processor.ingest(report("r2", "i2"))
    # Inferences resolve in the opposite order; candidate order follows resolution.
    processor.ingest(inference("i2"))
    processor.ingest(inference("i1"))

    batch = processor.build_batch()
    assert batch.samples == ("r2", "r1")


@pytest.mark.unit
def test_compaction_scrubs_the_ledger_for_reingest() -> None:
    processor = engine()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    batch = processor.build_batch()
    processor.acknowledge(batch.batch_id)
    compacted = frozenset({"i1", "r1"})
    processor.compaction_applied(compacted)

    decision = processor.retention_decision()
    assert decision.protected_agent_record_ids == frozenset()
    assert decision.releasable_agent_record_ids == frozenset()

    # The same ids arriving again are new records, not duplicates.
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    assert processor.ready()


@pytest.mark.unit
def test_duplicate_report_ids_are_ingested_once() -> None:
    processor = engine(batch_size=2)
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    processor.ingest(report("r1", "i1"))
    assert not processor.ready()
