"""Rounds: the record stage decides when a task-generation round is due, and from which sessions.

Task generation reads a description and, optionally, recorded sessions.
``RoundRecord2Record`` paces the rounds and holds the source sessions for
them. Otherwise it works as a ``ReportRecord2Record`` for the reports that
the rounds' plays produce. The record-to-task stage runs the rounds: each
time the pipeline looks for a batch, it asks ``round_due``, starts a due
round with ``start_round``, and reports ``round_finished``; when no round
can run any more, it calls ``stop_rounds``.

A round is due when no round is running and one of these is true:

* no round has started yet (the first round after a start);
* the last round played no task, so it makes no batch to wait for;
* ``batches_per_round`` batches trained since the last round started. A batch
  that the backend dropped as stale trained nothing and does not count.

A source session is a report that does not train and that the recipe's
``RoundGrouping.is_source_report`` accepts, with its inference records. With
``sources_per_round`` above zero, the stage holds source sessions in arrival
order. Each round takes at most ``sources_per_round`` of them; a round
without a waiting session has no sources. The stage keeps a session's records
until the round that uses it finishes, then releases them.
"""

from __future__ import annotations

from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record.reports import ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2record.sets import RecordSet


class RoundGrouping(ReportGrouping):
    """The report policy of a task-generation recipe, and which reports are source sessions."""

    def is_source_report(self, context: ReportContext) -> bool:
        """Whether a report that does not train is a recorded session to generate tasks from. The default is no."""
        return False


class RoundRecord2Record(ReportRecord2Record):
    """A report stage that paces task-generation rounds and holds the source sessions for them.

    The record-to-task stage starts each due round and tells the stage when
    it finished. After ``stop_rounds``, the stage holds no more source sessions.
    """

    def __init__(
        self,
        context: PipelineContext,
        grouping: RoundGrouping,
        *,
        batches_per_round: int,
        sources_per_round: int = 0,
        exclusive_sources: bool = False,
    ) -> None:
        super().__init__(context, grouping, exclusive_sources=exclusive_sources)
        if batches_per_round < 1:
            raise ValueError("batches_per_round must be positive")
        if sources_per_round < 0:
            raise ValueError("sources_per_round must be non-negative")
        self.round_grouping = grouping
        self.batches_per_round = batches_per_round
        self.sources_per_round = sources_per_round
        self.started_rounds = 0
        self.round_running = False
        self.follow_at_once = False
        self.rounds_stopped = False
        # Batches that trained since the last round started; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.waiting_sources: list[ReportContext] = []
        self.round_sources: tuple[ReportContext, ...] = ()

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round finishes.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        context = self.report_context(report, None)
        if self.round_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    # ---------------------------------------------------------------- rounds

    def round_due(self) -> bool:
        """Whether the record-to-task stage should start a round now."""
        if self.rounds_stopped or self.round_running:
            return False
        return self.started_rounds == 0 or self.follow_at_once or self.trained_batches >= self.batches_per_round

    def start_round(self) -> tuple[ReportContext, ...]:
        """A round starts: return its source sessions, at most ``sources_per_round`` of the waiting ones.

        The batches that the round's plays make count toward the next one.
        """
        sources = tuple(self.waiting_sources[: self.sources_per_round])
        self.waiting_sources = self.waiting_sources[self.sources_per_round :]
        self.started_rounds += 1
        self.round_running = True
        self.follow_at_once = False
        self.trained_batches = 0
        self.round_sources = sources
        return sources

    def round_finished(self, played_none: bool) -> None:
        """A round finished. A round that played no task makes no batch, so the next one is due at once."""
        self.round_running = False
        self.follow_at_once = played_none
        self.round_sources = ()

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.round_running = False
        self.round_sources = ()
        self.waiting_sources = []

    def consume(self, record_sets: tuple[RecordSet, ...], *, trained: bool) -> frozenset[str]:
        if trained:
            self.trained_batches += 1
        return super().consume(record_sets, trained=trained)

    def releasable_record_ids(self) -> frozenset[str]:
        held = {
            record_id
            for context in (*self.waiting_sources, *self.round_sources)
            for record_id in (*context.references, context.report.agent_record_id)
        }
        return super().releasable_record_ids() - held

    def operational_metrics(self) -> dict[str, float | int]:
        return {**super().operational_metrics(), "waiting_source_sessions": len(self.waiting_sources)}
