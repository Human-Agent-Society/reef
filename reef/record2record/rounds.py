"""Rounds: the record stage decides when a task-generation round is due, and from which sessions.

Task generation reads a description and, optionally, recorded sessions.
``RoundRecord2Record`` paces the rounds and holds the source sessions for
them. Otherwise it works as a ``ReportRecord2Record`` for the reports that
the rounds' plays produce.

When a round is due, the stage forms a ``RoundSet`` with the round number and
the round's source sessions. The record-to-task stage takes the set like any
other formed set. It usually returns ``TaskStatus.PENDING`` and hands the
result back later. The round ends when the set settles.

A round is due when no round is running, the round limit is not reached, and
one of these is true:

* no round has started yet (the first round after a start);
* the last round set settled as anything but ``READY``, for example a round
  that played no task, so it makes no batch to wait for;
* ``batches_per_round`` batches trained since the last round started. A batch
  that the backend dropped as stale does not count.

A source session is a report that does not train and that the recipe's
``RoundGrouping.is_source_report`` accepts, with its inference records. With
``sources_per_round`` above zero, the stage holds source sessions in arrival
order. Each round takes at most ``sources_per_round`` of them; a round
without a waiting session has no sources. The pipeline holds a round set's
records while the set is pending, and releases them when the set settles.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record.base import TaskStatus
from reef.record2record.reports import ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2record.sets import RecordSet


class RoundGrouping(ReportGrouping):
    """The report policy of a task-generation recipe, and which reports are source sessions."""

    def is_source_report(self, context: ReportContext) -> bool:
        """Whether a report that does not train is a recorded session to generate tasks from. The default is no."""
        return False


@dataclass(frozen=True, kw_only=True)
class RoundSet(RecordSet):
    """One task-generation round: its number and its source sessions. ``set_id`` is ``("round", round_number)``."""

    round_number: int
    sources: tuple[ReportContext, ...] = ()

    def record_ids(self) -> tuple[str, ...]:
        return tuple(
            record_id for source in self.sources for record_id in (*source.references, source.report.agent_record_id)
        )


class RoundRecord2Record(ReportRecord2Record):
    """A report stage that forms task-generation round sets, paces them, and holds their source sessions.

    The stage forms no round until ``resume_rounds`` sets the first round
    number and the round limit. At the limit, the stage stops the rounds and
    holds no more source sessions. The subclass's settings check that
    ``batches_per_round`` is positive and ``sources_per_round`` is not negative.
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
        self.round_grouping = grouping
        self.batches_per_round = batches_per_round
        self.sources_per_round = sources_per_round
        self.next_round = 0
        self.round_limit = 0
        self.started_rounds = 0
        self.round_running = False
        self.follow_at_once = False
        self.rounds_stopped = False
        # Batches that trained since the last round started; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.waiting_sources: list[ReportContext] = []

    def resume_rounds(self, *, next_round: int, round_limit: int) -> None:
        """Set the number of the next round and the number of rounds in all; a restart continues from its summaries."""
        self.next_round = next_round
        self.round_limit = round_limit

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round takes it.
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
        """Whether the next round should start now."""
        if self.rounds_stopped or self.round_running:
            return False
        return self.started_rounds == 0 or self.follow_at_once or self.trained_batches >= self.batches_per_round

    def poll(self, now: float) -> None:
        """Form the next round set when a round is due, or stop the rounds at the limit."""
        super().poll(now)
        if not self.round_due():
            return
        if self.next_round >= self.round_limit:
            self.stop_rounds()
            return
        sources = tuple(self.waiting_sources[: self.sources_per_round])
        self.waiting_sources = self.waiting_sources[self.sources_per_round :]
        self.formed.append(
            RoundSet(
                set_id=("round", self.next_round),
                arrival_index=self.pool.next_arrival_index(),
                round_number=self.next_round,
                sources=sources,
            )
        )
        self.next_round += 1
        self.started_rounds += 1
        self.round_running = True
        self.follow_at_once = False
        # The batches that the round's plays make count toward the next one.
        self.trained_batches = 0

    def settle(
        self,
        record_set: RecordSet,
        status: TaskStatus,
        producing_runtime_load_ids: frozenset[str | None] = frozenset(),
    ) -> None:
        """End a settled round set. A round that settled as anything but ``READY`` makes no batch to wait for."""
        if not isinstance(record_set, RoundSet):
            super().settle(record_set, status, producing_runtime_load_ids)
            return
        self.ended_set_ids.add(record_set.set_id)
        self.round_running = False
        self.follow_at_once = status is not TaskStatus.READY

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.waiting_sources = []

    def consume(self, record_sets: tuple[RecordSet, ...], *, trained: bool) -> frozenset[str]:
        if trained:
            self.trained_batches += 1
        return super().consume(record_sets, trained=trained)

    def releasable_record_ids(self) -> frozenset[str]:
        held = {
            record_id
            for context in self.waiting_sources
            for record_id in (*context.references, context.report.agent_record_id)
        }
        return super().releasable_record_ids() - held

    def operational_metrics(self) -> dict[str, float | int]:
        return {**super().operational_metrics(), "waiting_source_sessions": len(self.waiting_sources)}
