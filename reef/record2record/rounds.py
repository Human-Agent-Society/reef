"""Rounds: the record stage decides when a task-generation round is due, from which sessions, and which saved tasks play.

Task generation reads a description and, optionally, recorded sessions.
``RoundRecord2Record`` paces the rounds, holds the source sessions for them,
and holds the saved tasks that wait to play in its task pool. Otherwise it
works as a ``ReportRecord2Record`` for the reports that the plays produce.

When a round is due, the stage forms a ``RoundSet`` with the round number and
the round's source sessions. The record-to-task stage takes the set like any
other formed set. It usually returns ``TaskStatus.PENDING`` and hands the
result back later. The round ends when the set settles. A round set that
settles ``READY`` puts one ``TaskSet`` for each of its tasks in the task pool.

When no play runs and the task pool holds tasks, the stage takes every task
in the pool into one ``PlaySet``. The record-to-task stage plays it, and the
play ends when the set settles.

The round limit counts the rounds that finished a generation: a round set that
settles ``READY`` or ``SKIPPED``. A round that made nothing new (``INGESTED``)
or was deferred (``DEFERRED``) does not count.

A round is due when no round is running, the round limit is not reached, the
task pool holds fewer than ``refill_tasks`` tasks, and one of these is true:

* no round has started yet (the first round after a start);
* the last round set settled ``SKIPPED`` or ``INGESTED``, for example a round
  that made no train task, or the last play played no task, so no batch comes;
* the last round was deferred, and ``deferred_retry_seconds`` passed or new
  sessions wait;
* ``batches_per_round`` batches trained since the last round started. A batch
  that the backend dropped as stale does not count.

A source session is a report that does not train and that the recipe's
``RoundGrouping.is_source_report`` accepts, with its inference records. With
``sources_per_round`` above zero, the stage holds source sessions in arrival
order. Each round takes at most ``sources_per_round`` of them; a round
without a waiting session has no sources. The pipeline holds a round set's
records while the set is pending, and releases them when the set settles.

Rounds take their sessions from a run. A run starts with the sessions that
wait at that moment; sessions that arrive later wait for the next run. A run
ends when it has no session left, after ``max_rounds_per_run`` rounds, or
after ``max_run_seconds``; a limit of 0 is no limit. Sessions that a run did
not reach wait for the next run. A round set that settles ``DEFERRED`` keeps
its sessions. The next run, which starts when new sessions wait, runs that
round again with the same sessions before it takes new ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.batches import TaskItem, TrainDataItem
from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record.base import TaskStatus
from reef.record2record.pool import Pool
from reef.record2record.reports import ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2record.sets import RecordSet, TaskSet


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
        return session_record_ids(self.sources)


@dataclass(frozen=True, kw_only=True)
class PlaySet(RecordSet):
    """Saved tasks to play together. ``set_id`` is ``("play", play_number)``; the set holds no records."""

    play_number: int
    tasks: tuple[TaskItem, ...]

    def record_ids(self) -> tuple[str, ...]:
        return ()


class RoundRecord2Record(ReportRecord2Record):
    """A report stage that forms round sets and play sets, paces them, and holds their sessions and tasks.

    The stage forms no round until ``resume_rounds`` sets the first round
    number and the round limit. At the limit, the stage stops the rounds and
    holds no more source sessions. The subclass's settings check that
    ``batches_per_round`` and ``refill_tasks`` are positive, and that
    ``sources_per_round`` and the run limits are not negative.
    """

    def __init__(
        self,
        context: PipelineContext,
        grouping: RoundGrouping,
        *,
        batches_per_round: int,
        sources_per_round: int = 0,
        exclusive_sources: bool = False,
        refill_tasks: int = 1,
        max_rounds_per_run: int = 0,
        max_run_seconds: float = 0.0,
        deferred_retry_seconds: float = 600.0,
    ) -> None:
        super().__init__(context, grouping, exclusive_sources=exclusive_sources)
        self.round_grouping = grouping
        self.batches_per_round = batches_per_round
        self.sources_per_round = sources_per_round
        self.refill_tasks = refill_tasks
        self.max_rounds_per_run = max_rounds_per_run
        self.max_run_seconds = max_run_seconds
        self.deferred_retry_seconds = deferred_retry_seconds
        self.next_round = 0
        self.completed_rounds = 0
        self.round_limit = 0
        self.is_retry_waiting = False
        self.retry_at: float | None = None
        self.started_rounds = 0
        self.round_running = False
        self.follow_at_once = False
        self.rounds_stopped = False
        # Batches that trained since the last round started; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.waiting_sources: list[ReportContext] = []
        #: The sessions of each deferred round; each one runs again as one round of the next run.
        self.deferred_rounds: list[tuple[ReportContext, ...]] = []
        #: The deferred rounds that the current run has not run again yet.
        self.run_deferred: list[tuple[ReportContext, ...]] = []
        #: The sessions that the current run has not given to a round yet; None between runs.
        self.run_sources: list[ReportContext] | None = None
        self.run_started_at = 0.0
        self.run_rounds = 0
        self.task_pool: Pool[TaskSet] = Pool()
        self.next_play = 0
        self.play_running = False

    def resume_rounds(self, *, completed_rounds: int, round_limit: int) -> None:
        """Set the rounds that finished a generation and the limit; a restart continues from its store."""
        self.completed_rounds = completed_rounds
        self.round_limit = round_limit

    def add_tasks(self, tasks: tuple[TrainDataItem, ...], *, round_number: int) -> None:
        """Put saved tasks in the task pool; a restart puts back the tasks that have not played."""
        for task in tasks:
            if not isinstance(task, TaskItem):
                raise TypeError("a round's tasks must be TaskItem values")
            self.task_pool.put(
                TaskSet(
                    set_id=task.task_path.name,
                    arrival_index=self.task_pool.next_arrival_index(),
                    group_key=round_number,
                    item=task,
                )
            )

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

    def round_due(self, now: float) -> bool:
        """Whether the next round should start now."""
        if self.rounds_stopped or self.round_running or len(self.task_pool) >= self.refill_tasks:
            return False
        if self.retry_at is not None and (now >= self.retry_at or self.waiting_sources):
            return True
        return self.started_rounds == 0 or self.follow_at_once or self.trained_batches >= self.batches_per_round

    def poll(self, now: float) -> None:
        """Form a play set when no play runs, then the next round set when a round is due."""
        super().poll(now)
        if not self.play_running and len(self.task_pool) > 0:
            self.task_pool.reserve(len(self.task_pool))
            self.formed.append(
                PlaySet(
                    set_id=("play", self.next_play),
                    arrival_index=self.pool.next_arrival_index(),
                    play_number=self.next_play,
                    tasks=tuple(task_set.item for task_set in self.task_pool.consume_reserved()),
                )
            )
            self.next_play += 1
            self.play_running = True
        if self.is_retry_waiting:
            self.retry_at = now + self.deferred_retry_seconds
            self.is_retry_waiting = False
        if not self.round_due(now):
            return
        if self.completed_rounds >= self.round_limit:
            self.stop_rounds()
            return
        self.formed.append(
            RoundSet(
                set_id=("round", self.next_round),
                arrival_index=self.pool.next_arrival_index(),
                round_number=self.next_round,
                sources=self.take_run_sources(now),
            )
        )
        self.next_round += 1
        self.started_rounds += 1
        self.round_running = True
        self.follow_at_once = False
        self.retry_at = None
        # The batches that the round's plays make count toward the next one.
        self.trained_batches = 0

    def take_run_sources(self, now: float) -> tuple[ReportContext, ...]:
        """The next round's sessions from the current run; a run that is over gives way to a new one."""
        if self.sources_per_round == 0:
            return ()
        if self.run_sources is not None and (
            not (self.run_sources or self.run_deferred)
            or 0 < self.max_rounds_per_run <= self.run_rounds
            or 0 < self.max_run_seconds <= now - self.run_started_at
        ):
            self.waiting_sources = [*self.run_sources, *self.waiting_sources]
            self.deferred_rounds = [*self.run_deferred, *self.deferred_rounds]
            self.run_sources, self.run_deferred = None, []
        if self.run_sources is None:
            # Deferred rounds wait for a run that new sessions start.
            if not self.waiting_sources:
                return ()
            self.run_sources, self.waiting_sources = self.waiting_sources, []
            self.run_deferred, self.deferred_rounds = self.deferred_rounds, []
            self.run_started_at = now
            self.run_rounds = 0
        self.run_rounds += 1
        if self.run_deferred:
            return self.run_deferred.pop(0)
        sources = tuple(self.run_sources[: self.sources_per_round])
        self.run_sources = self.run_sources[self.sources_per_round :]
        return sources

    def settle(
        self,
        record_set: RecordSet,
        status: TaskStatus,
        producing_runtime_load_ids: frozenset[str | None] = frozenset(),
        tasks: tuple[TrainDataItem, ...] = (),
    ) -> None:
        """End a settled round or play set; a ``READY`` round puts its tasks in the task pool."""
        if isinstance(record_set, PlaySet):
            self.ended_set_ids.add(record_set.set_id)
            self.play_running = False
            # A play that played no task makes no batch to wait for.
            self.follow_at_once = self.follow_at_once or status is not TaskStatus.READY
            return
        if not isinstance(record_set, RoundSet):
            super().settle(record_set, status, producing_runtime_load_ids, tasks)
            return
        self.ended_set_ids.add(record_set.set_id)
        self.round_running = False
        self.follow_at_once = status in (TaskStatus.SKIPPED, TaskStatus.INGESTED)
        if status in (TaskStatus.READY, TaskStatus.SKIPPED):
            self.completed_rounds += 1
        if status is TaskStatus.READY:
            self.add_tasks(tasks, round_number=record_set.round_number)
        elif status is TaskStatus.DEFERRED:
            self.is_retry_waiting = True
            if record_set.sources and not self.rounds_stopped:
                self.deferred_rounds.append(record_set.sources)

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.waiting_sources = []
        self.deferred_rounds, self.run_deferred = [], []
        self.run_sources = None

    def consume(self, record_sets: tuple[RecordSet, ...], *, trained: bool) -> frozenset[str]:
        if trained:
            self.trained_batches += 1
        return super().consume(record_sets, trained=trained)

    def releasable_record_ids(self) -> frozenset[str]:
        deferred = [session for sources in (*self.deferred_rounds, *self.run_deferred) for session in sources]
        held = session_record_ids((*self.waiting_sources, *deferred, *(self.run_sources or ())))
        return super().releasable_record_ids() - frozenset(held)

    def status(self) -> dict[str, object]:
        return {**super().status(), "task_pool": len(self.task_pool)}

    def operational_metrics(self) -> dict[str, float | int]:
        return {
            **super().operational_metrics(),
            "waiting_source_sessions": len(self.waiting_sources),
            "deferred_rounds": len(self.deferred_rounds) + len(self.run_deferred),
            "task_pool": len(self.task_pool),
        }


def session_record_ids(sessions: tuple[ReportContext, ...]) -> tuple[str, ...]:
    """The record IDs of source sessions: each session's inference records, then its report."""
    return tuple(
        record_id for session in sessions for record_id in (*session.references, session.report.agent_record_id)
    )
