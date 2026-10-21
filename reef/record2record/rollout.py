"""Rollout groups: the record stage's half of rolling out saved tasks and training on their episodes.

A record stage that rolls out tasks holds two pools. Its task pool holds the saved tasks that wait to roll
out. Its ordinary pool holds the groups of rollout episodes: the training reports of one task use in one
rollout batch, keyed by ``(batch, task_use)`` from the episode labels, form one group.

The pipeline begins a rollout batch with its task uses. When the plays end, it hands over the reports that
each use's training episodes produced. A group settles when each of those reports joined it or ended, or
when the pipeline closes the batch's groups after its settle deadline. A report whose batch is not active, whose
task use is not one of the batch's, or whose arm is not the training arm, never trains.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from reef.core.batches import TaskItem
from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record.reports import GroupDecision, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2record.sets import RecordSetUnit

#: The episode label that names the rollout batch of an episode.
BATCH_LABEL = "batch"
#: The episode label that names the task use of an episode within its rollout batch.
TASK_USE_LABEL = "task_use"


def reported_label(report: AgentRecord, name: str) -> str | None:
    """Read one of the episode's labels from a task-player report."""
    labels = reported_episode(report).get("labels")
    label = labels.get(name) if isinstance(labels, Mapping) else None
    return label if isinstance(label, str) else None


def reported_episode_id(report: AgentRecord) -> str | None:
    """Read the episode ID from a task-player report; several reports of one episode share it."""
    episode_id = reported_episode(report).get("id")
    return episode_id if isinstance(episode_id, str) and episode_id else None


def reported_episode(report: AgentRecord) -> Mapping[str, object]:
    metadata = report.payload.get("metadata")
    episode = metadata.get("episode") if isinstance(metadata, Mapping) else None
    return episode if isinstance(episode, Mapping) else {}


def rollout_key(report: AgentRecord) -> tuple[str, str] | None:
    """The ``(batch, task_use)`` labels of a rollout episode's report, or None for another report."""
    batch = reported_label(report, BATCH_LABEL)
    task_use = reported_label(report, TASK_USE_LABEL)
    return (batch, task_use) if batch is not None and task_use is not None else None


def is_batch_group(unit: RecordSetUnit, batch: str) -> bool:
    """Whether a pool unit is a group of the rollout batch ``batch``."""
    key = unit.group_key
    return isinstance(key, tuple) and len(key) == 2 and key[0] == batch


@dataclass(frozen=True)
class PooledTask:
    """A saved task that waits to roll out.

    Values: ``task_id`` identifies the task in its pool. ``arrival_index``
    orders the tasks of a pool. ``group_key`` names what the task came from,
    such as its generation. ``item`` is the task that a rollout plays.
    Purpose: a record stage puts it in its ``TaskPool``; the pipeline takes it
    from the pool to submit it in a rollout batch, and puts it back when the
    batch did not train it.
    """

    task_id: Hashable
    arrival_index: int
    group_key: Hashable | None
    item: TaskItem


class TaskPool:
    """The saved tasks that wait to roll out.

    Values: a finite set of ``PooledTask`` values with distinct ``task_id``; ``arrival_count`` is larger than
    the ``arrival_index`` of each task that ``add`` made. ``take`` removes the oldest tasks by
    ``arrival_index``; ``put_back`` restores taken tasks with their indices; ``add`` gives a new index.
    Purpose: a rollout record stage holds it; the pipeline takes a rollout batch's tasks from it and puts back
    the tasks that the batch did not train.
    """

    def __init__(self) -> None:
        self.tasks_by_id: dict[Hashable, PooledTask] = {}
        self.arrival_count = 0

    def __len__(self) -> int:
        return len(self.tasks_by_id)

    def add(self, task_id: Hashable, item: TaskItem, *, group_key: Hashable | None = None) -> None:
        """Add a task after the held ones. A task with a held ID replaces it."""
        self.arrival_count += 1
        self.tasks_by_id[task_id] = PooledTask(task_id, self.arrival_count, group_key, item)

    def put_back(self, tasks: tuple[PooledTask, ...]) -> None:
        """Return taken tasks to the pool, in their old arrival order."""
        for task in tasks:
            self.tasks_by_id[task.task_id] = task

    def take(self, max_count: int) -> tuple[PooledTask, ...]:
        """Remove and return at most ``max_count`` tasks, oldest first."""
        taken = tuple(sorted(self.tasks_by_id.values(), key=lambda task: task.arrival_index)[:max_count])
        for task in taken:
            self.tasks_by_id.pop(task.task_id)
        return taken


class RolloutGrouping(ReportGrouping):
    """The training episodes of one task use in one rollout batch form one group.

    Values: ``active_keys`` holds the ``(batch, task_use)`` keys of the active batches. ``expected_reports``
    maps a key whose plays ended to the report IDs of its training episodes; ``ended_reports`` maps a key to
    its reports that ended without training; ``closed`` holds the keys whose batch's settle deadline passed.
    Every key in the last three sets is in ``active_keys`` until its batch ends.
    Purpose: ``RolloutReportRecord2Record`` owns it and updates it as batches begin, end their plays and end;
    the report stage asks it which reports train and when a group settles.

    A group settles when every report that its plays returned joined it or ended without training, or when
    its batch's groups are closed. A group of fewer than two training episodes has no relative reward and is
    discarded; several reports of one episode count as one episode. A report without an episode ID never
    trains, because its episode cannot be counted; it ends, and so still counts toward its group. With a
    ``training_arm``, only reports with that ``arm`` label train.
    """

    def __init__(self, training_arm: str | None = None) -> None:
        self.training_arm = training_arm
        #: The ``(batch, task_use)`` keys of the active rollout batches; only their episodes may train.
        self.active_keys: set[tuple[str, str]] = set()
        #: The reports that each group's plays returned, by group key, once its plays ended.
        self.expected_reports: dict[Hashable, frozenset[str]] = {}
        #: The reports of each group that ended without training, such as a skipped sample.
        self.ended_reports: dict[Hashable, set[str]] = {}
        #: The groups that settle with the reports they hold, because their batch's settle deadline passed.
        self.closed: set[Hashable] = set()

    def is_training_report(self, report: AgentRecord) -> bool:
        key = rollout_key(report)
        return (
            key is not None
            and key in self.active_keys
            and (self.training_arm is None or reported_label(report, "arm") == self.training_arm)
            and reported_episode_id(report) is not None
        )

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return rollout_key(context.report), None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if not self.is_complete(key, {member.report.agent_record_id for member in members}):
            return GroupDecision.INCOMPLETE
        episodes = {reported_episode_id(member.report) for member in members}
        return GroupDecision.READY if len(episodes) >= 2 else GroupDecision.DISCARD

    def is_complete(self, key: Hashable, member_report_ids: set[str]) -> bool:
        """Whether a group with these member reports can settle."""
        if key in self.closed:
            return True
        expected = self.expected_reports.get(key)
        return expected is not None and expected <= member_report_ids | self.ended_reports.get(key, set())


class RolloutReportRecord2Record(ReportRecord2Record):
    """A report stage that groups rollout episodes by ``(batch, task_use)``, with a task pool."""

    def __init__(
        self, context: PipelineContext, grouping: RolloutGrouping, *, exclusive_sources: bool = False
    ) -> None:
        super().__init__(context, grouping, exclusive_sources=exclusive_sources)
        self.rollout_grouping = grouping
        #: The saved tasks that wait to roll out.
        self.task_pool = TaskPool()
        #: The task uses of each active rollout batch.
        self.rollout_uses: dict[str, tuple[str, ...]] = {}

    def begin_rollout(self, batch: str, task_uses: tuple[str, ...]) -> None:
        """A rollout batch starts with these task uses; their episodes may train in it."""
        self.rollout_uses[batch] = task_uses
        self.rollout_grouping.active_keys.update((batch, task_use) for task_use in task_uses)

    def plays_ended(self, key: Hashable, report_ids: tuple[str, ...]) -> None:
        """The plays of the group ``key`` ended; these are the reports of its training episodes."""
        self.rollout_grouping.expected_reports[key] = frozenset(report_ids)
        # The plays can end before the group's first report arrives.
        if key in self.groups:
            self.refresh_group(key)

    def close_groups(self, batch: str) -> None:
        """Settle the batch's groups with the reports they hold: their settle deadline passed."""
        for task_use in self.rollout_uses.get(batch, ()):
            key = (batch, task_use)
            self.rollout_grouping.closed.add(key)
            if key in self.groups:
                self.refresh_group(key)

    def batch_groups_settled(self, batch: str) -> bool:
        """Whether every group of the batch is ready, discarded, or settled without a report."""
        for task_use in self.rollout_uses.get(batch, ()):
            key = (batch, task_use)
            group = self.groups.get(key)
            if group is None:
                if not self.rollout_grouping.is_complete(key, set()):
                    return False
            elif not group.discarded and ("group", key) not in self.pool.units_by_id:
                return False
        return True

    def end_report(self, report: AgentRecord) -> None:
        """End a report without training; a training report of the active batch still counts toward its group."""
        super().end_report(report)
        key = rollout_key(report)
        if key is None or key not in self.rollout_grouping.active_keys:
            return
        training_arm = self.rollout_grouping.training_arm
        if training_arm is not None and reported_label(report, "arm") != training_arm:
            return
        self.rollout_grouping.ended_reports.setdefault(key, set()).add(report.agent_record_id)
        group = self.groups.get(key)
        if group is not None and not group.discarded:
            self.refresh_group(key)

    def end_rollout(self, batch: str, *, trained: bool) -> None:
        """The batch trained, or was given up when ``trained`` is False; later reports of it never train."""
        for task_use in self.rollout_uses.pop(batch, ()):
            key = (batch, task_use)
            self.rollout_grouping.active_keys.discard(key)
            self.rollout_grouping.expected_reports.pop(key, None)
            self.rollout_grouping.ended_reports.pop(key, None)
            self.rollout_grouping.closed.discard(key)
            # A report that arrived after the batch was reserved ends with the batch.
            if key in self.groups:
                self.discard_group(key)
                self.groups.pop(key)

    def status(self) -> Mapping[str, object]:
        return {**super().status(), "task_pool": len(self.task_pool)}

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {**super().operational_metrics(), "task_pool": len(self.task_pool)}
