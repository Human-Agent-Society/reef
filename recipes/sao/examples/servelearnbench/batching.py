"""Keep a Retail optimizer batch together, including a window's smaller tail."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.pipeline_context import PipelineContext
from reef.record2record import GroupDecision, Record2RecordSettings, ReportContext, ReportGrouping, ReportRecord2Record


@dataclass(frozen=True)
class BatchSettings(Record2RecordSettings):
    """A positive maximum task count per batch, plus the inherited staleness limit."""

    max_batch_size: int = config_option(64, help="Maximum number of distinct tasks in one declared batch.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.max_batch_size, bool)
            or not isinstance(self.max_batch_size, int)
            or self.max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")


class BatchGrouping(ReportGrouping):
    """One ordered task-ID tuple is one batch; each task owns exactly one retry slot.

    A report must name its own task in a nonempty, unique list no longer than
    max_batch_size. The batch becomes ready only when every slot has a scored
    trajectory. This is a collection barrier, not a reward comparison group.
    """

    def __init__(self, max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size

    def placement(self, context: ReportContext) -> tuple[Hashable, Hashable]:
        metadata = context.report.payload.get("metadata", {})
        task_ids = metadata.get("batch_task_ids")
        task_id = metadata.get("task", {}).get("name")
        if (
            not isinstance(task_ids, list)
            or not 0 < len(task_ids) <= self.max_batch_size
            or any(not isinstance(value, str) or not value for value in task_ids)
            or len(set(task_ids)) != len(task_ids)
            or task_id not in task_ids
        ):
            raise ValueError("Retail reports require a unique batch_task_ids list containing their task name")
        return tuple(task_ids), task_id

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < len(key):
            return GroupDecision.INCOMPLETE
        versions = {record.payload.get("runtime_load_id") for member in members for record in member.inferences}
        if len(versions) != 1 or not all(isinstance(version, str) and version for version in versions):
            raise ValueError("A Retail batch must contain trajectories from one serving weight version")
        return GroupDecision.READY


class RetailBatchReports(ReportRecord2Record):
    """Exclusive episode reports held until their declared optimizer batch is complete.

    Reef persists and replays the reports, including their batch membership.
    Its existing group consumption handles retries and checkpoint recovery.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        settings = BatchSettings.parse(config, plugin_class=cls)
        return cls(
            context,
            BatchGrouping(settings.max_batch_size),
            exclusive_sources=True,
            max_staleness=settings.max_staleness,
        )
