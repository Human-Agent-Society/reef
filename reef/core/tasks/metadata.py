"""Task identity and arm labels shared by generation and training stages."""

from collections.abc import Mapping

from reef.core.records_types import AgentRecord

TRAINING_ARM = "plain"


def reported_task_name(report: AgentRecord) -> str | None:
    """Read the task player's task name from a report."""
    metadata = report.payload.get("metadata")
    task = metadata.get("task") if isinstance(metadata, Mapping) else None
    name = task.get("name") if isinstance(task, Mapping) else None
    return name if isinstance(name, str) and name else None


def reported_arm(report: AgentRecord) -> str | None:
    """Read the episode's arm label from a task-player report."""
    metadata = report.payload.get("metadata")
    episode = metadata.get("episode") if isinstance(metadata, Mapping) else None
    labels = episode.get("labels") if isinstance(episode, Mapping) else None
    arm = labels.get("arm") if isinstance(labels, Mapping) else None
    return arm if isinstance(arm, str) else None
