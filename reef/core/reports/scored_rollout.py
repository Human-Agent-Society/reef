"""Score-only report contracts."""

from dataclasses import dataclass

from reef.core.reports.base import ReportBase

__all__ = ["ScoredRolloutReport"]


@dataclass(frozen=True)
class ScoredRolloutReport(ReportBase):
    """One rollout carrying a finite numeric score."""

    score: float
