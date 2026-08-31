"""TTT-Discover's grouped rollout report contract."""

from dataclasses import dataclass

from reef.core.reports import ReportValidationError, ScoredRolloutReport

__all__ = ["TTTDGroupedRolloutReport"]


@dataclass(frozen=True)
class TTTDGroupedRolloutReport(ScoredRolloutReport):
    """One rollout addressed into a TTT-Discover comparison grid."""

    score: float
    step: int
    group: int
    rollout: int
    groups_per_step: int
    rollouts_per_group: int
    algorithm: str = "tttd"
    comparison_set: str = ""

    def __post_init__(self) -> None:
        if not self.comparison_set:
            object.__setattr__(self, "comparison_set", f"tttd-step-{self.step}-group-{self.group}")
        super().__post_init__()

    def validate(self) -> None:
        if self.algorithm not in ("tttd", "ttt-discover"):
            raise ReportValidationError("metadata.algorithm must be 'tttd' or 'ttt-discover'")
        if self.step < 0 or self.groups_per_step < 1 or self.rollouts_per_group < 2:
            raise ReportValidationError(
                "metadata grid requires step >= 0, groups_per_step >= 1, rollouts_per_group >= 2"
            )
        if not 0 <= self.group < self.groups_per_step:
            raise ReportValidationError("metadata.group must sit inside metadata.groups_per_step")
        if not 0 <= self.rollout < self.rollouts_per_group:
            raise ReportValidationError("metadata.rollout must sit inside metadata.rollouts_per_group")
        if self.comparison_set != f"tttd-step-{self.step}-group-{self.group}":
            raise ReportValidationError("metadata.comparison_set must echo the coordinates")
