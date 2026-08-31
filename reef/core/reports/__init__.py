"""Typed report contracts: the schema of the feedback a harness sends.

A report is the second thing a harness sends Reef, after the inference it
refers to (``AgentRecord``); both live in ``core`` because the service
validates a report at ingress, the scenario binds its type, the recipe
declares it, and the processors rehydrate it — packages that do not depend
on each other. ``ReportBase`` is the contract and ``ScoredRolloutReport`` is
the shared score-only vocabulary; method-specific contracts live with their
method packages.
"""

from reef.core.reports.base import ReportBase, ReportValidationError
from reef.core.reports.scored_rollout import ScoredRolloutReport

__all__ = [
    "ReportBase",
    "ReportValidationError",
    "ScoredRolloutReport",
]
