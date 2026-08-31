"""Core backend-neutral output types for step preparers.

Optional implementation helpers live in :mod:`reef.train.algos.helpers`;
the schedule materializer backends share lives in :mod:`reef.train.algos.schedule`.
Registered-preparer APIs live in :mod:`reef.train.algos.registry`.
"""

from reef.train.algos.signals import StepScheduling, StepSignal

__all__ = [
    "StepScheduling",
    "StepSignal",
]
