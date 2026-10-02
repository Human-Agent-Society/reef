"""Core backend-neutral output types for training objectives.

Optional implementation helpers live in :mod:`reef.train.algos.helpers`;
the schedule materializer backends share lives in :mod:`reef.train.algos.schedule`.
Registered-objective APIs live in :mod:`reef.train.algos.registry`.
"""

from reef.core.batches import StepScheduling
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.signals import StepSignal

__all__ = [
    "StepScheduling",
    "StepSignal",
    "TrainingObjective",
]
