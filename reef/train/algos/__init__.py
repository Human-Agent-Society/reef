"""Core backend-neutral output types for training objectives.

Optional implementation helpers live in :mod:`reef.train.algos.helpers`;
the schedule materializer backends share lives in :mod:`reef.train.algos.schedule`.
Registered-objective APIs live in :mod:`reef.train.algos.registry`. How a
recipe picks each job's objective and learning-rate schedule lives in
:mod:`reef.train.algos.methods`.
"""

from reef.core.batches import StepScheduling
from reef.core.training_method import LearningRateSchedule, TrainingMethod
from reef.train.algos.methods import FixedTrainingMethod, TrainingMethodSelector
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.signals import StepSignal

__all__ = [
    "FixedTrainingMethod",
    "LearningRateSchedule",
    "StepScheduling",
    "StepSignal",
    "TrainingMethod",
    "TrainingMethodSelector",
    "TrainingObjective",
]
