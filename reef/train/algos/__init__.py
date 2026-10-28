"""Backend-neutral training objectives and their output.

A recipe names a ``TrainingObjective``; its ``prepare`` returns a
``TrainingSignal`` for one job's data. Optional implementation helpers live
in :mod:`reef.train.algos.helpers`, the registered-objective APIs in
:mod:`reef.train.algos.registry`, and the step schedule that training
components share in :mod:`reef.train.algos.schedule`.
"""

from reef.train.algos.objective import TrainingObjective
from reef.train.algos.signals import TrainingSignal

__all__ = [
    "TrainingObjective",
    "TrainingSignal",
]
