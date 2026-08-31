"""Training domain: the loop, and the pluggable backends that execute it.

One package, mirroring how a well-scoped training subsystem is usually
organized: a coordinator (:class:`Trainer`) that turns raw records into
reserved, typed batches (``processors/``, ``types/``), the recipe's candidate
evaluation that gates each produced update (``evaluation/``), and one
:class:`TrainingBackend` lifecycle. Harness evolution implements it directly;
Slime binds its remote runtime through ``SlimeTrainingBackend``. The GPU stack
is reached by full path so importing ``reef.train`` itself stays light.

``algos/`` turns a reserved batch into a ``StepSignal`` — the same
computation no matter which backend executes it — and its ``StepScheduling``
says how the runtime cuts one batch into optimizer steps. ``evaluation/`` is
the recipe's candidate gate; ``Trainer`` runs it between prepare and settle.

Tests and deployment configuration stay at repository level, never inside an
integration subtree: ``tests/slime_backend/`` for runtime internals,
``tests/plugin_contracts/`` for package boundaries, ``tests/reef_service/``
for service-facing contracts, with runnable examples owned by their method
packages.
"""

from reef.train.backend import PreparedStep, StepExecution, TrainingBackend
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.trainer import Trainer
from reef.train.types import ProcessorContext, TrainingBatch, TrainStepResult

__all__ = [
    "DataProcessor",
    "PreparedStep",
    "ProcessorContext",
    "RetentionDecision",
    "StepExecution",
    "TrainStepResult",
    "Trainer",
    "TrainingBackend",
    "TrainingBatch",
]
