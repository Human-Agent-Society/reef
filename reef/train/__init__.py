"""Training domain: the loop, and the pluggable backends that execute it.

One package, mirroring how a well-scoped training subsystem is usually
organized: a coordinator (:class:`Trainer`) that turns raw records into
reserved, typed batches (``processors/``, ``reef.core.batches``), the recipe's candidate
evaluation that gates each produced update (``evaluation/``), and one
:class:`CandidateBackend` lifecycle. Harness evolution implements it directly;
Weight recipes map their batches through ``RuntimeCandidateBackend`` and delegate
coordination of separate training and inference runtimes to ``RuntimeScheduler``.
The GPU stack is reached by full path so importing ``reef.train`` itself stays light.

``algos/`` turns a reserved batch into a ``StepSignal`` — the same
computation no matter which backend executes it; the recipe's ``StepScheduling``
(``reef.core.batches``) says how the runtime cuts one batch into optimizer
steps. ``evaluation/`` is
the recipe's candidate gate; ``Trainer`` runs it between prepare and settle.

The package does not import recipes or service assembly. ``CordisRecipe``
lives in ``reef.recipe.cordis``. Slime component preparation lives in
``reef.train.slime_backend.driver``; Reef's service layer assembles components
and ``reef.runtime.deployment`` owns their lifecycle. ``tinker_backend`` owns
remote LoRA training, immutable samplers, and checkpoint manifests as a
separate training runtime and inference runtime over one checkpoint store;
SDK imports stay inside its client adapter, and method loss policy remains
in the recipe package.

Tests and deployment configuration stay at repository level, never inside an
integration subtree: ``tests/slime_backend/`` for runtime internals,
``tests/plugin_contracts/`` for package boundaries, ``tests/reef_service/``
for service-facing contracts, with runnable examples owned by their method
packages.
"""

from reef.train.backend import CandidateBackend, PreparedStep, StepExecution
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import ProcessorContext, TaskItem, TrainDataItem, TrainingBatch, TrainStepResult, TrajectoryItem

__all__ = [
    "CandidateBackend",
    "ComponentTrainer",
    "DataProcessor",
    "PreparedStep",
    "ProcessorContext",
    "RetentionDecision",
    "StepExecution",
    "TaskItem",
    "TrainDataItem",
    "TrainStepResult",
    "Trainer",
    "TrainingBatch",
    "TrajectoryItem",
]
