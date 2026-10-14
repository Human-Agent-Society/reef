"""Training domain: the loop, and the pluggable backends that execute it.

One package, mirroring how a well-scoped training subsystem is usually
organized: a coordinator (:class:`Trainer`) that turns raw records into
reserved, typed batches (:class:`TrainingPipeline` and its three stages), and one
:class:`CandidateBackend` lifecycle. Harness evolution implements it directly;
Weight recipes map their batches through ``RuntimeCandidateBackend`` and delegate
coordination of separate training and inference runtimes to ``RuntimeScheduler``.
The GPU stack is reached by full path so importing ``reef.train`` itself stays light.

``algos/`` turns a reserved batch into a ``StepSignal`` — the same
computation no matter which backend executes it; the recipe's ``StepScheduling``
(``reef.core.batches``) says how the runtime cuts one batch into optimizer
steps. ``Trainer`` runs the recipe's candidate evaluation (``reef.evaluation``)
between prepare and settle.

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

from reef.pipeline_context import PipelineContext
from reef.task2batch import Task2Batch

# Registers the ``mlx`` runtime kind. The module imports no MLX at module
# scope, so this costs a CUDA deployment nothing; reef.runtime cannot do the
# import itself without pointing runtime back at train.
from reef.train import mlx_backend as _mlx_backend  # noqa: F401
from reef.train.backend import CandidateBackend, PreparedStep, StepExecution
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch, TrainStepResult, TrajectoryItem

__all__ = [
    "CandidateBackend",
    "ComponentTrainer",
    "PipelineContext",
    "PreparedStep",
    "StepExecution",
    "Task2Batch",
    "TaskItem",
    "TrainDataItem",
    "TrainStepResult",
    "Trainer",
    "TrainingBatch",
    "TrainingPipeline",
    "TrajectoryItem",
]
