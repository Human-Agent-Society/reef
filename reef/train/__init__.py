"""Training domain: the loop, and the pluggable backends that execute it.

One package, mirroring how a well-scoped training subsystem is usually
organized: a :class:`Trainer` that turns raw records into reserved, typed
job data (:class:`TrainingPipeline` and its three stages) and trains it. A
trainer submits each job's data to a training component through
``reef.train.jobs.JobLoop``; a harness evolution step is a job of the harness
training component (``reef.train.harness_backend.component``).
The GPU stack is reached by full path so importing ``reef.train`` itself stays light.

``algos/`` turns a job's data into a ``TrainingSignal``, the same computation
whichever training component runs it; how the component cuts the data into
optimizer steps is the job's ``StepScheduling`` (``reef.train.algos.schedule``).
A harness step runs the recipe's candidate evaluation (``reef.evaluation``)
inside its job, between prepare and settle.

The package does not import recipes or service assembly. The harness
evolution recipe lives in ``reef.recipe.harness``. ``slime_backend.component`` and
``tinker_backend.component`` are training components over
``reef.train.local_jobs.LocalJobComponent``; Reef's service layer assembles
them and ``reef.runtime.deployment`` owns their lifecycle. Each component
saves checkpoints, and an inference runtime serves the published ones:
Tinker's sampler (``reef.inference.tinker``) reads the checkpoint manifest of
a release, and ``reef.runtime.serving`` loads a release's weight files.
Tinker SDK imports stay inside its client adapter, and method loss policy
remains in the recipe package.

Tests and deployment configuration stay at repository level, never inside an
integration subtree: ``tests/slime_backend/`` for runtime internals,
``tests/plugin_contracts/`` for package boundaries, ``tests/reef_service/``
for service-facing contracts, with runnable examples owned by their method
packages.
"""

from reef.pipeline_context import PipelineContext
from reef.sample2job import Sample2Job
from reef.train.backend import PreparedStep
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import ComponentTrainer, Trainer
from reef.train.types import JobData, Sample, TaskSample, TrainStepResult, TrajectorySample

__all__ = [
    "ComponentTrainer",
    "JobData",
    "PipelineContext",
    "PreparedStep",
    "Sample",
    "Sample2Job",
    "TaskSample",
    "TrainStepResult",
    "Trainer",
    "TrainingPipeline",
    "TrajectorySample",
]
