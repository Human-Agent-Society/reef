"""Harness evolution backend: candidate selection scored by real episodes.

Per training step, ``propose`` reads the current composition and the batched
traces and returns one ``Mutation``, a sequence of them, or a ``StepProposal``
(the mutations plus notes the step records); the backend
applies the proposal to the compose Entry tree under one snapshot; the
candidate and current compositions render through the adapter descriptor and
each run one headless episode per task in ``reef.evaluation.harness``, where
the method's ``EpisodeScorer`` scores individual results. The recipe picks the
``CandidateEvaluationPlugin`` class that pairs this measurement with a
selection policy and hands it to ``Trainer``. A sequence is one composite
proposal: it applies atomically and receives one selection decision, never one
per mutation.

Versioning goes through reef's native artifact stack: a selected mutation
renders to a directory and returns a ``TrainStepResult`` with the artifact
set, so ``ScenarioCommitter`` stages and publishes it through
``Repository``. The composition tree state travels in the algorithm state
(``"entries"`` key), which commit records persist and recover.

``reef.recipe.cordis.CordisRecipe`` assembles this backend. Composition and
mutation admission live in ``reef.harness`` so serving uses them independently
of the training loop.
"""

from reef.harness.tree.mutations import Mutation, MutationError
from reef.train.cordis_backend.backend import CordisBackend
from reef.train.cordis_backend.contracts import StepProgress
from reef.train.cordis_backend.manifest import FailureManifest, FailureRecord
from reef.train.cordis_backend.processor import InferenceTasks, TraceBatches, TraceRecord2Record, TraceTasks
from reef.train.cordis_backend.strategies import Promoter, Proposer, StepProposal

__all__ = [
    "CordisBackend",
    "FailureManifest",
    "FailureRecord",
    "InferenceTasks",
    "Mutation",
    "MutationError",
    "Promoter",
    "Proposer",
    "StepProgress",
    "StepProposal",
    "TraceBatches",
    "TraceRecord2Record",
    "TraceTasks",
]
