"""Harness evolution backend: candidate selection scored by real episodes.

Per training step, ``propose`` reads the current composition and the batched
traces and returns one ``Mutation``, a sequence of them, or a ``StepProposal``
(the mutations plus notes the step records); the backend
applies the proposal to the compose Entry tree under one snapshot; the
candidate and current compositions render through the adapter descriptor and
each run one headless episode per task; the method's ``EpisodeScorer`` scores
individual results. The recipe picks the ``CandidateEvaluationPlugin`` class
that pairs this measurement with a selection policy and hands it to ``Trainer``.
A sequence is one
composite proposal: it applies
atomically and receives one selection decision, never one per mutation. The
default policy selects a candidate with more task wins than losses; the
``floor`` policy runs the candidate alone and selects it when every task
scores at least ``floor_score``; the ``paired_confidence`` policy reruns
pairs whose current episode an infrastructure fault hit and selects when a
sign test over tasks passes and the bootstrap lower bound of the mean gain is
above ``min_effect``. When the tasks are a task manifest's eval split, a task
a consumed batch named, or whose source record a commit consumed, is not
evaluated again, and eval failures never reach the proposer.

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
from reef.train.cordis_backend.backend import (
    CordisBackend,
    EvalSplitTask,
    FloorMixin,
    FloorPlugin,
    FloorPluginFactory,
    HarnessCandidate,
    PairedConfidencePlugin,
    PairedConfidencePluginFactory,
    ScoreComparisonMixin,
    ScoreComparisonPlugin,
    ScoreComparisonPluginFactory,
)
from reef.train.cordis_backend.contracts import StepProgress
from reef.train.cordis_backend.manifest import FailureManifest, FailureObservation, FailureRecord
from reef.train.cordis_backend.processor import CordisProcessor
from reef.train.cordis_backend.strategies import (
    EpisodeScorer,
    Promoter,
    Proposer,
    ScoreUnavailable,
    StepProposal,
    untrusted_text,
)

__all__ = [
    "CordisBackend",
    "CordisProcessor",
    "EpisodeScorer",
    "EvalSplitTask",
    "FailureManifest",
    "FailureObservation",
    "FailureRecord",
    "FloorMixin",
    "FloorPlugin",
    "FloorPluginFactory",
    "HarnessCandidate",
    "Mutation",
    "MutationError",
    "PairedConfidencePlugin",
    "PairedConfidencePluginFactory",
    "Promoter",
    "Proposer",
    "ScoreComparisonMixin",
    "ScoreComparisonPlugin",
    "ScoreComparisonPluginFactory",
    "ScoreUnavailable",
    "StepProgress",
    "StepProposal",
    "untrusted_text",
]
