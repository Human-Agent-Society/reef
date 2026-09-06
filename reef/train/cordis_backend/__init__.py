"""Harness evolution backend: candidate selection scored by real episodes.

Per training step, ``propose`` reads the current composition and the batched
traces and returns one ``Mutation`` or a sequence of them; the backend
applies the proposal to the compose Entry tree under one snapshot; the
candidate and current compositions render through the adapter descriptor and
each run one headless episode per task; the method's ``EpisodeScorer`` scores
individual results. The recipe composes this evaluator and its configured
``CandidateSelector`` into one ``DefaultCandidateEvaluationPlugin`` for ``Trainer``.
A sequence is one
composite proposal: it applies
atomically and receives one selection decision, never one per mutation. The
default policy selects a candidate with more task wins than losses.

Versioning goes through reef's native artifact stack: a selected mutation
renders to a directory and returns a ``TrainStepResult`` with the artifact
set, so ``ScenarioCommitProtocol`` stages and publishes it through
``Repository``. The composition tree state travels in the algorithm state
(``"entries"`` key), which the commit log and snapshot metadata persist and
recover.
"""

from reef.train.cordis_backend.backend import CordisBackend, HarnessCandidate, ScoreComparisonSelector
from reef.train.cordis_backend.manifest import FailureManifest, FailureObservation, FailureRecord
from reef.train.cordis_backend.processor import CordisProcessor
from reef.train.cordis_backend.recipe import CordisRecipe
from reef.train.cordis_backend.strategies import (
    EpisodeScorer,
    Mutation,
    MutationError,
    Promoter,
    Proposer,
    untrusted_text,
)

__all__ = [
    "CordisBackend",
    "CordisProcessor",
    "CordisRecipe",
    "EpisodeScorer",
    "FailureManifest",
    "FailureObservation",
    "FailureRecord",
    "HarnessCandidate",
    "Mutation",
    "MutationError",
    "Promoter",
    "Proposer",
    "ScoreComparisonSelector",
    "untrusted_text",
]
