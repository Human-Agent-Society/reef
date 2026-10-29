"""Harness evolution: the backend that evolves one harness tree, and the contract its methods implement.

A method imports this package's root and nothing below it::

    from reef.train.harness_backend import Proposer, ProposalContext, TreeProposal

``contract`` defines what a method implements (``Proposer``, ``Promoter``)
and what a step hands it (``ProposalContext``, ``TrialRunner``,
``AgentHost``). ``backend`` holds ``HarnessBackend``, which the recipe
``reef.recipe.harness.HarnessEvolutionRecipe`` builds; methods never subclass
it. ``state`` is the backend's algorithm state, ``tasks`` the evaluation
tasks of each step, ``calls`` the step's model-call budget, ``trials`` the
trial rollouts, ``inbox`` the clients' pending patches, ``records`` the step
record, ``manifest`` the failure manifest, and ``capabilities`` what the
service routes may ask a backend. The tree itself is ``reef.harness.tree``,
and the evaluation is ``reef.evaluation.harness``.
"""

from reef.train.harness_backend.contract import (
    AgentHost,
    Promoter,
    ProposalContext,
    Proposer,
    ProposerCalls,
    RejectedProposal,
    SampleSource,
    StepOutcome,
    TreeProposal,
    TrialRunner,
)
from reef.train.harness_backend.manifest import FailureManifest, FailureRecord

__all__ = [
    "AgentHost",
    "FailureManifest",
    "FailureRecord",
    "Promoter",
    "ProposalContext",
    "Proposer",
    "ProposerCalls",
    "RejectedProposal",
    "SampleSource",
    "StepOutcome",
    "TreeProposal",
    "TrialRunner",
]
