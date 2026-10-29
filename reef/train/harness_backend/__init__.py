"""Harness evolution: the backend that evolves one harness tree, and the contract its methods implement.

A method imports this package's root and nothing below it::

    from reef.train.harness_backend import Method, StepContext

One step goes from the head tree to a release in two modules of this package:

``backend``
    ``HarnessBackend``, the step as the trainer runs it: prepare, evaluate,
    settle. The recipe ``reef.recipe.harness.HarnessEvolutionRecipe`` builds
    it; methods never subclass it.
``evolve``
    The step's candidate: the tree the method leaves in the step's
    workspace.

The evaluation and the release are Reef's, shared with weight training:
``reef.evaluation`` decides whether a candidate is served, and
``reef.train.release`` makes the decision a release.

What a method sees: ``contract`` defines what a method implements
(``Method``) and what a step gives it (``StepContext``: the ``Workspace`` and
the ``Services``); ``workspace`` is the step's directory (``Workspace``,
``Trace``), where the method also writes what it says about its step
(``write_output``). ``agent`` runs the deployment's coding agent for a method
(``evolution.agent``) and holds ``AgentMethod``, a method that is a folder of
prompts.

The rest supports a step: ``services`` implements the services of a step, ``calls``
its model budget, ``state`` the committed algorithm state, ``records`` the
step records, and ``capabilities`` what the service routes may ask a backend.
The tree itself is ``reef.harness.tree``; the evaluation, the history and the
publication policies are ``reef.evaluation.harness``.
"""

from reef.harness.gateway import ModelCalls
from reef.train.harness_backend.agent.method import AgentMethod
from reef.train.harness_backend.contract import AgentOutcome, Method, StepContext, Services
from reef.train.harness_backend.workspace import Trace, Workspace

__all__ = [
    "AgentMethod",
    "AgentOutcome",
    "Method",
    "ModelCalls",
    "Services",
    "StepContext",
    "Trace",
    "Workspace",
]
