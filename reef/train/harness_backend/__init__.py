"""Harness evolution: the backend that evolves one harness tree, and the contract its methods implement.

A method imports this package's root and nothing below it::

    from reef.train.harness_backend import Method, StepContext

One step goes from the head tree to a release in three modules:

``backend``
    ``HarnessBackend``, the step as the trainer runs it: prepare, evaluate,
    settle. The recipe ``reef.recipe.harness.HarnessEvolutionRecipe`` builds
    it; methods never subclass it.
``evolve``
    The method edits the head tree in the step's workspace, and the tree it
    leaves is the candidate.
``release``
    The evaluation's decision becomes the step's result: review, recheck,
    and the artifact that the scenario committer publishes.

What a method sees: ``contract`` defines what a method implements
(``Method``) and what a step gives it (``StepContext``: the ``Workspace`` and
the ``Tools``); ``workspace`` is the step's directory (``Workspace``,
``Trace``), where the method also writes what it says about its step
(``write_output``). ``agent`` runs the deployment's coding agent for a method
(``evolution.agent``) and holds ``AgentMethod``, a method that is a folder of
prompts.

The rest supports a step: ``tools`` implements the tools of a step, ``calls``
its model budget, ``state`` the committed algorithm state, ``records`` the
step records, and ``capabilities`` what the service routes may ask a backend.
The tree itself is ``reef.harness.tree``; the evaluation, the history and the
publication policies are ``reef.evaluation.harness``.
"""

from reef.harness.gateway import ModelCalls
from reef.train.harness_backend.agent.method import AgentMethod
from reef.train.harness_backend.contract import AgentOutcome, Method, StepContext, Tools
from reef.train.harness_backend.workspace import Trace, Workspace

__all__ = [
    "AgentMethod",
    "AgentOutcome",
    "Method",
    "ModelCalls",
    "StepContext",
    "Tools",
    "Trace",
    "Workspace",
]
