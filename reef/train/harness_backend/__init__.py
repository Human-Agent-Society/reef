"""Harness evolution: the backend that evolves one harness tree, and the contract its methods implement.

A method imports this package's root and nothing below it::

    from reef.train.harness_backend import Method, StepContext

``contract`` defines what a method implements (``Method``) and what a step
gives it (``StepContext``: the ``Workspace`` and the ``Tools``);
``workspace`` is the step's directory (``Workspace``, ``Trace``), where the
method also writes what it says about its step (``write_output``).
``agent`` runs the deployment's coding agent for a method (``evolution.agent``)
and holds ``AgentMethod``, a method that is a folder of prompts.
``backend`` holds ``HarnessBackend``, which the recipe
``reef.recipe.harness.HarnessEvolutionRecipe`` builds; methods never
subclass it. ``tools`` implements the tools of a step, ``calls`` its model
budget, ``state`` the committed algorithm state, ``inbox`` the clients'
pending patches, ``records`` the step records, ``manifest`` the failure
manifest, and ``capabilities`` what the service routes may ask a backend.
The tree itself is ``reef.harness.tree``; the evaluation, the history and the
publication policies are ``reef.evaluation.harness``.
"""

from reef.harness.gateway import ModelCalls
from reef.train.harness_backend.agent.method import AgentMethod
from reef.train.harness_backend.contract import AgentOutcome, Method, StepContext, Tools
from reef.train.harness_backend.manifest import FailureManifest, FailureRecord
from reef.train.harness_backend.workspace import Trace, Workspace

__all__ = [
    "AgentMethod",
    "AgentOutcome",
    "FailureManifest",
    "FailureRecord",
    "Method",
    "ModelCalls",
    "StepContext",
    "Tools",
    "Trace",
    "Workspace",
]
