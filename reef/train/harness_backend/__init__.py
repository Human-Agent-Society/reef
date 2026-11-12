"""Harness evolution: the backend that evolves one harness tree, and the contract its methods implement.

A method imports this package's root and nothing below it::

    from reef.train.harness_backend import Method, MethodJob

One job goes from the head tree to a kept tree in ``backend``:
``HarnessBackend`` is the job as the harness training
component (``component``) runs it (prepare, evaluate, settle). It holds the deployment's ``Method`` and calls ``Method.run`` once
per job in ``propose``; the tree the method leaves in the job's workspace is
the candidate. The recipe ``reef.recipe.harness.HarnessEvolutionRecipe``
builds the backend; methods never subclass it.

The evaluation and the activation are Reef's, shared with weight training:
``reef.evaluation`` decides whether a candidate is kept, and the scenario's
activation rule decides whether a kept tree serves.

What a method sees: ``contract`` defines what a method implements
(``Method``) and what a job gives it (``MethodJob``: the ``Workspace`` and
the ``Services``); ``workspace`` is the job's directory (``Workspace``,
``Trace``), where the method also writes what it says about its job
(``write_output``). ``agent`` runs the deployment's coding agent for a method
(``evolution.agent``) and holds ``AgentMethod``, a method that is a folder of
prompts.

The rest supports a job: ``services`` implements the services of a job, ``calls``
its model budget, ``state`` the committed algorithm state, and ``records`` the
job records.
The tree itself is ``reef.harness.tree``; the evaluation, the history and the
publication policies are ``reef.evaluation.harness``.
"""

from reef.harness.gateway import ModelCalls
from reef.train.harness_backend.agent.method import AgentMethod
from reef.train.harness_backend.contract import AgentOutcome, Method, MethodJob, Services
from reef.train.harness_backend.workspace import Trace, Workspace

__all__ = [
    "AgentMethod",
    "AgentOutcome",
    "Method",
    "MethodJob",
    "ModelCalls",
    "Services",
    "Trace",
    "Workspace",
]
