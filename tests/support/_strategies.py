"""Test-only adapters that wrap a plain function as a Proposer or an EpisodeScorer.

Reef takes harness evolution plugins as classes only. Tests keep their short
functions through these adapters. A proposer function takes the step's
``ProposalContext`` and returns a ``TreeProposal``, a ``TreePatch`` (which the
adapter wraps in a proposal), or ``None``. ``reads_requests`` and
``runs_agent`` declare what the wrapped function handles. A scorer function
takes ``(task, result)``, and ``models`` too when its signature names it.
"""

import inspect

from reef.evaluation.harness import EpisodeScorer
from reef.harness.tree import TreePatch
from reef.train.harness_backend.contract import Proposer, TreeProposal


def _accepts(fn, name):
    parameters = inspect.signature(fn).parameters.values()
    return any(parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


class FunctionProposer(Proposer):
    def __init__(self, fn, *, reads_requests=False, runs_agent=False):
        self.fn = fn
        self.answers_requests = reads_requests
        self.uses_agent = runs_agent

    @property
    def reads_requests(self):
        return self.answers_requests

    @property
    def runs_agent(self):
        return self.uses_agent

    def propose(self, context):
        proposal = self.fn(context)
        return TreeProposal(proposal) if isinstance(proposal, TreePatch) else proposal


class FunctionScorer(EpisodeScorer):
    def __init__(self, fn):
        self.fn = fn

    def score(self, task, result):
        return self.fn(task, result)

    def score_with_models(self, task, result, models):
        if _accepts(self.fn, "models"):
            return self.fn(task, result, models=models)
        return self.fn(task, result)


def resolve_proposer(value, *, reads_requests=False, runs_agent=False):
    if isinstance(value, Proposer):
        return value
    return FunctionProposer(value, reads_requests=reads_requests, runs_agent=runs_agent)


def resolve_episode_scorer(value):
    return value if isinstance(value, EpisodeScorer) else FunctionScorer(value)
