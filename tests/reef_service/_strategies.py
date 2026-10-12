"""Test-only adapters that wrap a plain function as a Proposer or an EpisodeScorer.

Reef takes harness evolution plugins as classes only. Tests keep their short
lambdas through these adapters. As before, a keyword reaches the function only
when its signature names it (or takes ``**kwargs``), and the function declares
``reads_requests`` and ``runs_agent`` by naming ``requests`` and ``agent_host``.
"""

import inspect

from reef.train.cordis_backend.strategies import EpisodeScorer, Proposer


def _accepts(fn, name):
    parameters = inspect.signature(fn).parameters.values()
    return any(parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _names(fn, name):
    parameter = inspect.signature(fn).parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class FunctionProposer(Proposer):
    def __init__(self, fn):
        self.fn = fn

    @property
    def reads_requests(self):
        return _names(self.fn, "requests")

    @property
    def runs_agent(self):
        return _names(self.fn, "agent_host")

    def propose(self, nodes, samples, models, **keywords):
        forwarded = {name: value for name, value in keywords.items() if _accepts(self.fn, name)}
        if not self.reads_requests:
            forwarded.pop("requests", None)
        if not self.runs_agent:
            forwarded.pop("agent_host", None)
        return self.fn(nodes, samples, models, **forwarded)


class FunctionScorer(EpisodeScorer):
    def __init__(self, fn):
        self.fn = fn

    def score(self, task, result):
        return self.fn(task, result)

    def score_with_models(self, task, result, models):
        if _accepts(self.fn, "models"):
            return self.fn(task, result, models=models)
        return self.fn(task, result)


def resolve_proposer(value):
    return value if isinstance(value, Proposer) else FunctionProposer(value)


def resolve_episode_scorer(value):
    return value if isinstance(value, EpisodeScorer) else FunctionScorer(value)
