"""Test-only adapters that wrap a plain function as a Method or an EpisodeScorer.

Reef takes harness evolution plugins as classes only. Tests keep their short
functions through these adapters. A method function takes the step's
``Workspace`` and ``Tools`` and returns a ``StepResult``, a ``TreePatch``
(which the adapter applies to the workspace's tree), or ``None`` (the tree
stays, so the step skips). A scorer function takes ``(task, result)``, and
``models`` too when its signature names it.
"""

import inspect

from reef.evaluation.harness import EpisodeScorer
from reef.harness.tree import TreePatch
from reef.train.harness_backend import Method, StepResult


def _accepts(fn, name):
    parameters = inspect.signature(fn).parameters.values()
    return any(parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


class FunctionMethod(Method):
    def __init__(self, fn):
        self.fn = fn

    def step(self, workspace, tools):
        result = self.fn(workspace, tools)
        if isinstance(result, TreePatch):
            workspace.write_tree(workspace.tree().apply(result))
            return StepResult()
        return StepResult() if result is None else result


class FunctionScorer(EpisodeScorer):
    def __init__(self, fn):
        self.fn = fn

    def score(self, task, result):
        return self.fn(task, result)

    def score_with_models(self, task, result, models):
        if _accepts(self.fn, "models"):
            return self.fn(task, result, models=models)
        return self.fn(task, result)


def resolve_method(value):
    return value if isinstance(value, Method) else FunctionMethod(value)


def resolve_episode_scorer(value):
    return value if isinstance(value, EpisodeScorer) else FunctionScorer(value)
