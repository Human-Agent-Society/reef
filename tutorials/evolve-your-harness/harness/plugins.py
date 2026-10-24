"""The plugin classes the serve files name: ``evolution.propose`` and ``evolution.evaluate``.

They import Reef, so they live apart from ``evolution.py`` and
``native_evolution.py``, which ``run.py`` imports with only reef-client
installed. The pi variant proposes with the built-in Reefine text proposer;
these classes are the scorers of both variants and the native proposer.
"""

from reef.evaluation.harness import EpisodeScorer
from reef.harness.episodes.run import EpisodeResult
from reef.train.harness_backend.contract import ProposalContext, Proposer, TreeProposal

from . import evolution, native_evolution


class AnswerScorer(EpisodeScorer):
    """``evolution.evaluate``: exact last-line answer grading of the episode's final assistant text."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return evolution.evaluate(task, result)


class NativeSelfProposer(Proposer):
    """``native_evolution.propose``: one change to a skill, tool, hook, the loop graph or its agents."""

    def propose(self, context: ProposalContext) -> TreeProposal | None:
        patch = native_evolution.propose(context.tree, context.samples, context.models)
        return None if patch is None else TreeProposal(patch)


class NativeAnswerScorer(EpisodeScorer):
    """``native_evolution.evaluate``: the same grading over the native-jsonl trajectory."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return native_evolution.evaluate(task, result)
