"""The plugin classes skillclaw.yaml names: ``evolution.propose`` and ``evolution.evaluate``.

Each delegates to its function in ``skillclaw.py``, which keeps its Reef
training imports lazy.
"""

from __future__ import annotations

from reef.evaluation.harness import RolloutScorer
from reef.harness.rollouts.run import RolloutResult
from reef.train.harness_backend.contract import ProposalContext, Proposer, TreeProposal

from . import skillclaw


class NightProposer(Proposer):
    """The sealed night flow (:func:`skillclaw.propose`): one patch of the skill pool per night."""

    def propose(self, context: ProposalContext) -> TreeProposal | None:
        patch = skillclaw.propose(context.tree.files, context.samples, context.models)
        return None if patch is None else TreeProposal(patch)


class ProbeScorer(RolloutScorer):
    """Exact last-line grading of the probe rollouts (:func:`skillclaw.evaluate`)."""

    def score(self, task: str, result: RolloutResult) -> float:
        return skillclaw.evaluate(task, result)
