"""The plugin classes skillclaw.yaml names: ``evolution.propose`` and ``evolution.evaluate``.

Each delegates to its function in ``skillclaw.py``, which keeps its Reef
training imports lazy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.evaluation.harness import RolloutScorer
from reef.harness.rollouts.model_binding import ModelBindings
from reef.harness.rollouts.run import RolloutResult
from reef.harness.tree.mutations import Mutation
from reef.train.cordis_backend.manifest import FailureManifest
from reef.train.cordis_backend.strategies import AgentHost, Proposer
from reef.train.types import TrajectorySample

from . import skillclaw


class NightProposer(Proposer):
    """The sealed night flow (:func:`skillclaw.propose`): one composite mutation sequence per night."""

    def propose(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TrajectorySample, ...],
        models: ModelBindings,
        *,
        manifest: FailureManifest | None = None,
        rejected: Sequence[Mapping[str, Any]] = (),
        sources: Sequence[Mapping[str, Any]] = (),
        requests: Sequence[Mapping[str, Any]] = (),
        entries: Sequence[Mapping[str, Any]] = (),
        agent_host: AgentHost | None = None,
    ) -> Sequence[Mutation] | None:
        return skillclaw.propose(nodes, samples, models)


class ProbeScorer(RolloutScorer):
    """Exact last-line grading of the probe rollouts (:func:`skillclaw.evaluate`)."""

    def score(self, task: str, result: RolloutResult) -> float:
        return skillclaw.evaluate(task, result)
