"""The plugin classes the serve files name: ``evolution.propose`` and ``evolution.evaluate`` for both variants.

They import Reef, so they live apart from ``evolution.py`` and
``native_evolution.py``, which ``run.py`` imports with only reef-client
installed. Each class delegates to its module's ``propose`` or ``evaluate``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from reef.evaluation.harness import RolloutScorer
from reef.harness.rollouts.model_binding import ModelBindings
from reef.harness.rollouts.run import RolloutResult
from reef.harness.tree.mutations import Mutation
from reef.train.cordis_backend.manifest import FailureManifest
from reef.train.cordis_backend.strategies import AgentHost, Proposer, StepProposal
from reef.train.types import TrajectorySample

from . import evolution, native_evolution


class SelfProposer(Proposer):
    """``evolution.propose``: the served model proposes one skill change over its failures, or answers a request."""

    @property
    def reads_requests(self) -> bool:
        return True

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
    ) -> Mutation | Sequence[Mutation] | StepProposal | None:
        return evolution.propose(nodes, samples, models, requests=requests, entries=entries)


class AnswerScorer(RolloutScorer):
    """``evolution.evaluate``: exact last-line answer grading of the rollout's final assistant text."""

    def score(self, task: str, result: RolloutResult) -> float:
        return evolution.evaluate(task, result)


class NativeSelfProposer(Proposer):
    """``native_evolution.propose``: one change to a skill, tool, hook, the loop graph or its agents."""

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
    ) -> Mutation | Sequence[Mutation] | StepProposal | None:
        return native_evolution.propose(nodes, samples, models)


class NativeAnswerScorer(RolloutScorer):
    """``native_evolution.evaluate``: the same grading over the native-jsonl trajectory."""

    def score(self, task: str, result: RolloutResult) -> float:
        return native_evolution.evaluate(task, result)
