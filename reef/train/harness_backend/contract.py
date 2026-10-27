"""The method contract of harness evolution: what a method implements, and what each step hands it.

A harness evolution method implements a ``Proposer`` and, where it needs its
own, an ``EpisodeScorer`` (``reef.evaluation.harness``), a selection plugin
(``reef.evaluation.SelectionPlugin``) and a ``Promoter``. A method never
subclasses the backend. Everything a step gives a method is in
``ProposalContext``, and everything a method gives back is in
``TreeProposal``, including the state the method keeps between steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.job_data import TrajectorySample
from reef.core.plugins import ConfiguredPlugin
from reef.core.requirements import parse_requires
from reef.core.training_request import TrainingRequest
from reef.evaluation.harness import ScoredEpisode
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, TreePatch
from reef.sandbox.process import EpisodeExecutor
from reef.train.harness_backend.manifest import FailureManifest


@dataclass(frozen=True)
class SampleSource:
    """Where one trace sample came from.

    Values: ``record`` is the agent record id of the sample's last request,
    or ``None`` for a sample with none; ``client`` is the request's
    ``x-reef-tag-client`` tag, else its session tag, else ``"untagged"``.
    Every sample is client text, so a proposer treats its text as untrusted
    data and wraps it with ``reef.core.prompts.untrusted_text`` before it
    enters a prompt.
    Purpose: the backend makes one per sample, in sample order, for
    ``ProposalContext.sources``.
    """

    record: str | None
    client: str


@dataclass(frozen=True)
class RejectedProposal:
    """One earlier proposal the evaluation rejected.

    Values: ``step`` is the positive step that evaluated it, ``patch`` is the
    patch as proposed, and ``reason`` is the selection's non-empty reason.
    Purpose: the backend keeps the latest ones in its state so a proposer
    can stop proposing what the evaluation refused.
    """

    step: int
    patch: TreePatch
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {"step": self.step, "patch": self.patch.to_json(), "reason": self.reason}

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> RejectedProposal:
        return cls(int(value["step"]), TreePatch.from_json(value["patch"]), str(value["reason"]))


@dataclass(frozen=True)
class StepOutcome:
    """What an evaluated step did with a candidate the method proposed.

    Values: ``step`` is that step's positive number, ``patch`` the change it
    evaluated, ``selected`` whether the candidate was published, ``reason``
    the selection's reason, and the two score tuples the evaluation's
    per-task scores in pairing order, ``None`` for an episode that could not
    run. ``current_scores`` is empty when the selection ran the candidate alone.
    Purpose: the backend hands the outcome of the latest step that evaluated
    the method's own proposal to the next proposal, so a method updates its
    own state from the result.
    """

    step: int
    patch: TreePatch
    selected: bool
    reason: str
    candidate_scores: tuple[float | None, ...]
    current_scores: tuple[float | None, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "patch": self.patch.to_json(),
            "selected": self.selected,
            "reason": self.reason,
            "candidate_scores": list(self.candidate_scores),
            "current_scores": list(self.current_scores),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> StepOutcome:
        return cls(
            step=int(value["step"]),
            patch=TreePatch.from_json(value["patch"]),
            selected=bool(value["selected"]),
            reason=str(value["reason"]),
            candidate_scores=tuple(value["candidate_scores"]),
            current_scores=tuple(value["current_scores"]),
        )


class ProposerCalls(ABC):
    """The step's model-call budget, record and live activity, for model traffic a proposer makes outside ``models``.

    An agent proposer's own process reaches its model through a gateway, not
    through the bindings. The gateway spends from and records into the same
    budget and ``proposer.json`` as the bindings, and notes what the agent
    does, for the request page to show while the step runs.
    Purpose: ``AgentHost.calls`` hands it to an agent proposer.
    """

    @abstractmethod
    def spend(self) -> None:
        """Count one call; raise ``RuntimeError`` when the step's budget is exhausted."""

    @abstractmethod
    def record(self, entry: Mapping[str, Any]) -> None:
        """Append one call to the step's proposer record."""

    @abstractmethod
    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        """Add one line to the step's live activity: ``kind`` is what acted (``model``, ``agent``, ``check``,
        ``trial``, ``provider``, ``proposer``), ``text`` what it did, ``failed`` when that went wrong."""


class TrialRunner(ABC):
    """Run trial episodes of a tree before the proposer settles on its patch.

    Purpose: a method that probes its candidates (harness2's probes, GEPA's
    minibatches) runs them here instead of rendering and launching episodes
    itself; the trials use the evaluation's worker pool, scorer and model
    binding, and they count against the step's budget and record.
    """

    @abstractmethod
    def run(self, tree: HarnessTree, tasks: Sequence[str]) -> tuple[ScoredEpisode, ...]:
        """One scored episode of ``tree`` per task, in task order."""


@dataclass(frozen=True)
class AgentHost:
    """What a proposer that runs a coding agent needs from the step.

    Values: ``descriptor`` and ``binary`` are the harness adapter and its
    installed binary, the same the episodes run; ``executor`` is the one the
    deployment built for the agent (its isolation, not the episodes');
    ``step_dir`` is the step's record directory, ``None`` with the record
    off; ``calls`` is the step's budget and record; ``timeout_s`` bounds the
    whole agent run and ``trial_timeout_s`` each trial, both positive.
    Purpose: the backend makes one per step for a proposer whose
    ``runs_agent`` is true, when the deployment configured an agent executor.
    """

    descriptor: AdapterDescriptor
    binary: str
    executor: EpisodeExecutor
    step_dir: Path | None
    calls: ProposerCalls
    timeout_s: float
    trial_timeout_s: float

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.trial_timeout_s <= 0:
            raise ValueError("AgentHost timeouts must be positive")


@dataclass(frozen=True)
class ProposalContext:
    """Everything one step gives a proposer.

    Values: ``step`` ≥ 1 is this step's number, the one ``StepOutcome.step``
    names once the step is evaluated. ``tree`` is the head tree the patch
    applies to, and ``descriptor`` the adapter it renders for, which says the
    kinds and config targets a tree may hold there. ``samples`` are
    the step's trace samples in the order of its job data, and ``sources`` holds one
    ``SampleSource`` per sample in the same order; a sample's ``score`` is
    ``None`` when the deployment trains on unscored traffic. ``request`` is the
    training instruction this step answers, ``None`` for an automatic step.
    ``evaluation_tasks`` are the tasks this step's evaluation runs, the seed
    tasks first, then the promoted ones. ``models`` are the bindings the
    proposer may call, every call counted against the step's budget. ``manifest`` is the previous step's failure
    manifest, ``None`` before the first. ``rejected`` holds the latest
    rejected proposals, oldest first. ``problems`` is empty on the first
    proposal of a step and holds the validation problems of the previous
    proposal on the retry. ``method_state`` is the state the method committed
    with the latest step, empty before its first. ``last_outcome`` is the
    result of the latest step that evaluated this method's proposal, ``None``
    before the first; a recheck and a client's proposal from the inbox leave it
    as it was.
    ``trials`` runs trial episodes, ``None`` when the deployment allows none.
    ``agent_host`` is set exactly when the proposer runs an agent and the
    deployment configured an agent executor.
    Purpose: the backend makes one per proposal; a proposer reads it and
    returns a ``TreeProposal`` or ``None``.
    """

    step: int
    tree: HarnessTree
    descriptor: AdapterDescriptor
    samples: tuple[TrajectorySample, ...]
    sources: tuple[SampleSource, ...]
    request: TrainingRequest | None
    evaluation_tasks: tuple[str, ...]
    models: ModelBindings
    manifest: FailureManifest | None = None
    rejected: tuple[RejectedProposal, ...] = ()
    problems: tuple[str, ...] = ()
    method_state: Mapping[str, Any] = field(default_factory=dict)
    last_outcome: StepOutcome | None = None
    trials: TrialRunner | None = None
    agent_host: AgentHost | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("ProposalContext.step must be a positive integer")
        if len(self.sources) != len(self.samples):
            raise ValueError("ProposalContext needs one source per sample")


@dataclass(frozen=True)
class TreeProposal:
    """What a proposer returns for one step.

    Values: ``patch`` changes the head tree; the empty patch changes nothing.
    ``requires`` are what the change needs from the person's machine, beyond
    the request's own (``reef.core.requirements`` items). ``notes`` is the
    proposer's JSON record of the step, which the commit metrics carry and
    the backend never reads. ``method_state`` replaces the method's committed
    state; ``None`` keeps it.
    Purpose: ``Proposer.propose`` makes it. The backend commits
    ``method_state`` and ``notes`` with the step whatever happens next. It
    skips the step for the empty patch, and otherwise applies, validates and
    evaluates ``patch``.
    """

    patch: TreePatch
    requires: tuple[Mapping[str, Any], ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)
    method_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", tuple(parse_requires(list(self.requires))))


class Proposer(ConfiguredPlugin, ABC):
    """Base class of a harness evolution method's proposer.

    ``propose`` returns the patch for the step, or ``None`` to skip it with
    nothing to record; a ``TreeProposal`` with the empty patch skips it and
    records the method's state.
    Purpose: a deployment names the class in ``evolution.propose``, the
    recipe builds it with ``from_config``, and the backend calls ``propose``
    once per step, and once more with the validation problems.
    """

    @property
    def reads_requests(self) -> bool:
        """Whether this proposer can answer a training instruction (``manual`` and ``hybrid``); override to say so."""
        return False

    @property
    def runs_agent(self) -> bool:
        """Whether this proposer runs a coding agent on ``context.agent_host``; override to say so."""
        return False

    @abstractmethod
    def propose(self, context: ProposalContext) -> TreeProposal | None:
        """The patch for this step, or ``None`` to skip it."""


class Promoter(ConfiguredPlugin, ABC):
    """Base class for choosing which trace prompts become permanent evaluation tasks.

    ``promote`` receives the step's trace samples and the previous step's
    failure manifest, and returns the prompts to promote. Reef still dedupes,
    screens for credentials and instruction overrides, and caps them.
    Purpose: a deployment names the class in ``evolution.promote``; the
    backend calls it in a step that promotes failures.
    """

    @abstractmethod
    def promote(
        self, samples: tuple[TrajectorySample, ...], *, manifest: FailureManifest | None = None
    ) -> Sequence[str]:
        """The prompts this step should promote into the evaluation tasks."""


__all__ = [
    "AgentHost",
    "Promoter",
    "ProposalContext",
    "Proposer",
    "ProposerCalls",
    "RejectedProposal",
    "SampleSource",
    "StepOutcome",
    "TreeProposal",
    "TrialRunner",
]
