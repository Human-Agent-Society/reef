"""Strategy contracts for harness evolution: proposers and promoters.

Each is a :class:`~reef.core.plugins.ConfiguredPlugin`. A deployment names
the class (``evolution.propose``, ``evolution.promote``), and the recipe
builds it with ``from_config``; Python callers may pass a built instance. The
episode scorer that ``evolution.evaluate`` names is
:class:`reef.evaluation.harness.EpisodeScorer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.plugins import ConfiguredPlugin
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.tree.mutations import Mutation
from reef.sandbox.process import EpisodeExecutor
from reef.train.cordis_backend.manifest import FailureManifest
from reef.train.types import TrajectoryItem


@dataclass(frozen=True)
class StepProposal:
    """A proposer's mutations with notes the step records and never reads.

    ``notes`` is a JSON mapping (a plan, a review result, what the method
    could not honor) that the commit metrics carry under ``proposal_notes``,
    bounded like the rest of the step record. Empty ``mutations`` skip the
    step as ``None`` does.
    """

    mutations: tuple[Mutation, ...]
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutations", tuple(self.mutations))
        if not isinstance(self.notes, Mapping):
            raise TypeError("StepProposal notes must be a mapping")


class ProposerCalls(ABC):
    """The step's model-call budget, record and live activity, for traffic a proposer makes outside ``models``.

    An agent proposer's own process reaches its model through a gateway, not
    through the bindings; the gateway spends from and records into the same
    per-step budget and ``proposer.json`` the bindings do, and notes what the
    agent does as it does it, for the request page to show while the step runs.
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


@dataclass(frozen=True)
class AgentHost:
    """What a proposer that runs a coding agent needs from the step, handed over as the ``agent_host`` keyword.

    ``descriptor`` and ``binary`` are the harness adapter and its installed
    binary, the same the episodes run; ``executor`` is the one the deployment
    built for the agent (its isolation, not the episodes'); ``step_dir`` is the
    step's record directory, ``None`` with the record off; ``calls`` is the
    step's budget and record; the timeouts bound the whole agent run and each
    trial run of a candidate tree.
    """

    descriptor: AdapterDescriptor
    binary: str
    executor: EpisodeExecutor
    step_dir: Path | None
    calls: ProposerCalls
    timeout_s: float
    trial_timeout_s: float


class Proposer(ConfiguredPlugin, ABC):
    """Base class for harness-evolution mutation proposers.

    Implement:
        ``propose`` — given the current composition (as ``(kind, config)``
        pairs in tree order), a batch of trace samples (a sample's ``score``
        is ``None`` when the deployment batches recorded traffic without
        reports, so a method must handle unscored samples), and the
        :class:`~reef.harness.episodes.model_binding.ModelBindings` the method may
        call, return one :class:`~reef.train.cordis_backend.Mutation`, a
        sequence of them (one composite proposal, applied under one snapshot
        and settled by one selection decision), a :class:`StepProposal`
        (the mutations plus notes the step records), or ``None`` to skip.

    ``models`` is the only way a method reaches a model: ``models.served``
    is the model under test and ``models["name"]`` one the recipe declared
    under ``evolution.models``. ``.chat(...)`` goes to the endpoint the
    deployment configured, and the method never needs to know where that is.

    ``manifest`` is the previous step's
    :class:`~reef.train.cordis_backend.FailureManifest`, or ``None`` when
    no step has settled one yet. ``rejected`` is the recent rejected
    proposals, oldest first, each a mapping of ``step``, ``mutations``
    (``{"op", "id", "options"}`` records, the options as proposed) and the
    selector's ``reason``. ``sources`` is
    one mapping per sample, in sample order: ``record`` (the agent record
    id the sample came from), ``client`` (the ``x-reef-tag-client`` header's
    value, else the session tag, else ``untagged``) and ``untrusted``
    (always true: a sample is client text, never the operator's). Wrap sample text with
    :func:`reef.core.prompts.untrusted_text` before it enters a model prompt. ``entries`` is
    the step's tree as entry options, ``{"id", "name", "config"}`` mappings
    in tree order, so a method can name the entry an ``update`` or ``remove``
    targets. Every keyword is always passed.

    In ``manual`` and ``hybrid`` mode, when an instruction is queued,
    ``requests`` contains exactly one mapping with ``id``, ``text``,
    ``session``, ``release_id``, ``requires`` and ``untrusted=True``. It is the
    instruction that owns this step; ``samples`` is empty in ``manual``, and
    in ``hybrid`` it is what an automatic batch would take next, up to
    ``batch_size`` and possibly none (scored traces, or unscored records
    when the ``pipeline`` section names the inference stages). A proposer that takes
    instructions sets ``reads_requests``. It generates mutations against the current tree, then the
    same evaluation and publication policy used by automatic evolution apply.

    ``requires`` is what the person said the change needs from their
    machine: a list of ``{name, kind, check}`` items, ``kind`` one of
    ``permission``, ``env`` or ``service``, ``check`` optional. The mapping
    is a plain dict the method may extend: when the change it wrote needs
    something of its own (an extension that reads a variable, say), it
    adds items of the same shape to ``request["requires"]``, and the
    backend merges them by name into the commit's
    ``training_request.requires`` after the same shape and text screens
    admission runs; the person's items stand as sent, a bad item of the
    method's is dropped alone, and the mutations still stand. No check
    ever runs on the service.
    """

    @property
    def reads_requests(self) -> bool:
        """Whether this proposer can honor a training instruction (``manual`` and ``hybrid``); override to say so."""
        return False

    @property
    def runs_agent(self) -> bool:
        """Whether this proposer uses the ``agent_host`` it is given to run a coding agent; override to say so."""
        return False

    @abstractmethod
    def propose(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TrajectoryItem, ...],
        models: ModelBindings,
        *,
        manifest: FailureManifest | None = None,
        rejected: Sequence[Mapping[str, Any]] = (),
        sources: Sequence[Mapping[str, Any]] = (),
        requests: Sequence[Mapping[str, Any]] = (),
        entries: Sequence[Mapping[str, Any]] = (),
        agent_host: AgentHost | None = None,
    ) -> Mutation | Sequence[Mutation] | StepProposal | None:
        """Propose mutations for the current composition and trace batch."""


class Promoter(ConfiguredPlugin, ABC):
    """Base class for choosing which trace prompts become permanent evaluation tasks.

    Implement ``promote``: given the step's trace samples and the previous
    step's failure manifest, return the prompts to promote. Reef still
    dedupes, screens for credentials, and caps them.
    """

    @abstractmethod
    def promote(
        self,
        samples: tuple[TrajectoryItem, ...],
        *,
        manifest: FailureManifest | None = None,
    ) -> Sequence[str]:
        """Return the prompts this step should promote into the evaluation."""
