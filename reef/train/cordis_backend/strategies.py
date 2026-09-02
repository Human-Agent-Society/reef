"""Strategy contracts for harness evolution: proposers and episode scorers.

Both are resolved from recipe config as either a Python callable passed
directly or a dotted ``"module:attribute"`` reference. Plain callables are
wrapped by ``_CallableProposer`` / ``_CallableEpisodeScorer`` so the backend
always receives a typed instance.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.errors import ReefError
from reef.harness.episode import EpisodeResult
from reef.harness.model_binding import ModelBindings
from reef.train.cordis_backend.manifest import FailureManifest
from reef.train.types import TraceSample


class MutationError(ReefError):
    """A proposed mutation could not be applied."""


@dataclass(frozen=True)
class Mutation:
    """One proposed change to the composition tree, by entry id.

    ``create`` and ``update`` carry ``options`` (entry options without the
    id, e.g. ``{"name": "rules", "config": {"text": ...}}``); an ``update``
    merges them into the entry, with ``None`` values deleting keys, exactly
    as the compose loader reconciles. Ids are root-level: the minimal layer
    composes a flat tree, so nested (``:``-qualified) ids are rejected.
    """

    op: str
    id: str
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _MUTATION_OPS = ("create", "update", "remove")
        if self.op not in _MUTATION_OPS:
            raise MutationError(f"mutation op must be one of {_MUTATION_OPS}, got {self.op!r}")
        if not self.id or ":" in self.id:
            raise MutationError(f"mutation id must be a non-empty root-level id, got {self.id!r}")
        if self.op in ("create", "update") and not isinstance(self.options, Mapping):
            raise MutationError(f"{self.op} mutation requires options")
        if self.op == "remove" and self.options is not None:
            raise MutationError("remove mutation takes no options")


class Proposer(ABC):
    """Base class for harness-evolution mutation proposers.

    Implement:
        ``__call__`` — given the current composition (as ``(kind, config)``
        pairs in tree order), a batch of trace samples (a sample's ``score``
        is ``None`` when the deployment batches recorded traffic without
        reports, so a method must handle unscored samples), and the
        :class:`~reef.harness.model_binding.ModelBindings` the method may
        call, return one :class:`~reef.train.cordis_backend.Mutation`, a
        sequence of them (one composite proposal, applied under one snapshot
        and settled by one selection decision), or ``None`` to skip.

    ``models`` is the only way a method reaches a model: ``models.served``
    is the model under test and ``models["name"]`` one the recipe declared
    under ``evolution.models``. ``.chat(...)`` goes to the endpoint the
    deployment configured, and the method never needs to know where that is.

    ``manifest`` is the previous step's
    :class:`~reef.train.cordis_backend.FailureManifest`, or ``None`` when
    no step has settled one yet. The keyword is only forwarded to callables
    whose signature names it, so pre-manifest proposers run unchanged.
    """

    @abstractmethod
    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TraceSample, ...],
        models: ModelBindings,
        *,
        manifest: FailureManifest | None = None,
    ) -> Mutation | Sequence[Mutation] | None:
        """Propose mutations for the current composition and trace batch."""


class EpisodeScorer(ABC):
    """Base class for scoring one harness-evolution episode.

    Implement:
        ``__call__`` — given a task name and its episode result, return a
        float score. Higher is better; non-finite values raise.
    """

    @abstractmethod
    def __call__(self, task: str, result: EpisodeResult) -> float:
        """Score one episode result for a task."""


def accepts_manifest(fn: Callable[..., Any]) -> bool:
    """Whether ``fn``'s signature names ``manifest`` or takes ``**kwargs``.

    Signature inspection is the compatibility gate: the manifest keyword is
    only ever passed to code that declared it, so every three-argument
    proposer written before the manifest existed runs unchanged.
    """
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "manifest" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )


class _CallableProposer(Proposer):
    """Adapter wrapping a plain callable as a :class:`Proposer` instance."""

    def __init__(
        self,
        fn: Callable[..., Mutation | Sequence[Mutation] | None],
    ) -> None:
        self._fn = fn
        self._forward_manifest = accepts_manifest(fn)

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TraceSample, ...],
        models: ModelBindings,
        *,
        manifest: FailureManifest | None = None,
    ) -> Mutation | Sequence[Mutation] | None:
        if self._forward_manifest:
            return self._fn(nodes, samples, models, manifest=manifest)
        return self._fn(nodes, samples, models)


class _CallableEpisodeScorer(EpisodeScorer):
    """Adapter wrapping a plain callable as an episode scorer."""

    def __init__(self, fn: Callable[[str, EpisodeResult], float]) -> None:
        self._fn = fn

    def __call__(self, task: str, result: EpisodeResult) -> float:
        return self._fn(task, result)


def resolve_proposer(value: object) -> Proposer:
    """Resolve a callable or dotted reference into a :class:`Proposer` instance.

    ``value`` may be a :class:`Proposer` instance, a plain callable with the
    proposer signature, or a dotted ``"module:attribute"`` string reference.
    """
    if isinstance(value, Proposer):
        return value
    if callable(value):
        return _CallableProposer(value)
    if isinstance(value, str) and ":" in value:
        import importlib

        module_name, _, attribute = value.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot import proposer {value!r}: {exc}") from exc
        if isinstance(resolved, Proposer):
            return resolved
        if callable(resolved):
            return _CallableProposer(resolved)
    raise ValueError("propose must be a Proposer instance, a callable, or a dotted 'module:attribute' reference")


def resolve_episode_scorer(value: object) -> EpisodeScorer:
    """Resolve a callable or dotted reference into an episode scorer.

    ``value`` may be an ``EpisodeScorer``, a plain callable with the scorer
    signature, or a dotted ``"module:attribute"`` string reference.
    """
    if isinstance(value, EpisodeScorer):
        return value
    if callable(value):
        return _CallableEpisodeScorer(value)
    if isinstance(value, str) and ":" in value:
        import importlib

        module_name, _, attribute = value.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot import episode scorer {value!r}: {exc}") from exc
        if isinstance(resolved, EpisodeScorer):
            return resolved
        if callable(resolved):
            return _CallableEpisodeScorer(resolved)
    raise ValueError(
        "evaluate must be an EpisodeScorer instance, a callable, or a dotted 'module:attribute' reference"
    )
