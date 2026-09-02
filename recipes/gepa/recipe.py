"""GEPA's recipe: the config block, and the per-scenario archive binding.

``GEPARecipe`` is a ``CordisRecipe`` that supplies its own ``propose`` and
``selection`` objects instead of asking the config for them. Both halves of
GEPA share one :class:`~recipes.gepa.archive.Archive`, and an archive is
per scenario, so the binding cannot happen at config time: ``build`` opens
``<archive>/<scenario>.json``, constructs the proposer and the selector
against it, and boots the stock backend with those two in place.

The config adds one ``gepa:`` block under ``evolution``::

    evolution:
      adapter: pi
      evaluate: harness.aime:evaluate
      feedback: harness.aime:feedback     # optional; the score restated when absent
      tasks: [...]                        # the validation set
      models:
        reflection: {url: ..., model: ..., api_key_env: OPENAI_API_KEY}
      gepa:
        archive: ${REEF_WORK}/gepa
        minibatch_size: 3
        seed: 0
        skip_perfect_score: true
        perfect_score: 1.0
        max_metric_calls: 150
        components: [rules, skill]

A deployment that names its own ``evolution.propose`` or
``evolution.selection`` keeps them: the recipe only fills the seams it finds
empty, so the archive and the reflection loop can be driven from one side
while the other stays under the operator's control.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

from reef.harness.adapters import get_adapter
from reef.observability import ExperimentLogger
from reef.recipe.errors import RecipeConfigError
from reef.records import RecordStore
from reef.train.cordis_backend import CordisRecipe
from reef.train.cordis_backend.strategies import Mutation, Proposer, resolve_proposer
from reef.train.evaluation.contracts import EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.trainer import Trainer
from reef.train.types import TraceSample

from .archive import Archive
from .components import EVOLVABLE_KINDS
from .method import Feedback, GEPAProposer, GEPASelector, default_feedback

#: Node kinds a deployment evolves unless ``gepa.components`` says otherwise.
#: Rules and skills are the instruction surface both surveyed harnesses
#: share; agent commands are opt in because they only fire when invoked.
DEFAULT_KINDS = ("rules", "skill")

#: The seam the seed id convention protects: ``propose`` receives nodes with
#: entry ids stripped, so a component key can only name the entry it came
#: from while these ids hold. A seed that breaks the convention would have
#: every mutation land on the wrong entry, or on none.
_NAMED_KINDS = ("skill", "agent_command")


class _UnboundProposer(Proposer):
    """The placeholder ``build`` swaps for a proposer bound to the archive."""

    def __call__(
        self,
        nodes: tuple[tuple[str, object], ...],
        samples: tuple[TraceSample, ...],
        models: Any,
        *,
        manifest: Any = None,
    ) -> Mutation | Sequence[Mutation] | None:
        raise RecipeConfigError("the GEPA proposer is bound by GEPARecipe.build; this recipe was not built")


class _UnboundSelector:
    """The placeholder ``build`` swaps for a selector bound to the archive."""

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        raise RecipeConfigError("the GEPA selector is bound by GEPARecipe.build; this recipe was not built")


@dataclass(frozen=True)
class GEPARecipe(CordisRecipe):
    """Harness evolution driven by GEPA's reflective search."""

    _: KW_ONLY
    archive_dir: Path
    minibatch_size: int = 3
    rng_seed: int = 0
    skip_perfect_score: bool = True
    perfect_score: float = 1.0
    max_metric_calls: int | None = None
    kinds: tuple[str, ...] = DEFAULT_KINDS
    feedback: Feedback | None = None
    name: str = "gepa"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.minibatch_size <= 0:
            raise ValueError("evolution.gepa.minibatch_size must be positive")
        unknown = [kind for kind in self.kinds if kind not in EVOLVABLE_KINDS]
        if unknown or not self.kinds:
            raise ValueError(f"evolution.gepa.components must name kinds from {EVOLVABLE_KINDS}, got {self.kinds}")

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("gepa requires an 'evolution' config section")
        # The base recipe insists on both seams; the placeholders keep it
        # satisfied and tell build() which of them it may still bind.
        supplied = dict(evolution)
        supplied.setdefault("propose", _UnboundProposer())
        supplied.setdefault("selection", _UnboundSelector())
        kwargs = super()._recipe_kwargs({**settings, "evolution": supplied}, values)
        _check_seed_ids(kwargs["seed"])

        gepa = evolution.get("gepa")
        if not isinstance(gepa, Mapping):
            raise RecipeConfigError("gepa requires an 'evolution.gepa' config section")
        archive = gepa.get("archive")
        if not isinstance(archive, str) or not archive.strip():
            raise RecipeConfigError("evolution.gepa.archive must be a directory path string")
        components = gepa.get("components", DEFAULT_KINDS)
        if not isinstance(components, Sequence) or isinstance(components, str) or not components:
            raise RecipeConfigError(
                f"evolution.gepa.components must be a non-empty list of kinds from {EVOLVABLE_KINDS}"
            )
        max_calls = gepa.get("max_metric_calls")
        feedback = evolution.get("feedback")
        return {
            **kwargs,
            "archive_dir": Path(archive.strip()),
            "minibatch_size": _int(gepa, "minibatch_size", 3),
            "rng_seed": _int(gepa, "seed", 0),
            "skip_perfect_score": _bool(gepa, "skip_perfect_score", True),
            "perfect_score": _float(gepa, "perfect_score", 1.0),
            "max_metric_calls": None if max_calls is None else _int(gepa, "max_metric_calls", 0),
            "kinds": tuple(str(kind) for kind in components),
            "feedback": None if feedback is None else _resolve_feedback(feedback),
        }

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        archive = Archive(self.archive_dir / f"{scenario}.json")
        propose = self.propose
        if isinstance(propose, _UnboundProposer):
            propose = resolve_proposer(
                GEPAProposer(
                    archive=archive,
                    descriptor=get_adapter(self.adapter),
                    binary=self.binary,
                    score_episode=self.score_episode,
                    feedback=self.feedback or default_feedback,
                    minibatch_size=self.minibatch_size,
                    rng_seed=self.rng_seed,
                    skip_perfect_score=self.skip_perfect_score,
                    perfect_score=self.perfect_score,
                    max_metric_calls=self.max_metric_calls,
                    kinds=self.kinds,
                    valset_size=len(self.tasks),
                )
            )
        selector = self.candidate_selector
        if isinstance(selector, _UnboundSelector):
            selector = GEPASelector(archive)
        bound = dataclasses.replace(self, propose=propose, candidate_selector=selector)
        # ``bound`` is a GEPARecipe, so the stock build is named explicitly:
        # calling it through ``bound`` would come straight back here.
        return CordisRecipe.build(
            bound,
            scenario,
            records,
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
        )


def _check_seed_ids(seed: Sequence[Mapping[str, Any]]) -> None:
    """Every seeded node's entry id must be the id its component key implies."""
    for entry in seed:
        kind, entry_id = str(entry.get("name")), str(entry.get("id"))
        config = entry.get("config")
        config = config if isinstance(config, Mapping) else {}
        if kind == "rules" and entry_id != "rules":
            raise RecipeConfigError(f"a seeded rules node must have entry id 'rules', got {entry_id!r}")
        if kind in _NAMED_KINDS and entry_id != str(config.get("name")):
            raise RecipeConfigError(
                f"seed entry {entry_id!r} must have id equal to its config name {config.get('name')!r}; "
                "GEPA addresses a node by the name it composes under"
            )


def _resolve_feedback(value: Any) -> Feedback:
    """``evolution.feedback`` as a callable, or a dotted reference to one."""
    if callable(value):
        return value
    if isinstance(value, str) and ":" in value:
        module_name, _, attribute = value.partition(":")
        try:
            resolved = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise RecipeConfigError(f"cannot import evolution.feedback {value!r}: {exc}") from exc
        if callable(resolved):
            return resolved
    raise RecipeConfigError("evolution.feedback must be a callable or a dotted 'module:attribute' reference")


def _int(section: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(section.get(key, default))
    except (TypeError, ValueError) as exc:
        raise RecipeConfigError(f"evolution.gepa.{key} must be an integer") from exc


def _float(section: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(section.get(key, default))
    except (TypeError, ValueError) as exc:
        raise RecipeConfigError(f"evolution.gepa.{key} must be a number") from exc


def _bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise RecipeConfigError(f"evolution.gepa.{key} must be a boolean")
    return value
