"""GEPA's recipe: the harness evolution recipe with GEPA's proposer and selection.

``GEPARecipe`` reads one ``gepa:`` block under ``evolution``, builds the
proposer from it, and selects with :class:`~recipes.gepa.selection.GEPASelection`
unless the deployment names its own ``evolution.selection``. The search state
(the archive) travels as the method state Reef commits with each step, so
nothing of it lives in a file of its own::

    evolution:
      adapter: pi
      seed: seed/                         # the first tree
      evaluate: harness.aime:AIMEAnswerScorer
      feedback: harness.aime:AIMEFeedback # optional; the score restated when absent
      tasks: [...]                        # the validation set
      models:
        reflection: {url: ..., model: ..., api_key_env: OPENAI_API_KEY}
      gepa:
        minibatch_size: 3
        seed: 0
        skip_perfect_score: true
        perfect_score: 1.0
        max_metric_calls: 150
        components: [rules, skill]
        trainset_size: 45                 # optional; plans GEPA's minibatch order
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reef.core.plugins import PluginReference
from reef.harness.tree import TreeKind
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe, HarnessEvolutionSpec

from .components import EVOLVABLE_KINDS
from .proposer import Feedback, GEPAProposer, GEPASettings
from .selection import GEPASelection


#: The keys of the ``evolution.gepa`` section.
GEPA_KEYS = (
    "minibatch_size",
    "seed",
    "skip_perfect_score",
    "perfect_score",
    "max_metric_calls",
    "components",
    "trainset_size",
)


@dataclass(frozen=True)
class GEPARecipe(HarnessEvolutionRecipe):
    """Harness evolution driven by GEPA's reflective search."""

    name: str = field(default="gepa", kw_only=True)

    @classmethod
    def evolution_spec(cls) -> HarnessEvolutionSpec:
        return HarnessEvolutionSpec(selection=PluginReference(GEPASelection))

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("gepa requires an 'evolution' config section")
        gepa = evolution.get("gepa")
        if not isinstance(gepa, Mapping):
            raise RecipeConfigError("gepa requires an 'evolution.gepa' config section")
        unknown = sorted(set(gepa) - set(GEPA_KEYS))
        if unknown:
            raise RecipeConfigError(f"evolution.gepa has unknown key(s) {', '.join(map(str, unknown))}")
        components = gepa.get("components", ("rules", "skill"))
        if not isinstance(components, Sequence) or isinstance(components, str) or not components:
            evolvable = ", ".join(kind.value for kind in EVOLVABLE_KINDS)
            raise RecipeConfigError(f"evolution.gepa.components must be a non-empty list of kinds from {evolvable}")
        max_calls = gepa.get("max_metric_calls")
        trainset_size = gepa.get("trainset_size")
        try:
            gepa_settings = GEPASettings(
                minibatch_size=integer(gepa, "minibatch_size", 3),
                rng_seed=integer(gepa, "seed", 0),
                skip_perfect_score=boolean(gepa, "skip_perfect_score", True),
                perfect_score=number(gepa, "perfect_score", 1.0),
                max_metric_calls=None if max_calls is None else integer(gepa, "max_metric_calls", 0),
                kinds=tuple(TreeKind(str(kind)) for kind in components),
                trainset_size=None if trainset_size is None else integer(gepa, "trainset_size", 0),
            )
            feedback_setting = evolution.get("feedback")
            feedback = (
                None
                if feedback_setting is None
                else Feedback.from_setting(feedback_setting, setting="evolution.feedback")
            )
        except ValueError as exc:
            raise RecipeConfigError(str(exc)) from exc
        # A deployment that names its own evolution.propose keeps it; GEPA fills the seam it finds empty.
        supplied = {key: value for key, value in evolution.items() if key not in ("gepa", "feedback")}
        supplied.setdefault("propose", GEPAProposer(gepa_settings, feedback=feedback))
        return super()._recipe_kwargs({**settings, "evolution": supplied}, values)


def integer(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeConfigError(f"evolution.gepa.{key} must be an integer")
    return value


def number(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeConfigError(f"evolution.gepa.{key} must be a number")
    return float(value)


def boolean(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise RecipeConfigError(f"evolution.gepa.{key} must be a boolean")
    return value


__all__ = ["GEPARecipe"]
