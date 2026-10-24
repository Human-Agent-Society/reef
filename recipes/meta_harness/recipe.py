"""Meta-Harness's recipe: the harness evolution recipe with the population proposer and selection.

``MetaHarnessRecipe`` reads one ``meta_harness:`` block under ``evolution``,
builds the history-aware proposer from it, and owns the selection, which
judges each candidate against the population's committed incumbent. The
population travels as the method state Reef commits with each step.

Example recipe section::

    recipe:
      implementation: recipes.meta_harness.recipe:MetaHarnessRecipe
      config:
        evolution:
          adapter: pi
          seed: seed/
          evaluate: harness.scoring:score_episode
          tasks: ["validation task one", "validation task two"]
          models:
            proposer: {url: https://api.openai.com, model: gpt-5, api_key_env: OPENAI_API_KEY}
          meta_harness:
            mode: full_history
            max_candidates: 20
            max_target_episodes: 200
            max_files: 32
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reef.core.plugins import PluginReference
from reef.harness.tree import TreeKind
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe, HarnessEvolutionSpec

from .proposer import MetaHarnessProposer, MetaHarnessSettings
from .selection import MetaHarnessSelection

#: The keys of the ``evolution.meta_harness`` section.
META_HARNESS_KEYS = ("mode", "model", "components", "max_candidates", "max_target_episodes", "max_files")


@dataclass(frozen=True)
class MetaHarnessRecipe(HarnessEvolutionRecipe):
    """Full-history population search over harness trees."""

    name: str = field(default="meta_harness", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.publish != "auto" or self.review_kinds:
            raise ValueError("Meta-Harness requires automatic publication so selection and serving commit together")
        if self.recheck_every or self.promote_failures or self.promote is not None:
            raise ValueError("Meta-Harness requires a fixed evaluation suite without rechecks or task promotion")

    @classmethod
    def evolution_spec(cls) -> HarnessEvolutionSpec:
        return HarnessEvolutionSpec(selection=PluginReference(MetaHarnessSelection))

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("meta_harness requires an 'evolution' config section")
        if "selection" in evolution:
            raise RecipeConfigError("Meta-Harness owns evolution.selection, which reads its population")
        block = evolution.get("meta_harness")
        if not isinstance(block, Mapping):
            raise RecipeConfigError("meta_harness requires an 'evolution.meta_harness' config section")
        unknown = sorted(set(block) - set(META_HARNESS_KEYS))
        if unknown:
            raise RecipeConfigError(f"evolution.meta_harness has unknown key(s) {', '.join(map(str, unknown))}")
        kinds = block.get("components", ())
        if isinstance(kinds, str) or not isinstance(kinds, Sequence):
            raise RecipeConfigError("evolution.meta_harness.components must be a list of tree kinds")
        tasks = evolution.get("tasks") or ()
        repeats = evolution.get("episode_repeats", 1)
        try:
            meta_settings = MetaHarnessSettings(
                # Both sides of every task, every repeat.
                evaluation_cost=2 * len(tasks) * (repeats if isinstance(repeats, int) else 1),
                mode=str(block.get("mode", "full_history")),
                model=str(block.get("model", "proposer")).strip(),
                kinds=tuple(TreeKind(str(kind)) for kind in kinds),
                max_candidates=block.get("max_candidates", 0),
                max_target_episodes=block.get("max_target_episodes", 0),
                max_files=block.get("max_files", 32),
            )
        except ValueError as exc:
            raise RecipeConfigError(str(exc)) from exc
        supplied = {key: value for key, value in evolution.items() if key != "meta_harness"}
        supplied.setdefault("propose", MetaHarnessProposer(meta_settings))
        return super()._recipe_kwargs({**settings, "evolution": supplied}, values)


__all__ = ["MetaHarnessRecipe"]
