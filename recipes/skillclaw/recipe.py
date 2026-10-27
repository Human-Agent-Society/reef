"""SkillClaw's recipe: harness evolution plus the pool's delivery surface.

``skillclaw.yaml`` names this class as its dotted recipe reference
(``recipes.skillclaw.recipe:SkillClawRecipe``) and the night method in
``evolution.method``. The recipe changes three things over the harness
evolution recipe:

- ``build_surface`` returns ``create_skill_surface([SkillCatalogModule(...)])``, so
  every ``/v1`` request the service proxies carries the served pool's
  catalog section - the sealed campaign's design, where the published pool
  reaches the day's traffic through the proxy, not just through client pull.
  The same surface binds the catalog tree's admission rules, which the
  scenario runs before a pool is published or restored.
- No harness tree is installed, so ``harness_adapter`` names none.
- ``evolution.seed_skills`` (optional) names a directory of
  ``<name>/SKILL.md`` files - the benchmark's shipped skills library - and
  adds each as ``skills/<name>/SKILL.md`` to the seed tree, so the first
  release is the bootstrap pool, not an empty one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recipes.skillclaw.harness.catalog import SkillCatalogModule
from recipes.skillclaw.harness.config import PUBLIC_SKILL_ROOT
from reef.harness.tree import TreePatch
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.surface import Surface
from reef.surface.skills import create_skill_surface


def seed_skills(skills_dir: Path) -> TreePatch:
    """Every ``<name>/SKILL.md`` under ``skills_dir`` as the tree file ``skills/<name>/SKILL.md``.

    Directory names are carried verbatim, as the benchmark's bootstrap copy
    does: the shipped pool includes ``self-improving-agent-3.0.5`` and the
    served tree must land it under that exact name. Night-created skills
    still go through the night's own sanitize_name rule.
    """
    if not skills_dir.is_dir():
        raise RecipeConfigError(f"evolution.seed_skills directory does not exist: {skills_dir}")
    return TreePatch(
        {
            f"skills/{skill_md.parent.name}/SKILL.md": skill_md.read_text(encoding="utf-8")
            for skill_md in sorted(skills_dir.glob("*/SKILL.md"))
        }
    )


@dataclass(frozen=True)
class SkillClawRecipe(HarnessEvolutionRecipe):
    """Harness evolution of a skill pool that every proxied request carries as a catalog.

    Values: the fields of ``HarnessEvolutionRecipe``, with ``name``
    ``skillclaw``; ``seed`` holds the ``evolution.seed_skills`` pool.
    Purpose: skillclaw.yaml names it, run.py builds it, and the service
    serves the pool through its skill surface.
    """

    name: str = field(default="skillclaw", kw_only=True)

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("evolution must be an object")
        # seed_skills is this recipe's own key: the base recipe refuses a key it does not know.
        supplied = dict(evolution)
        skills = supplied.pop("seed_skills", None)
        kwargs = super()._recipe_kwargs({**settings, "evolution": supplied}, values)
        if skills:
            if not isinstance(skills, str):
                raise RecipeConfigError("evolution.seed_skills must be a directory path string")
            kwargs["seed"] = kwargs["seed"].apply(seed_skills(Path(skills)))
        return kwargs

    def build_surface(self, scenario: str) -> Surface:
        return create_skill_surface([SkillCatalogModule(public_root=PUBLIC_SKILL_ROOT)])

    @property
    def harness_adapter(self) -> str | None:
        return None
