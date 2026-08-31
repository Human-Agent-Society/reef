"""SkillClaw's recipe: harness evolution plus the pool's own delivery surface.

``skillclaw.yaml`` names this class as its dotted recipe reference
(``harness.skillclaw_recipe:SkillClawRecipe``). It changes exactly two things over
the stock ``CordisRecipe`` implementation:

- ``build_surface`` returns ``create_skill_surface([SkillCatalogModule(...)])``, so
  every ``/v1`` request the service proxies carries the served pool's
  catalog section - the sealed campaign's design, where the published pool
  reaches the day's traffic through the proxy, not just through client pull.
- ``build_artifact_validator`` separately binds the catalog tree's admission
  rules; validation is scenario commit policy, not a serving capability.
- ``evolution.seed_skills`` (optional) names a directory of
  ``<name>/SKILL.md`` files - the benchmark's shipped skills library - and
  seeds one skill node per file, so the first-boot composition is the
  bootstrap pool, not an empty one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any

from reef.recipe.errors import RecipeConfigError
from reef.surface import Surface
from reef.surface.skills import SkillValidator, create_skill_surface
from reef.train.cordis_backend import CordisRecipe

from .catalog import SkillCatalogModule
from .config import PUBLIC_SKILL_ROOT


def seed_skill_entries(skills_dir: Path) -> tuple[dict[str, Any], ...]:
    """One ``skill`` node per ``<name>/SKILL.md``, id = skill name, name order.

    Directory names are carried verbatim, as the benchmark's bootstrap copy
    does: the shipped pool includes ``self-improving-agent-3.0.5`` and the
    served tree must land it under that exact name. Night-created skills
    still go through the night's own sanitize_name rule.
    """
    if not skills_dir.is_dir():
        raise RecipeConfigError(f"evolution.seed_skills directory does not exist: {skills_dir}")
    return tuple(
        {
            "id": skill_md.parent.name,
            "name": "skill",
            "config": {"name": skill_md.parent.name, "text": skill_md.read_text(encoding="utf-8")},
        }
        for skill_md in sorted(skills_dir.glob("*/SKILL.md"))
    )


@dataclass(frozen=True)
class SkillClawRecipe(CordisRecipe):
    """Harness evolution with catalog-injecting delivery of the skill pool."""

    _: KW_ONLY
    name: str = "skillclaw"

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        kwargs = super()._recipe_kwargs(settings, values)
        evolution = settings.get("evolution")
        if not isinstance(evolution, Mapping):
            raise RecipeConfigError("evolution must be an object")
        seed_skills = evolution.get("seed_skills")
        if seed_skills:
            if not isinstance(seed_skills, str):
                raise RecipeConfigError("evolution.seed_skills must be a directory path string")
            kwargs["seed"] = tuple(kwargs["seed"]) + seed_skill_entries(Path(seed_skills))
        return kwargs

    def build_surface(self, scenario: str) -> Surface:
        return create_skill_surface([SkillCatalogModule(public_root=PUBLIC_SKILL_ROOT)])

    def build_artifact_validator(self) -> SkillValidator:
        return SkillValidator((SkillCatalogModule(public_root=PUBLIC_SKILL_ROOT),))
