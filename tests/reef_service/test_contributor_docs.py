"""Keep contributor routing and playbook paths tied to the repository."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
CONTRIBUTOR_GUIDES = (
    REPO_ROOT / "docs" / "contributing" / "codebase-structure.rst",
    REPO_ROOT / "docs" / "contributing" / "adding-components.rst",
)
REPOSITORY_PATH = re.compile(r"``((?:(?:reef|tests|docs|examples|docker)/[^`]*|pyproject\.toml))``")


def documented_repository_paths(source: str) -> set[str]:
    """Return concrete repository paths from RST inline literals."""
    return {
        value.rstrip("/")
        for value in REPOSITORY_PATH.findall(source)
        if not any(marker in value for marker in ("<", ">", "*", "{"))
    }


def test_contributor_guide_paths_exist() -> None:
    documented = set()
    for guide in CONTRIBUTOR_GUIDES:
        documented.update(documented_repository_paths(guide.read_text(encoding="utf-8")))

    missing = sorted(path for path in documented if not (REPO_ROOT / path).exists())
    assert not missing, f"contributor guides reference missing repository paths: {missing}"


def test_contributor_guides_are_linked_from_entry_points_and_site_navigation() -> None:
    # The docs site has no landing page: the sidebar navigation is the index,
    # so CONTRIBUTING.md and the navigation are the two entry points.
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    navigation = (REPO_ROOT / "docs" / "site" / "lib" / "docs.ts").read_text(encoding="utf-8")

    for filename in ("codebase-structure.rst", "adding-components.rst"):
        slug = Path(filename).stem
        assert f"https://reefinfra.ai/docs/contributing/{slug}/" in contributing
        assert f'"contributing/{filename}"' in navigation


def test_component_playbooks_name_the_live_registration_points() -> None:
    playbooks = CONTRIBUTOR_GUIDES[1].read_text(encoding="utf-8")
    recipe_registry = (REPO_ROOT / "reef" / "recipe" / "registry.py").read_text(encoding="utf-8")
    runtime_registry = (REPO_ROOT / "reef" / "runtime" / "registry.py").read_text(encoding="utf-8")
    route_registry = (REPO_ROOT / "reef" / "service" / "routes" / "__init__.py").read_text(encoding="utf-8")

    assert "@register_kind" in playbooks and "def register_kind" in recipe_registry
    assert "@register_runtime_kind" in playbooks and "def register_runtime_kind" in runtime_registry
    assert "register_routes" in playbooks and "def register_routes" in route_registry
