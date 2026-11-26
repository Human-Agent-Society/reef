"""Keep contributor guide paths and navigation tied to the repository."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
CONTRIBUTOR_GUIDES = (
    REPO_ROOT / "docs" / "contributing" / "codebase-structure.rst",
    REPO_ROOT / "docs" / "contributing" / "development.rst",
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

    for guide in CONTRIBUTOR_GUIDES:
        slug = guide.stem
        assert f"https://reefinfra.ai/docs/contributing/{slug}/" in contributing
        assert f'"contributing/{guide.name}"' in navigation
