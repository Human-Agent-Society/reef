"""The mirror is complete: every reference case is mirrored or omitted.

This is what makes an upstream bump visible. Bumping the submodule without
re-porting leaves a new ``it()`` matched by nothing, and this test names it.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections import Counter
from pathlib import Path

import pytest

from . import omissions

# The port conforms to these two packages only (UPSTREAM.md); the reference
# repository also ships hmr, include, timer and logger-console.
PACKAGES = ("core", "loader")

_IT = re.compile(r"^\s*it\((['\"])(.*?)\1", re.M)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE = _REPO_ROOT / "third_party" / "cordis" / "packages"


def _upstream_cases() -> Counter[tuple[str, str]]:
    found: Counter[tuple[str, str]] = Counter()
    for package in PACKAGES:
        for spec in sorted((_REFERENCE / package / "tests").glob("*.spec.ts")):
            name = f"{package}/tests/{spec.name}"
            for match in _IT.finditer(spec.read_text()):
                found[(name, match.group(2))] += 1
    return found


def _mirrored_cases() -> Counter[tuple[str, str]]:
    found: Counter[tuple[str, str]] = Counter()
    package = importlib.import_module(__package__)
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.endswith("_spec"):
            continue
        module = importlib.import_module(f"{__package__}.{info.name}")
        for _, member in inspect.getmembers(module):
            if inspect.isfunction(member):
                marker = getattr(member, "__upstream__", None)
                if marker is not None:
                    found[marker] += 1
            elif inspect.isclass(member):
                for _, method in inspect.getmembers(member, inspect.isfunction):
                    marker = getattr(method, "__upstream__", None)
                    if marker is not None:
                        found[marker] += 1
    return found


@pytest.fixture(scope="module")
def upstream_cases() -> Counter[tuple[str, str]]:
    if not _REFERENCE.is_dir():
        pytest.skip("reference submodule absent: run `git submodule update --init third_party/cordis`")
    cases = _upstream_cases()
    if not cases:
        pytest.skip("reference submodule is empty")
    return cases


def test_every_reference_case_is_mirrored_or_omitted(upstream_cases: Counter[tuple[str, str]]) -> None:
    accounted = _mirrored_cases() + Counter(dict.fromkeys(omissions.OMITTED, 1))
    missing = upstream_cases - accounted
    assert not missing, (
        "reference cases nothing accounts for -- mirror them, or record them in omissions.py "
        f"with the UPSTREAM.md omission they fall under: {sorted(missing.elements())}"
    )


def test_no_mirror_or_omission_outlives_its_reference_case(upstream_cases: Counter[tuple[str, str]]) -> None:
    accounted = _mirrored_cases() + Counter(dict.fromkeys(omissions.OMITTED, 1))
    stale = accounted - upstream_cases
    assert not stale, (
        "mirrors or omissions with no reference case left -- the upstream test was renamed or "
        f"deleted, so re-read the hunk before dropping them here: {sorted(stale.elements())}"
    )


def test_omissions_are_disjoint_from_mirrors() -> None:
    overlap = set(omissions.OMITTED) & set(_mirrored_cases())
    assert not overlap, f"cases both mirrored and recorded as omitted: {sorted(overlap)}"
