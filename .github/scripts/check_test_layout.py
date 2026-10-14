#!/usr/bin/env python3
"""Check test ownership directories and the installed-wheel suite selection."""

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIRECTORIES = {
    "artifact",
    "cli",
    "contributing",
    "core",
    "evaluation",
    "harness",
    "inference",
    "observability",
    "packaging",
    "plugin_contracts",
    "recipes",
    "record2record",
    "record2task",
    "reef_client",
    "reef_recipe",
    "reef_service",
    "runtime",
    "scenario",
    "slime_backend",
    "smoke",
    "storage",
    "surface",
    "task2batch",
    "tinker_backend",
    "train",
}


def is_test_module(path: Path) -> bool:
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py"))


def layout_errors(root: Path, paths: Iterable[Path]) -> list[str]:
    """Validate repository-relative paths, excluding rendered harness fixtures."""
    errors = []
    for path in sorted(paths):
        if path.is_relative_to("tests/harness/data/harness_goldens") or not is_test_module(path):
            continue
        parts = path.parts
        if len(parts) < 3 or parts[0] != "tests" or parts[1] not in TEST_DIRECTORIES:
            errors.append(f"{path}: place tests in an owner directory from tests/AGENTS.md")
        elif parts[1] == "recipes" and (
            len(parts) < 4
            or not any((root / location / parts[2]).is_dir() for location in ("recipes", "recipes/beta"))
        ):
            errors.append(f"{path}: use tests/recipes/<recipe>/ for an existing cookbook method")
    return errors


def wheel_errors(root: Path, entries: Iterable[str]) -> list[str]:
    """Reject stale, duplicate, or non-test paths before CI selects the wheel suite."""
    errors = []
    seen: set[str] = set()
    for entry in entries:
        path = Path(entry)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to("tests")
            or not is_test_module(path)
            or not (root / path).is_file()
        ):
            errors.append(f"tests/packaging/wheel-tests.txt: invalid test module {entry!r}")
        elif entry in seen:
            errors.append(f"tests/packaging/wheel-tests.txt: duplicate test module {entry!r}")
        seen.add(entry)
    if not seen:
        errors.append("tests/packaging/wheel-tests.txt: the installed-wheel suite must not be empty")
    return errors


def main() -> int:
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "tests"], cwd=ROOT
    ).decode()
    paths = {Path(name) for name in files.split("\0") if name and (ROOT / name).is_file()}
    errors = layout_errors(ROOT, paths)
    entries = (ROOT / "tests/packaging/wheel-tests.txt").read_text().splitlines()
    errors.extend(wheel_errors(ROOT, entries))
    for error in errors:
        print(error)
    if errors:
        return 1
    print("Test placement and installed-wheel selection passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
