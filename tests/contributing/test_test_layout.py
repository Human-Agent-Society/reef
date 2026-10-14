"""Test placement checks reject accidental omissions and misplaced modules."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / ".github/scripts"))
    return importlib.import_module("check_test_layout")


def test_owner_suites_and_fixture_trees_are_accepted(tmp_path: Path, policy: ModuleType) -> None:
    (tmp_path / "recipes/beta/coral").mkdir(parents=True)
    paths = [
        Path("tests/runtime/test_scheduler.py"),
        Path("tests/reef_recipe/test_composite_recipe.py"),
        Path("tests/reef_client/test_reef_client.py"),
        Path("tests/recipes/coral/test_reports.py"),
        Path("tests/harness/data/harness_goldens/native/test_task.py"),
        Path("tests/conftest.py"),
    ]
    assert policy.layout_errors(tmp_path, paths) == []


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_reports.py",
        "tests/reports_test.py",
        "tests/misc/test_reports.py",
        "tests/support/test_reports.py",
        "tests/recipes/test_reports.py",
        "tests/recipes/missing/test_reports.py",
    ],
)
def test_misplaced_modules_are_rejected(tmp_path: Path, policy: ModuleType, path: str) -> None:
    errors = policy.layout_errors(tmp_path, [Path(path)])
    assert len(errors) == 1
    assert path in errors[0]


def test_wheel_selection_rejects_missing_duplicate_and_non_test_entries(tmp_path: Path, policy: ModuleType) -> None:
    module = "tests/runtime/test_scheduler.py"
    (tmp_path / module).parent.mkdir(parents=True)
    (tmp_path / module).touch()
    assert policy.wheel_errors(tmp_path, [module]) == []
    errors = policy.wheel_errors(tmp_path, [module, module, "tests/runtime/test_missing.py", "../test_outside.py"])
    assert len(errors) == 3
    assert "duplicate" in errors[0]
    assert "test_missing.py" in errors[1]
    assert "test_outside.py" in errors[2]
    assert policy.wheel_errors(tmp_path, [])
