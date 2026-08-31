"""The task-suite contract, enforced for every first-party task.

Each task under tasks/ that ships a ``tests/grader_tests.json`` gets,
automatically:

- judge scoring cases: every declared solution scores its declared reward
  (each case says why; zero-reward cases must also produce a reason);
- a stock-Harbor check: its ``task.toml`` validates under Harbor's
  ``TaskConfig`` (skipped when harbor isn't installed).

Adding a task therefore means adding a folder; the suite picks it up with no
test code written. This is the anti-hack wall as a standing contract, not a
per-task effort.
"""

import importlib.util
import json
from pathlib import Path

import pytest

TASKS_ROOT = Path(__file__).parent.parent / "tasks"
# The template ships as a complete working task and carries its own
# grader_tests.json, so this glob already holds it to the same contract and
# copying it always starts from green.
GRADER_TESTED_TASKS = sorted(
    p.parent.parent for p in TASKS_ROOT.glob("**/tests/grader_tests.json")
)
# Skip `_`- and `.`-prefixed directories inside the catalog. The test looks
# only at the parts below TASKS_ROOT, so a checkout living under a
# dot-directory still collects every task.
ALL_TASKS = sorted(
    p.parent
    for p in TASKS_ROOT.glob("**/task.toml")
    if not any(part.startswith(("_", ".")) for part in p.relative_to(TASKS_ROOT).parts)
)
assert ALL_TASKS, f"no tasks collected under {TASKS_ROOT}"


def _load_grader(task_dir: Path, filename: str = "score.py"):
    spec = importlib.util.spec_from_file_location(
        f"{filename.removesuffix('.py')}_{task_dir.name}",
        task_dir / "environment" / filename,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_artifact(tmp_path: Path, payload) -> Path:
    artifact = tmp_path / "solution.json"
    artifact.write_text(json.dumps(payload))
    return artifact


def _grader_tests(task_dir: Path) -> dict:
    return json.loads((task_dir / "tests" / "grader_tests.json").read_text())


@pytest.mark.parametrize("task_dir", GRADER_TESTED_TASKS, ids=lambda p: p.name)
def test_grader_cases(task_dir, tmp_path):
    grader = _load_grader(task_dir)
    for case in _grader_tests(task_dir)["cases"]:
        result = grader.grade(_write_artifact(tmp_path, case["solution"]))
        assert result["reward"] == pytest.approx(
            case["reward"], abs=case.get("tolerance", 1e-9)
        ), (case["why"], result)
        if case["reward"] == 0.0:
            assert result.get("reason"), (
                f"zero-reward case gave no reason: {case['why']}"
            )


@pytest.mark.parametrize("task_dir", GRADER_TESTED_TASKS, ids=lambda p: p.name)
def test_final_judge_cases(task_dir, tmp_path):
    """Tasks with a final judge (final.py: hidden tests, run once on the
    best submission) declare its expectations as final_cases."""
    cases = _grader_tests(task_dir).get("final_cases", [])
    if not cases:
        pytest.skip("no final judge")
    grader = _load_grader(task_dir, "final.py")
    for case in cases:
        result = grader.grade(_write_artifact(tmp_path, case["solution"]))
        assert result["reward"] == pytest.approx(
            case["reward"], abs=case.get("tolerance", 1e-9)
        ), (case["why"], result)


@pytest.mark.parametrize("task_dir", GRADER_TESTED_TASKS, ids=lambda p: p.name)
def test_missing_artifact_scores_zero(task_dir):
    grader = _load_grader(task_dir)
    assert grader.grade(None)["reward"] == 0.0


@pytest.mark.parametrize("task_dir", GRADER_TESTED_TASKS, ids=lambda p: p.name)
def test_task_is_stock_harbor(task_dir):
    pytest.importorskip("harbor")
    import tomllib

    from harbor.models.task.config import TaskConfig

    TaskConfig.model_validate(tomllib.loads((task_dir / "task.toml").read_text()))
    for required in (
        "instruction.md",
        "environment/Dockerfile",
        "environment/Dockerfile.judge",
        "environment/judge_server.py",
        "environment/score.py",
        "environment/docker-compose.yaml",
        "tests/test.sh",
        "solution/solve.sh",
    ):
        assert (task_dir / required).exists(), f"{task_dir.name} missing {required}"


@pytest.mark.parametrize(
    "task_dir", ALL_TASKS, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_every_committed_task_validates(task_dir):
    """Every task in the catalog, first-party and synced external alike, is a
    valid stock Harbor task."""
    pytest.importorskip("harbor")
    import tomllib

    from harbor.models.task.config import TaskConfig

    TaskConfig.model_validate(tomllib.loads((task_dir / "task.toml").read_text()))
    assert (task_dir / "instruction.md").exists()


@pytest.mark.parametrize(
    "task_dir",
    sorted(TASKS_ROOT.glob("autoresearch/first-party/*/")),
    ids=lambda p: p.name,
)
def test_first_party_tasks_ship_grader_tests(task_dir):
    """Policy: grader unit tests are optional for your own tasks, required
    for first-party ones, whose graders hand out continuous scores from
    agent-written files, so untested leniency is free points."""
    assert (task_dir / "tests" / "grader_tests.json").is_file(), (
        f"{task_dir.name} is first-party and must ship tests/grader_tests.json"
    )
