from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
TTTD_HARBOR = REPO_ROOT / "recipes" / "tttd" / "examples" / "tttd" / "harbor"
PACKING_TASKS = {
    "circle_packing_26": {
        "count": 26,
        "instruction_length": 3_105,
        "instruction_sha256": "76111e7a7d9c594b4c4dac12df7d0851f1b1ad1daabce9e80c93633f689d27dc",
    },
    "circle_packing_32": {
        "count": 32,
        "instruction_length": 3_105,
        "instruction_sha256": "352187e63a5ad14d91358bf6742a9aba7d2f5222ebe283d672c31ab8317a230e",
    },
}
VALIDATOR_SHA256 = "f7aa4deec93f5bb7c0703df5b29a4aff9c2d9e4f03471ec1911c0e37bf4fb628"


def test_tttd_harbor_tasks_have_a_uniform_layout() -> None:
    assert {path.name for path in TTTD_HARBOR.iterdir() if path.is_dir()} == {
        "circle_packing_26",
        "circle_packing_32",
        "erdos_min_overlap",
    }
    for task in ("erdos_min_overlap", *PACKING_TASKS):
        task_root = TTTD_HARBOR / task
        assert (task_root / "task.toml").is_file()
        assert (task_root / "instruction.md").is_file()
        assert (task_root / "environment" / "score.py").is_file()
        assert (task_root / "tests" / "test.sh").is_file()


def _load_score(task: str) -> ModuleType:
    path = TTTD_HARBOR / task / "environment" / "score.py"
    spec = importlib.util.spec_from_file_location(f"tttd_{task}_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("task", "expected"), PACKING_TASKS.items())
def test_packing_harbor_tasks_are_self_contained(task: str, expected: dict[str, object]) -> None:
    task_root = TTTD_HARBOR / task
    for relative_path in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/Dockerfile.judge",
        "environment/docker-compose.yaml",
        "environment/judge_config.json",
        "environment/judge_server.py",
        "environment/score.py",
        "tests/grade.py",
        "tests/test.sh",
    ):
        assert (task_root / relative_path).is_file()

    config = tomllib.loads((task_root / "task.toml").read_text())
    assert config["task"]["name"] == f"reef-eval/tttd-circle-packing-{expected['count']}"
    assert config["environment"]["cpus"] == 1


@pytest.mark.parametrize(("task", "expected"), PACKING_TASKS.items())
def test_packing_static_instructions_are_stable(task: str, expected: dict[str, object]) -> None:
    instruction = (TTTD_HARBOR / task / "instruction.md").read_text()

    # Parent code, reward, and output are dynamic search state appended by the
    # shared harness; instruction.md contains only the immutable task contract.
    assert len(instruction) == expected["instruction_length"]
    assert hashlib.sha256(instruction.encode()).hexdigest() == expected["instruction_sha256"]


@pytest.mark.parametrize("task", PACKING_TASKS)
def test_packing_validator_source_matches_ttt_discover(task: str) -> None:
    module = _load_score(task)
    source = inspect.getsource(module.validate_packing)

    assert hashlib.sha256(source.encode()).hexdigest() == VALIDATOR_SHA256


@pytest.mark.parametrize(("task", "expected"), PACKING_TASKS.items())
def test_packing_grade_verifies_geometry_and_ignores_reported_sum(
    task: str,
    expected: dict[str, object],
    tmp_path: Path,
) -> None:
    count = int(expected["count"])
    centers, radii = _grid_packing(count)
    artifact = tmp_path / "solution.py"
    artifact.write_text(
        "import numpy as np\n\n"
        "def run_packing():\n"
        f"    centers = np.array({centers.tolist()!r}, dtype=np.float64)\n"
        f"    radii = np.array({radii.tolist()!r}, dtype=np.float64)\n"
        "    return centers, radii, -999.0\n"
    )

    result = _load_score(task).grade(artifact)

    assert result["reward"] == pytest.approx(float(np.sum(radii)))
    assert "sum of radii" in result["reason"]


@pytest.mark.parametrize(("task", "expected"), PACKING_TASKS.items())
def test_packing_validator_rejects_shape_overlap_boundary_and_nan(
    task: str,
    expected: dict[str, object],
) -> None:
    module = _load_score(task)
    count = int(expected["count"])
    centers, radii = _grid_packing(count)

    assert module.check_packing_correctness(centers, radii, count)
    assert not module.check_packing_correctness(centers[:-1], radii[:-1], count)

    overlapping = centers.copy()
    overlapping[1] = overlapping[0]
    assert not module.validate_packing(overlapping, radii)

    outside = centers.copy()
    outside[0, 0] = 0.0
    assert not module.validate_packing(outside, radii)

    nonfinite = radii.copy()
    nonfinite[0] = np.nan
    assert not module.validate_packing(centers, nonfinite)


def _grid_packing(num_circles: int) -> tuple[np.ndarray, np.ndarray]:
    columns = 8
    rows = 4
    centers = np.array(
        [((column + 0.5) / columns, (row + 0.5) / rows) for row in range(rows) for column in range(columns)][
            :num_circles
        ],
        dtype=np.float64,
    )
    return centers, np.full(num_circles, 0.04, dtype=np.float64)
