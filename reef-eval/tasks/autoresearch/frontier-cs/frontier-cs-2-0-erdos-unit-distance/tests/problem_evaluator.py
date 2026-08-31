"""Evaluator for the Erdos unit distance 2.0 problem."""

from __future__ import annotations

import importlib.util
import math
import os
import pickle
import pwd
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

N_POINTS = 65536
BASELINE_EDGES = N_POINTS
TIMEOUT_SECONDS = 10800
UNIT_DISTANCE = 1.0
DISTANCE_REL_TOL = 1e-10
DISTANCE_ABS_TOL = 1e-10
MIN_SEPARATION = 1e-3
SCORE_POWER = 3.0


def _protect_evaluator_source() -> None:
    """Hide evaluator source from unprivileged submitted solutions in containers."""
    try:
        evaluator_path = Path(__file__).resolve()
        if str(evaluator_path).startswith(("/judge/", "/tests/")) and os.geteuid() == 0:
            evaluator_path.chmod(0o600)
    except Exception:
        pass


_protect_evaluator_source()


def _solution_preexec():
    """Return a preexec_fn that runs submitted code as nobody when possible."""
    if os.name != "posix":
        return None
    try:
        if os.geteuid() != 0:
            return None
        nobody = pwd.getpwnam("nobody")
    except Exception:
        return None

    def demote() -> None:
        os.setgid(nobody.pw_gid)
        os.setuid(nobody.pw_uid)

    return demote


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _to_points(raw: Any) -> list[tuple[float, float]]:
    try:
        values = raw.tolist()
    except Exception:
        values = list(raw)

    points: list[tuple[float, float]] = []
    for index, item in enumerate(values):
        try:
            pair = item.tolist()
        except Exception:
            pair = item
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"point {index} is not a 2D coordinate pair")
        x, y = pair
        if not _is_number(x) or not _is_number(y):
            raise ValueError(f"point {index} has a non-finite coordinate")
        points.append((float(x), float(y)))
    return points


def _load_points(solution_path: str) -> Any:
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import solution from {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("solve", "generate_points", "run"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn(N_POINTS)

    points = getattr(module, "POINTS", None)
    if points is not None:
        return points

    raise RuntimeError("solution must define solve(n), generate_points(n), run(n), or POINTS")


def _run_solution(solution_path: str) -> tuple[Any, str]:
    with tempfile.TemporaryDirectory(prefix="erdos_unit_distance_") as tmp:
        tmp_path = Path(tmp)
        isolated_solution_path = tmp_path / "solution.py"
        result_path = Path(tmp) / "result.pkl"
        runner_path = Path(tmp) / "runner.py"
        shutil.copy2(solution_path, isolated_solution_path)
        runner_path.write_text(
            """
import importlib.util
import pickle
from pathlib import Path

solution_path = __SOLUTION_PATH__
result_path = Path(__RESULT_PATH__)
n_points = __N_POINTS__


def load_points():
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import solution")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("solve", "generate_points", "run"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn(n_points)

    points = getattr(module, "POINTS", None)
    if points is not None:
        return points

    raise RuntimeError("solution must define solve(n), generate_points(n), run(n), or POINTS")

try:
    points = load_points()
    with result_path.open("wb") as f:
        pickle.dump({"points": points}, f)
except Exception:
    with result_path.open("wb") as f:
        pickle.dump({"error": "solution failed while generating points"}, f)
""".replace("__SOLUTION_PATH__", repr(str(isolated_solution_path)))
            .replace("__RESULT_PATH__", repr(str(result_path)))
            .replace("__N_POINTS__", repr(N_POINTS)),
            encoding="utf-8",
        )
        preexec_fn = _solution_preexec()
        if preexec_fn is not None:
            nobody = pwd.getpwnam("nobody")
            os.chown(tmp, nobody.pw_uid, nobody.pw_gid)
            os.chown(isolated_solution_path, nobody.pw_uid, nobody.pw_gid)
            os.chown(runner_path, nobody.pw_uid, nobody.pw_gid)
        os.chmod(tmp, 0o700 if preexec_fn is not None else 0o755)

        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            preexec_fn=preexec_fn,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"solution runner exited with code {proc.returncode}")
        if not result_path.exists():
            raise RuntimeError("solution did not produce a result")
        with result_path.open("rb") as f:
            payload = pickle.load(f)
        if "error" in payload:
            raise RuntimeError("solution failed while generating points")
        return payload["points"], ""


def _validate_points(points: list[tuple[float, float]]) -> None:
    if len(points) != N_POINTS:
        raise ValueError(f"expected {N_POINTS} points, got {len(points)}")

    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    min_sep2 = MIN_SEPARATION * MIN_SEPARATION
    for index, (x, y) in enumerate(points):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"point {index} has a non-finite coordinate")
        key = (math.floor(x / MIN_SEPARATION), math.floor(y / MIN_SEPARATION))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in buckets.get((key[0] + dx, key[1] + dy), ()):
                    sep2 = (x - px) * (x - px) + (y - py) * (y - py)
                    if sep2 < min_sep2:
                        raise ValueError(
                            f"point {index} is closer than {MIN_SEPARATION:g} to another point"
                        )
        buckets.setdefault(key, []).append((x, y))


def _count_unit_distance_pairs(points: list[tuple[float, float]]) -> int:
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    target2 = UNIT_DISTANCE * UNIT_DISTANCE
    tol = max(DISTANCE_ABS_TOL, DISTANCE_REL_TOL * target2)
    neighbor_radius = math.ceil((UNIT_DISTANCE + tol) / UNIT_DISTANCE) + 1
    unit_pairs = 0

    for x, y in points:
        key = (math.floor(x / UNIT_DISTANCE), math.floor(y / UNIT_DISTANCE))
        for dx in range(-neighbor_radius, neighbor_radius + 1):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                for px, py in buckets.get((key[0] + dx, key[1] + dy), ()):
                    d2 = (x - px) * (x - px) + (y - py) * (y - py)
                    if not math.isfinite(d2):
                        raise ValueError("pairwise distance overflowed")
                    if abs(d2 - target2) <= tol:
                        unit_pairs += 1
        buckets.setdefault(key, []).append((x, y))

    return unit_pairs


def evaluate(solution_path: str) -> tuple[float, float, str]:
    raw_points, _ = _run_solution(solution_path)
    points = _to_points(raw_points)
    _validate_points(points)
    unit_pairs = _count_unit_distance_pairs(points)

    if unit_pairs <= BASELINE_EDGES:
        raw_score = 0.0
    else:
        raw_score = 100.0 * (unit_pairs - BASELINE_EDGES) / unit_pairs
    score = 100.0 * (raw_score / 100.0) ** SCORE_POWER
    score_unbounded = score
    message = (
        f"N={N_POINTS}; unit_pairs={unit_pairs}; unit_distance={UNIT_DISTANCE:.12g}; "
        f"baseline={BASELINE_EDGES}; score_power={SCORE_POWER:.12g}; "
        f"raw_score={raw_score:.6f}; "
        f"score={score:.6f}; score_unbounded={score_unbounded:.6f}"
    )
    return score, score_unbounded, message


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: evaluator.py /path/to/solution.py", file=sys.stderr)
        return 1
    try:
        score, score_unbounded, message = evaluate(argv[1])
        print(message, file=sys.stderr)
        print(f"{score:.12f} {score_unbounded:.12f}")
        return 0
    except subprocess.TimeoutExpired:
        print(f"timed out after {TIMEOUT_SECONDS}s", file=sys.stderr)
        print("0.0 0.0")
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        print("0.0 0.0")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
