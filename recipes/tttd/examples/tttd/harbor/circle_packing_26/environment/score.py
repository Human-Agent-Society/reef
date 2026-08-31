"""Official-style verifier and reward for the 26-circle packing task."""

# The validator source is preserved verbatim from TTT-Discover.
# ruff: noqa: RET505

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

NUM_CIRCLES = 26
TIMEOUT_SEC = 530
PROGRAM_CPUS = 1
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _program_environment():
    environment = os.environ.copy()
    for name in _THREAD_ENV_VARS:
        environment[name] = str(PROGRAM_CPUS)
    return environment

_RUNNER = """
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from solution import run_packing

centers, radii, _reported_sum = run_packing()
print(json.dumps([np.asarray(centers).tolist(), np.asarray(radii).tolist()]))
"""


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def check_packing_correctness(centers, radii, num_circles: int) -> bool:
    """Check the task-specific shapes before running the reference validator."""
    shape_valid = centers.shape == (num_circles, 2) and radii.shape == (num_circles,)
    if not shape_valid:
        return False
    return validate_packing(centers, radii)


def grade(artifact):
    """Run ``run_packing()``, verify its geometry, and reward sum of radii."""
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution file"}

    source = Path(artifact).read_text()
    if not source.strip():
        return {"reward": 0.0, "reason": "empty solution"}

    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "solution.py"
        program.write_text(source)
        try:
            run = subprocess.run(
                [sys.executable, "-c", _RUNNER, tmp],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
                cwd=tmp,
                env=_program_environment(),
            )
        except subprocess.TimeoutExpired:
            return {"reward": 0.0, "reason": f"program exceeded {TIMEOUT_SEC}s"}

    if run.returncode != 0:
        return {"reward": 0.0, "reason": f"program failed: {run.stderr[-200:]}"}

    try:
        centers_data, radii_data = json.loads(run.stdout.strip().splitlines()[-1])
        centers = np.asarray(centers_data, dtype=np.float64)
        radii = np.asarray(radii_data, dtype=np.float64)
    except Exception as exc:
        return {"reward": 0.0, "reason": f"cannot parse result: {exc}"}

    try:
        valid = check_packing_correctness(centers, radii, NUM_CIRCLES)
    except Exception as exc:
        return {"reward": 0.0, "reason": f"verification failed: {exc}"}
    if not valid:
        return {"reward": 0.0, "reason": "packing is not valid"}

    reward = float(np.sum(radii))
    if not math.isfinite(reward):
        return {"reward": 0.0, "reason": "sum of radii must be finite"}
    return {"reward": reward, "reason": f"sum of radii = {reward:.12f}"}
