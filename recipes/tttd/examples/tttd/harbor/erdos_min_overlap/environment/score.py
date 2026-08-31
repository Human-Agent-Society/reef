"""The scoring rule — run by the judge on every submission.

The submitted file is a Python program whose ``run(seed=42, budget_s=1000)``
function returns ``(h_values, c5_bound, n_points)``. The judge runs it in a
subprocess with a hard timeout, verifies the C₅ bound independently, and
returns ``reward = 1 / (ε + C₅)``.
"""

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TIMEOUT_SEC = 1100
REWARD_EPSILON = 1e-8
PROGRAM_CPUS = 2
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


def verify_c5_solution(h_values, c5_achieved, n_points):
    """The reference task verifier, including its normalization semantics."""
    if not isinstance(h_values, np.ndarray):
        h_values = np.array(h_values, dtype=np.float64)

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    target_sum = n_points / 2.0
    current_sum = np.sum(h_values)
    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(
                f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]"
            )

    dx = 2.0 / n_points
    correlation = np.correlate(h_values, 1.0 - h_values, mode="full") * dx
    computed_c5 = np.max(correlation)
    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    return float(computed_c5)


def grade(artifact):
    """Run the submitted program and return ``{"reward": float, "reason": str}``."""
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
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.path.insert(0, '{tmp}'); from solution import run; import json; print(json.dumps(run(seed=42, budget_s=1000)))",
                ],
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

    import json

    try:
        result = json.loads(run.stdout.strip().splitlines()[-1])
        h_values, c5_bound, n_points = result
    except Exception as e:
        return {"reward": 0.0, "reason": f"cannot parse result: {e}"}

    try:
        c5 = verify_c5_solution(h_values, c5_bound, n_points)
    except Exception as e:
        return {"reward": 0.0, "reason": f"verification failed: {e}"}

    if c5 <= 0 or math.isnan(c5) or math.isinf(c5):
        return {"reward": 0.0, "reason": "C5 must be positive and finite"}

    reward = 1.0 / (REWARD_EPSILON + c5)
    return {"reward": reward, "reason": f"C5 = {c5:.6f}, reward = {reward:.6f}"}
