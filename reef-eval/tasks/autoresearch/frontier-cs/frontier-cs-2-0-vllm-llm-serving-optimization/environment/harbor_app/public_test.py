#!/usr/bin/env python3
"""Async public-test client for the vLLM serving optimization task.

Deploys the current `/app/vllm` working tree to a Modal H100, runs the public
instance subset (a strict subset of the final eval set), and reports per-instance
and aggregate end-to-end latency plus an accuracy signal versus the baseline.
The returned feedback is the same kind the judge uses — never just a compile/OK
flag — so it can drive the optimization loop.

Usage:
    python3 /app/public_test.py launch        # start async run -> prints run id
    python3 /app/public_test.py status <id>   # poll for the result
    python3 /app/public_test.py run           # run synchronously
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

SERVING_EVAL_ROOT = "/opt"
TASK_CONFIG_PATH = Path("/app/task_config.json")
RESULTS_DIR = Path("/app/.public_test")
VLLM_SRC = os.environ.get("VLLM_DIR", "/app/vllm")
DEFAULT_BASELINE_CACHE = "/opt/vllm-baseline/baseline_metrics.json"


def load_eval_config() -> dict:
    try:
        payload = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    evaluation = payload.get("evaluation") if isinstance(payload, dict) else {}
    return evaluation if isinstance(evaluation, dict) else {}


def baseline_cache_path(config: dict) -> str:
    return str(config.get("baseline_cache_path", DEFAULT_BASELINE_CACHE))


def run_sync() -> dict:
    if SERVING_EVAL_ROOT not in sys.path:
        sys.path.insert(0, SERVING_EVAL_ROOT)
    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        return {"ok": False, "error": "Modal credentials are not configured (MODAL_TOKEN_ID/SECRET)"}
    try:
        import serving_eval  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"serving harness unavailable: {type(exc).__name__}"}

    config = load_eval_config()
    try:
        return serving_eval.run_public_test(
            src=VLLM_SRC,
            config=config,
            baseline_cache_path=baseline_cache_path(config),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"public test failed: {type(exc).__name__}"}


def cmd_run() -> int:
    print(json.dumps(run_sync(), indent=2, sort_keys=True))
    return 0


def cmd_launch() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_worker", run_id],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps({"status": "launched", "run_id": run_id}, indent=2))
    print(f"poll with: bash /app/public_test.sh status {run_id}")
    return 0


def cmd_worker(run_id: str) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_id}.json"
    try:
        result = run_sync()
        out.write_text(json.dumps({"status": "done", "result": result}), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        out.write_text(json.dumps({"status": "error", "error": type(exc).__name__}), encoding="utf-8")
    return 0


def cmd_status(run_id: str) -> int:
    out = RESULTS_DIR / f"{run_id}.json"
    if not out.exists():
        print(json.dumps({"status": "unknown", "run_id": run_id}, indent=2))
        return 0
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
    except Exception:
        payload = {"status": "running", "run_id": run_id}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    command = argv[1]
    if command == "run":
        return cmd_run()
    if command == "launch":
        return cmd_launch()
    if command == "status":
        if len(argv) < 3:
            print("Usage: public_test.py status <run_id>", file=sys.stderr)
            return 2
        return cmd_status(argv[2])
    if command == "_worker":
        if len(argv) < 3:
            return 2
        return cmd_worker(argv[2])
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
