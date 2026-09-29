"""Bridge the official H100 evaluator's JSON endpoint to the common judge protocol."""

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

if __package__:
    from .protocol import Judge, serve
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import Judge, serve


class TriMulJudge(Judge):
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def __call__(self, pid: str, language: str, code: str) -> dict:
        if pid != "trimul" or language != "python":
            return {"valid": False, "score": 0, "message": "expected trimul and Python"}
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("TRIMUL_JUDGE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"solution": code, "runner_timeout_s": 1100}).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1150) as response:
            payload = json.loads(response.read())
        return score_report(payload)


def score_report(payload: dict) -> dict:
    report = payload["report"]
    if report.get("all_correct") is not True:
        return {"valid": False, "score": 0, "message": report.get("error") or "correctness check failed"}
    latency = float(report["score_us"])
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("H100 evaluator returned an invalid latency")
    if report["test_count"] != 18 or report["benchmark_count"] != 7:
        raise ValueError("H100 evaluator must run 18 correctness tests and seven benchmarks")
    return {
        "valid": True,
        "score": 1500.0 / latency,
        "score_unbounded": latency,
        "message": f"H100 geometric-mean runtime: {latency:g} microseconds",
        "artifacts": {"benchmarks": report["benchmarks"], "reward_formula": "1500 / runtime_us"},
    }


class ModalTriMulJudge(Judge):
    def __init__(self, app_name: str) -> None:
        import modal

        self.evaluator = modal.Function.from_name(app_name, "evaluate")

    def __call__(self, pid: str, language: str, code: str) -> dict:
        if pid != "trimul" or language != "python":
            return {"valid": False, "score": 0, "message": "expected trimul and Python"}
        return score_report(self.evaluator.remote(code))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("TRIMUL_JUDGE_URL"))
    parser.add_argument("--modal-app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    if args.modal_app:
        judge = ModalTriMulJudge(args.modal_app)
    elif args.endpoint:
        judge = TriMulJudge(args.endpoint)
    else:
        parser.error("set --modal-app or --endpoint to the official H100 evaluator")
    serve(judge, host=args.host, port=args.port, max_workers=1)


if __name__ == "__main__":
    main()
