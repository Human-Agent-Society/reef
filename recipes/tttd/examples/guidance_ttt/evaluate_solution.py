"""Evaluate a released solution without starting models or training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent


def load_solution(task: str, solution_dir: Path) -> str:
    """Read the program listed for a task in the manifest."""
    manifest = json.loads((solution_dir / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["solutions"][task]
    return (solution_dir / record["file"]).read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("polyomino_packing", "lasso_path", "ahc058", "trimul"))
    parser.add_argument("--judge-url", required=True, help="URL of the task judge, such as http://127.0.0.1:8082")
    parser.add_argument("--repeats", type=int, default=1, help="Number of sequential fixed-program evaluations")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.timeout_seconds < float("inf"):
        parser.error("repeats and timeout-seconds must be positive; timeout-seconds must be finite")
    source = load_solution(args.task, EXAMPLE_DIR / "solutions")

    from harness.scorer import JudgeScorer, JudgeUnavailableError

    contract = json.loads((EXAMPLE_DIR / "harbor" / args.task / "contract.json").read_text(encoding="utf-8"))
    scorer = JudgeScorer(
        args.judge_url,
        problem_id=contract["judge_problem_id"],
        language=contract["solution_language"],
        timeout_s=args.timeout_seconds,
    )
    all_valid = True
    for repeat in range(1, args.repeats + 1):
        try:
            result = scorer.evaluate(source)
        except JudgeUnavailableError as exc:
            print(f"Judge infrastructure error: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "task": args.task,
                    "repeat": repeat,
                    "raw_score_label": contract["raw_score_label"],
                    "score_direction": contract["score_direction"],
                    "result": asdict(result),
                }
            ),
            flush=True,
        )
        all_valid = all_valid and result.valid
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
