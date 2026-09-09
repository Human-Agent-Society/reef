#!/usr/bin/env python3
"""Assert the smoke run's bundle proves the loop closed. Usage: check_bundle.py <bundle.json>"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    bundle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    problems: list[str] = []

    if bundle["attempt_count"] < 1:
        problems.append("no attempts were reported")
    scored = [s for s in bundle["score_over_time"] if s["score"] is not None]
    if not scored:
        problems.append("no attempt carried a score")
    if bundle["best_attempt"] is None:
        problems.append("no best attempt")
    referenced = [a for a in bundle["attempts"] if a["inference_record_ids"]]
    if not referenced:
        problems.append("no attempt resolved inference references (receipts did not flow)")
    if bundle["token_accounting"]["inference_calls"] < 1:
        problems.append("journal captured no calls")
    agents = {s["agent_id"] for s in bundle["score_over_time"]}
    if len(agents) < 2:
        problems.append(f"expected attempts from 2 agents, saw {sorted(agents)}")

    if problems:
        print("bundle check FAILED:")
        for p in problems:
            print(f"  - {p}")
        print(json.dumps(bundle, indent=2)[:2000])
        return 1
    print(
        f"bundle check OK: {bundle['attempt_count']} attempts, "
        f"{len(referenced)} with references, best={bundle['best_attempt']['score']}, "
        f"{bundle['token_accounting']['inference_calls']} journaled calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
