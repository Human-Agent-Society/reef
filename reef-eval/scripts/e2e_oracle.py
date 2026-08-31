"""The E2E gate: run the oracle agent through real containers on every task
that declares ``solve_sh_scores`` in its grader_tests.json, and require each
score to match. The expectation is a number for an exact match, or
``{"min", "max"}`` where the live run legitimately varies (compression
ratios differ across zlib builds, for instance).

    python scripts/e2e_oracle.py                 # all tasks with grader_tests.json
    python scripts/e2e_oracle.py circle-packing  # a subset, by name
"""

import asyncio
import json
import sys
from pathlib import Path

from reef_eval import Lab

TASKS_ROOT = Path(__file__).parent.parent / "tasks"


def discover(names: list[str]) -> list[Path]:
    # The template carries its own grader_tests.json, so this glob already
    # includes it. It is a working task and is kept working.
    tasks = sorted(
        {p.parent.parent for p in TASKS_ROOT.glob("**/tests/grader_tests.json")},
        key=lambda p: p.name,
    )
    return [t for t in tasks if t.name in names] if names else tasks


async def main(names: list[str]) -> None:
    tasks = discover(names)
    if not tasks:
        known = [t.name for t in discover([])]
        sys.exit(f"no tasks matched {names!r}; known tasks: {known}")

    lab = Lab("runs/e2e-oracle")
    failures = []
    for task_dir in tasks:
        expect = json.loads((task_dir / "tests" / "grader_tests.json").read_text())[
            "solve_sh_scores"
        ]
        row = await lab.run(str(task_dir), {"name": "oracle"}, tags={"gate": "e2e"})
        reward = row.rewards.get("reward")
        if isinstance(expect, dict):
            ok = reward is not None and expect["min"] <= reward <= expect["max"]
            expected = f"[{expect['min']}, {expect['max']}]"
        else:
            ok = reward is not None and abs(reward - expect) <= 1e-6
            expected = f"{expect} ± 1e-6"
        status = "OK " if ok else "FAIL"
        print(f"{status} {task_dir.name}: reward={reward} expected {expected}")
        if not ok:
            failures.append(task_dir.name)

    if failures:
        print(f"\nE2E FAILED for: {failures}")
        sys.exit(1)
    print("\nE2E oracle gate passed for all tasks")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
