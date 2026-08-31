## Treant's Forest - Maze Obstruction Strategy (AHC054)

Write `solution.py` in the project root that reads from stdin and writes to stdout.

---

## Problem Overview

Read `README.md` and `tools/README.md` for full problem details. A baseline `solution.py` already exists (it produces syntactically valid but low-quality output). Your job is to improve it.

---

## Evaluation

Your solution is scored on **50 fixed test cases**. Final score = sum of individual case scores. **Higher is better.**

---

## Local Testing

Generate local random tests with `./tools/bin/gen <seed>`, using seeds in the range **0..10000** only.

```bash
# Generate a random test case (seed-based, deterministic)
./tools/bin/gen 0 > input.txt

# Run your solution
python3 solution.py < input.txt > output.txt

# Score output (Higher is better)
./tools/bin/tester input.txt output.txt
# Outputs to stderr: Score = <N>
```

---

## Rules

- Write your solution as `solution.py` in the project root directory
- Do NOT modify files in `tools/`
- Use `tools/bin/gen` and `tools/bin/tester` for local testing
- For local scoring, use only `./tools/bin/tester`; do not use `tools/src/verifier.py` for scores
- Your program should read from stdin and write to stdout
- Run your solution to completion and verify with the tester before finishing
