## Warehouse Manager - Forklift Operations Optimization (CodeChef WAREHOUS)

Write `solution.py` in the project root that reads from stdin and writes to stdout.

---

## Problem Overview

Chef's warehouse is an R x C grid. A forklift starts at the northwest corner (0,0). R*C-1 goods arrive one by one in a given order, must be loaded and placed in the warehouse, then later dispatched from the entrance in order 1, 2, ..., R*C-1.

The forklift can:
- Move: N/S/E/W
- Pick up (P) arriving goods at entrance
- Dispatch (D) goods at entrance
- Load (Lx) from adjacent cell in direction x
- Unload (Ux) to adjacent cell in direction x

A forklift carrying a good cannot enter an occupied cell. Minimize the total instruction string length.

---

## Constraints

- T=5 test cases per input
- 6 <= R, C <= 20
- Output string length <= 500,000 per test case

## Scoring Formula

Per test case: (S + 2) / (R + C - 1) - 2*R*C + 20, where S = instruction string length. Score is averaged over T cases. **Minimize.**

---

## Input Format

```
T
R_1 C_1
a_1 a_2 ... a_{R1*C1-1}
R_2 C_2
a_1 a_2 ... a_{R2*C2-1}
...
```

## Output Format

One line per test case: the forklift command string.

Commands: N, S, E, W (move), P (pick up arrival), D (dispatch at entrance), Lx (load from direction x), Ux (unload to direction x).

---

## Runtime Limits

- Time limit: 30 seconds per test case
- Memory limit: 1 GB
- No GPU

---

## Local Testing

Use `./tools/bin/gen <seed>` for local testing with seeds in the range **0..10000** only.

```bash
./tools/bin/gen 1 > input.txt
python3 solution.py < input.txt > output.txt
./tools/bin/tester input.txt output.txt
# stderr: Score = <number>
```

A baseline `solution.py` exists. Improve it.

---

## Strategy Hints

- Plan storage positions to minimize movement during both placement and retrieval.
- Consider placing goods that will be dispatched first (low IDs) near the entrance.
- Use BFS for pathfinding in the grid.
- The arrival order is random; you may need to temporarily rearrange goods.
- Read `README.md` and `tools/README.md` for full problem details.

## Rules

- Write your solution as `solution.py` in the project root directory
- Do NOT modify files in `tools/`
- Use `tools/bin/gen` and `tools/bin/tester` for local testing
- Your program should read from stdin and write to stdout
- Run your solution to completion and verify with the tester before finishing