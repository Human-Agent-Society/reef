## Wireless Electricity - Wire Placement Optimization (CodeChef WIREL)

Write `solution.cpp` in the project root. It must read from stdin and write to stdout.

---

## Problem Overview

Chefland lies on a 2D plane with N=10,000 cities and M=2,500 wire segments.

**Poles:**
- Positive plate: (-1,0) to (0,-1)
- Negative plate: (1000001,1000000) to (1000000,1000001)

A wire is 'positive' if it intersects the positive plate or another positive wire (transitively). Similarly for 'negative'. No wire may be both.

**Costs:**
- For each city i: transmission cost = Si^2 + Ti^2, where Si=min dist to any positive wire endpoint, Ti=min dist to any negative wire endpoint
- For each wire i: movement cost = hi^2 + vi^2, where (hi,vi) is your chosen translation
- Total score = sum(transmission costs) + sum(movement costs). **Minimize.**

**Constraints:**
- N=10000, M=2500
- City coords: random in [0, 1000000]
- Wire lengths: L in {8000,10000,12000,14000,16000}
- |hi|, |vi| <= 1000000
- Must have >= 1 positive and >= 1 negative wire, no short circuits

---

## Input Format

```
N M
X_1 Y_1
... (N city lines)
A_1 B_1 C_1 D_1
... (M wire lines, endpoints (A,B)-(C,D))
```

## Output Format

M lines, each: `h_i v_i` (integer translation for wire i)

---

## Evaluation

Your solution is scored on **50 fixed hidden test cases**. Final score = sum of (P+Q) across all hidden cases. **Lower is better.**

Local generated cases are for validation only and are not the judge cases. Do not assume any local seed range corresponds to the judge cases.

---

## Local Testing

```bash
# Generate a random test case
./tools/bin/gen <seed> > input.txt

# Compile your solution
g++ -std=c++17 -O2 -o solution solution.cpp

# Run your solution
./solution < input.txt > output.txt

# Score output (lower is better)
./tools/bin/tester input.txt output.txt
# Outputs to stderr: Score = <N>
```

A valid baseline `solution.cpp` already exists. Improve it.

---

## Strategy Hints

- Use the provided baseline as a validity sanity check before optimizing.
- Reduce transmission cost by moving wires near city clusters.
- Carefully choose which wires become positive/negative vs neutral.
- Read `README.md` and `tools/README.md` for full problem details.
- Iterate: test locally with `gen` + `tester`, then submit.

## Rules

- Write your solution as `solution.cpp` in the project root directory
- Do NOT modify files in `tools/`
- Use `tools/bin/gen` and `tools/bin/tester` for local testing
- Your program should read from stdin and write to stdout
- Run your solution to completion and verify with the tester before finishing