## Ad Placement Optimization

You need to place rectangular ads for n companies on a 10000x10000 grid.

Each company i wants an ad space containing point (x_i+0.5, y_i+0.5) with area as close to r_i as possible. Maximize the total satisfaction.

Full problem description, constraints, scoring formula, and input/output format are in `README.md`.

## Local Testing Tools

`tools/` provides an input generator and scoring program. The generator reads a seed file and writes generated cases into a directory; it does not accept a raw seed directly.

```bash
# Generate one test input
printf '0\n' > /tmp/seeds.txt
rm -rf /tmp/ad_cases && mkdir -p /tmp/ad_cases
./tools/bin/gen /tmp/seeds.txt -d /tmp/ad_cases
cp /tmp/ad_cases/0000.txt input.txt

# Run your solution
./my_solution < input.txt > output.txt

# Score it
./tools/bin/tester input.txt output.txt
# stderr: Score = <number>
```

For multiple local cases, put one seed per line in the seed file, for example `seq 0 9 > /tmp/seeds.txt`, then run `./tools/bin/gen /tmp/seeds.txt -d /tmp/ad_cases`.

You can generate unlimited test data with any seed value. Use this extensively for local testing and optimization.

## Compilation

Recommended: C++17 with `g++ -std=c++17 -O2`.

Time limit: 5 seconds per test case. Memory limit: 1 GB. No GPU.

## Rules

- Write your solution as a single C++ file in the project root directory
- Do NOT modify files in `tools/`
- Use `tools/bin/gen` with a seed file and `tools/bin/tester` for local testing
- Your program should read from stdin and write to stdout
- Run your solution to completion and verify with the tester before finishing