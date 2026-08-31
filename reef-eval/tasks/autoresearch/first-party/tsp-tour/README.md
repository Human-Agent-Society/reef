# tsp-tour

Shortest closed tour over 40 fixed cities; reward = identity length / yours.

| | |
|---|---|
| **Oracle (baseline)** | 1.0 (identity tour) |
| **Optimum** | ~2.0 (strong heuristics) |
| **Run it** | `await lab.run("tasks/autoresearch/first-party/tsp-tour", {"name": "oracle"})` |
| **Verify standalone** | `harbor trial start -p tasks/autoresearch/first-party/tsp-tour` |

**What this task teaches:** Classic combinatorial search with a continuous improvement signal: 2-opt and beyond.

