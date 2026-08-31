# circle-packing

Pack 3 circles in the unit square, maximize the sum of radii.

| | |
|---|---|
| **Oracle (baseline)** | 0.75 (greedy) |
| **Optimum** | 1.007626 (known optimum) |
| **Run it** | `await lab.run("tasks/autoresearch/first-party/circle-packing", {"name": "oracle"})` |
| **Verify standalone** | `harbor trial start -p tasks/autoresearch/first-party/circle-packing` |

**What this task teaches:** The reference task: the full autoresearch protocol in its simplest form. Grader uses exact rational arithmetic; a 5e-9 overlap scores zero.

