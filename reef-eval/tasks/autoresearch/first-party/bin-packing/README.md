# bin-packing

Pack 60 items into capacity-100 bins; reward = first-fit bins / yours.

| | |
|---|---|
| **Oracle (baseline)** | 1.0 (first-fit) |
| **Optimum** | > 1.0 (each bin saved) |
| **Run it** | `await lab.run("tasks/autoresearch/first-party/bin-packing", {"name": "oracle"})` |
| **Verify standalone** | `harbor trial start -p tasks/autoresearch/first-party/bin-packing` |

**What this task teaches:** Exact constraint checking: every item exactly once, no overfull bins; one violation scores zero.

