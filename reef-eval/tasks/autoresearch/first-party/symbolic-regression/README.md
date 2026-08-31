# symbolic-regression

Recover a hidden formula from samples; reward = 1/(1+RMSE) on HELD-OUT points.

| | |
|---|---|
| **Oracle (baseline)** | 0.610 (plain quadratic) |
| **Optimum** | 1.0 (the true formula) |
| **Run it** | `await lab.run("tasks/autoresearch/first-party/symbolic-regression", {"name": "oracle"})` |
| **Verify standalone** | `harbor trial start -p tasks/autoresearch/first-party/symbolic-regression` |

**What this task teaches:** The final judge: the session judge scores training points, and a final judge scores the best submission once on held-out points the agent never saw; no submission budget can probe them. Expressions run through an AST allowlist, never eval().

