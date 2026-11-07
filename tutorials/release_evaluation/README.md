# Audit retained Agent releases after reflective updates

This experiment exercises issue #355 with actual model-generated harness updates and a sealed held-out comparison. It uses Reef's native Agent, a deterministic `apply_swaps` tool, and an explicit model/tool graph with at most three model steps per episode. The tool has no file, shell or network operations. Only prompt rules evolve.

The reusable feature is the retained-release evaluator and its failure-aware, resumable, task-paired reports. The reflective update method is an experimental workload, not a new optimization algorithm or a GEPA reproduction.

## Protocol

The workload is two [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) task families: seven-object state tracking and seven-object logical deduction. Data is downloaded from commit `9ee07bd481feebf959a6b59d61ea57bdcf30964d`; `dataset.json` records source hashes, source indices and split assignments before any provider call. The data authors distribute the repository under the MIT license. Downloaded task bodies are kept in the private run, not vendored here.

With the defaults, each family has 8 training, 4 validation and 12 test cases, selected using seed 355. There are 24 distinct held-out tasks, each repeated twice per retained release. Repeats are grouped by task in the comparison; they are not 48 independent tasks.

1. Freeze an ordinary reasoning prompt and the tool/graph as the baseline release.
2. Execute the first family's training cases; give the model the actual output, score and reference answer. Ask for one reusable rules update. The proposer cannot read validation/test labels through its input.
3. Evaluate current and candidate harnesses on that family's validation cases. Accept only when every case is nondegrading and there are no execution failures; ties are accepted. Rejections do not create a synthetic release.
4. Repeat once for the second family. There are no retries or searches driven by held-out scores.
5. After optimization finishes, evaluate every accepted release plus the initial baseline on the same held-out tasks. Stop after three episodes and resume, checking completed checkpoint bytes are preserved. Run a small changed-suite comparison in a new directory.

The gate intentionally observes only the current family's validation set. Historical measurement can therefore reveal a regression the local gate missed. This example does not change Reef's default gate or introduce an automatic publication policy.

## Run

Use the repository's Python 3.12 development environment from the repository root:

```bash
python -m tutorials.release_evaluation.run \
  --api-key-file /private/path/deepseek-key.txt \
  --output /private/path/new-run-directory
```

The default model is `deepseek-flash` using the official DeepSeek endpoint. The key remains inside a serial loopback transport, not in harness artifacts or subprocess environments. The transport permits at most 320 provider attempts and at most 1,536 output tokens per attempt, with temperature 0 and thinking disabled. Failed attempts consume the request budget. This does not fix currency spend or cap the total input tokens. No external service or third-party credential is required beyond the model provider.

For a plumbing preflight, add `--train-per-family 1 --validation-per-family 1 --test-per-family 1 --repeats 1 --max-calls 30`. A preflight is not a benchmark result.

The experiment starts a **new** private run and uses a local artifact backend. The three-episode stop/resume demonstrates the evaluator's checkpoint contract within the run; restarting the entire optimization driver is not supported. The reusable evaluator can resume against a caller's durable scenario and the same manifest. The driver never replaces a rejected candidate with a manually authored improvement.

## Inspect

- `protocol.json`, `dataset.json`: conditions and preselected data.
- `reflection-*.json`, `gate-*.json`, `phases.json`: actual proposed rules and publication decisions.
- `heldout/{manifest.json,run.json,report.md}`: release/content identities, per-case results, coverage, baseline/previous deltas, paired task bootstrap intervals and observed usage.
- `heldout/episodes/`: normalized execution traces, referenced by result ordinal and verified checksum.
- `changed-suite/`: common re-evaluation under changed suite conditions.
- `provider-usage.json`: every admitted provider attempt and returned counters, including optimization/gating traffic.
- `summary.json`: completed coverage, resume checks and unchanged serving history.

Keep raw runs private. They include task text and model outputs. To share results, export reviewed aggregate data and hashes; do not publish provider credentials, local databases or artifact directories.

## Interpretation

This is a small, single-seed experiment using a mutable provider model alias, with potential public-benchmark pretraining contamination. No accuracy improvement is promised. Task bootstrap intervals are descriptive, not corrected for multiple comparisons. A confidence interval spanning zero does not establish an improvement. Repeated tasks are clustered together; missing or failed evaluations are shown as missing, not zero or free.

Scores test the explicit `FINAL: (X)` answer contract. Reasoning that mentions the right option without a valid final answer is not counted as success. The report measures retained harness behavior; it does not claim a weight-training improvement, autonomous recursive self-improvement, or algorithmic novelty.

The [recorded seed-355 result](results/seed-355/README.md) includes all case outcomes, actual updates, source/data hashes and observed usage.
