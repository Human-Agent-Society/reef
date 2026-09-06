# Terminal-Bench Meta-Harness example

Run full-history harness search through Reef's committed scenario lifecycle.
The campaign measures its baseline, proposes one candidate at a time, and
retains every evaluated candidate. Selection accepts only a strict improvement
on the incumbent's recorded mean score.

[RESULTS.md](RESULTS.md) reports the matched upstream comparison. The
[chosen Reef harness](results/reef_harness.py) and
[recorded scores](results/scores.json) are the result artifacts.

To replay the recorded histories without model calls, check out upstream at
the commit listed in `RESULTS.md`, then run the selector tests in the example's
Python environment with `pytest` installed:

```sh
METAHARNESS_DIR=/path/to/meta-harness/reference_examples/terminal_bench_2 \
  python -m pytest tests/reef_service/test_meta_harness_replay.py
```

## Setup

Use Python 3.12, `uv` and Git LFS. The example's lockfile pins Harbor 0.20.0,
LiteLLM 1.99.0, OpenAI 2.54.0 and E2B 2.46.4. Supply `OPENAI_API_KEY` and
`E2B_API_KEY` through the environment.

```sh
EXAMPLE=recipes/meta_harness/examples/terminal_bench
uv sync --locked --project "$EXAMPLE" --python 3.12.14
uv run --locked --project "$EXAMPLE" \
  python -m recipes.meta_harness.examples.terminal_bench.runtime
uv run --locked --project "$EXAMPLE" \
  python -m recipes.meta_harness.examples.terminal_bench.e2b_runtime \
  --prepare /tmp/tb-runtime.json --verifier-compat tb2-torch-gloo-cleanup-v1
```

Runtime preparation applies a pinned Harbor tool-installation compatibility
fix. The verifier option applies a shared cleanup fix to the distributed Torch
task without changing its correctness assertions. Both are source checked.
Snapshot preparation uploads an explicit source allowlist without credentials,
local environments or experiment outputs.

Executable harnesses run inside an E2B runner and control a second E2B task
sandbox. The recorded comparison used a 4 GiB runner. Prepare that resource
allocation from the runtime snapshot:

```sh
uv run --locked --project "$EXAMPLE" \
  python -m recipes.meta_harness.examples.terminal_bench.runner_resources start \
  --original /tmp/tb-runtime.json --build /tmp/tb-runner-build.json \
  --name reef-meta-harness-4g
uv run --locked --project "$EXAMPLE" \
  python -m recipes.meta_harness.examples.terminal_bench.runner_resources finish \
  --build /tmp/tb-runner-build.json --output /tmp/tb-runtime-4g.json
```

If the build is still running, the finish command reports its status. Run it
again after the build completes. It verifies source and runtime identity,
resource allocation and the original unprivileged runner account.

## Run

```sh
"$EXAMPLE/run.sh" \
  --tasks-file "$EXAMPLE/tasks-hard30.txt" --trials 2 --iterations 4 \
  --concurrency 12 --benchmark-sandbox-limit 64 \
  --target-model gpt-5.6-luna --proposer-model gpt-5.6-sol \
  --proposer-api responses --proposer-effort xhigh \
  --max-observed-cost-usd 100 --max-proposer-cost-usd 30 \
  --e2b-runtime-receipt /tmp/tb-runtime-4g.json --executable-harness \
  --agent-failure-policy upstream-completed-terminal-error-zero-v5 \
  --output-dir /tmp/tb-search
```

`--dry-run` validates and prints the plan without launching model work.
`--stop-after-baseline` ends at the committed baseline boundary. To evaluate
the reproduced harness separately, use a new output directory and add
`--seed-agent "$EXAMPLE/results/reef_harness.py" --stop-after-baseline`.

The cost allowance includes target and proposer usage. Costs come from completed
trial bills and proposer token usage; in-flight work can exceed the allowance.
The campaign retains unknown usage as unknown and stops further admission.
Each active worker consumes two sandbox slots. Capacity checks leave unrelated
sandboxes untouched, and task phase timeouts are retained.

## State and failures

The journal owns the plan, full candidate population, observations, costs and
served composition. `run.json`, `observed-cost.json` and `population.json` are
post-commit mirrors. Restart reads the journal and repairs stale mirrors.
Preparation stages state; a failed commit leaves the prior committed state
unchanged and a retry reuses the prepared result. Compressed journals retain
complete states, validate checksums and refuse appends beyond a torn tail.

Only completed, admissible measurements enter selection. A verifier result
returned after an agent timeout is still a measurement. The optional shared
terminal-loss policy admits an evidenced, completed, billed terminal loss as a
benchmark zero while preserving its raw invalid/null verifier outcome.
Unevidenced infrastructure failures do not become benchmark zeros.

The E2B transport reconnects to the original process. Completion requires a
protected receipt bound to the exact command and inputs; a missing process ID
alone never implies success. Interrupted reservations stop instead of silently
repeating work.

The standalone campaign serves its composition from committed algorithm state.
`terminal_bench.yaml` separately demonstrates the generic service recipe,
whose paired evaluator measures both the proposed and incumbent compositions.
