# SDPO on an arithmetic grid

The smallest complete run of the `sdpo` recipe (`recipes/sdpo/`): a synchronous
sampling grid, a teacher that reads a successful sibling or the environment's
feedback, one optimizer step per grid, and a new weight release before the next
grid is sampled.

The task is arithmetic with a formatting instruction, so the protocol is visible
without a dataset: a wrong answer earns the feedback the teacher reads, and a
right one becomes its siblings' demonstration. The grid is two questions by two
attempts, far below the paper's 32 by 8.

**This example shows the protocol and the training cycle. It is not a benchmark
result and must not be reported as one.** SDPO's paper reproduction, SciKnowEval
Chemistry on OLMo-3-7B-Instruct, is [Reef #428](https://github.com/Human-Agent-Society/reef/issues/428);
its example and its recorded curve land with that work.

```text
run.sh           starts the stack, runs the campaign against it, stops the stack
run.py           the reef-client campaign: samples each grid, reports every rollout, waits for the release
serve.yaml       the training stack: Qwen3-0.6B on four GPUs, the sdpo recipe and its Slime loss family
```

## The protocol

Each step samples every question of the grid `--rollouts-per-group` times from
one policy release, then reports each rollout with its coordinates and score:

```python
client.report(scenario, {
    "references": [receipt],
    "score": score,
    "metadata": {"step": step, "group": question, "rollout": attempt, "teacher_context": feedback},
})
```

The processor holds the step until every coordinate has arrived, then builds the
whole grid as one batch: each rollout's teacher reads the original question plus
the first successful sibling's response, or the feedback when the question had no
success. A rollout whose teacher reads neither keeps the plain request and a
sample weight of 0, so the step still takes its optimizer step. A step sampled
from more than one release, or with different requests inside one question group,
is discarded whole and listed in the processor's `failed_steps` status.

`max-staleness: 0` is what makes the grid on-policy: the campaign waits for each
step's release before sampling the next.

## Run it

In a configured Reef GPU environment with four GPUs, download the model at a
fixed revision, then run from this directory:

```bash
hf download Qwen/Qwen3-0.6B --local-dir /tmp/models/Qwen3-0.6B
./run.sh
```

`run.sh` mints a token per run directory, waits for `/healthz`, and runs the
campaign in an ephemeral `uv` environment with `reef-client` alone: no Harbor
and no reef-eval. State goes to `$RUN_DIR` (default `./work`); use a fresh one
per run, because the campaign refuses a scenario that has already trained.
`SDPO_STEPS` sets the number of grids.

Each step's record holds the receipts, the scores, the sampling and training
times, and the release the step published.
