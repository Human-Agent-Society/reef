# SDPO on an arithmetic grid

The smallest complete run of the `sdpo` recipe (`recipes/sdpo/`): a synchronous
sampling grid, a teacher that reads a successful sibling or the environment's
feedback, one optimizer step per grid, and a new weight release before the next
grid is sampled. One Harbor task, run through
[reef-eval](https://github.com/Human-Agent-Society/reef-eval), is one training
run.

The task is arithmetic with a formatting instruction, so the protocol is visible
without a dataset: a wrong answer earns the feedback the teacher reads, and a
right one becomes its siblings' demonstration. The grid is two questions by two
attempts, far below the paper's 32 by 8.

**This example shows the protocol and the training cycle. It is not a benchmark
result and must not be reported as one.** SDPO's paper reproduction, SciKnowEval
Chemistry on OLMo-3-7B-Instruct, is [Reef #428](https://github.com/Human-Agent-Society/reef/issues/428);
its example and its recorded learning curve land with that work.

```text
run.sh                 starts the Reef stack, then runs the episode
run.py                 reef-eval runs the harbor/ task under harness:HarborAgent
harness/agent.py       runs the grid runner in the task container; makes no model calls itself
harbor/
  instruction.md       what the task trains and how
  task.toml            timeouts and resource limits
  environment/
    Dockerfile         the runner's image: python plus reef-client
    grid.py            the runner: samples each grid, reports every rollout, waits for the release
  tests/grade.py       the verifier: the last grid's accuracy from the runner's record
serve.yaml             the training stack: Qwen3-0.6B on four GPUs, the sdpo recipe and its loss family
```

## The protocol

Each step samples every question of the grid `SDPO_ROLLOUTS_PER_GROUP` times
from one policy release, then reports each rollout with its coordinates and
score:

```python
client.report(
    scenario,
    {
        "references": [receipt],
        "score": score,
        "metadata": {"step": step, "group": question, "rollout": attempt, "teacher_context": feedback},
    },
)
```

The processor holds the step until every coordinate has arrived, then builds the
whole grid as one batch: each rollout's teacher reads the original question plus
the first successful sibling's response, or the feedback when the question had no
success. A rollout whose teacher reads neither keeps the plain request and a
sample weight of 0, so the step still takes its optimizer step. A step sampled
from more than one release, or with different requests inside one question group,
is discarded whole and listed in the processor's `failed_steps` status.

`max-staleness: 0` is what makes the grid on-policy: the runner waits for each
step's release before sampling the next.

The runner reports every rollout as training feedback, so the harness posts no
report of its own. The Harbor reward is the last grid's accuracy and stays in the
trial result, as an evaluation rather than a training signal.

## Run it

In a configured Reef GPU environment with four GPUs and Docker, download the
model at a fixed revision, then run from this directory:

```bash
hf download Qwen/Qwen3-0.6B --local-dir /tmp/models/Qwen3-0.6B
./run.sh
```

`run.sh` mints a token per run directory, waits for `/healthz`, and runs the
episode in an ephemeral `uv` environment. State goes to `$RUN_DIR` (default
`./work`); the trial's rows, the runner's log and the Reef log all land there.

The runner's settings come from the host environment and the harness forwards
them into the container:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `SDPO_STEPS` | 2 | sampling grids, one optimizer step each |
| `SDPO_GROUPS_PER_STEP` | 2 | questions per grid; MUST equal `serve.yaml` |
| `SDPO_ROLLOUTS_PER_GROUP` | 2 | attempts per question; MUST equal `serve.yaml` |
| `SDPO_SEED` | 42 | draws the questions |
| `SDPO_TRAIN_TIMEOUT_S` | 1800 | how long a step waits for its weight release |
